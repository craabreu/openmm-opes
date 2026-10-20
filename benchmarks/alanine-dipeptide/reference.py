"""
Ground-truth free energy surface of alanine dipeptide from a static bias.

The bias is the converged well-tempered metadynamics bias, V = -(1 - 1/gamma)
F_metad, built from the probability average of the final surfaces of long
metadynamics runs. Plain simulations under this fixed bias sample both basins
often, and since the bias never changes, reweighting each sample by
exp(V / kT) is exact. Many independent runs give the estimate and its spread.

Usage:
    python reference.py run METAD_DIR SEED OUTFILE [--ns 20]
    python reference.py kde RUNS_DIR OUTFILE [--kappa 800]
    python reference.py table NPZ OUTFILE
    python reference.py plot NPZ OUTFILE

``run`` simulates under the static bias built from METAD_DIR/metad_*.npz and
saves phi, psi and the bias energy felt every ps. ``kde`` reweights all
samples in RUNS_DIR and estimates F(phi, psi) on the shared grid with product
von Mises kernels of concentrations kappa and 2 kappa (1/rad^2; kappa = 800
means a width of about 0.035 rad), extrapolated to zero kernel width, with
jackknife standard errors over runs, and dF between the phi > 0 and phi < 0
halves straight from the weighted samples. ``table`` writes that surface as a
plain-text table with a header describing the system and the method, and
``plot`` draws it with its standard error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openmm
from analyze import KCAL, KT, combine, interior
from openmm import app, unit
from run import BIAS_FACTOR, GRID, PDB, PHI, PSI, TEMPERATURE, TIMESTEP, torsion

RECORD = 250  # steps between samples: 1 ps


def static_bias(metad_dir):
    """The averaged converged metadynamics bias on the full periodic grid, kJ/mol."""
    surfaces = [
        interior(np.load(path)["fes"])[-1]
        for path in sorted(Path(metad_dir).glob("metad_*.npz"))
    ]
    fes = combine(surfaces)  # psi along rows, phi along columns, without pi
    full = np.empty((GRID, GRID))
    full[:-1, :-1] = fes
    full[-1, :-1] = fes[0]  # periodic copies at psi = pi and phi = pi
    full[:, -1] = full[:, 0]
    return -(1 - 1 / BIAS_FACTOR) * full, len(surfaces)


def simulate(metad_dir, seed, outfile, ns):
    bias, _ = static_bias(metad_dir)
    pdb = app.PDBFile(str(PDB))
    forcefield = app.ForceField("amber14-all.xml")
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        hydrogenMass=4 * unit.amu,
    )
    force = openmm.CustomCVForce("table(phi, psi)")
    force.addCollectiveVariable("phi", torsion(PHI))
    force.addCollectiveVariable("psi", torsion(PSI))
    # Continuous2DFunction takes x (phi) as the fastest index: row-major (psi, phi).
    force.addTabulatedFunction(
        "table",
        openmm.Continuous2DFunction(
            GRID, GRID, bias.ravel(), -np.pi, np.pi, -np.pi, np.pi, True
        ),
    )
    force.setForceGroup(1)
    system.addForce(force)
    integrator = openmm.LangevinMiddleIntegrator(
        TEMPERATURE, 1 / unit.picosecond, TIMESTEP * unit.picoseconds
    )
    integrator.setRandomNumberSeed(seed)
    simulation = app.Simulation(
        pdb.topology, system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(TEMPERATURE, seed)
    simulation.step(25_000)  # 100 ps of equilibration under the bias
    samples = round(ns * 1000 / (RECORD * TIMESTEP))
    data = np.empty((samples, 3))
    for k in range(samples):
        simulation.step(RECORD)
        phi, psi = force.getCollectiveVariableValues(simulation.context)
        energy = simulation.context.getState(getEnergy=True, groups={1})
        data[k] = (
            phi,
            psi,
            energy.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
        )
    np.save(outfile, data)


def von_mises_density(samples, weights, grid, kappa, chunk=20_000):
    """Weighted product von Mises density on grid, rows psi, columns phi."""
    density = np.zeros((len(grid), len(grid)))
    for start in range(0, len(samples), chunk):
        phi, psi = samples[start : start + chunk, 0], samples[start : start + chunk, 1]
        w = weights[start : start + chunk]
        kernel_phi = np.exp(kappa * (np.cos(grid[:, None] - phi[None, :]) - 1))
        kernel_psi = np.exp(kappa * (np.cos(grid[:, None] - psi[None, :]) - 1))
        density += (kernel_psi * w[None, :]) @ kernel_phi.T
    return density


def normalized_fes(density):
    """Free energy of a density on the grid, kJ/mol, normalized over the grid."""
    return -KT * np.log(density / density.sum())


def estimate(runs_dir, outfile, kappa):
    """Reweight all runs and extrapolate the surface to zero kernel width.

    The smoothing bias of a kernel estimate grows as the kernel variance,
    1/kappa, so 2 F(2 kappa) - F(kappa) cancels it to leading order
    (Richardson extrapolation). The standard error of each grid value comes
    from a jackknife over the runs. Basin dF is also computed straight from
    the weighted samples, with no kernel at all.
    """
    grid = np.linspace(-np.pi, np.pi, GRID)[:-1]
    paths = sorted(Path(runs_dir).glob("*.npy"))
    runs = [np.load(path) for path in paths]
    shift = max(run[:, 2].max() for run in runs)
    perRun = {kappa: [], 2 * kappa: []}
    sampleDeltaF = []
    for run in runs:
        weights = np.exp((run[:, 2] - shift) / KT)
        for k in perRun:
            perRun[k].append(von_mises_density(run[:, :2], weights, grid, k))
        positive = run[:, 0] > 0
        sampleDeltaF.append(
            -KT * np.log(weights[positive].sum() / weights[~positive].sum())
        )
    perRun = {k: np.array(v) for k, v in perRun.items()}

    def extrapolated(sums):
        return 2 * normalized_fes(sums[2 * kappa]) - normalized_fes(sums[kappa])

    totals = {k: v.sum(axis=0) for k, v in perRun.items()}
    fes = extrapolated(totals)
    n = len(runs)
    jackknife = np.array(
        [extrapolated({k: totals[k] - perRun[k][i] for k in totals}) for i in range(n)]
    )
    stderr = np.sqrt((n - 1) / n * np.sum((jackknife - jackknife.mean(0)) ** 2, axis=0))
    offset = fes.min()
    fes -= offset
    sampleDeltaF = np.array(sampleDeltaF)
    dfError = sampleDeltaF.std(ddof=1) / np.sqrt(n)
    np.savez(
        outfile,
        fes=fes,
        stderr=stderr,
        grid=grid,
        kappa=kappa,
        delta_f=sampleDeltaF.mean(),
        delta_f_stderr=dfError,
        delta_f_per_run=sampleDeltaF,
        samples=sum(len(run) for run in runs),
        runs=n,
    )
    print(
        f"{n} runs, {sum(len(r) for r in runs)} samples, kappa {kappa} and "
        f"{2 * kappa}: dF = {sampleDeltaF.mean():.3f} +/- {dfError:.3f} kJ/mol; "
        f"median standard error of F < 5 kcal/mol: "
        f"{np.median(stderr[fes < 5 * KCAL]):.3f} kJ/mol"
    )


def write_table(npz, outfile):
    """Write the reference surface as a plain-text table with a descriptive header."""
    data = np.load(npz)
    fes, stderr, grid = data["fes"], data["stderr"], data["grid"]
    psi, phi = np.meshgrid(grid, grid, indexing="ij")
    header = TABLE_HEADER.format(
        runs=int(data["runs"]),
        samples=int(data["samples"]),
        kappa=float(data["kappa"]),
        kappa2=2 * float(data["kappa"]),
        dF=float(data["delta_f"]) / KCAL,
        dFerr=float(data["delta_f_stderr"]) / KCAL,
        n=len(grid),
    )
    table = np.column_stack(
        [phi.ravel(), psi.ravel(), fes.ravel() / KCAL, stderr.ravel() / KCAL]
    )
    np.savetxt(outfile, table, fmt="%9.5f %9.5f %9.5f %8.5f", header=header)


def plot(npz, outfile):
    """Draw the reference surface and its standard error."""
    import cmcrameri.cm as cmc
    import matplotlib.pyplot as plt

    data = np.load(npz)
    fes, stderr, grid = data["fes"] / KCAL, data["stderr"] / KCAL, data["grid"]
    closed = np.append(grid, np.pi)

    def wrap(values):
        return np.pad(values, ((0, 1), (0, 1)), mode="wrap")

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8})
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True, layout="constrained")
    levels = np.arange(0, 12.5, 1)
    filled = axs[0].contourf(
        closed, closed, wrap(fes), levels=levels, cmap=cmc.devon, extend="max"
    )
    axs[0].contour(closed, closed, wrap(fes), levels=levels, colors="k", linewidths=0.3)
    fig.colorbar(
        filled, ax=axs[0], label=r"$F$ (kcal mol$^{-1}$)", ticks=range(0, 13, 2)
    )
    for name, (x, y) in {"C7eq": (-1.5, 1.2), "C7ax": (1.2, -1.0)}.items():
        axs[0].annotate(name, (x, y), color="w", ha="center", fontsize="small")
    axs[0].set(title="Free energy")
    shown = np.where(fes < 12, stderr, np.nan)
    axs[1].set_facecolor("0.85")
    mesh = axs[1].pcolormesh(
        grid,
        grid,
        shown,
        cmap=cmc.lajolla,
        vmin=0,
        vmax=0.05,
        shading="nearest",
        rasterized=True,
    )
    axs[1].contour(closed, closed, wrap(fes), levels=levels, colors="k", linewidths=0.3)
    fig.colorbar(
        mesh, ax=axs[1], extend="max", label=r"Standard error (kcal mol$^{-1}$)"
    )
    axs[1].set(title=r"Standard error, where $F < 12$ kcal mol$^{-1}$")
    for ax in axs:
        ax.set_xlabel(r"$\phi$ (rad)")
    axs[0].set_ylabel(r"$\psi$ (rad)")
    for suffix in (".png", ".svg"):
        fig.savefig(outfile.with_suffix(suffix), dpi=200)


TABLE_HEADER = """\
Reference free energy surface of alanine dipeptide in vacuum, F(phi, psi)
Ramachandran free energy landscape (potential of mean force) of the backbone
dihedrals phi and psi of alanine dipeptide (Ace-Ala-Nme), from openmm-opes,
benchmarks/alanine-dipeptide. Useful as ground truth for enhanced sampling
benchmarks (OPES, metadynamics, umbrella sampling, ABF, ...).

System: Amber ff14SB (OpenMM amber14-all.xml), vacuum, no cutoff, bonds to
hydrogen constrained, hydrogen mass 4 amu, Langevin middle integrator at
300 K, friction 1/ps, 4 fs step. phi = C-N-CA-C (atoms 4,6,8,14), psi =
N-CA-C-N (atoms 6,8,14,16) of the OpenMM alanine-dipeptide test structure.

Method: {runs} independent runs of 200 ns under a fixed bias, the converged
well-tempered metadynamics bias, sampled every ps ({samples} samples) and
reweighted exactly by exp(beta V). Densities from product von Mises kernels
at kappa = {kappa:g} and {kappa2:g} rad^-2, Richardson-extrapolated to zero
kernel width: F = 2 F(kappa2) - F(kappa). Standard errors by jackknife over
runs; they leave out the residual smoothing bias, a few thousandths of a
kcal/mol where F < 5 kcal/mol (a three-point extrapolation that also includes
kappa / 2 differs by 0.002 kcal/mol RMS).

Free energy difference between the phi > 0 (C7ax, alpha_L) and phi < 0
(C7eq, C5, beta, alpha_R) regions, from the weighted samples:
dF = {dF:.4f} +/- {dFerr:.4f} kcal/mol.

Grid: {n} x {n} points, phi and psi from -pi in steps of 2 pi / {n} (radians);
F relative to its minimum. Values above about 12 kcal/mol are less reliable.
Columns: phi (rad), psi (rad), F (kcal/mol), standard error of F (kcal/mol)."""


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("metad_dir", type=Path)
    run.add_argument("seed", type=int)
    run.add_argument("outfile", type=Path)
    run.add_argument("--ns", type=float, default=20)
    kde = commands.add_parser("kde")
    kde.add_argument("runs_dir", type=Path)
    kde.add_argument("outfile", type=Path)
    kde.add_argument("--kappa", type=float, default=800)
    table = commands.add_parser("table")
    table.add_argument("npz", type=Path)
    table.add_argument("outfile", type=Path)
    figure = commands.add_parser("plot")
    figure.add_argument("npz", type=Path)
    figure.add_argument("outfile", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        simulate(args.metad_dir, args.seed, args.outfile, args.ns)
    elif args.command == "kde":
        estimate(args.runs_dir, args.outfile, args.kappa)
    elif args.command == "table":
        write_table(args.npz, args.outfile)
    else:
        plot(args.npz, args.outfile)


if __name__ == "__main__":
    main()
