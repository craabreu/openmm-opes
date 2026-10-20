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

``run`` simulates under the static bias built from METAD_DIR/metad_*.npz and
saves phi, psi and the bias energy felt every ps. ``kde`` reweights all
samples in RUNS_DIR and estimates F(phi, psi) on the shared grid with product
von Mises kernels of concentration kappa (1/rad^2, so kappa = 800 means a
width of about 0.035 rad, below the grid spacing), and saves the surface and,
per run, dF between the phi > 0 and phi < 0 halves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openmm
from analyze import KT, combine, delta_f, interior
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


def estimate(runs_dir, outfile, kappa):
    grid = np.linspace(-np.pi, np.pi, GRID)[:-1]
    runs = [np.load(path) for path in sorted(Path(runs_dir).glob("*.npy"))]
    shift = max(run[:, 2].max() for run in runs)
    densities, delta_fs = [], []
    for run in runs:
        weights = np.exp((run[:, 2] - shift) / KT)
        density = von_mises_density(run[:, :2], weights, grid, kappa)
        densities.append(density)
        delta_fs.append(delta_f(-KT * np.log(density), grid))
    fes = -KT * np.log(np.sum(densities, axis=0))
    fes -= fes.min()
    np.savez(
        outfile,
        fes=fes,
        grid=grid,
        delta_f_per_run=np.array(delta_fs),
        delta_f=delta_f(fes, grid),
        kappa=kappa,
        samples=sum(len(run) for run in runs),
    )
    sem = np.std(delta_fs, ddof=1) / np.sqrt(len(delta_fs))
    print(
        f"{len(runs)} runs, {sum(len(r) for r in runs)} samples, kappa {kappa}: "
        f"dF = {delta_f(fes, grid):.3f} kJ/mol "
        f"(runs: {np.mean(delta_fs):.3f} +/- {sem:.3f})"
    )


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
    args = parser.parse_args()
    if args.command == "run":
        simulate(args.metad_dir, args.seed, args.outfile, args.ns)
    else:
        estimate(args.runs_dir, args.outfile, args.kappa)


if __name__ == "__main__":
    main()
