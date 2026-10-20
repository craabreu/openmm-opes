"""
Convergence of OPES in PLUMED and in openmm-opes on a model potential.

A single particle moves in OpenMM on

    U(x, y) = A (1 - x^2)^2 + B x + C (y - D x)^2

an asymmetric double well along x, tilted so that the two basins differ in
free energy, with a stiff coupled y. OPES biases x only. PLUMED runs through
openmm-plumed; openmm-opes runs as a CustomCVForce in the same System. Both
use the same integrator, seeds and deposition settings, and both record the
CV and the bias halfway between depositions, so that every sample is weighted
by the bias that acted on it. The exact free energy difference between the
basins x < 0 and x > 0 comes from quadrature (see analyze_dynamics.py).

Usage:
    python dynamics.py IMPLEMENTATION MODE SEED OUTFILE [--ns 20]

IMPLEMENTATION is plumed, openmm-opes, or openmm-opes-plumedmerge (openmm-opes
with PLUMED's height-weighted kernel merge, see plumed_merge.py). MODE is
opes, explore, or com (OPES and OPES-explore together, a CombinedOPES in
openmm-opes). OUTFILE gets the columns time (ps), x (nm) and one bias (kJ/mol)
per OPES variant acting, and OUTFILE.bias the final total bias on a grid of x,
with columns x (nm) and bias (kJ/mol).

PLUMED must be built with the opes module; see README.md. PLUMED runs
evaluate their final bias with `plumed driver`, found at $PLUMED/bin/plumed
unless --plumed-exe says otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import openmm
from common import KB
from openmm import app, unit

A, B, C, D = 30.0, 3.0, 50.0, 0.5  # kJ/mol and nm
TEMPERATURE = 300.0  # K
TIMESTEP = 0.002  # ps
PACE = 500  # steps between depositions
VARIANCE_STRIDE = 50  # steps between adaptive-variance samples in openmm-opes
BARRIERS = {"opes": (35.0,), "explore": (35.0,), "com": (35.0, 20.0)}
BIAS_GRID = np.linspace(-2.0, 2.0, 401)  # where the final bias is reported, nm


def exact_delta_f():
    """F(x > 0) - F(x < 0) by quadrature, in kJ/mol."""
    kT = KB * TEMPERATURE
    x = np.linspace(-2.5, 2.5, 5001)
    # The y integral of exp(-C (y - D x)^2 / kT) does not depend on x.
    w = np.exp(-(A * (1 - x**2) ** 2 + B * x) / kT)
    return -kT * np.log(w[x > 0].sum() / w[x < 0].sum())


def model_system():
    system = openmm.System()
    system.addParticle(12.0)
    potential = openmm.CustomExternalForce(
        f"{A}*(1-x^2)^2 + {B}*x + {C}*(y-{D}*x)^2 + 100*z^2"
    )
    potential.addParticle(0, [])
    system.addForce(potential)
    return system


def x_variable():
    cv = openmm.CustomExternalForce("x")
    cv.addParticle(0, [])
    return app.BiasVariable(cv, -2.5, 2.5, 0.05, False, 501)


def plumed_actions(mode, outfile, steps=None):
    """The OPES actions of a PLUMED input, and the labels of their biases.

    With `steps`, each action writes its state at the last deposition,
    step steps - PACE: openmm-plumed calls PLUMED before each step, so the
    last step it sees is steps - 1. Without `steps`, each action restarts
    from that state with a pace too long to deposit anything.
    Each run gets its own files: PLUMED backs up an existing file before
    overwriting it and aborts after 100 backups.
    """
    actions = []
    if mode in ("opes", "com"):
        actions.append(("o", "OPES_METAD", BARRIERS[mode][0], "opes"))
    if mode in ("explore", "com"):
        actions.append(("e", "OPES_METAD_EXPLORE", BARRIERS[mode][-1], "explore"))
    lines = []
    for label, action, barrier, tag in actions:
        common = (
            f"{label}: {action} ARG=p.x BARRIER={barrier} TEMP={TEMPERATURE}"
            f" FILE={outfile}.{tag}-kernels"
        )
        if steps is None:
            lines.append(
                f"{common} PACE={100 * len(BIAS_GRID)}"
                f" STATE_RFILE={outfile}.{tag}-state"
            )
        else:
            lines.append(
                f"{common} PACE={PACE}"
                f" STATE_WFILE={outfile}.{tag}-state STATE_WSTRIDE={steps - PACE}"
            )
    return lines, [f"{label}.bias" for label, *_ in actions]


def plumed_script(mode, outfile, steps):
    """PLUMED input that prints every half PACE; the midpoints are kept later."""
    actions, biases = plumed_actions(mode, outfile, steps)
    return "\n".join(
        [
            "UNITS LENGTH=nm ENERGY=kj/mol",
            "p: POSITION ATOM=1 NOPBC",
            *actions,
            f"PRINT ARG=p.x,{','.join(biases)} STRIDE={PACE // 2}"
            f" FILE={outfile}.plumed",
            "",
        ]
    )


def plumed_final_bias(plumed_exe, mode, outfile):
    """Evaluate the saved PLUMED states on BIAS_GRID with plumed driver.

    A one-atom trajectory visits the grid, so that the restarted actions see
    the same argument, p.x, as during the run.
    """
    frames = Path(f"{outfile}.grid.xyz")
    with open(frames, "w") as file:
        for x in BIAS_GRID:
            file.write(f"1\n100 100 100\nX {x:.10f} 0 0\n")  # the driver needs a box
    actions, biases = plumed_actions(mode, outfile)
    script = Path(f"{outfile}.grid.dat")
    script.write_text(
        "\n".join(
            [
                "RESTART",
                "UNITS LENGTH=nm ENERGY=kj/mol",
                "p: POSITION ATOM=1 NOPBC",
                *actions,
                f"PRINT ARG={','.join(biases)} FILE={outfile}.grid.bias FMT=%.10g",
                "",
            ]
        )
    )
    subprocess.run(
        [plumed_exe, "driver", "--ixyz", str(frames), "--plumed", str(script)],
        check=True,
        capture_output=True,
    )
    data = np.loadtxt(f"{outfile}.grid.bias", comments="#", ndmin=2)
    return data[:, 1:].sum(axis=1)


@contextlib.contextmanager
def silenced_stdout():
    """Discard the process's standard output, including that of C++ libraries.

    openmm-plumed sends PLUMED's log there, about 1 GB per 20 ns run.
    """
    libc = ctypes.CDLL(None)
    sys.stdout.flush()
    saved = os.dup(1)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 1)
    try:
        yield
    finally:
        libc.fflush(None)  # empty C stdio buffers while they still go nowhere
        os.dup2(saved, 1)
        os.close(saved)


def make_simulation(system, seed):
    integrator = openmm.LangevinMiddleIntegrator(TEMPERATURE, 10.0, TIMESTEP)
    integrator.setRandomNumberSeed(seed)
    topology = app.Topology()
    chain = topology.addChain()
    topology.addAtom("A", app.element.carbon, topology.addResidue("M", chain))
    simulation = app.Simulation(
        topology,
        system,
        integrator,
        openmm.Platform.getPlatformByName("CPU"),
        {"Threads": "1"},
    )
    simulation.context.setPositions([openmm.Vec3(-1.0, -0.5, 0.0)])
    simulation.context.setVelocitiesToTemperature(TEMPERATURE, seed)
    return simulation


def run_plumed(mode, seed, outfile, steps, plumed_exe):
    from openmmplumed import PlumedForce

    system = model_system()
    system.addForce(PlumedForce(plumed_script(mode, outfile, steps)))
    with silenced_stdout():
        simulation = make_simulation(system, seed)
        simulation.step(steps)
        # Destroying the Context finalizes PLUMED, which flushes its files and
        # prints a timing summary, so do it while the output is still silenced.
        del simulation
        gc.collect()
    printed = Path(f"{outfile}.plumed")
    data = np.loadtxt(printed, comments="#", ndmin=2)
    steps_printed = np.rint(data[:, 0] / TIMESTEP).astype(int)
    midpoints = data[steps_printed % PACE == PACE // 2]
    np.savetxt(outfile, midpoints, fmt="%.6f", header="time x biases")
    bias = plumed_final_bias(plumed_exe, mode, outfile)
    np.savetxt(f"{outfile}.bias", np.column_stack([BIAS_GRID, bias]), fmt="%.6f")
    name, folder = Path(outfile).name, Path(outfile).parent
    for path in [*folder.glob(f"{name}.*"), *folder.glob(f"bck.*.{name}.*")]:
        if path.name != f"{name}.bias":
            path.unlink()


def run_openmm_opes(mode, seed, outfile, steps, plumed_merge=False):
    from openmm_opes import OPES, CombinedOPES

    if plumed_merge:
        from plumed_merge import use_plumed_merge

        use_plumed_merge()
    system = model_system()
    kT = TEMPERATURE * unit.kelvin
    if mode == "com":
        barrier, exploreBarrier = BARRIERS[mode]
        sampler = CombinedOPES(
            system, [x_variable()], kT, barrier, exploreBarrier, PACE, VARIANCE_STRIDE
        )
    else:
        sampler = OPES(
            system,
            [x_variable()],
            kT,
            BARRIERS[mode][0],
            PACE,
            VARIANCE_STRIDE,
            exploreMode=mode == "explore",
        )

    def biases():
        if isinstance(sampler, CombinedOPES):
            energies = sampler.getComponentBiasEnergies(simulation.context)
        else:
            energies = [sampler.getBiasEnergy(simulation.context)]
        return [e.value_in_unit(unit.kilojoules_per_mole) for e in energies]

    simulation = make_simulation(system, seed)
    rows = []
    sampler.step(simulation, PACE // 2)
    while simulation.currentStep + PACE <= steps:
        state = simulation.context.getState(getPositions=True)
        x = state.getPositions(asNumpy=True)[0][0].value_in_unit(unit.nanometer)
        rows.append([simulation.currentStep * TIMESTEP, x, *biases()])
        sampler.step(simulation, PACE)
    np.savetxt(outfile, np.array(rows), fmt="%.6f", header="time x biases")
    grid = np.linspace(-2.5, 2.5, 501)  # the grid of x_variable()
    total = sampler.getBias().value_in_unit(unit.kilojoules_per_mole)
    bias = np.interp(BIAS_GRID, grid, total)
    np.savetxt(f"{outfile}.bias", np.column_stack([BIAS_GRID, bias]), fmt="%.6f")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "implementation", choices=["plumed", "openmm-opes", "openmm-opes-plumedmerge"]
    )
    parser.add_argument("mode", choices=sorted(BARRIERS))
    parser.add_argument("seed", type=int)
    parser.add_argument("outfile")
    parser.add_argument("--ns", type=float, default=20)
    parser.add_argument(
        "--plumed-exe",
        default=os.path.join(os.environ.get("PLUMED", ""), "bin", "plumed"),
    )
    args = parser.parse_args()
    steps = round(args.ns * 1000 / TIMESTEP)
    if args.implementation == "plumed":
        run_plumed(args.mode, args.seed, args.outfile, steps, args.plumed_exe)
    else:
        run_openmm_opes(
            args.mode,
            args.seed,
            args.outfile,
            steps,
            plumed_merge=args.implementation == "openmm-opes-plumedmerge",
        )


if __name__ == "__main__":
    main()
