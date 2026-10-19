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
opes, explore, or com (OPES and OPES-explore together). OUTFILE gets the
columns time (ps), x (nm) and one bias (kJ/mol) per sampler.

PLUMED must be built with the opes module; see README.md.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import os
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


def plumed_script(mode, outfile):
    """PLUMED input that prints every half PACE; the midpoints are kept later.

    Each run gets its own kernel files: PLUMED backs up an existing file
    before overwriting it and aborts after 100 backups.
    """
    lines = ["UNITS LENGTH=nm ENERGY=kj/mol", "p: POSITION ATOM=1 NOPBC"]
    names = []
    if mode in ("opes", "com"):
        lines.append(
            f"o: OPES_METAD ARG=p.x PACE={PACE} BARRIER={BARRIERS[mode][0]}"
            f" TEMP={TEMPERATURE} FILE={outfile}.kernels"
        )
        names.append("o.bias")
    if mode in ("explore", "com"):
        lines.append(
            f"e: OPES_METAD_EXPLORE ARG=p.x PACE={PACE} BARRIER={BARRIERS[mode][-1]}"
            f" TEMP={TEMPERATURE} FILE={outfile}.explore-kernels"
        )
        names.append("e.bias")
    lines.append(
        f"PRINT ARG=p.x,{','.join(names)} STRIDE={PACE // 2} FILE={outfile}.plumed"
    )
    return "\n".join(lines) + "\n"


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


def run_plumed(mode, seed, outfile, steps):
    from openmmplumed import PlumedForce

    system = model_system()
    system.addForce(PlumedForce(plumed_script(mode, outfile)))
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
    for suffix in (".plumed", ".kernels", ".explore-kernels"):
        Path(f"{outfile}{suffix}").unlink(missing_ok=True)


def run_openmm_opes(mode, seed, outfile, steps, plumed_merge=False):
    from openmm_opes import OPES

    if plumed_merge:
        from plumed_merge import use_plumed_merge

        use_plumed_merge()
    system = model_system()
    kT = TEMPERATURE * unit.kelvin
    samplers = []
    if mode in ("opes", "com"):
        samplers.append(
            OPES(system, [x_variable()], kT, BARRIERS[mode][0], PACE, VARIANCE_STRIDE)
        )
    if mode in ("explore", "com"):
        samplers.append(
            OPES(
                system,
                [x_variable()],
                kT,
                BARRIERS[mode][-1],
                PACE,
                VARIANCE_STRIDE,
                exploreMode=True,
            )
        )
    simulation = make_simulation(system, seed)
    rows = []
    OPES.stepTogether(simulation, samplers, PACE // 2)
    while simulation.currentStep + PACE <= steps:
        state = simulation.context.getState(getPositions=True)
        x = state.getPositions(asNumpy=True)[0][0].value_in_unit(unit.nanometer)
        biases = [
            s.getBiasEnergy(simulation.context).value_in_unit(unit.kilojoules_per_mole)
            for s in samplers
        ]
        rows.append([simulation.currentStep * TIMESTEP, x, *biases])
        OPES.stepTogether(simulation, samplers, PACE)
    np.savetxt(outfile, np.array(rows), fmt="%.6f", header="time x biases")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "implementation", choices=["plumed", "openmm-opes", "openmm-opes-plumedmerge"]
    )
    parser.add_argument("mode", choices=sorted(BARRIERS))
    parser.add_argument("seed", type=int)
    parser.add_argument("outfile")
    parser.add_argument("--ns", type=float, default=20)
    args = parser.parse_args()
    steps = round(args.ns * 1000 / TIMESTEP)
    if args.implementation == "plumed":
        run_plumed(args.mode, args.seed, args.outfile, steps)
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
