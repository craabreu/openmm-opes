"""
Free energy surface of alanine dipeptide in vacuum with OPES or with
well-tempered metadynamics, biasing the backbone torsions phi and psi.

Usage:
    python run.py METHOD SEED OUTFILE [--ns 50] [--every 1]

METHOD is opes (openmm-opes) or metad (openmm.app.Metadynamics). OUTFILE, an
.npz file, gets the free energy surface estimated by the method itself every
--every ns, in kJ/mol on the shared phi-psi grid (GRID points per axis), with
the simulation times in ns. For OPES it also gets the surface with the
smoothing bias corrected, as fes_corrected.

Settings follow the alanine dipeptide inputs of Invernizzi & Parrinello,
J. Phys. Chem. Lett. 11, 2731 (2020), PLUMED-NEST plumID:19.068:
    OPES:  PACE=500 SIGMA=0.15 BARRIER=50 BIASFACTOR=10
    METAD: PACE=500 SIGMA=0.35 HEIGHT=1.2 BIASFACTOR=10, 100 x 100 grid
The paper deposits every 500 steps of 2 fs, i.e. every ps. Here hydrogen
mass repartitioning (4 amu, the value in OpenMM's user guide) allows a 4 fs
step, so PACE is 250 steps, still one deposition per ps. The system uses Amber
ff14SB without cutoff, bonds to hydrogen constrained, and Langevin dynamics
at 300 K with a 1/ps friction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openmm
from openmm import app, unit

TEMPERATURE = 300 * unit.kelvin
TIMESTEP = 0.004  # ps, with hydrogen mass repartitioning
PACE = 250  # steps, one deposition per ps
BIAS_FACTOR = 10.0
PHI, PSI = (4, 6, 8, 14), (6, 8, 14, 16)
PDB = Path(__file__).resolve().parent / "alanine-dipeptide.pdb"

# Both methods tabulate their bias on this grid, and the reported free energy
# surfaces share it. Its spacing, 2 pi / 100, matches PLUMED's GRID_BIN=100
# for metadynamics.
GRID = 101


def torsion(atoms):
    force = openmm.CustomTorsionForce("theta")
    force.addTorsion(*atoms, [])
    return force


def variables(width, grid):
    return [
        app.BiasVariable(torsion(atoms), -np.pi, np.pi, width, True, grid)
        for atoms in (PHI, PSI)
    ]


def build_simulation(method, seed):
    pdb = app.PDBFile(str(PDB))
    forcefield = app.ForceField("amber14-all.xml")
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        hydrogenMass=4 * unit.amu,
    )
    if method == "opes":
        from openmm_opes import OPES

        sampler = OPES(
            system,
            variables(0.15, GRID),
            TEMPERATURE,
            50 * unit.kilojoules_per_mole,
            PACE,
            None,  # fixed initial bandwidth, SIGMA=0.15 as in the paper
            biasFactor=BIAS_FACTOR,
        )
    else:
        sampler = app.Metadynamics(
            system,
            variables(0.35, GRID),
            TEMPERATURE,
            BIAS_FACTOR,
            1.2 * unit.kilojoules_per_mole,
            PACE,
        )
    integrator = openmm.LangevinMiddleIntegrator(
        TEMPERATURE, 1 / unit.picosecond, TIMESTEP * unit.picoseconds
    )
    integrator.setRandomNumberSeed(seed)
    simulation = app.Simulation(
        pdb.topology,
        system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(TEMPERATURE, seed)
    return simulation, sampler


def free_energy(sampler, correctSmoothing=False):
    """The method's own estimate of F(phi, psi) on the shared grid, kJ/mol."""
    if correctSmoothing:
        fes = sampler.getFreeEnergy(correctSmoothing=True)
    else:
        fes = sampler.getFreeEnergy()
    fes = fes.value_in_unit(unit.kilojoules_per_mole)
    return fes - fes.min()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["opes", "metad"])
    parser.add_argument("seed", type=int)
    parser.add_argument("outfile", type=Path)
    parser.add_argument("--ns", type=float, default=50)
    parser.add_argument("--every", type=float, default=1)
    args = parser.parse_args()
    simulation, sampler = build_simulation(args.method, args.seed)
    chunk = round(args.every * 1000 / TIMESTEP)
    times, surfaces, corrected = [], [], []
    for k in range(1, round(args.ns / args.every) + 1):
        sampler.step(simulation, chunk)
        times.append(k * args.every)
        surfaces.append(free_energy(sampler))
        if args.method == "opes":
            corrected.append(free_energy(sampler, correctSmoothing=True))
    extra = {"fes_corrected": np.array(corrected)} if corrected else {}
    np.savez_compressed(
        args.outfile,
        times=np.array(times),
        fes=np.array(surfaces),
        grid=np.linspace(-np.pi, np.pi, GRID),
        **extra,
    )


if __name__ == "__main__":
    main()
