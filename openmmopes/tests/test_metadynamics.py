"""
Unit tests for Metadynamics.
"""

import openmm as mm
from openmm import app, unit


def test_metadynamics():
    """
    Test the Metadynamics class.
    """
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    force = mm.HarmonicBondForce()
    force.addBond(0, 1, 1.0, 100000.0)
    system.addForce(force)
    cv = mm.CustomBondForce("r")
    cv.addBond(0, 1)
    bias = app.BiasVariable(cv, 0.94, 1.06, 0.00431, gridWidth=31)
    meta = app.Metadynamics(system, [bias], 300 * unit.kelvin, 3.0, 5.0, 10)
    integrator = mm.LangevinIntegrator(
        300 * unit.kelvin, 10 / unit.picosecond, 0.001 * unit.picosecond
    )
    integrator.setRandomNumberSeed(4321)
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("H2", chain)
    topology.addAtom("H1", app.element.hydrogen, residue)
    topology.addAtom("H2", app.element.hydrogen, residue)
    simulation = app.Simulation(
        topology, system, integrator, mm.Platform.getPlatformByName("Reference")
    )
    simulation.context.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(1, 0, 0)])
    meta.step(simulation, 200000)
    fe = meta.getFreeEnergy()
    center = bias.gridWidth // 2
    fe -= fe[center]

    # Energies should be reasonably well converged over the central part of the range.

    for i in range(center - 3, center + 4):
        r = bias.minValue + i * (bias.maxValue - bias.minValue) / (bias.gridWidth - 1)
        e = 0.5 * 100000.0 * (r - 1.0) ** 2 * unit.kilojoules_per_mole
        assert abs(fe[i] - e) < 1.0 * unit.kilojoules_per_mole
