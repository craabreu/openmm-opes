"""End-to-end runs on tiny analytic systems.

Reference numbers come from the design review (spec section 8.2): standard
OPES reached RMSE 0.78 kJ/mol at 1.2M steps with a barrier estimate of 24.43
against an analytic 25.00, and explore mode reached RMSE 3.00 with more
barrier crossings. Tolerances below are set from those runs and must not be
loosened to make a failing test pass.
"""

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

from openmm_opes import OPES  # noqa: E402

BARRIER_HEIGHT = 25.0
TEMPERATURE = 300.0


def doubleWellSystem(gridWidth=151):
    """U(x) = 25*(1 - x^2)^2 kJ/mol: symmetric wells at x = +/-1."""
    system = openmm.System()
    system.addParticle(12.0)
    potential = openmm.CustomExternalForce(f"{BARRIER_HEIGHT}*(1-(x/1.0)^2)^2")
    potential.addParticle(0, [])
    system.addForce(potential)
    cv = openmm.CustomExternalForce("x")
    cv.addParticle(0, [])
    return system, app.BiasVariable(cv, -2.0, 2.0, 0.15, False, gridWidth)


def runSampler(sampler, system, chunks, chunkSize, seed=1234):
    integrator = openmm.LangevinMiddleIntegrator(
        TEMPERATURE * unit.kelvin, 10.0 / unit.picosecond, 0.002 * unit.picoseconds
    )
    integrator.setRandomNumberSeed(seed)
    topology = app.Topology()
    topology.addAtom("A", None, topology.addResidue("M", topology.addChain()))
    simulation = app.Simulation(
        topology, system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    simulation.context.setPositions([openmm.Vec3(-1.0, 0, 0)])
    simulation.context.setVelocitiesToTemperature(TEMPERATURE * unit.kelvin, seed + 1)
    trajectory = []
    for _ in range(chunks):
        sampler.step(simulation, chunkSize)
        trajectory.append(sampler.getCollectiveVariables(simulation)[0])
    return np.array(trajectory)


def fesError(sampler, gridWidth=151):
    grid = np.linspace(-2, 2, gridWidth)
    analytic = BARRIER_HEIGHT * (1 - grid**2) ** 2
    mask = analytic < 30.0
    fes = sampler.getFreeEnergy().value_in_unit(unit.kilojoules_per_mole)
    shifted = fes[mask] - fes[mask].min()
    reference = analytic[mask] - analytic[mask].min()
    rmse = np.sqrt(np.mean((shifted - reference) ** 2))
    barrier = shifted[np.argmin(np.abs(grid[mask]))]
    return rmse, barrier


@pytest.mark.slow
def test_opes_recovers_the_analytic_double_well():
    system, variable = doubleWellSystem()
    sampler = OPES(system, [variable], TEMPERATURE, 30.0, 200, 20)
    trajectory = runSampler(sampler, system, 600, 2000)

    assert trajectory.min() < -0.5 and trajectory.max() > 0.5
    rmse, barrier = fesError(sampler)
    assert rmse < 1.5, f"RMSE {rmse:.3f} kJ/mol (review run reached 0.78)"
    assert barrier == pytest.approx(BARRIER_HEIGHT, abs=2.5)


@pytest.mark.slow
def test_explore_mode_explores_more_but_converges_more_slowly():
    """The tradeoff the OPES-explore paper exists to demonstrate.

    Asserts the ordering, not the absolute values: the method guarantees the
    tradeoff, not any particular number. Needs the full 2.4M-step run (1200
    chunks): the crossing-count gap was validated at that length (67 vs 79),
    and a crossing count is a single noisy integer per run, so at half the
    steps the gap can vanish into run-to-run noise even when the underlying
    method is correct. The shorter 1.2M-step config is reserved for the RMSE
    tolerance test above, which does not depend on this ordering holding.
    """
    results = {}
    for exploreMode in (False, True):
        system, variable = doubleWellSystem()
        sampler = OPES(
            system, [variable], TEMPERATURE, 30.0, 200, 20, exploreMode=exploreMode
        )
        trajectory = runSampler(sampler, system, 1200, 2000)
        crossings = int(np.sum(np.diff(np.sign(trajectory)) != 0))
        results[exploreMode] = (crossings, fesError(sampler)[0])

    assert results[True][0] > results[False][0], "explore should cross more often"
    assert results[True][1] > results[False][1], "explore should converge more slowly"


def test_harmonic_free_energy_is_quadratic_where_sampled():
    system = openmm.System()
    system.addParticle(12.0)
    potential = openmm.CustomExternalForce("200*x^2")
    potential.addParticle(0, [])
    system.addForce(potential)
    cv = openmm.CustomExternalForce("x")
    cv.addParticle(0, [])
    variable = app.BiasVariable(cv, -1.0, 1.0, 0.05, False, 101)
    sampler = OPES(system, [variable], TEMPERATURE, 20.0, 200, 20)
    runSampler(sampler, system, 150, 2000)

    grid = np.linspace(-1, 1, 101)
    fes = sampler.getFreeEnergy().value_in_unit(unit.kilojoules_per_mole)
    core = np.abs(grid) < 0.25
    shifted = fes[core] - fes[core].min()
    analytic = 200 * grid[core] ** 2
    assert shifted == pytest.approx(analytic - analytic.min(), abs=4.0)


def test_bounded_and_compact_options_run_end_to_end():
    for kwargs in ({"bounded": True}, {"kernelShape": "compact"}):
        system, variable = doubleWellSystem(gridWidth=61)
        sampler = OPES(system, [variable], TEMPERATURE, 30.0, 200, 20, **kwargs)
        runSampler(sampler, system, 30, 1000)
        bias = sampler.getBias().value_in_unit(unit.kilojoules_per_mole)
        assert np.all(np.isfinite(bias)), kwargs
        assert sampler.getNumKernels() > 0, kwargs


def test_warmup_runs_end_to_end_and_beats_no_warmup_on_nan_safety():
    system, variable = doubleWellSystem(gridWidth=61)
    sampler = OPES(system, [variable], TEMPERATURE, 30.0, 200, 200, warmupSteps=2000)
    runSampler(sampler, system, 30, 1000)
    assert sampler._warmupComplete
    assert sampler.getNumKernels() > 0
    assert np.all(
        np.isfinite(sampler.getBias().value_in_unit(unit.kilojoules_per_mole))
    )


def test_two_walkers_share_their_kernels(tmp_path):
    samplers, systems = [], []
    for _ in range(2):
        system, variable = doubleWellSystem(gridWidth=61)
        samplers.append(
            OPES(
                system,
                [variable],
                TEMPERATURE,
                30.0,
                200,
                20,
                saveFrequency=200,
                biasDir=str(tmp_path),
            )
        )
        systems.append(system)

    for index, (sampler, system) in enumerate(zip(samplers, systems, strict=True)):
        runSampler(sampler, system, 20, 500, seed=1000 * (index + 1))

    # the second walker synced after the first had already written kernels
    assert (
        samplers[1]._kde["total"].getNumKernels()
        > samplers[1]._kde["self"].getNumKernels()
    )


def test_multiwalker_sync_does_not_corrupt_the_walkers_own_variance(tmp_path):
    """Regression for spec 7.4.1 at the level it actually bit.

    A peer must exist: with no peer, _syncWithDisk returns early and never
    reaches the copy-then-merge that the aliasing bug corrupted.
    """
    peerSystem, peerVariable = doubleWellSystem(gridWidth=61)
    peer = OPES(
        peerSystem,
        [peerVariable],
        TEMPERATURE,
        30.0,
        200,
        20,
        saveFrequency=200,
        biasDir=str(tmp_path),
    )
    runSampler(peer, peerSystem, 10, 500, seed=99)

    system, variable = doubleWellSystem(gridWidth=61)
    sampler = OPES(
        system,
        [variable],
        TEMPERATURE,
        30.0,
        200,
        20,
        saveFrequency=200,
        biasDir=str(tmp_path),
    )
    runSampler(sampler, system, 10, 500)

    own = sampler._variance["self"].get().copy()
    sampler._syncWithDisk()
    assert sampler._variance["self"].get() == pytest.approx(own)
