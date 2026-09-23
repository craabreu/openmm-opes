"""End-to-end runs on tiny analytic systems.

The double-well convergence tests drive OPES with an ideal sampler instead of
molecular dynamics (see idealSampler). An MD trajectory is reproducible for a
fixed seed but not stable: any change to the bias, however small, alters which
barrier crossings happen, so a single-seed MD run passed or failed on luck.
The remaining tests use MD to exercise step() and the multi-walker path, and
assert only properties that do not hinge on individual crossings.
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


GOLDEN_RATIO = (np.sqrt(5) - 1) / 2


def idealSampler(sampler, numDepositions, gridWidth=151):
    """Deposit kernels at CVs drawn from the current biased distribution.

    Each draw inverts the CDF of exp(-(U + V)/kT) on a fine grid at the next
    point of the golden-ratio sequence, then deposits there with the bias
    energy at that point. This is the adiabatic limit OPES theory assumes.
    Nothing is random and there is no trajectory, and a draw is a continuous
    function of the bias, so a small change to the code moves the result a
    little instead of rerolling it: across sequence offsets, RMSE after 4000
    depositions spans 1.20-1.31 (standard) and 1.33-1.38 (explore).
    """
    kT = (unit.MOLAR_GAS_CONSTANT_R * TEMPERATURE * unit.kelvin).value_in_unit(
        unit.kilojoules_per_mole
    )
    fine = np.linspace(-2, 2, 4001)
    potential = BARRIER_HEIGHT * (1 - fine**2) ** 2
    grid = np.linspace(-2, 2, gridWidth)
    for k in range(1, numDepositions + 1):
        bias = sampler.getBias().value_in_unit(unit.kilojoules_per_mole)
        bias = np.interp(fine, grid, bias)
        energy = potential + bias
        cdf = np.cumsum(np.exp(-(energy - energy.min()) / kT))
        x = np.interp(k * GOLDEN_RATIO % 1.0, cdf / cdf[-1], fine)
        sampler.addKernel([x], float(np.interp(x, fine, bias)))


@pytest.fixture(scope="module")
def doubleWellErrors():
    """(RMSE, barrier) after 4000 ideal depositions, keyed by exploreMode."""
    errors = {}
    for exploreMode in (False, True):
        system, variable = doubleWellSystem()
        sampler = OPES(
            system, [variable], TEMPERATURE, 30.0, 200, None, exploreMode=exploreMode
        )
        idealSampler(sampler, 4000)
        errors[exploreMode] = fesError(sampler)
    return errors


@pytest.mark.slow
@pytest.mark.parametrize("exploreMode", [False, True])
def test_opes_recovers_the_analytic_double_well(doubleWellErrors, exploreMode):
    rmse, barrier = doubleWellErrors[exploreMode]
    assert rmse < 1.5, f"RMSE {rmse:.3f} kJ/mol"
    assert barrier == pytest.approx(BARRIER_HEIGHT, abs=1.0)


@pytest.mark.slow
def test_explore_mode_converges_more_slowly(doubleWellErrors):
    """The convergence half of the tradeoff the OPES-explore paper describes.

    The other half, that explore crosses the barrier more often, is a
    property of the dynamics and has no counterpart in the ideal sampler.
    """
    assert doubleWellErrors[True][0] > doubleWellErrors[False][0]


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

    # The second walker synced after the first had written its final state, so
    # its shared estimates hold every sample of both walkers. Counted in
    # samples, not kernels: compression merges overlapping peer kernels into
    # the walker's own, so the kernel count can even drop when sharing works.
    def numSamples(kde):
        return sum(kernel.numSamples for kernel in kde._kernels)

    first, second = samplers
    for total, own in (("total", "self"), ("total.rw", "self.rw")):
        assert numSamples(second._kde[total]) == numSamples(
            second._kde[own]
        ) + numSamples(first._kde[own])
        assert numSamples(first._kde[own]) > 0


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
