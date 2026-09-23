import warnings

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

from openmm_opes.opes import OPES, RunningAverage  # noqa: E402


def makeSystemAndVariable(sigma=0.1, gridWidth=51):
    system = openmm.System()
    system.addParticle(1.0)
    force = openmm.CustomExternalForce("x")
    force.addParticle(0, [])
    variable = app.BiasVariable(force, -2.0, 2.0, sigma, False, gridWidth)
    return system, variable


def makeOPES(**kwargs):
    system, variable = makeSystemAndVariable(kwargs.pop("sigma", 0.1))
    kwargs.setdefault("varianceFrequency", 10)
    frequency = kwargs.pop("frequency", 100)
    return OPES(system, [variable], 300.0, 20.0, frequency, **kwargs)


def test_running_average_copies_do_not_share_their_accumulator():
    """Regression for spec 7.4.1.

    _syncWithDisk copies the 'self' accumulator to seed 'total', then adds
    peers into it. With a shared array that in-place add silently corrupted
    the walker's own variance, affecting every multi-walker run.
    """
    own = RunningAverage(1)
    own.update(np.array([5.0]))
    peer = RunningAverage(1)
    peer.update(np.array([1.0]))

    total = own.copy()
    before = own.get().copy()
    total += peer

    assert own.get() == pytest.approx(before)
    assert total.get() == pytest.approx([3.0])
    assert not np.shares_memory(own._total, total._total)


def test_running_average_state_round_trips():
    average = RunningAverage(2)
    average.update(np.array([1.0, 2.0]))
    average.update(np.array([3.0, 4.0]))
    restored = RunningAverage(2)
    restored.setState(average.getState())
    assert restored.get() == pytest.approx(average.get())


def test_constructor_adds_a_force_in_a_free_group():
    """The OPES force must land in a group no pre-existing force occupies.

    system.getForce(i) returns a fresh SWIG proxy object on every call, so
    identity comparison against sampler._force never matches even for the
    same underlying force. Groups (plain ints) compare correctly instead.
    """
    system, variable = makeSystemAndVariable()
    existing = openmm.CustomExternalForce("0")
    existing.addParticle(0, [])
    existing.setForceGroup(5)
    system.addForce(existing)

    before = system.getNumForces()
    sampler = OPES(system, [variable], 300.0, 20.0, 100, 10)
    assert system.getNumForces() == before + 1
    assert sampler._force.getForceGroup() != 5


def test_bias_factor_defaults_to_barrier_over_kT():
    sampler = makeOPES()
    kbt = (unit.MOLAR_GAS_CONSTANT_R * 300 * unit.kelvin).value_in_unit(
        unit.kilojoules_per_mole
    )
    assert sampler._biasFactor == pytest.approx(20.0 / kbt)


def test_variance_frequency_must_divide_frequency():
    with pytest.raises(ValueError, match="varianceFrequency must be a divisor"):
        makeOPES(varianceFrequency=30)


def test_zero_variance_frequency_is_rejected():
    """Regression for spec 7.6: this used to reach a ZeroDivisionError."""
    with pytest.raises(ValueError, match="varianceFrequency must be positive"):
        makeOPES(varianceFrequency=0)


def test_zero_save_frequency_is_rejected(tmp_path):
    """Regression for spec 7.6: this used to construct, then fail inside step."""
    with pytest.raises(ValueError, match="saveFrequency must be positive"):
        makeOPES(saveFrequency=0, biasDir=str(tmp_path))


def test_save_frequency_and_bias_dir_must_come_together(tmp_path):
    with pytest.raises(ValueError, match="Must specify both"):
        makeOPES(saveFrequency=100)
    with pytest.raises(ValueError, match="Must specify both"):
        makeOPES(biasDir=str(tmp_path))


def test_explicit_bias_factor_error_names_bias_factor():
    """Regression for spec 7.6: the message used to blame 'barrier'."""
    with pytest.raises(ValueError, match="biasFactor must be greater than 1"):
        makeOPES(biasFactor=0.5)


def test_low_barrier_error_names_the_barrier():
    system, variable = makeSystemAndVariable()
    with pytest.raises(ValueError, match="barrier must be greater than 1 kT"):
        OPES(system, [variable], 300.0, 0.5, 100, 10)


def test_mixed_periodicity_is_rejected():
    system = openmm.System()
    system.addParticle(1.0)
    variables = []
    for periodic in (True, False):
        force = openmm.CustomExternalForce("x")
        force.addParticle(0, [])
        variables.append(app.BiasVariable(force, -2.0, 2.0, 0.1, periodic, 31))
    with pytest.raises(ValueError, match="mixed periodic"):
        OPES(system, variables, 300.0, 20.0, 100, 10)


def test_more_than_three_cvs_is_rejected():
    system = openmm.System()
    system.addParticle(1.0)
    variables = []
    for _ in range(4):
        force = openmm.CustomExternalForce("x")
        force.addParticle(0, [])
        variables.append(app.BiasVariable(force, -2.0, 2.0, 0.1, False, 11))
    with pytest.raises(ValueError, match="1, 2, or 3 collective variables"):
        OPES(system, variables, 300.0, 20.0, 100, 10)


def test_fixed_bandwidth_stores_gamma_times_the_unbiased_variance():
    """Spec 7.5: biasWidth is the UNBIASED sigma^0, per both papers.

    _variance holds a sampled-distribution variance by convention, and the
    sampled distribution is wider than the unbiased one by sqrt(gamma).
    """
    sampler = makeOPES(varianceFrequency=None, sigma=0.1)
    expected = sampler._biasFactor * 0.1**2
    assert sampler.getVariance() == pytest.approx([expected])


def referenceBias(sampler):
    """Eq. iter_bias / iter_bias-explore, evaluated independently."""
    kde = sampler._kde["total" if sampler.exploreMode else "total.rw"]
    probability = np.exp(kde.getLogPDF())
    normalization = np.exp(kde.getLogMeanDensity())
    prefactor = sampler._prefactor.value_in_unit(unit.kilojoules_per_mole)
    epsilon = np.exp(sampler._logEpsilon)
    return prefactor * np.log(probability / normalization + epsilon)


@pytest.mark.parametrize("exploreMode", [False, True])
def test_bias_matches_the_published_equation(exploreMode):
    sampler = makeOPES(exploreMode=exploreMode)
    rng = np.random.default_rng(1)
    for _ in range(40):
        sampler.addKernel(
            np.array([rng.uniform(-1.5, 1.5)]),
            rng.uniform(0, 5) * unit.kilojoules_per_mole,
            variance=np.array([0.01]),
        )
    got = sampler.getBias().value_in_unit(unit.kilojoules_per_mole)
    assert got == pytest.approx(referenceBias(sampler), abs=1e-10)


def test_prefactor_follows_the_mode():
    kbt = (unit.MOLAR_GAS_CONSTANT_R * 300 * unit.kelvin).value_in_unit(
        unit.kilojoules_per_mole
    )
    plain = makeOPES()
    explore = makeOPES(exploreMode=True)
    gamma = plain._biasFactor
    assert plain._prefactor.value_in_unit(unit.kilojoules_per_mole) == pytest.approx(
        (1 - 1 / gamma) * kbt
    )
    assert explore._prefactor.value_in_unit(unit.kilojoules_per_mole) == pytest.approx(
        (gamma - 1) * kbt
    )


def test_epsilon_is_exp_minus_barrier_over_prefactor():
    sampler = makeOPES()
    prefactor = sampler._prefactor.value_in_unit(unit.kilojoules_per_mole)
    assert np.exp(sampler._logEpsilon) == pytest.approx(np.exp(-20.0 / prefactor))


def test_free_energy_always_uses_the_reweighted_estimate():
    sampler = makeOPES(exploreMode=True)
    sampler.addKernel(
        np.array([0.0]), 0.0 * unit.kilojoules_per_mole, variance=np.array([0.01])
    )
    kbt = sampler._kbt.value_in_unit(unit.kilojoules_per_mole)
    expected = -kbt * sampler._kde["total.rw"].getLogPDF()
    got = sampler.getFreeEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert got == pytest.approx(expected)


def test_reweighted_kde_uses_variance_divided_by_the_bias_factor():
    sampler = makeOPES()
    sampler.addKernel(
        np.array([0.0]), 0.0 * unit.kilojoules_per_mole, variance=np.array([0.04])
    )
    plain = sampler._kde["total"]._kernels[0].bandwidth[0]
    reweighted = sampler._kde["total.rw"]._kernels[0].bandwidth[0]
    assert plain / reweighted == pytest.approx(np.sqrt(sampler._biasFactor))


def test_non_positive_variance_skips_deposition_instead_of_producing_nan():
    """Regression for spec 7.4.4.

    A zero variance made a zero-bandwidth kernel whose logHeight is -inf,
    while its weight still entered logSumW. getLogPDF and getLogMeanDensity
    were then both -inf and their difference NaN, which reached the forces.
    """
    sampler = makeOPES()
    with pytest.warns(UserWarning, match="variance"):
        sampler.addKernel(
            np.array([0.0]),
            0.0 * unit.kilojoules_per_mole,
            variance=np.array([0.0]),
        )
    assert sampler.getNumKernels() == 0
    sampler.addKernel(
        np.array([0.0]), 0.0 * unit.kilojoules_per_mole, variance=np.array([0.01])
    )
    assert np.all(
        np.isfinite(sampler.getBias().value_in_unit(unit.kilojoules_per_mole))
    )


def test_opes_is_exported_from_the_package():
    import openmm_opes

    assert openmm_opes.OPES is OPES


def runSteps(sampler, system, nsteps, seed=1234, stepsBefore=0):
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 10.0 / unit.picosecond, 0.002 * unit.picoseconds
    )
    integrator.setRandomNumberSeed(seed)
    topology = app.Topology()
    topology.addAtom("A", None, topology.addResidue("M", topology.addChain()))
    simulation = app.Simulation(
        topology, system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    simulation.context.setPositions([openmm.Vec3(0.1, 0, 0)])
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, seed + 1)
    # stepsBefore stands in for equilibration run before OPES takes over
    simulation.step(stepsBefore)
    sampler.step(simulation, nsteps)
    return simulation


def makeHarmonicSystem():
    system = openmm.System()
    system.addParticle(12.0)
    potential = openmm.CustomExternalForce("500*x^2")
    potential.addParticle(0, [])
    system.addForce(potential)
    cv = openmm.CustomExternalForce("x")
    cv.addParticle(0, [])
    variable = app.BiasVariable(cv, -1.0, 1.0, 0.05, False, 51)
    return system, variable


def test_warmup_requires_variance_frequency():
    with pytest.raises(ValueError, match="warmupSteps requires varianceFrequency"):
        makeOPES(varianceFrequency=None, warmupSteps=100)


def test_warmup_must_span_at_least_two_intervals():
    with pytest.raises(ValueError, match="at least two varianceFrequency"):
        makeOPES(varianceFrequency=50, warmupSteps=50)


def test_no_kernels_are_deposited_during_warmup():
    system, variable = makeHarmonicSystem()
    sampler = OPES(system, [variable], 300.0, 20.0, 100, 10, warmupSteps=500)
    runSteps(sampler, system, 400)
    assert sampler.getNumKernels() == 0
    assert not sampler._warmupComplete


def test_deposition_starts_after_warmup_and_variance_freezes():
    system, variable = makeHarmonicSystem()
    sampler = OPES(system, [variable], 300.0, 20.0, 100, 10, warmupSteps=500)
    runSteps(sampler, system, 1500)
    assert sampler._warmupComplete
    assert sampler.getNumKernels() > 0
    frozen = sampler.getVariance().copy()
    runSteps(sampler, system, 500)
    assert sampler.getVariance() == pytest.approx(frozen)


def test_frozen_variance_is_gamma_times_the_measured_unbiased_variance():
    """Spec 7.7: warm-up measures an UNBIASED variance, while _variance holds
    a sampled one, so freezing stores gamma times the measurement."""
    system, variable = makeHarmonicSystem()
    sampler = OPES(system, [variable], 300.0, 20.0, 100, 10, warmupSteps=500)
    runSteps(sampler, system, 500)
    assert sampler._warmupComplete
    reweighted = sampler._kde["total.rw"]
    sampler.addKernel(np.array([0.0]), 0.0 * unit.kilojoules_per_mole)
    sigma0 = np.sqrt(sampler.getVariance() / sampler._biasFactor)
    # Silverman still applies to the very first kernel, where Neff == 1
    silverman = (1 * (1 + 2) / 4) ** (-1 / (1 + 4))
    assert reweighted._kernels[0].bandwidth[0] == pytest.approx(sigma0[0] * silverman)


def test_warmup_state_round_trips_without_rescaling_twice():
    system, variable = makeHarmonicSystem()
    sampler = OPES(system, [variable], 300.0, 20.0, 100, 10, warmupSteps=500)
    runSteps(sampler, system, 600)
    frozen = sampler.getVariance().copy()

    system2, variable2 = makeHarmonicSystem()
    restored = OPES(system2, [variable2], 300.0, 20.0, 100, 10, warmupSteps=500)
    restored.setState(sampler.getState())
    assert restored._warmupComplete
    assert restored.getVariance() == pytest.approx(frozen)


# --- Regression tests for the code-review findings -------------------------


def test_fresh_sampler_bias_is_the_flat_barrier_floor_not_nan():
    """Regression for review finding 1.

    With no kernels the log PDF and log mean density are both -inf, and
    their difference is NaN. getBias must return the well-defined limit --
    the -barrier floor the tabulated function starts at -- because this
    value is pushed straight into the forces.
    """
    sampler = makeOPES()
    bias = sampler.getBias().value_in_unit(unit.kilojoules_per_mole)
    assert sampler.getNumKernels() == 0
    assert np.all(np.isfinite(bias))
    assert bias == pytest.approx(np.full(51, -20.0))


def test_skipped_deposition_never_pushes_nan_into_the_context():
    """Regression for review finding 1, at the level it actually bit.

    varianceFrequency == frequency leaves exactly one stats sample before
    the first deposition, so the variance is 0 and the kernel is skipped.
    The bias must stay finite and the trajectory must survive.
    """
    system, variable = makeHarmonicSystem()
    sampler = OPES(system, [variable], 300.0, 20.0, 100, 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = runSteps(sampler, system, 300)
    position = simulation.context.getState(positions=True).getPositions()[0]
    assert np.isfinite(position.x), "trajectory went NaN"
    bias = sampler.getBias().value_in_unit(unit.kilojoules_per_mole)
    assert np.all(np.isfinite(bias))


def test_add_kernel_reports_whether_it_deposited():
    sampler = makeOPES()
    assert sampler.addKernel(
        np.array([0.0]), 0.0 * unit.kilojoules_per_mole, variance=np.array([0.01])
    )
    with pytest.warns(UserWarning, match="variance"):
        assert not sampler.addKernel(
            np.array([0.0]), 0.0 * unit.kilojoules_per_mole, variance=np.array([0.0])
        )


def test_set_state_preserves_kernel_shape_and_compression_threshold():
    """Regression for review finding 2: the rebuilt KDE used the defaults."""
    sysA, varA = makeSystemAndVariable()
    a = OPES(
        sysA,
        [varA],
        300.0,
        20.0,
        100,
        10,
        kernelShape="compact",
        compressionThreshold=0.0,
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        a.addKernel(
            np.array([rng.uniform(-1, 1)]),
            0.0 * unit.kilojoules_per_mole,
            variance=np.array([0.01]),
        )

    sysB, varB = makeSystemAndVariable()
    b = OPES(
        sysB,
        [varB],
        300.0,
        20.0,
        100,
        10,
        kernelShape="compact",
        compressionThreshold=0.0,
    )
    b.setState(a.getState())

    restored = b._kde["total"]
    assert restored._shape.name == "compact"
    assert restored._compressionThreshold == 0.0
    assert b.getBias().value_in_unit(unit.kilojoules_per_mole) == pytest.approx(
        a.getBias().value_in_unit(unit.kilojoules_per_mole)
    )


def test_set_state_restores_the_own_contribution_accumulators(tmp_path):
    """Regression for review finding 3.

    _syncWithDisk rebuilds "total" from "self" plus peers, so an unrestored
    "self" threw away the restored history and republished an empty state
    to the other walkers.
    """
    system, variable = makeHarmonicSystem()
    a = OPES(
        system,
        [variable],
        300.0,
        20.0,
        100,
        10,
        saveFrequency=100,
        biasDir=str(tmp_path),
    )
    runSteps(a, system, 2000)
    assert a._kde["self"].getNumKernels() > 0

    system2, variable2 = makeHarmonicSystem()
    b = OPES(
        system2,
        [variable2],
        300.0,
        20.0,
        100,
        10,
        saveFrequency=100,
        biasDir=str(tmp_path),
    )
    b.setState(a.getState())
    assert b._kde["self"].getNumKernels() == a._kde["self"].getNumKernels()
    assert b._kde["self.rw"].getNumKernels() == a._kde["self.rw"].getNumKernels()
    assert b._variance["self"].get() == pytest.approx(a._variance["self"].get())


def test_frequency_and_stats_window_size_must_be_positive():
    """Regression for review finding 5: both reached a ZeroDivisionError."""
    with pytest.raises(ValueError, match="frequency must be positive"):
        makeOPES(frequency=0)
    with pytest.raises(ValueError, match="statsWindowSize must be positive"):
        makeOPES(statsWindowSize=0)


def test_free_energy_docstring_grid_spacing_matches_the_actual_axis():
    """Regression for review finding 6: the docstring stated the wrong axis."""
    sampler = makeOPES()
    fes = sampler.getFreeEnergy()
    gridWidth = 51
    assert len(fes) == gridWidth
    documented = np.linspace(-2.0, 2.0, gridWidth)
    spacing = (2.0 - -2.0) / (gridWidth - 1)
    assert documented[1] - documented[0] == pytest.approx(spacing)
    assert "gridWidth-1" in OPES.getFreeEnergy.__doc__


# --- Regression tests for the second code review ---------------------------


def test_warmup_counts_its_own_steps_not_the_simulation_clock():
    """warmupSteps was compared to simulation.currentStep, so a simulation
    equilibrated beforehand ended warm-up at the first interval, on one
    zero-deviation sample, and raised."""
    system, variable = makeHarmonicSystem()
    sampler = OPES(system, [variable], 300.0, 20.0, 100, 10, warmupSteps=500)
    simulation = runSteps(sampler, system, 490, stepsBefore=1000)
    assert not sampler._warmupComplete
    sampler.step(simulation, 10)
    assert sampler._warmupComplete


def test_warmup_progress_survives_a_restart():
    system, variable = makeHarmonicSystem()
    sampler = OPES(system, [variable], 300.0, 20.0, 100, 10, warmupSteps=500)
    runSteps(sampler, system, 300)
    assert not sampler._warmupComplete

    system2, variable2 = makeHarmonicSystem()
    restored = OPES(system2, [variable2], 300.0, 20.0, 100, 10, warmupSteps=500)
    restored.setState(sampler.getState())
    runSteps(restored, system2, 200)
    assert restored._warmupComplete


def test_set_state_restores_the_running_cv_mean():
    """The running CV mean and its sample count were not saved, so a restored
    sampler restarted its mean from the next sample and its window from zero."""
    system, variable = makeHarmonicSystem()
    sampler = OPES(system, [variable], 300.0, 20.0, 100, 10)
    runSteps(sampler, system, 500)

    system2, variable2 = makeHarmonicSystem()
    restored = OPES(system2, [variable2], 300.0, 20.0, 100, 10)
    restored.setState(sampler.getState())
    assert restored._counter == sampler._counter
    assert restored._sampleMean == pytest.approx(sampler._sampleMean)


def test_add_kernel_accepts_a_plain_list_variance():
    sampler = makeOPES()
    assert sampler.addKernel([0.0], 0.0, variance=[0.01])
    assert sampler.getNumKernels() == 1


@pytest.mark.parametrize("barrier", [0.0, -20.0])
def test_non_positive_barrier_is_rejected_even_with_an_explicit_bias_factor(barrier):
    """Regression: with biasFactor given, a barrier <= 0 slipped through.

    That makes log(epsilon) = -barrier/prefactor >= 0, i.e. epsilon >= 1,
    which swamps P/Z and leaves the bias meaningless.
    """
    system, variable = makeSystemAndVariable()
    with pytest.raises(ValueError, match="barrier must be positive"):
        OPES(system, [variable], 300.0, barrier, 100, 10, biasFactor=5.0)


def test_average_density_of_an_empty_estimate_is_nan_without_warnings():
    """Regression: log(0) on the empty kernel list raised RuntimeWarnings."""
    sampler = makeOPES()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(sampler.getAverageDensity())


def test_a_sync_does_not_resize_the_walkers_own_kernels(tmp_path):
    """Regression: each walker sized kernels by its OWN effective sample size.

    Kernels deposited between syncs went into "total" sized by the shared
    sample size, but "self" got a wider copy sized by the walker's own. The
    next sync rebuilt "total" from "self", so the bias jumped at every sync
    and the shared estimate ended up wider by numWalkers^(1/(d+4)).
    """

    def walker(walkerId):
        system, variable = makeSystemAndVariable()
        return OPES(
            system,
            [variable],
            300.0,
            20.0,
            100,
            None,
            saveFrequency=100,
            biasDir=str(tmp_path),
            walkerId=walkerId,
            compressionThreshold=0.0,
        )

    rng = np.random.default_rng(0)

    def deposit(sampler, n):
        for _ in range(n):
            sampler.addKernel([rng.normal()], rng.normal())

    a, b = walker(1), walker(2)
    deposit(b, 20)
    b._syncWithDisk()
    deposit(a, 20)
    a._syncWithDisk()  # a's total now holds a's 20 kernels, then b's 20
    deposit(a, 5)
    before = {
        key: [k.bandwidth.copy() for k in a._kde[key]._kernels[-5:]]
        for key in ("total", "total.rw")
    }
    deposit(b, 1)
    b._syncWithDisk()
    a._syncWithDisk()  # b advanced, so this rebuilds a's total: a's 25 first
    for key, bandwidths in before.items():
        after = [k.bandwidth for k in a._kde[key]._kernels[20:25]]
        assert np.concatenate(after) == pytest.approx(np.concatenate(bandwidths))


def test_adaptive_variance_waits_one_stats_window_before_the_first_kernel():
    """Regression: the first kernel went down after a single deposition stride.

    With frequency=100 and varianceFrequency=10 that sized it from ten
    correlated samples, the first of which always contributes zero, and
    compression kept the resulting too-narrow kernels. The first deposition
    now waits for statsWindowSize strides (tau = 100 samples, 1000 steps).
    """
    system, variable = makeHarmonicSystem()
    sampler = OPES(system, [variable], 300.0, 20.0, 100, 10)
    runSteps(sampler, system, 900)
    assert sampler.getNumKernels() == 0
    runSteps(sampler, system, 100)
    assert sampler.getNumKernels() == 1


def test_the_first_kernel_wait_does_not_apply_to_a_fixed_bandwidth():
    system, variable = makeHarmonicSystem()
    sampler = OPES(system, [variable], 300.0, 20.0, 100, None)
    runSteps(sampler, system, 100)
    assert sampler.getNumKernels() == 1


def test_a_sync_that_loads_peer_kernels_refreshes_the_context(tmp_path):
    """Regression: the context was refreshed before the sync, not after.

    Kernels loaded from peers only reached the forces at the next
    deposition, one frequency later.
    """

    def walker(walkerId):
        system, variable = makeHarmonicSystem()
        sampler = OPES(
            system,
            [variable],
            300.0,
            20.0,
            100,
            None,
            saveFrequency=100,
            biasDir=str(tmp_path),
            walkerId=walkerId,
        )
        return sampler, system

    b, _ = walker(2)
    b.addKernel([0.5], 0.0)
    b._syncWithDisk()
    a, system = walker(1)
    runSteps(a, system, 100)
    assert a._kde["total"].getNumKernels() == 2
    tabulated = np.array(a._force.getTabulatedFunction(0).getFunctionParameters()[0])
    bias = a.getBias().value_in_unit(unit.kilojoules_per_mole)
    assert tabulated == pytest.approx(bias.ravel())
