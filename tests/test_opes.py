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
    return OPES(system, [variable], 300.0, 20.0, 100, **kwargs)


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
