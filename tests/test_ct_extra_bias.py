import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

from openmm_opes import OPES, CombinedOPES  # noqa: E402

TEMPERATURE = 300.0
KT = (unit.MOLAR_GAS_CONSTANT_R * TEMPERATURE * unit.kelvin).value_in_unit(
    unit.kilojoules_per_mole
)
KJ = unit.kilojoules_per_mole
EXTRA_GROUP = 5


def doubleWell(withExtra=True, periodic=False):
    """U(x) = 25*(1 - x^2)^2 kJ/mol, plus a tilt 3*x in group EXTRA_GROUP."""
    system = openmm.System()
    system.addParticle(12.0)
    potential = openmm.CustomExternalForce("25*(1-x^2)^2")
    potential.addParticle(0, [])
    system.addForce(potential)
    tilt = None
    if withExtra:
        tilt = openmm.CustomExternalForce("3*x")
        tilt.addParticle(0, [])
        tilt.setForceGroup(EXTRA_GROUP)
        system.addForce(tilt)
    return system, tilt


def variable(periodic=False):
    cv = openmm.CustomExternalForce("x")
    cv.addParticle(0, [])
    if periodic:
        return app.BiasVariable(cv, -np.pi, np.pi, 0.15, True, 61)
    return app.BiasVariable(cv, -2.0, 2.0, 0.15, False, 61)


def simulate(system, seed=1234):
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
    return simulation


def recordWeights(kdeOwner, monkeypatch):
    """Record the bias energy each addKernel call receives."""
    energies = []
    original = kdeOwner.addKernel

    def spy(values, biasEnergy, variance=None):
        energies.append(biasEnergy.value_in_unit(KJ))
        return original(values, biasEnergy, variance)

    monkeypatch.setattr(kdeOwner, "addKernel", spy)
    return energies


def extraEnergy(context):
    state = context.getState(getEnergy=True, groups={EXTRA_GROUP})
    return state.getPotentialEnergy().value_in_unit(KJ)


@pytest.mark.parametrize("form", ["force", "group"])
def test_kernel_weights_include_the_extra_bias(form, monkeypatch):
    system, tilt = doubleWell()
    extra = tilt if form == "force" else EXTRA_GROUP
    sampler = OPES(
        system, [variable()], TEMPERATURE, 30.0, 100, None, extraBias=[extra]
    )
    simulation = simulate(system)
    expected = []
    original = sampler.addKernel

    def spy(values, biasEnergy, variance=None):
        own = sampler.getBiasEnergy(simulation.context).value_in_unit(KJ)
        expected.append(own + extraEnergy(simulation.context))
        assert biasEnergy.value_in_unit(KJ) == pytest.approx(expected[-1], abs=1e-9)
        return original(values, biasEnergy, variance)

    monkeypatch.setattr(sampler, "addKernel", spy)
    sampler.step(simulation, 2000)
    assert len(expected) == 20


def test_another_sampler_can_be_an_extra_bias(monkeypatch):
    system, _ = doubleWell(withExtra=False)
    other = OPES(system, [variable()], TEMPERATURE, 30.0, 100, None)
    sampler = OPES(
        system, [variable()], TEMPERATURE, 30.0, 100, None, extraBias=[other]
    )
    simulation = simulate(system)
    expected = []
    original = sampler.addKernel

    def spy(values, biasEnergy, variance=None):
        context = simulation.context
        own = sampler.getBiasEnergy(context).value_in_unit(KJ)
        expected.append(own + other.getBiasEnergy(context).value_in_unit(KJ))
        assert biasEnergy.value_in_unit(KJ) == pytest.approx(expected[-1], abs=1e-9)
        return original(values, biasEnergy, variance)

    monkeypatch.setattr(sampler, "addKernel", spy)
    OPES.stepTogether(simulation, [other, sampler], 2000)
    assert len(expected) == 20
    assert other.getNumKernels() > 0


def test_the_extra_bias_cannot_include_the_samplers_own_group():
    system, tilt = doubleWell()
    sampler = OPES(system, [variable()], TEMPERATURE, 30.0, 100, None, extraBias=[tilt])
    tilt.setForceGroup(sampler._force.getForceGroup())
    with pytest.raises(ValueError, match="force group of this sampler"):
        sampler.step(simulate(system), 200)


def test_extra_bias_items_must_be_groups_forces_or_samplers():
    system, _ = doubleWell()
    with pytest.raises(TypeError, match="extraBias items"):
        OPES(system, [variable()], TEMPERATURE, 30.0, 100, None, extraBias=["tilt"])


def test_in_explore_mode_only_the_reweighted_estimate_sees_the_extra_bias():
    samplers = []
    for extra in (None, [EXTRA_GROUP]):
        system, _ = doubleWell()
        sampler = OPES(
            system,
            [variable()],
            TEMPERATURE,
            30.0,
            100,
            None,
            exploreMode=True,
            extraBias=extra,
        )
        sampler.step(simulate(system), 2000)
        samplers.append(sampler)
    plain, extra = samplers
    assert extra.getBias().value_in_unit(KJ) == pytest.approx(
        plain.getBias().value_in_unit(KJ), abs=1e-9
    )
    assert not np.allclose(
        extra.getFreeEnergy().value_in_unit(KJ),
        plain.getFreeEnergy().value_in_unit(KJ),
    )


def test_running_ct_is_kT_ln_of_the_mean_kernel_weight(monkeypatch):
    system, _ = doubleWell()
    sampler = OPES(system, [variable()], TEMPERATURE, 30.0, 100, None, extraBias=[5])
    weights = recordWeights(sampler, monkeypatch)
    sampler.step(simulate(system), 3000)
    expected = KT * np.log(np.mean(np.exp(np.array(weights) / KT)))
    assert sampler.getCt("running").value_in_unit(KJ) == pytest.approx(expected)


@pytest.mark.parametrize("periodic", [False, True])
def test_current_ct_averages_exp_minus_beta_v_over_the_estimate(periodic):
    system, _ = doubleWell(withExtra=False)
    sampler = OPES(system, [variable(periodic)], TEMPERATURE, 30.0, 100, None)
    sampler.step(simulate(system), 3000)
    fes = sampler.getFreeEnergy().value_in_unit(KJ)
    bias = sampler.getBias().value_in_unit(KJ)
    if periodic:
        fes, bias = fes[:-1], bias[:-1]
    p = np.exp(-(fes - np.nanmin(fes)) / KT)
    expected = -KT * np.log(np.sum(p * np.exp(-bias / KT)) / np.sum(p))
    assert sampler.getCt().value_in_unit(KJ) == pytest.approx(expected)


def test_ct_is_nan_before_the_first_kernel_and_checks_its_argument():
    system, _ = doubleWell()
    sampler = OPES(system, [variable()], TEMPERATURE, 30.0, 100, None)
    assert np.isnan(sampler.getCt().value_in_unit(KJ))
    assert np.isnan(sampler.getCt("running").value_in_unit(KJ))
    with pytest.raises(ValueError, match="estimate must be"):
        sampler.getCt("final")


def test_ct_estimates_agree_once_the_bias_settles():
    system, _ = doubleWell(withExtra=False)
    sampler = OPES(system, [variable()], TEMPERATURE, 30.0, 500, None)
    sampler.step(simulate(system), 200_000)
    current = sampler.getCt().value_in_unit(KJ)
    running = sampler.getCt("running").value_in_unit(KJ)
    assert current == pytest.approx(running, abs=1.0)


def test_combined_weights_include_the_extra_bias(monkeypatch):
    system, tilt = doubleWell()
    combined = CombinedOPES(
        system, [variable()], TEMPERATURE, 30.0, 15.0, 100, None, extraBias=[tilt]
    )
    simulation = simulate(system)
    opes, explore = combined._components.values()
    opesWeights = recordWeights(opes, monkeypatch)
    exploreWeights = recordWeights(explore, monkeypatch)
    expected = []
    original = combined.getComponentBiasEnergies

    def spy(context):
        energies = original(context)
        values = [e.value_in_unit(KJ) for e in energies]
        extra = extraEnergy(context)
        expected.append((values[0] + extra, values[0] + values[1] + extra))
        return energies

    monkeypatch.setattr(combined, "getComponentBiasEnergies", spy)
    combined.step(simulation, 2000)
    assert len(opesWeights) == 20
    assert opesWeights == pytest.approx([o for o, _ in expected], abs=1e-9)
    assert exploreWeights == pytest.approx([t for _, t in expected], abs=1e-9)


def test_combined_ct_uses_the_total_bias_and_the_free_energy_estimate(monkeypatch):
    system, _ = doubleWell(withExtra=False)
    combined = CombinedOPES(system, [variable()], TEMPERATURE, 30.0, 15.0, 100, None)
    explore = combined._components["explore"]
    weights = recordWeights(explore, monkeypatch)
    combined.step(simulate(system), 3000)
    expected = KT * np.log(np.mean(np.exp(np.array(weights) / KT)))
    assert combined.getCt("running").value_in_unit(KJ) == pytest.approx(expected)
    fes = combined.getFreeEnergy().value_in_unit(KJ)
    bias = combined.getBias().value_in_unit(KJ)
    p = np.exp(-(fes - np.nanmin(fes)) / KT)
    current = -KT * np.log(np.sum(p * np.exp(-bias / KT)) / np.sum(p))
    assert combined.getCt().value_in_unit(KJ) == pytest.approx(current)
