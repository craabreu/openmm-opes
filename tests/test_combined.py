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


def doubleWell():
    """U(x) = 25*(1 - x^2)^2 kJ/mol, with x as the collective variable."""
    system = openmm.System()
    system.addParticle(12.0)
    potential = openmm.CustomExternalForce("25*(1-x^2)^2")
    potential.addParticle(0, [])
    system.addForce(potential)
    cv = openmm.CustomExternalForce("x")
    cv.addParticle(0, [])
    return system, app.BiasVariable(cv, -2.0, 2.0, 0.15, False, 61)


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


def makeCombined(varianceFrequency=None, **kwargs):
    system, variable = doubleWell()
    sampler = CombinedOPES(
        system, [variable], TEMPERATURE, 30.0, 15.0, 100, varianceFrequency, **kwargs
    )
    return sampler, simulate(system)


def recordWeights(sampler, monkeypatch):
    """Record the bias energy every addKernel call of a sampler receives."""
    energies = []
    original = sampler.addKernel

    def spy(values, biasEnergy, variance=None):
        energies.append(biasEnergy.value_in_unit(KJ))
        return original(values, biasEnergy, variance)

    monkeypatch.setattr(sampler, "addKernel", spy)
    return energies


@pytest.fixture
def pairAndCombined(monkeypatch):
    """Separate OPES and OPES-explore samplers and a CombinedOPES, same run."""
    system, variable = doubleWell()
    _, exploreVariable = doubleWell()
    opes = OPES(system, [variable], TEMPERATURE, 30.0, 100, None)
    explore = OPES(
        system, [exploreVariable], TEMPERATURE, 15.0, 100, None, exploreMode=True
    )
    pairWeights = (
        recordWeights(opes, monkeypatch),
        recordWeights(explore, monkeypatch),
    )
    simulation = simulate(system)
    OPES.stepTogether(simulation, [opes, explore], 2000)

    combined, combinedSimulation = makeCombined()
    components = combined._components.values()
    combinedWeights = tuple(recordWeights(c, monkeypatch) for c in components)
    combined.step(combinedSimulation, 2000)
    return (opes, explore, pairWeights), (combined, combinedWeights)


def test_matches_opes_and_explore_stepped_together(pairAndCombined):
    (opes, explore, (opesWeights, exploreWeights)), (combined, weights) = (
        pairAndCombined
    )
    assert len(opesWeights) == 20
    # Kernels of the OPES component are weighted by the OPES bias alone.
    assert weights[0] == pytest.approx(opesWeights, abs=1e-6)
    # The explore component receives the total bias, for the Eq. 14 estimate.
    total = np.add(opesWeights, exploreWeights)
    assert weights[1] == pytest.approx(total, abs=1e-6)
    for separate, joint in zip(
        (opes.getBias(), explore.getBias()), combined.getComponentBiases(), strict=True
    ):
        assert joint.value_in_unit(KJ) == pytest.approx(
            separate.value_in_unit(KJ), abs=1e-6
        )
    assert combined.getNumKernels() == (opes.getNumKernels(), explore.getNumKernels())


def test_free_energy_reweights_samples_by_the_total_bias(pairAndCombined):
    (_, _, (opesWeights, exploreWeights)), (combined, _) = pairAndCombined
    total = np.add(opesWeights, exploreWeights) / KT
    estimate = combined._components["explore"]._kde["total.rw"]
    logSumW = total.max() + np.log(np.exp(total - total.max()).sum())
    assert estimate._logSumW == pytest.approx(logSumW, abs=1e-9)
    assert combined.getFreeEnergy().value_in_unit(KJ) == pytest.approx(
        -KT * estimate.getLogPDF(), abs=1e-9
    )


def test_component_bias_energies_add_up_and_leave_the_force_switched_on():
    combined, simulation = makeCombined()
    combined.step(simulation, 1000)
    opesEnergy, exploreEnergy = combined.getComponentBiasEnergies(simulation.context)
    assert simulation.context.getParameter(combined._scale) == 1.0
    total = combined.getBiasEnergy(simulation.context)
    assert (opesEnergy + exploreEnergy).value_in_unit(KJ) == pytest.approx(
        total.value_in_unit(KJ), abs=1e-9
    )
    x = combined.getCollectiveVariables(simulation)[0]
    grid = np.linspace(-2, 2, 61)
    for energy, bias in zip(
        (opesEnergy, exploreEnergy), combined.getComponentBiases(), strict=True
    ):
        assert energy.value_in_unit(KJ) == pytest.approx(
            np.interp(x, grid, bias.value_in_unit(KJ)), abs=0.5
        )


def test_the_bias_is_the_sum_of_the_components():
    combined, simulation = makeCombined()
    combined.step(simulation, 1000)
    opes, explore = combined.getComponentBiases()
    assert combined.getBias().value_in_unit(KJ) == pytest.approx(
        (opes + explore).value_in_unit(KJ)
    )


def test_an_explore_barrier_above_the_opes_barrier_warns():
    system, variable = doubleWell()
    with pytest.warns(UserWarning, match="exploreBarrier exceeds barrier"):
        CombinedOPES(system, [variable], TEMPERATURE, 15.0, 30.0, 100, None)


def test_bias_factors_default_to_each_barrier_over_kT():
    combined, _ = makeCombined()
    opes, explore = combined._components.values()
    assert opes.biasFactor == pytest.approx(30.0 / KT)
    assert explore.biasFactor == pytest.approx(15.0 / KT)
    assert not opes.exploreMode
    assert explore.exploreMode


def test_adaptive_variance_waits_then_deposits_in_both_components():
    combined, simulation = makeCombined(varianceFrequency=20)
    combined.step(simulation, 900)
    assert combined.getNumKernels() == (0, 0)
    combined.step(simulation, 1100)
    numOpes, numExplore = combined.getNumKernels()
    assert numOpes > 0
    assert numExplore > 0


def test_state_round_trips_into_a_fresh_sampler():
    combined, simulation = makeCombined(varianceFrequency=20)
    combined.step(simulation, 3000)
    restored, restoredSimulation = makeCombined(varianceFrequency=20)
    restored.setState(combined.getState())
    for before, after in zip(
        combined.getComponentBiases(), restored.getComponentBiases(), strict=True
    ):
        assert after.value_in_unit(KJ) == pytest.approx(before.value_in_unit(KJ))
    assert restored.getFreeEnergy().value_in_unit(KJ) == pytest.approx(
        combined.getFreeEnergy().value_in_unit(KJ)
    )
    restored.updateContext(restoredSimulation.context)
    restoredSimulation.context.setPositions(
        simulation.context.getState(getPositions=True).getPositions()
    )
    assert restored.getBiasEnergy(restoredSimulation.context).value_in_unit(
        KJ
    ) == pytest.approx(combined.getBiasEnergy(simulation.context).value_in_unit(KJ))


def test_two_combined_samplers_in_one_system_get_distinct_switches():
    system, variable = doubleWell()
    _, other = doubleWell()
    first = CombinedOPES(system, [variable], TEMPERATURE, 30.0, 15.0, 100, None)
    second = CombinedOPES(system, [other], TEMPERATURE, 30.0, 15.0, 100, None)
    assert first._scale != second._scale


def test_walkers_share_both_components(tmp_path):
    samplers = []
    for index in range(2):
        system, variable = doubleWell()
        sampler = CombinedOPES(
            system,
            [variable],
            TEMPERATURE,
            30.0,
            15.0,
            100,
            None,
            saveFrequency=200,
            biasDir=str(tmp_path),
        )
        sampler.step(simulate(system, seed=1000 * (index + 1)), 2000)
        samplers.append(sampler)

    def numSamples(kde):
        return sum(kernel.numSamples for kernel in kde._kernels)

    first, second = samplers
    for name, key in (("opes", "rw"), ("explore", "")):
        suffix = f".{key}" if key else ""
        mine = second._components[name]._kde
        theirs = first._components[name]._kde
        assert numSamples(mine[f"total{suffix}"]) == numSamples(
            mine[f"self{suffix}"]
        ) + numSamples(theirs[f"self{suffix}"])
        assert (tmp_path / name).is_dir()
    walkerIds = {c._sharer.walkerId for c in second._components.values()}
    assert len(walkerIds) == 1


def test_step_together_accepts_a_combined_sampler_with_another_sampler():
    system, variable = doubleWell()
    _, other = doubleWell()
    combined = CombinedOPES(system, [variable], TEMPERATURE, 30.0, 15.0, 100, None)
    opes = OPES(system, [other], TEMPERATURE, 30.0, 150, None)
    simulation = simulate(system)
    OPES.stepTogether(simulation, [combined, opes], 3000)
    opesComponent = combined._components["opes"]._kde["total.rw"]
    assert sum(kernel.numSamples for kernel in opesComponent._kernels) == 30
    assert opes.getNumKernels() > 0


def test_free_energy_can_correct_the_smoothing_bias():
    combined, simulation = makeCombined()
    combined.step(simulation, 3000)
    estimate = combined._components["explore"]._kde["total.rw"]
    kT = combined._components["explore"]._kbt.value_in_unit(KJ)
    corrected = combined.getFreeEnergy(correctSmoothing=True, pilotFactor=1.5)
    assert corrected.value_in_unit(KJ) == pytest.approx(
        -kT * estimate.getCorrectedLogPDF(1.5)
    )
