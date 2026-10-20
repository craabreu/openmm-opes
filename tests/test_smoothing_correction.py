"""The multiplicative smoothing correction of the kernel density estimate."""

import numpy as np
import pytest

from openmm_opes.kde import OnlineKDE
from tests.helpers import makeSpace

GRID = np.linspace(-2.0, 2.0, 401)
EXACT = 5 * (1 - GRID**2) ** 2  # free energy in kT: a double well


def doubleWellKDE(numSamples=5_000, seed=4, bounded=False):
    """An estimate of exp(-EXACT) from exact samples, with 0.05-wide kernels.

    The pilot kernels must stay narrow next to the wells, about 0.3 wide:
    with 0.1-wide kernels, the correction no longer helps.
    """
    p = np.exp(-EXACT) / np.exp(-EXACT).sum()
    rng = np.random.default_rng(seed)
    samples = rng.choice(GRID, size=numSamples, p=p)
    samples += rng.uniform(-0.005, 0.005, numSamples)
    kde = OnlineKDE(makeSpace((-2.0, 2.0, 401, False), bounded=bounded))
    for x in np.clip(samples, -2.0, 2.0):
        kde.update(np.array([x]), 0.0, np.array([0.0025]), 1.0)
    return kde


def error(logPDF):
    """RMS deviation from the exact free energy below 6 kT, offset removed."""
    diff = (-logPDF - EXACT)[EXACT < 6]
    return np.sqrt(np.mean((diff - diff.mean()) ** 2))


def basinFreeEnergy(logPDF):
    """F(x > 0) - F(x < 0) in kT."""
    p = np.exp(logPDF - np.nanmax(logPDF))
    return -np.log(p[GRID > 0].sum() / p[GRID < 0].sum())


@pytest.mark.parametrize("bounded", [False, True])
def test_the_correction_reduces_the_error_on_a_double_well(bounded):
    kde = doubleWellKDE(bounded=bounded)
    assert error(kde.getCorrectedLogPDF()) < 0.75 * error(kde.getLogPDF())


def test_the_correction_keeps_the_basin_free_energy():
    kde = doubleWellKDE()
    assert basinFreeEnergy(kde.getCorrectedLogPDF()) == pytest.approx(
        basinFreeEnergy(kde.getLogPDF()), abs=0.01
    )


def test_without_rounds_and_broadening_it_reduces_to_the_standard_estimate():
    kde = doubleWellKDE(numSamples=1_000)
    assert kde.getCorrectedLogPDF(pilotFactor=1.0, iterations=0) == pytest.approx(
        kde.getLogPDF(), abs=1e-9
    )


def test_more_rounds_and_wider_pilots_run_and_stay_finite():
    kde = doubleWellKDE(numSamples=5_000)
    for pilotFactor, iterations in ((1.5, 1), (3.0, 2)):
        corrected = kde.getCorrectedLogPDF(pilotFactor, iterations)
        assert np.all(np.isfinite(corrected))


def test_the_corrected_density_is_normalized_like_the_standard_one():
    kde = doubleWellKDE(numSamples=5_000)
    spacing = GRID[1] - GRID[0]
    assert np.exp(kde.getCorrectedLogPDF()).sum() * spacing == pytest.approx(
        np.exp(kde.getLogPDF()).sum() * spacing, rel=1e-2
    )


def test_periodic_grids_repeat_the_first_node_at_the_end():
    space = makeSpace((-np.pi, np.pi, 61, True), (-np.pi, np.pi, 61, True))
    kde = OnlineKDE(space)
    rng = np.random.default_rng(5)
    for _ in range(300):
        kde.update(rng.normal(0.0, 0.8, 2), 0.0, np.array([0.04, 0.04]))
    corrected = kde.getCorrectedLogPDF()
    assert np.all(np.isfinite(corrected))
    assert corrected[-1, :] == pytest.approx(corrected[0, :])
    assert corrected[:, -1] == pytest.approx(corrected[:, 0])


def test_an_empty_estimate_stays_nan():
    kde = OnlineKDE(makeSpace((-1.0, 1.0, 11, False)))
    assert np.all(np.isnan(kde.getCorrectedLogPDF()))
