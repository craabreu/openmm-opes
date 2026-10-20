import functools

import numpy as np
import pytest

from openmm_opes.kde import OnlineKDE
from tests.helpers import makeSpace


def loopedGrid(kde):
    return functools.reduce(
        np.logaddexp,
        (k.evaluateOnGrid() for k in kde._kernels),
        np.full(kde._cvSpace.gridShape, -np.inf),
    )


def filled(space, shape="gaussian", count=300, scale=0.4, seed=0, threshold=1.0):
    kde = OnlineKDE(space, compressionThreshold=threshold, kernelShape=shape)
    rng = np.random.default_rng(seed)
    d = space.numDimensions
    for _ in range(count):
        kde.update(rng.normal(0.0, 1.0, d), rng.normal(), np.full(d, scale**2))
    return kde


CASES = {
    "1d": (makeSpace((-4.0, 4.0, 81, False)),),
    "1d reflective": (makeSpace((-2.0, 2.0, 61, False), bounded=True),),
    "2d periodic": (makeSpace((-np.pi, np.pi, 51, True), (-np.pi, np.pi, 41, True)),),
    "2d reflective": (
        makeSpace((-2.0, 2.0, 31, False), (-3.0, 3.0, 41, False), bounded=True),
    ),
    "3d": (
        makeSpace(
            (-3.0, 3.0, 21, False), (-3.0, 3.0, 17, False), (-3.0, 3.0, 13, False)
        ),
    ),
}


@pytest.mark.parametrize("shape", ["gaussian", "compact"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_vectorized_grid_matches_the_per_kernel_loop(case, shape):
    (space,) = CASES[case]
    kde = filled(space, shape)
    expected = loopedGrid(kde)
    actual = kde._logMixtureOnGrid(kde._positions, kde._bandwidths, kde._logHeights)
    finite = np.isfinite(expected)
    assert np.array_equal(finite, np.isfinite(actual))
    assert actual[finite] == pytest.approx(expected[finite], rel=1e-10, abs=1e-10)


def test_far_tails_fall_back_to_log_space_instead_of_underflowing():
    space = makeSpace((-60.0, 60.0, 121, False), (-60.0, 60.0, 121, False))
    kde = filled(space, count=50, scale=0.3, threshold=0.0)
    expected = loopedGrid(kde)
    assert expected.min() < -800  # beyond the range of a double
    actual = kde._logMixtureOnGrid(kde._positions, kde._bandwidths, kde._logHeights)
    assert np.all(np.isfinite(actual))
    assert actual == pytest.approx(expected, rel=1e-10)


def test_an_empty_mixture_is_minus_infinity_everywhere():
    space = makeSpace((-1.0, 1.0, 11, False))
    kde = OnlineKDE(space)
    empty = np.empty((0, 1))
    assert np.all(kde._logMixtureOnGrid(empty, empty, np.empty(0)) == -np.inf)


def test_rebuilt_caches_match_the_incremental_ones():
    space = makeSpace((-np.pi, np.pi, 61, True), (-np.pi, np.pi, 61, True))
    kde = filled(space, count=400, scale=0.3)
    incremental = kde._logPG.copy()
    kde._rebuildCaches()
    assert kde._logPG == pytest.approx(incremental, rel=1e-8, abs=1e-8)


def test_periodic_grid_ends_are_exactly_equal_after_a_rebuild():
    """OpenMM rejects a periodic table whose first and last values differ."""
    space = makeSpace((-np.pi, np.pi, 101, True), (-np.pi, np.pi, 101, True))
    kde = OnlineKDE(space)
    rng = np.random.default_rng(7)
    for _ in range(2000):
        kde.update(rng.uniform(-np.pi, np.pi, 2), rng.normal(), np.full(2, 0.04**2))
    kde._rebuildCaches()
    assert np.array_equal(kde._logPG[-1, :], kde._logPG[0, :])
    assert np.array_equal(kde._logPG[:, -1], kde._logPG[:, 0])
    corrected = kde.getCorrectedLogPDF()
    assert np.array_equal(corrected[-1, :], corrected[0, :])
    assert np.array_equal(corrected[:, -1], corrected[:, 0])
