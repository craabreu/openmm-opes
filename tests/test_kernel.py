import numpy as np
import pytest
from scipy import integrate

from openmm_opes.kde import COMPACT, Kernel
from tests.helpers import makeSpace


def test_compact_shape_normalization_constant_is_exact():
    value, _ = integrate.quad(lambda x: (9 - x * x) ** 4, -3, 3)
    assert np.exp(COMPACT.logNorm) == pytest.approx(value)
    assert np.exp(COMPACT.logNorm) == pytest.approx(559872 / 35)


def test_gaussian_kernel_integrates_to_its_weight():
    space = makeSpace((-10.0, 10.0, 2001, False))
    kernel = Kernel(space, [0.3], [0.7], np.log(2.5))
    xs = np.linspace(-10, 10, 200001)
    values = np.exp(kernel.evaluate(xs.reshape(-1, 1)))
    # Riemann sum rather than np.trapezoid, which needs numpy >= 2.0
    mass = values.sum() * (xs[1] - xs[0])
    assert mass == pytest.approx(2.5, rel=1e-5)


def test_compact_kernel_integrates_to_its_weight():
    space = makeSpace((-10.0, 10.0, 2001, False))
    kernel = Kernel(space, [0.0], [0.7], np.log(2.5), shape=COMPACT)
    xs = np.linspace(-10, 10, 200001)
    values = np.exp(kernel.evaluate(xs.reshape(-1, 1)))
    mass = values.sum() * (xs[1] - xs[0])
    assert mass == pytest.approx(2.5, rel=1e-5)


def test_compact_kernel_has_support_of_three_bandwidths():
    space = makeSpace((-10.0, 10.0, 201, False))
    kernel = Kernel(space, [0.0], [0.5], 0.0, shape=COMPACT)
    assert np.isfinite(kernel.evaluate(np.array([[1.49]])))[0]
    assert np.isneginf(kernel.evaluate(np.array([[1.51]])))[0]


def test_zero_bandwidth_gives_a_null_kernel():
    space = makeSpace((-1.0, 1.0, 11, False))
    assert np.isneginf(Kernel(space, [0.0], [0.0], 0.0).logHeight)


def test_find_nearest_returns_index_and_squared_mahalanobis_distance():
    space = makeSpace((-10.0, 10.0, 101, False))
    kernel = Kernel(space, [0.0], [2.0], 0.0)
    centers = np.array([[6.0], [1.0], [-8.0]])
    bandwidths = np.array([[2.0], [2.0], [2.0]])
    index, sqDist = kernel.findNearest(centers, bandwidths)
    assert index == 1
    assert sqDist == pytest.approx(0.25)


def test_find_nearest_honours_ignored_indices():
    space = makeSpace((-10.0, 10.0, 101, False))
    kernel = Kernel(space, [0.0], [2.0], 0.0)
    centers = np.array([[6.0], [1.0], [-8.0]])
    bandwidths = np.full((3, 1), 2.0)
    index, _ = kernel.findNearest(centers, bandwidths, ignore=[1])
    assert index == 0


def test_find_nearest_on_empty_center_list():
    space = makeSpace((-10.0, 10.0, 101, False))
    index, sqDist = Kernel(space, [0.0], [1.0], 0.0).findNearest(
        np.empty((0, 1)), np.empty((0, 1))
    )
    assert index == -1
    assert np.isinf(sqDist)


def test_merge_preserves_weight_mean_and_second_moment():
    space = makeSpace((-10.0, 10.0, 501, False))
    merged = Kernel(space, [0.0], [1.0], np.log(3.0))
    other = Kernel(space, [2.0], [0.5], np.log(1.0))
    merged.merge(other)

    w1, w2 = 3.0, 1.0
    total = w1 + w2
    meanRef = (w1 * 0.0 + w2 * 2.0) / total
    varRef = (w1 * 1.0**2 + w2 * 0.5**2) / total + (w1 * w2 / total**2) * 2.0**2

    assert np.exp(merged.logWeight) == pytest.approx(total)
    assert merged.position[0] == pytest.approx(meanRef)
    assert merged.bandwidth[0] ** 2 == pytest.approx(varRef)
    assert merged.numSamples == 2


def test_evaluate_on_grid_matches_direct_evaluation():
    space = makeSpace((-4.0, 4.0, 41, False), (-4.0, 4.0, 31, False))
    kernel = Kernel(space, [0.5, -1.0], [0.8, 1.2], np.log(2.0))
    onGrid = kernel.evaluateOnGrid()
    assert onGrid.shape == space.gridShape

    xs = np.linspace(-4, 4, 41)
    ys = np.linspace(-4, 4, 31)
    points = np.stack(np.meshgrid(xs, ys), axis=-1)
    assert onGrid == pytest.approx(kernel.evaluate(points))


def test_copy_does_not_share_arrays():
    from copy import copy

    space = makeSpace((-4.0, 4.0, 41, False))
    kernel = Kernel(space, [0.5], [0.8], 0.0)
    duplicate = copy(kernel)
    duplicate.position[0] = 99.0
    assert kernel.position[0] == pytest.approx(0.5)
