import numpy as np
import pytest

from openmm_opes.kde import OnlineKDE
from tests.helpers import makeSpace


def directLogPDF(kde, grid):
    """Mixture density on a 1-D grid, computed straight from the kernel list."""
    total = np.zeros_like(grid)
    for kernel in kde._kernels:
        total += np.exp(kernel.evaluate(grid.reshape(-1, 1)))
    return np.log(total) - kde._logSumW


def test_empty_kde_is_falsy_and_has_no_kernels():
    kde = OnlineKDE(makeSpace((-4.0, 4.0, 41, False)))
    assert not kde
    assert kde.getNumKernels() == 0


def test_kde_is_truthy_after_a_deposit():
    kde = OnlineKDE(makeSpace((-4.0, 4.0, 41, False)))
    kde.update(np.array([0.0]), 0.0, np.array([0.25]))
    assert kde
    assert kde.getNumKernels() == 1


def test_grid_pdf_matches_a_direct_mixture_sum():
    space = makeSpace((-4.0, 4.0, 41, False))
    kde = OnlineKDE(space, compressionThreshold=0.0)
    rng = np.random.default_rng(0)
    for _ in range(12):
        kde.update(np.array([rng.uniform(-2, 2)]), rng.normal(), np.array([0.09]))
    grid = np.linspace(-4, 4, 41)
    assert kde.getLogPDF() == pytest.approx(directLogPDF(kde, grid), abs=1e-12)


def test_neff_matches_the_kish_definition():
    kde = OnlineKDE(makeSpace((-4.0, 4.0, 41, False)))
    rng = np.random.default_rng(1)
    logWeights = np.log(rng.uniform(0.5, 2.0, 25))
    for logWeight in logWeights:
        kde.update(np.array([rng.normal()]), logWeight, np.array([1.0]))
    weights = np.exp(logWeights)
    neff = np.exp(2 * kde._logSumW - kde._logSumWSq)
    assert neff == pytest.approx(weights.sum() ** 2 / (weights**2).sum())


def test_log_mean_density_is_the_mean_pdf_over_kernel_centers():
    kde = OnlineKDE(makeSpace((-4.0, 4.0, 41, False)))
    rng = np.random.default_rng(2)
    for _ in range(20):
        kde.update(np.array([rng.normal()]), rng.normal(), np.array([0.09]))
    centers = np.stack([k.position for k in kde._kernels])
    direct = np.mean([np.exp(kde.evaluate(c)) for c in centers])
    assert np.exp(kde.getLogMeanDensity()) == pytest.approx(direct, rel=1e-8)


def test_repeated_deposits_at_one_spot_are_compressed():
    kde = OnlineKDE(makeSpace((-4.0, 4.0, 41, False)))
    for _ in range(50):
        kde.update(np.array([0.0]), 0.0, np.array([0.25]))
    assert kde.getNumKernels() == 1


def test_well_separated_deposits_are_not_compressed():
    kde = OnlineKDE(makeSpace((-40.0, 40.0, 81, False)))
    for x in (-30.0, 0.0, 30.0):
        kde.update(np.array([x]), 0.0, np.array([0.01]))
    assert kde.getNumKernels() == 3


def test_zero_threshold_disables_merging():
    kde = OnlineKDE(makeSpace((-4.0, 4.0, 41, False)), compressionThreshold=0.0)
    for _ in range(25):
        kde.update(np.array([0.0]), 0.0, np.array([0.25]))
    assert kde.getNumKernels() == 25


def test_iadd_combines_weights_and_kernels():
    """__iadd__ transfers kernels verbatim (adjustBandwidth=False), because a
    peer's bandwidths already carry that peer's own Silverman factor."""
    from copy import copy

    space = makeSpace((-4.0, 4.0, 41, False))
    rng = np.random.default_rng(3)
    first = OnlineKDE(space, compressionThreshold=0.0)
    second = OnlineKDE(space, compressionThreshold=0.0)
    for _ in range(8):
        first.update(np.array([rng.normal()]), rng.normal(), np.array([0.09]))
        second.update(np.array([rng.normal()]), rng.normal(), np.array([0.09]))

    expectedLogSumW = np.logaddexp(first._logSumW, second._logSumW)
    expectedKernels = first.getNumKernels() + second.getNumKernels()
    union = [copy(k) for k in first._kernels] + [copy(k) for k in second._kernels]

    first += second
    assert first.getNumKernels() == expectedKernels
    assert first._logSumW == pytest.approx(expectedLogSumW)

    grid = np.linspace(-4, 4, 41)
    total = np.zeros_like(grid)
    for kernel in union:
        total += np.exp(kernel.evaluate(grid.reshape(-1, 1)))
    assert first.getLogPDF() == pytest.approx(
        np.log(total) - expectedLogSumW, abs=1e-12
    )


def test_copy_is_independent_of_the_original():
    from copy import copy

    space = makeSpace((-4.0, 4.0, 41, False))
    kde = OnlineKDE(space)
    kde.update(np.array([0.0]), 0.0, np.array([0.25]))
    duplicate = copy(kde)
    duplicate.update(np.array([2.0]), 0.0, np.array([0.25]))
    assert kde.getNumKernels() == 1
    assert duplicate.getNumKernels() == 2


def test_removal_cancellation_margin_stays_bounded_away_from_zero():
    """Pins the invariant behind spec section 12.4.

    _logsubexp(x, y) loses precision as x - y approaches zero. _pushKernel
    always adds the merged replacement kernel BEFORE removing what it absorbed,
    which bounds that margin. If a removal path without a compensating addition
    is ever introduced, this fails.
    """
    space = makeSpace((-4.0, 4.0, 21, False))
    kde = OnlineKDE(space)
    margins = []
    original = OnlineKDE._removeKernels

    def spy(self, centers, toRemove):
        for index in toRemove:
            margin = self._logPK - self._kernels[index].evaluate(centers)
            finite = np.isfinite(margin)
            if finite.any():
                margins.append(float(np.min(margin[finite])))
        return original(self, centers, toRemove)

    OnlineKDE._removeKernels = spy
    try:
        rng = np.random.default_rng(5)
        for _ in range(3000):
            kde.update(np.array([rng.normal()]), rng.normal(0, 4.0), np.array([0.01]))
    finally:
        OnlineKDE._removeKernels = original

    assert len(margins) > 1000
    assert min(margins) > 1e-3

    centers = np.stack([k.position for k in kde._kernels])
    recomputed = np.logaddexp.reduce(
        np.stack([k.evaluate(centers) for k in kde._kernels]), axis=0
    )
    assert kde._logPK == pytest.approx(recomputed, abs=1e-9)


def test_state_round_trip_preserves_the_density():
    space = makeSpace((-4.0, 4.0, 41, False))
    kde = OnlineKDE(space)
    rng = np.random.default_rng(7)
    for _ in range(40):
        kde.update(np.array([rng.normal()]), rng.normal(), np.array([0.09]))

    restored = OnlineKDE(space)
    restored.setState(kde.getState())

    assert restored.getNumKernels() == kde.getNumKernels()
    # setState rebuilds _logPG from scratch via a single reduce, while the
    # original built it incrementally through many merge/subtract operations;
    # a different summation order gives different (still tiny) roundoff.
    assert restored.getLogPDF() == pytest.approx(kde.getLogPDF(), rel=1e-6)
    assert restored.getLogMeanDensity() == pytest.approx(kde.getLogMeanDensity())


def test_empty_kde_state_round_trips():
    """Regression for spec 7.4.3: restoring an empty KDE raised TypeError.

    Routine on the warm-up path, where a walker syncs before its first deposit.
    """
    space = makeSpace((-4.0, 4.0, 41, False))
    restored = OnlineKDE(space)
    restored.setState(OnlineKDE(space).getState())
    assert restored.getNumKernels() == 0
    assert not restored
    # -inf - (-inf) = NaN: an undefined density with zero data, not a bug.
    assert np.all(np.isnan(restored.getLogPDF()))


def test_state_contains_only_arrays_and_scalars():
    """The state must be np.savez-able, so no objects allowed."""
    space = makeSpace((-4.0, 4.0, 41, False))
    kde = OnlineKDE(space)
    kde.update(np.array([0.0]), 0.0, np.array([0.25]))
    for key, value in kde.getState().items():
        assert isinstance(value, (np.ndarray, float, int)), key
        if isinstance(value, np.ndarray):
            assert value.dtype != object, key


def test_iadd_combines_the_weight_moments_rather_than_re_accumulating():
    """Regression for review finding 4.

    A compressed kernel carries the combined weight of every sample it
    absorbed. Re-deriving the sum of SQUARED weights from those merged
    kernels badly overstates it, collapsing the effective sample size and
    over-widening every kernel deposited afterwards.
    """
    space = makeSpace((-4.0, 4.0, 41, False))
    first = OnlineKDE(space)
    second = OnlineKDE(space)
    rng = np.random.default_rng(1)
    for _ in range(2000):
        first.update(np.array([rng.normal()]), 0.0, np.array([0.04]))
        second.update(np.array([rng.normal()]), 0.0, np.array([0.04]))

    expectedLogSumW = np.logaddexp(first._logSumW, second._logSumW)
    expectedLogSumWSq = np.logaddexp(first._logSumWSq, second._logSumWSq)
    first += second

    assert first._logSumW == pytest.approx(expectedLogSumW)
    assert first._logSumWSq == pytest.approx(expectedLogSumWSq)
    # 4000 unit-weight samples, so neff is 4000 and must not collapse
    neff = np.exp(2 * first._logSumW - first._logSumWSq)
    assert neff == pytest.approx(4000.0, rel=1e-6)


# --- Regression tests for the second code review ---------------------------


def reflectedDensity(kde, point):
    """Normalized mixture density at ``point``, summing every mirror image.

    Reflecting the evaluation point across each wall is equivalent to
    reflecting the kernel, and a diagonal kernel factorizes per dimension, so
    the reflected kernel is a product of three-image sums.
    """
    space = kde._cvSpace
    lower = np.array([cv.minValue for cv in space.variables])
    upper = np.array([cv.maxValue for cv in space.variables])
    point = np.asarray(point, dtype=float)
    total = 0.0
    for kernel in kde._kernels:
        product = np.exp(kernel.logWeight - kde._logSumW)
        for i, (sigma, center) in enumerate(
            zip(kernel.bandwidth, kernel.position, strict=True)
        ):
            images = (point[i], 2 * lower[i] - point[i], 2 * upper[i] - point[i])
            product *= (
                sum(
                    np.exp(-0.5 * ((x - center) / sigma) ** 2) / np.sqrt(2 * np.pi)
                    for x in images
                )
                / sigma
            )
        total += product
    return total


def test_bounded_density_at_the_wall_includes_the_mirror_image():
    """The wall node lost its mirror image, halving the density right there.

    Reflection doubles a kernel's value at the wall itself, since the kernel
    and its image coincide. Dropping that image left a sharp dip at the
    boundary node, which went straight into the tabulated bias.
    """
    space = makeSpace((0.0, 1.0, 11, False), bounded=True)
    kde = OnlineKDE(space, compressionThreshold=0.0)
    kde.update(np.array([0.0]), 0.0, np.array([0.04]))
    grid = np.linspace(0.0, 1.0, 11)
    expected = [reflectedDensity(kde, [x]) for x in grid]
    assert np.exp(kde.getLogPDF()) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "specs",
    [
        ((0.0, 1.0, 11, False),),
        ((0.0, 1.0, 11, False), (-1.0, 1.0, 21, False)),
    ],
)
def test_bounded_point_density_agrees_with_the_grid(specs):
    """Z_n must use the same reflected density the grid holds.

    Point evaluation ignored the walls while the grid folded them in, so the
    mean density at the kernel centers normalized a different estimate from
    the one the bias is built on.
    """
    space = makeSpace(*specs, bounded=True)
    kde = OnlineKDE(space, compressionThreshold=0.0)
    grids = [np.linspace(a, b, n) for a, b, n, _ in specs]
    rng = np.random.default_rng(11)
    nodes = []
    for _ in range(6):
        # centers sit on grid nodes, near the walls, so grid and point values
        # can be compared directly where reflection matters most
        index = tuple(int(rng.choice([0, 1, len(g) - 2, len(g) - 1])) for g in grids)
        nodes.append(index)
        position = np.array([g[i] for g, i in zip(grids, index, strict=True)])
        kde.update(position, rng.normal(), np.full(len(specs), 0.02))

    logPDF = kde.getLogPDF()
    for kernel, index in zip(kde._kernels, nodes, strict=True):
        direct = reflectedDensity(kde, kernel.position)
        assert np.exp(kde.evaluate(kernel.position)) == pytest.approx(direct)
        assert np.exp(logPDF[tuple(reversed(index))]) == pytest.approx(direct)
    centers = [k.position for k in kde._kernels]
    meanDensity = np.mean([reflectedDensity(kde, c) for c in centers])
    assert np.exp(kde.getLogMeanDensity()) == pytest.approx(meanDensity)


def test_merge_search_rescales_by_the_grown_bandwidth():
    """With useExistingBandwidths=False, distances are measured in units of the
    incoming kernel's bandwidth. After a merge that kernel is wider, so the
    search for a further neighbor must use the widened bandwidth, not the one
    it had before absorbing anything."""
    space = makeSpace((-4.0, 4.0, 41, False))
    kde = OnlineKDE(space, compressionThreshold=0.0, useExistingBandwidths=False)
    kde._addKernel(np.array([0.005]), np.array([1.0]), 0.0, adjustBandwidth=False)
    kde._addKernel(np.array([0.5]), np.array([0.01]), 0.0, adjustBandwidth=False)
    kde._compressionThreshold = 1.0
    # 0.5 bandwidths from the first kernel, so it merges; the result is about
    # 0.71 wide and 0.70 of that from the second kernel, which must merge too
    kde._addKernel(np.array([0.0]), np.array([0.01]), 0.0, adjustBandwidth=False)
    assert kde.getNumKernels() == 1


def test_empty_kde_log_pdf_is_nan_without_a_runtime_warning():
    import warnings

    kde = OnlineKDE(makeSpace((-4.0, 4.0, 41, False)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.all(np.isnan(kde.getLogPDF()))


def freshCaches(kde):
    """The grid and kernel-center densities recomputed from the kernel list."""
    grid = np.logaddexp.reduce(np.stack([k.evaluateOnGrid() for k in kde._kernels]))
    centers = np.stack([k.position for k in kde._kernels])
    atCenters = np.logaddexp.reduce(
        np.stack([k.evaluate(centers) for k in kde._kernels]), axis=0
    )
    return grid, atCenters


@pytest.mark.parametrize("bounded", [False, True])
def test_merging_into_a_narrow_heavy_kernel_keeps_the_wide_tails(bounded):
    """Subtracting the absorbed wide kernel cancelled catastrophically in its
    tails, where it held nearly all the density, leaving -inf behind."""
    kde = OnlineKDE(makeSpace((-3.0, 3.0, 201, False), bounded=bounded))
    kde.update(np.array([0.0]), 0.0, np.array([1.0]), factor=1.0)
    kde.update(np.array([0.1]), 20.0, np.array([0.01]), factor=1.0)
    assert kde.getNumKernels() == 1
    grid, atCenters = freshCaches(kde)
    assert np.all(np.isfinite(kde._logPG))
    assert kde._logPG == pytest.approx(grid, abs=1e-9)
    assert kde._logPK == pytest.approx(atCenters, abs=1e-9)


def test_cached_densities_stay_accurate_over_a_long_run():
    """Hopping between two wells while the weights grow, as a metastable OPES
    run does. Partial cancellations compound over successive merges, and the
    far tails of the grid drifted by orders of magnitude."""
    kde = OnlineKDE(makeSpace((-3.0, 3.0, 41, False), (-3.0, 3.0, 41, False)))
    rng = np.random.default_rng(0)
    phases = 8
    for phase in range(phases):
        center = (-1.5, 1.5)[phase % 2]
        for i in range(200):
            position = np.clip(center + rng.normal(0, 0.5, 2), -2.9, 2.9)
            logWeight = 20.0 * (phase + i / 200) / phases + rng.normal()
            kde.update(position, logWeight, np.full(2, 0.09))
    grid, atCenters = freshCaches(kde)
    assert kde._logPG == pytest.approx(grid, abs=1e-6)
    assert kde._logPK == pytest.approx(atCenters, abs=1e-6)
