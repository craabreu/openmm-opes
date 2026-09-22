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
