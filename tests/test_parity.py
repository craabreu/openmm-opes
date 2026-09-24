"""Pins the port against the original implementation's numerics.

The port reorganizes code but changes no arithmetic on the default path, so
these must agree up to floating-point roundoff. The tolerance is not
platform-bit-identical, though: the reference fixture was generated once on
one machine's numpy/BLAS build, and this test runs across CI's Python
3.11/3.13 conda-forge environments, which use separately-built numpy. exp/log
are not required to be bit-identical across builds, and with ~400 deposits
chained through many merges the last-ULP differences compound. Confirmed
against CI: Python 3.11 matched at rel=1e-12, but 3.13 diverged up to
rel=3e-7 on the same fixture -- a cross-build drift, not a code difference
(the same commit's src/openmm_opes passed on 3.11). rel=1e-5 is two orders of
magnitude looser than that observed worst case, while a genuine formula
error would show up as an O(1) or many-percent difference, not a few ULPs.
If this test fails outside that noise floor, the port changed the science.
Do NOT loosen the tolerance further to paper over a larger gap.

The one intended difference is in the far tails. The original's incremental
cache cancels catastrophically there when a merge removes a kernel that held
nearly all the density, leaving -inf or errors of tens of percent at log
densities below about 40 under the peak. The port rebuilds its caches before
that happens, so the grid is compared with the original only within
TAIL_CUTOFF of the peak, and everywhere with a from-scratch recompute.
"""

from pathlib import Path

import numpy as np
import pytest

from openmm_opes.kde import OnlineKDE
from tests.data.make_reference import CASES, depositSequence
from tests.helpers import makeSpace

REFERENCE = Path(__file__).parent / "data" / "reference_kde.npz"
TAIL_CUTOFF = 40.0


@pytest.fixture(scope="module")
def reference():
    return np.load(REFERENCE)


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_the_original_implementation(reference, name):
    case = CASES[name]
    kde = OnlineKDE(makeSpace(*case.cvs))
    positions, logWeights, variances = depositSequence(
        len(case.cvs), case.count, case.seed
    )
    for position, logWeight, variance in zip(
        positions, logWeights, variances, strict=True
    ):
        kde.update(position, logWeight, variance)

    assert kde.getNumKernels() == int(reference[f"{name}_numKernels"])
    logPDF = kde.getLogPDF()
    body = logPDF > logPDF.max() - TAIL_CUTOFF
    assert logPDF[body] == pytest.approx(reference[f"{name}_logPDF"][body], rel=1e-5)
    fresh = np.logaddexp.reduce(np.stack([k.evaluateOnGrid() for k in kde._kernels]))
    assert logPDF == pytest.approx(fresh - kde._logSumW, rel=1e-9)
    assert kde.getLogMeanDensity() == pytest.approx(
        float(reference[f"{name}_logMeanDensity"]), rel=1e-5
    )
    centers = np.stack([k.position for k in kde._kernels])
    bandwidths = np.stack([k.bandwidth for k in kde._kernels])
    assert centers == pytest.approx(reference[f"{name}_centers"], rel=1e-5)
    assert bandwidths == pytest.approx(reference[f"{name}_bandwidths"], rel=1e-5)
