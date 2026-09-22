"""Pins the port against the original implementation's numerics.

The port reorganizes code but changes no arithmetic on the default path, so
these must agree to floating-point tolerance. If this test fails, the port
changed the science. Do NOT loosen the tolerance to make it pass.
"""

from pathlib import Path

import numpy as np
import pytest

from openmm_opes.kde import OnlineKDE
from tests.data.make_reference import CASES, depositSequence
from tests.helpers import makeSpace

REFERENCE = Path(__file__).parent / "data" / "reference_kde.npz"


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
    assert kde.getLogPDF() == pytest.approx(reference[f"{name}_logPDF"], rel=1e-12)
    assert kde.getLogMeanDensity() == pytest.approx(
        float(reference[f"{name}_logMeanDensity"]), rel=1e-12
    )
    centers = np.stack([k.position for k in kde._kernels])
    bandwidths = np.stack([k.bandwidth for k in kde._kernels])
    assert centers == pytest.approx(reference[f"{name}_centers"], rel=1e-12)
    assert bandwidths == pytest.approx(reference[f"{name}_bandwidths"], rel=1e-12)
