"""Generate the parity fixture from the ORIGINAL online_kde.py.

Run once, commit the .npz, and do not run again unless the source
implementation itself changes:

    python tests/data/make_reference.py /path/to/opes-simulations
"""

import sys
from collections import namedtuple
from pathlib import Path

import numpy as np


def depositSequence(d, count, seed):
    """Deterministic deposits shared by the generator and the test."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-2.0, 2.0, (count, d))
    logWeights = rng.normal(0.0, 1.5, count)
    variances = np.full((count, d), 0.04)
    return positions, logWeights, variances


Case = namedtuple("Case", ["cvs", "count", "seed"])

CASES = {
    "1d": Case(cvs=[(-4.0, 4.0, 61, False)], count=400, seed=11),
    "1d_periodic": Case(cvs=[(-np.pi, np.pi, 61, True)], count=400, seed=12),
    "2d": Case(
        cvs=[(-4.0, 4.0, 31, False), (-4.0, 4.0, 31, False)], count=300, seed=13
    ),
}


def _patchedInit(self, variables, bounded=False):
    """Fixed copy of the ORIGINAL CVSpace.__init__, for generating this fixture only.

    The unmodified original crashes for ANY periodic CV configuration: it indexes
    `self._lengths[self._pdims]` where `self._pdims` is a bare tuple. For numpy,
    a 1-D array indexed by a length-1 tuple collapses to a scalar, and by a
    longer tuple raises "too many indices" outright -- confirmed against both
    the root module and the alanine_dipeptide/ and muller-brown/ frozen copies,
    all three identical. This means the source, as currently committed, cannot
    construct a CVSpace for the phi/psi (2 periodic CVs) configuration used in
    the OPES and OPES-explore papers' own example figures.

    This patch changes only the three indexing lines (tuple -> list, exactly
    openmm_opes.kde.CVSpace's fix) and touches nothing in OnlineKDE or Kernel,
    which is the code this fixture actually needs to pin. It exists solely so
    the "1d_periodic" case below can exercise the original deposit-side math at
    all; it is not a claim that the upstream bug is fixed there.
    """
    self.variables = [
        self._CV(cv.minValue, cv.maxValue, cv.gridWidth, cv.periodic)
        for cv in variables
    ]
    self.bounded = bounded
    self._periodic = any(cv.periodic for cv in variables)
    self._grid = []
    for cv in variables:
        a, b, n = cv.minValue, cv.maxValue, cv.gridWidth
        points = np.linspace(a, b, n)
        if bounded and not cv.periodic:
            left = np.linspace(2 * a - b, a, n)
            right = np.linspace(b, 2 * b - a, n)
            left[-1] = right[0] = np.inf
            points = np.concatenate((points, np.flip(left), np.flip(right)))
        self._grid.append(points)
    self._widths = np.array([cv.gridWidth for cv in variables])
    self._lbounds = np.array([cv.minValue for cv in variables])
    ubounds = np.array([cv.maxValue for cv in variables])
    self._lengths = ubounds - self._lbounds
    if self._periodic:
        self._pdims = tuple(i for i, cv in enumerate(variables) if cv.periodic)
        self._plbounds = self._lbounds[list(self._pdims)]
        self._plengths = self._lengths[list(self._pdims)]


def main(sourceDir):
    sys.path.insert(0, str(sourceDir))
    from online_kde import CVSpace, OnlineKDE  # the ORIGINAL implementation

    assert __import__("online_kde").COMPRESSION_THRESHOLD == 1.0
    assert __import__("online_kde").BOUNDED_KERNELS is False
    assert __import__("online_kde").UNCOMPRESSED_KDE is False
    assert __import__("online_kde").USE_EXISTING_BANDWIDTHS is True

    CVSpace.__init__ = _patchedInit  # see _patchedInit's docstring

    arrays = {}
    for name, case in CASES.items():
        space = CVSpace([CVSpace._CV(*cv) for cv in case.cvs])
        kde = OnlineKDE(space)
        positions, logWeights, variances = depositSequence(
            len(case.cvs), case.count, case.seed
        )
        for position, logWeight, variance in zip(
            positions, logWeights, variances, strict=True
        ):
            kde.update(position, logWeight, variance)
        arrays[f"{name}_logPDF"] = kde.getLogPDF()
        arrays[f"{name}_logMeanDensity"] = np.array(kde.getLogMeanDensity())
        arrays[f"{name}_numKernels"] = np.array(kde.getNumKernels())
        arrays[f"{name}_centers"] = np.stack([k.position for k in kde._kernels])
        arrays[f"{name}_bandwidths"] = np.stack([k.bandwidth for k in kde._kernels])

    out = Path(__file__).parent / "reference_kde.npz"
    np.savez(out, **arrays)
    print(f"wrote {out} with {len(arrays)} arrays")


if __name__ == "__main__":
    main(sys.argv[1])
