"""PLUMED's kernel merge rule, for comparison with openmm-opes's.

PLUMED's OPES_METAD weights the merged center and variance by kernel height
h = w / prod(sigma) and keeps h1 + h2 as the merged height, which does not
conserve the probability mass when the two bandwidths differ. openmm-opes
weights by mass instead (Kernel.merge). merge_accuracy.py compares the two.
"""

from __future__ import annotations

import numpy as np

from openmm_opes import kde


def plumed_merge(self, other):
    """Drop-in replacement for openmm_opes.kde.Kernel.merge with PLUMED's rule."""
    logH = np.logaddexp(self.logHeight, other.logHeight)
    w1, w2 = np.exp(self.logHeight - logH), np.exp(other.logHeight - logH)
    disp = self.cvSpace.displacement(self.position, other.position)
    self.position = self.cvSpace.endpoint(self.position, w2 * disp)
    self.bandwidth = np.sqrt(
        w1 * self.bandwidth**2
        + w2 * other.bandwidth**2
        + w1 * w2 * disp**2 / self.shape.variance
    )
    d = self.cvSpace.numDimensions
    self.logWeight = logH + d * self.shape.logNorm + np.sum(np.log(self.bandwidth))
    self.numSamples += other.numSamples
    self.logHeight = self._computeLogHeight()


def use_plumed_merge():
    """Make every openmm-opes kernel merge with PLUMED's rule from now on."""
    kde.Kernel.merge = plumed_merge
