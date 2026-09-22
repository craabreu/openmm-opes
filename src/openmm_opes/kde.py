"""Online kernel density estimation for OPES.

Pure NumPy/SciPy. This module must not import openmm.
"""

from __future__ import annotations

import functools
from collections import namedtuple

import numpy as np


class CVSpace:
    """The domain of the collective variables: grid, periodicity, boundaries.

    Parameters
    ----------
    variables
        Sequence of collective variable descriptions. Each must expose
        ``minValue``, ``maxValue``, ``gridWidth`` and ``periodic``; an
        ``openmm.app.BiasVariable`` qualifies.
    bounded
        Whether non-periodic variables have reflective boundaries. When true the
        internal grid is tripled with mirrored copies on each side, and
        :meth:`foldedGrid` folds them back.
    """

    CV = namedtuple("CV", ["minValue", "maxValue", "gridWidth", "periodic"])

    def __init__(self, variables, bounded: bool = False):
        self.variables = [
            self.CV(cv.minValue, cv.maxValue, cv.gridWidth, cv.periodic)
            for cv in variables
        ]
        self.bounded = bounded
        self._periodic = any(cv.periodic for cv in self.variables)
        self._grid = []
        for cv in self.variables:
            a, b, n = cv.minValue, cv.maxValue, cv.gridWidth
            points = np.linspace(a, b, n)
            if bounded and not cv.periodic:
                left = np.linspace(2 * a - b, a, n)
                right = np.linspace(b, 2 * b - a, n)
                left[-1] = right[0] = np.inf
                points = np.concatenate((points, np.flip(left), np.flip(right)))
            self._grid.append(points)
        self._widths = np.array([cv.gridWidth for cv in self.variables])
        self._lbounds = np.array([cv.minValue for cv in self.variables])
        self._lengths = np.array([cv.maxValue for cv in self.variables]) - self._lbounds
        if self._periodic:
            self._pdims = tuple(i for i, cv in enumerate(self.variables) if cv.periodic)
            self._plbounds = self._lbounds[list(self._pdims)]
            self._plengths = self._lengths[list(self._pdims)]

    @property
    def gridShape(self) -> tuple[int, ...]:
        """Shape of the CV-space grid, slowest axis first."""
        return tuple(int(width) for width in reversed(self._widths))

    @property
    def numDimensions(self) -> int:
        """Number of collective variables."""
        return len(self.variables)

    def displacement(self, position, endpoint):
        """Displacement from ``position`` to ``endpoint``, minimum-image if periodic."""
        disp = endpoint - position
        if self._periodic:
            disp[..., self._pdims] -= self._plengths * np.rint(
                disp[..., self._pdims] / self._plengths
            )
        return disp

    def endpoint(self, position, displacement):
        """Endpoint reached from ``position``, wrapped into the periodic domain."""
        end = position + displacement
        if self._periodic:
            end[..., self._pdims] = (
                self._plbounds
                + (end[..., self._pdims] - self._plbounds) % self._plengths
            )
        return end

    def gridDistances(self, position):
        """Per-axis distances from ``position`` to every grid node."""
        distances = [
            centers - x for centers, x in zip(self._grid, position, strict=True)
        ]
        if self._periodic:
            for dim, length in zip(self._pdims, self._plengths, strict=True):
                distances[dim] -= length * np.rint(distances[dim] / length)
                # a periodic grid's last node is the same point as its first
                distances[dim][-1] = distances[dim][0]
        return distances

    def closestNode(self, position) -> tuple[int, ...]:
        """Index of the grid node nearest to ``position``, in gridShape order."""
        indices = np.rint(
            (self._widths - 1) * (position - self._lbounds) / self._lengths
        ).astype(int)
        if self._periodic:
            indices[list(self._pdims)] %= self._widths[list(self._pdims)]
        indices = np.clip(indices, 0, self._widths - 1)
        return tuple(reversed(indices))

    def foldedGrid(self, values):
        """Fold a tripled bounded grid back onto the physical domain, in log space."""
        if self.bounded:
            for i, cv in enumerate(reversed(self.variables)):
                if not cv.periodic:
                    values, left, right = np.array_split(values, 3, axis=i)
                    values = np.logaddexp(left, np.logaddexp(values, right))
        return values


class KernelShape:
    """A kernel profile: its per-dimension log normalization and its exponent."""

    def __init__(self, name: str, logNorm: float, exponents):
        self.name = name
        self.logNorm = logNorm
        self._exponents = exponents

    def exponents(self, x):
        """Log of the kernel profile at scaled distances ``x``."""
        return self._exponents(x)


def _gaussianExponents(x):
    return -0.5 * x**2


def _compactExponents(x):
    values = 9 - x**2
    mask = values > 0
    result = np.empty_like(values)
    result[mask] = 4 * np.log(values[mask])
    result[~mask] = -np.inf
    return result


#: Unbounded Gaussian profile; the default and the one both papers use.
GAUSSIAN = KernelShape("gaussian", np.log(2 * np.pi) / 2, _gaussianExponents)

#: Compact quartic profile with support of +/- 3 bandwidths. Its normalization
#: constant is exactly the integral of (9 - x**2)**4 over [-3, 3].
COMPACT = KernelShape("compact", np.log(559872 / 35), _compactExponents)

KERNEL_SHAPES = {shape.name: shape for shape in (GAUSSIAN, COMPACT)}


class Kernel:
    """A multivariate kernel with diagonal covariance."""

    def __init__(
        self, cvSpace, position, bandwidth, logWeight, numSamples=1, shape=GAUSSIAN
    ):
        self.cvSpace = cvSpace
        self.position = np.array(position, dtype=float)
        self.bandwidth = np.array(bandwidth, dtype=float)
        self.logWeight = logWeight
        self.numSamples = numSamples
        self.shape = shape
        self.logHeight = self._computeLogHeight()

    def __copy__(self):
        return Kernel(
            self.cvSpace,
            self.position,
            self.bandwidth,
            self.logWeight,
            self.numSamples,
            self.shape,
        )

    def _computeLogHeight(self):
        if np.any(self.bandwidth == 0):
            return -np.inf
        d = self.cvSpace.numDimensions
        return self.logWeight - d * self.shape.logNorm - np.sum(np.log(self.bandwidth))

    def _scaledDistances(self, points, bandwidths):
        return self.cvSpace.displacement(self.position, points) / bandwidths

    def findNearest(self, centers, bandwidths, ignore=()):
        """Index of and squared Mahalanobis distance to the nearest center."""
        if centers.size == 0:
            return -1, np.inf
        sqDistances = np.sum(self._scaledDistances(centers, bandwidths) ** 2, axis=-1)
        if len(ignore):
            sqDistances[list(ignore)] = np.inf
        index = int(np.argmin(sqDistances))
        return index, sqDistances[index]

    def merge(self, other) -> None:
        """Absorb ``other``, preserving total weight, mean and second moment."""
        logSumWeights = np.logaddexp(self.logWeight, other.logWeight)
        w1 = np.exp(self.logWeight - logSumWeights)
        w2 = np.exp(other.logWeight - logSumWeights)
        disp = self.cvSpace.displacement(self.position, other.position)
        self.position = self.cvSpace.endpoint(self.position, w2 * disp)
        self.bandwidth = np.sqrt(
            w1 * self.bandwidth**2 + w2 * other.bandwidth**2 + w1 * w2 * disp**2
        )
        self.logWeight = logSumWeights
        self.numSamples += other.numSamples
        self.logHeight = self._computeLogHeight()

    def evaluate(self, points):
        """Log of the kernel at the given point or points."""
        return self.logHeight + np.sum(
            self.shape.exponents(self._scaledDistances(points, self.bandwidth)), axis=-1
        )

    def evaluateOnGrid(self):
        """Log of the kernel on the CV-space grid, in gridShape order."""
        distances = self.cvSpace.gridDistances(self.position)
        exponents = [
            self.shape.exponents(dist / sigma)
            for dist, sigma in zip(distances, self.bandwidth, strict=True)
        ]
        return self.cvSpace.foldedGrid(
            self.logHeight + functools.reduce(np.add.outer, reversed(exponents))
        )
