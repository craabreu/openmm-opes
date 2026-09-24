"""Online kernel density estimation for OPES.

Pure NumPy/SciPy. This module must not import openmm.
"""

from __future__ import annotations

import functools
from collections import namedtuple
from copy import copy

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

    def mirrorPositions(self, position):
        """Every image of ``position`` under the reflective boundaries.

        For each non-periodic dimension, reflecting a point off both walls
        gives two images in addition to the point itself; the images across
        every dimension combine multiplicatively (a corner sees images of
        images). Unbounded spaces, or spaces with only periodic CVs, have no
        images: ``[position]`` is returned unchanged.

        A kernel centered at any one of these images is, by the symmetry of
        reflection, exactly as far from a given query point as the original
        kernel is from that point's own mirror image -- so summing a
        kernel's density over these images is equivalent to summing the
        query point's images against the one true kernel, which is what the
        tripled evaluation grid does. Used to keep :meth:`Kernel.evaluate`
        (arbitrary points) consistent with :meth:`Kernel.evaluateOnGrid`.
        """
        positions = [np.array(position, dtype=float)]
        if not self.bounded:
            return positions
        for i, cv in enumerate(self.variables):
            if cv.periodic:
                continue
            expanded = []
            for p in positions:
                expanded.append(p)
                mirrorLow = p.copy()
                mirrorLow[i] = 2 * cv.minValue - p[i]
                expanded.append(mirrorLow)
                mirrorHigh = p.copy()
                mirrorHigh[i] = 2 * cv.maxValue - p[i]
                expanded.append(mirrorHigh)
            positions = expanded
        return positions

    def foldedGrid(self, values):
        """Fold a tripled bounded grid back onto the physical domain, in log space."""
        if self.bounded:
            for i, cv in enumerate(reversed(self.variables)):
                if not cv.periodic:
                    values, left, right = np.array_split(values, 3, axis=i)
                    values = np.logaddexp(left, np.logaddexp(values, right))
        return values


class KernelShape:
    """A kernel profile: its per-dimension log normalization, its variance at
    unit bandwidth, and its exponent."""

    def __init__(self, name: str, logNorm: float, variance: float, exponents):
        self.name = name
        self.logNorm = logNorm
        self.variance = variance
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
GAUSSIAN = KernelShape("gaussian", np.log(2 * np.pi) / 2, 1.0, _gaussianExponents)

#: Compact quartic profile with support of +/- 3 bandwidths. Its normalization
#: constant is exactly the integral of (9 - x**2)**4 over [-3, 3], and its
#: variance at unit bandwidth is 9/11.
COMPACT = KernelShape("compact", np.log(559872 / 35), 9 / 11, _compactExponents)

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

    def _scaledDistancesFrom(self, center, points, bandwidths):
        return self.cvSpace.displacement(center, points) / bandwidths

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
            w1 * self.bandwidth**2
            + w2 * other.bandwidth**2
            + w1 * w2 * disp**2 / self.shape.variance
        )
        self.logWeight = logSumWeights
        self.numSamples += other.numSamples
        self.logHeight = self._computeLogHeight()

    def evaluate(self, points):
        """Log of the kernel at the given point or points.

        Reflected at the domain walls when the space is bounded, by summing
        over the kernel's mirror images (see :meth:`CVSpace.mirrorPositions`)
        so this agrees with :meth:`evaluateOnGrid`. Unbounded spaces have no
        images, so this reduces to the single-term evaluation.
        """
        images = self.cvSpace.mirrorPositions(self.position)
        contributions = [
            np.sum(
                self.shape.exponents(
                    self._scaledDistancesFrom(image, points, self.bandwidth)
                ),
                axis=-1,
            )
            for image in images
        ]
        return self.logHeight + np.logaddexp.reduce(contributions, axis=0)

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


class OnlineKDE:
    """Online kernel density estimate with on-the-fly kernel compression.

    Parameters
    ----------
    cvSpace
        The :class:`CVSpace` the estimate lives on.
    compressionThreshold
        Merge a new kernel into any existing kernel closer than this, measured
        as a Mahalanobis distance. Zero disables merging.
    useExistingBandwidths
        Whether the nearest-neighbour search scales distances by the existing
        kernels' bandwidths rather than the incoming kernel's.
    kernelShape
        Either ``"gaussian"`` or ``"compact"``.
    """

    def __init__(
        self,
        cvSpace,
        compressionThreshold: float = 1.0,
        useExistingBandwidths: bool = True,
        kernelShape: str = "gaussian",
    ):
        if kernelShape not in KERNEL_SHAPES:
            raise ValueError(
                f"Unknown kernelShape {kernelShape!r}; "
                f"expected one of {sorted(KERNEL_SHAPES)}"
            )
        self._cvSpace = cvSpace
        self._compressionThreshold = compressionThreshold
        self._useExistingBandwidths = useExistingBandwidths
        self._shape = KERNEL_SHAPES[kernelShape]
        self._kernels: list[Kernel] = []
        self._logSumW = -np.inf
        self._logSumWSq = -np.inf
        self._logPK = np.empty(0)
        self._logPG = np.full(cvSpace.gridShape, -np.inf)
        self._d = cvSpace.numDimensions

    def __bool__(self) -> bool:
        return bool(self._kernels)

    def __copy__(self):
        new = self.__class__(
            self._cvSpace,
            self._compressionThreshold,
            self._useExistingBandwidths,
            self._shape.name,
        )
        new._kernels = list(map(copy, self._kernels))
        new._logSumW = self._logSumW
        new._logSumWSq = self._logSumWSq
        new._logPK = self._logPK.copy()
        new._logPG = self._logPG.copy()
        return new

    def __iadd__(self, other):
        """Absorb every kernel of ``other``.

        The weight moments are combined directly rather than re-accumulated
        from the absorbed kernels: a compressed kernel carries the combined
        weight of all its samples, so squaring it would overstate the sum of
        squared weights and collapse the effective sample size.
        """
        logSumW = np.logaddexp(self._logSumW, other._logSumW)
        logSumWSq = np.logaddexp(self._logSumWSq, other._logSumWSq)
        for kernel in other._kernels:
            self._addKernel(
                kernel.position,
                kernel.bandwidth,
                kernel.logWeight,
                kernel.numSamples,
                adjustBandwidth=False,
            )
        self._logSumW = logSumW
        self._logSumWSq = logSumWSq
        return self

    @staticmethod
    def _logsubexp(x, y):
        """log(exp(x) - exp(y)), elementwise and numerically stable."""
        result = np.full_like(x, -np.inf)
        valid = y < x
        inner = -np.exp(y[valid] - x[valid])
        representable = inner > -1.0
        inner[representable] = np.log1p(inner[representable])
        inner[~representable] = -np.inf
        result[valid] = inner + x[valid]
        return result

    def _removeKernels(self, centers, toRemove):
        """Remove kernels by index, subtracting their contributions in log space.

        Callers MUST add any replacement kernel's contribution to _logPK and
        _logPG *before* calling this. _logsubexp loses precision as its two
        arguments converge, and the replacement is what keeps the removed
        kernel from dominating the density at its own center.
        """
        toRemove = sorted(toRemove, reverse=True)
        removed = []
        for index in toRemove:
            kernel = self._kernels.pop(index)
            self._logPK = self._logsubexp(self._logPK, kernel.evaluate(centers))
            self._logPG = self._logsubexp(self._logPG, kernel.evaluateOnGrid())
            removed.append(kernel)
        self._logPK = np.delete(self._logPK, toRemove)
        return removed

    def _pushKernel(self, newKernel):
        centers = np.stack([k.position for k in self._kernels])

        def bandwidths():
            return (
                np.stack([k.bandwidth for k in self._kernels])
                if self._useExistingBandwidths
                else newKernel.bandwidth
            )

        threshold = self._compressionThreshold
        index, minSqDist = newKernel.findNearest(centers, bandwidths())
        toRemove = []
        while threshold > 0 and index >= 0 and minSqDist <= threshold**2:
            toRemove.append(index)
            newKernel.merge(self._kernels[index])
            index, minSqDist = newKernel.findNearest(centers, bandwidths(), toRemove)
        self._logPK = np.logaddexp(self._logPK, newKernel.evaluate(centers))
        self._logPG = np.logaddexp(self._logPG, newKernel.evaluateOnGrid())
        if toRemove:
            self._removeKernels(centers, toRemove)
        self._kernels.append(newKernel)
        self._logPK = np.append(
            self._logPK,
            np.logaddexp.reduce(
                [k.evaluate(newKernel.position) for k in self._kernels]
            ),
        )

    def _addKernel(
        self, position, bandwidth, logWeight, numSamples=1, adjustBandwidth=True
    ):
        if adjustBandwidth:
            bandwidth = bandwidth * self.bandwidthFactor(logWeight)
        self._logSumW = np.logaddexp(self._logSumW, logWeight)
        self._logSumWSq = np.logaddexp(self._logSumWSq, 2 * logWeight)
        newKernel = Kernel(
            self._cvSpace, position, bandwidth, logWeight, numSamples, self._shape
        )
        if self._kernels:
            self._pushKernel(newKernel)
        else:
            self._kernels = [newKernel]
            self._logPG = newKernel.evaluateOnGrid()
            self._logPK = np.array([newKernel.evaluate(newKernel.position)])

    def bandwidthFactor(self, logWeight) -> float:
        """Silverman shrink factor for a new kernel of the given log weight.

        Computed from this estimate's effective sample size with the new
        kernel's weight already counted, which is what :meth:`update` applies.
        """
        logSumW = np.logaddexp(self._logSumW, logWeight)
        logSumWSq = np.logaddexp(self._logSumWSq, 2 * logWeight)
        neff = np.exp(2 * logSumW - logSumWSq)
        return (neff * (self._d + 2) / 4) ** (-1 / (self._d + 4))

    def update(self, position, logWeight, variance, factor=None) -> None:
        """Deposit a kernel of the given log weight and per-CV variance.

        The bandwidth is shrunk by ``factor`` when given, and otherwise by
        this estimate's own :meth:`bandwidthFactor`. Passing one lets an
        estimate that is part of a larger one size kernels by the larger
        one's sample size.
        """
        if factor is None:
            self._addKernel(position, np.sqrt(variance), logWeight)
        else:
            self._addKernel(
                position, np.sqrt(variance) * factor, logWeight, adjustBandwidth=False
            )

    def getNumKernels(self) -> int:
        """Number of compressed kernels currently stored."""
        return len(self._kernels)

    def getLogPDF(self):
        """Log of the normalized probability density on the grid.

        With no kernels deposited yet, both operands are -inf and the
        result is the documented NaN (see :meth:`getState`'s empty-KDE
        note) rather than an error; the subtraction is expected to be
        undefined here, not a sign that something went wrong.
        """
        with np.errstate(invalid="ignore"):
            return self._logPG - self._logSumW

    def getLogMeanDensity(self) -> float:
        """Log of Z_n, the mean density over the compressed kernel centers.

        NaN with no kernels deposited yet, for the same reason as
        :meth:`getLogPDF`: the mean over an empty set is undefined, so the
        log(0) and inf - inf along the way are expected, not warned about.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                np.logaddexp.reduce(self._logPK)
                - np.log(len(self._kernels))
                - self._logSumW
            )

    def evaluate(self, point) -> float:
        """Log of the normalized density at a single point."""
        return (
            np.logaddexp.reduce([k.evaluate(point) for k in self._kernels])
            - self._logSumW
        )

    def getState(self) -> dict:
        """Flat, npz-writable snapshot of this estimate.

        The CVSpace is deliberately excluded: all walkers share identical CV
        definitions, so a reader constructs its own and calls setState.
        """
        d = self._cvSpace.numDimensions
        if self._kernels:
            positions = np.stack([k.position for k in self._kernels])
            bandwidths = np.stack([k.bandwidth for k in self._kernels])
            logWeights = np.array([k.logWeight for k in self._kernels])
            numSamples = np.array([k.numSamples for k in self._kernels])
        else:
            positions = np.empty((0, d))
            bandwidths = np.empty((0, d))
            logWeights = np.empty(0)
            numSamples = np.empty(0, dtype=int)
        return {
            "positions": positions,
            "bandwidths": bandwidths,
            "logWeights": logWeights,
            "numSamples": numSamples,
            "logSumW": float(self._logSumW),
            "logSumWSq": float(self._logSumWSq),
            "logPK": self._logPK.copy(),
        }

    def setState(self, state) -> None:
        """Restore from a :meth:`getState` snapshot, discarding current contents."""
        self._kernels = [
            Kernel(self._cvSpace, position, bandwidth, logWeight, int(n), self._shape)
            for position, bandwidth, logWeight, n in zip(
                state["positions"],
                state["bandwidths"],
                state["logWeights"],
                state["numSamples"],
                strict=True,
            )
        ]
        self._logSumW = float(state["logSumW"])
        self._logSumWSq = float(state["logSumWSq"])
        self._logPK = np.asarray(state["logPK"]).copy()
        self._logPG = functools.reduce(
            np.logaddexp,
            (k.evaluateOnGrid() for k in self._kernels),
            np.full(self._cvSpace.gridShape, -np.inf),
        )
