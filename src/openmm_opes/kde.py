"""Online kernel density estimation for OPES.

Pure NumPy/SciPy. This module must not import openmm.
"""

from __future__ import annotations

import functools
from collections import namedtuple
from copy import copy

import numpy as np
from scipy.special import logsumexp


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

    def axisGridDistances(self, index, centers):
        """Distances from each of ``centers`` to every node of one axis.

        A 2-D array, one row per center, matching :meth:`gridDistances` for a
        single center: minimum-image on a periodic axis, whose last node
        repeats the first, and over the tripled grid on a reflective axis.
        """
        distances = (
            self._grid[index][np.newaxis, :] - np.asarray(centers)[:, np.newaxis]
        )
        cv = self.variables[index]
        if cv.periodic:
            length = cv.maxValue - cv.minValue
            distances -= length * np.rint(distances / length)
            distances[:, -1] = distances[:, 0]
        return distances

    def periodicCopies(self, values):
        """Make the last node along every periodic axis an exact copy of the first.

        The two nodes are the same point. Evaluating them separately can
        round differently (a matrix product, for one, need not give equal
        columns equal sums), and OpenMM rejects a periodic table whose ends
        differ at all.
        """
        for i, cv in enumerate(self.variables):
            if cv.periodic:
                axis = self.numDimensions - 1 - i
                first = np.take(values, [0], axis=axis)
                index = [slice(None)] * values.ndim
                index[axis] = slice(-1, None)
                values[tuple(index)] = first
        return values

    def foldedAxisRows(self, index, values):
        """:meth:`foldedAxis` applied to every row of a 2-D array of log profiles."""
        if self.bounded and not self.variables[index].periodic:
            values, left, right = np.array_split(values, 3, axis=1)
            values = np.logaddexp(left, np.logaddexp(values, right))
        return values

    def closestNode(self, position) -> tuple[int, ...]:
        """Index of the grid node nearest to ``position``, in gridShape order."""
        indices = np.rint(
            (self._widths - 1) * (position - self._lbounds) / self._lengths
        ).astype(int)
        if self._periodic:
            indices[list(self._pdims)] %= self._widths[list(self._pdims)] - 1
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

    def axisImages(self, index, value):
        """The images of one coordinate under the reflective walls of its axis.

        ``value`` itself, plus its reflections off both walls when the space is
        bounded and the axis is not periodic. :meth:`mirrorPositions` returns
        every combination of these per-axis images.
        """
        cv = self.variables[index]
        if self.bounded and not cv.periodic:
            return (value, 2 * cv.minValue - value, 2 * cv.maxValue - value)
        return (value,)

    def axisDisplacement(self, index, start, end):
        """Displacement along one axis, minimum-image if that axis is periodic."""
        disp = end - start
        if self.variables[index].periodic:
            length = self._lengths[index]
            disp = disp - length * np.rint(disp / length)
        return disp

    def foldedAxis(self, index, values):
        """Fold one axis's tripled bounded profile back onto the domain, in log space.

        A kernel factorizes over the CVs, and its mirror images under the
        reflective walls are every combination of per-axis reflections, so
        folding each axis's one-dimensional profile before the outer product
        gives exactly what :meth:`foldedGrid` gives on the full tripled grid,
        at a fraction of the cost.
        """
        if self.bounded and not self.variables[index].periodic:
            values, left, right = np.array_split(values, 3)
            values = np.logaddexp(left, np.logaddexp(values, right))
        return values

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
        # The kernel factorizes over the axes and its images are every
        # combination of per-axis reflections, so the sum over images is the
        # product over axes of per-axis sums: 3d terms instead of 3^d.
        points = np.asarray(points, dtype=float)
        space = self.cvSpace
        result = self.logHeight
        for i, (center, sigma) in enumerate(
            zip(self.position, self.bandwidth, strict=True)
        ):
            terms = [
                self.shape.exponents(
                    space.axisDisplacement(i, image, points[..., i]) / sigma
                )
                for image in space.axisImages(i, center)
            ]
            result = result + np.logaddexp.reduce(terms, axis=0)
        return result

    def evaluateOnGrid(self):
        """Log of the kernel on the CV-space grid, in gridShape order."""
        distances = self.cvSpace.gridDistances(self.position)
        exponents = [
            self.cvSpace.foldedAxis(i, self.shape.exponents(dist / sigma))
            for i, (dist, sigma) in enumerate(
                zip(distances, self.bandwidth, strict=True)
            )
        ]
        return self.logHeight + functools.reduce(np.add.outer, reversed(exponents))


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
    minBandwidth
        Per-CV floor on the bandwidth of new kernels, applied after the
        Silverman shrink. Zero disables it.
    """

    #: How far, in log units, a cached density may fall below its peak since
    #: the last rebuild before both caches are recomputed from scratch.
    MAX_LOG_DROP = np.log(1e8)

    def __init__(
        self,
        cvSpace,
        compressionThreshold: float = 1.0,
        useExistingBandwidths: bool = True,
        kernelShape: str = "gaussian",
        minBandwidth=0.0,
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
        self._d = cvSpace.numDimensions
        self._minBandwidth = np.broadcast_to(
            np.asarray(minBandwidth, dtype=float), (self._d,)
        )
        self._setKernels([])
        self._logSumW = -np.inf
        self._logSumWSq = -np.inf
        self._logPK = np.empty(0)
        self._logPG = np.full(cvSpace.gridShape, -np.inf)
        self._peakPK = self._logPK.copy()
        self._peakPG = self._logPG.copy()

    def __bool__(self) -> bool:
        return bool(self._kernels)

    def __copy__(self):
        new = self.__class__(
            self._cvSpace,
            self._compressionThreshold,
            self._useExistingBandwidths,
            self._shape.name,
            self._minBandwidth,
        )
        new._setKernels(list(map(copy, self._kernels)))
        new._logSumW = self._logSumW
        new._logSumWSq = self._logSumWSq
        new._logPK = self._logPK.copy()
        new._logPG = self._logPG.copy()
        new._peakPK = self._peakPK.copy()
        new._peakPG = self._peakPG.copy()
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
        removed = [self._kernels[index] for index in toRemove]
        for kernel in removed:
            self._logPK = self._logsubexp(self._logPK, kernel.evaluate(centers))
            self._logPG = self._logsubexp(self._logPG, kernel.evaluateOnGrid())
        self._deleteKernels(toRemove)
        self._logPK = np.delete(self._logPK, toRemove)
        self._peakPK = np.delete(self._peakPK, toRemove)
        return removed

    def _setKernels(self, kernels):
        self._kernels = kernels
        if kernels:
            self._positions = np.stack([k.position for k in kernels])
            self._bandwidths = np.stack([k.bandwidth for k in kernels])
        else:
            self._positions = np.empty((0, self._d))
            self._bandwidths = np.empty((0, self._d))
        self._logHeights = np.array([k.logHeight for k in kernels], dtype=float)

    def _appendKernel(self, kernel):
        self._kernels.append(kernel)
        self._positions = np.vstack((self._positions, kernel.position))
        self._bandwidths = np.vstack((self._bandwidths, kernel.bandwidth))
        self._logHeights = np.append(self._logHeights, kernel.logHeight)

    def _deleteKernels(self, indices):
        for index in sorted(indices, reverse=True):
            del self._kernels[index]
        self._positions = np.delete(self._positions, indices, axis=0)
        self._bandwidths = np.delete(self._bandwidths, indices, axis=0)
        self._logHeights = np.delete(self._logHeights, indices)

    def _pushKernel(self, newKernel):
        centers = self._positions

        def bandwidths():
            return (
                self._bandwidths if self._useExistingBandwidths else newKernel.bandwidth
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
        self._peakPK = np.maximum(self._peakPK, self._logPK)
        self._peakPG = np.maximum(self._peakPG, self._logPG)
        if toRemove:
            self._removeKernels(centers, toRemove)
        self._appendKernel(newKernel)
        self._logPK = np.append(self._logPK, self._logDensityAt(newKernel.position))
        self._peakPK = np.append(self._peakPK, self._logPK[-1])
        if toRemove and self._maxLogDrop() > self.MAX_LOG_DROP:
            self._rebuildCaches()

    def _logDensityAt(self, point):
        """Log of the unnormalized mixture density at a single point.

        Reflects the point rather than each kernel, which is equivalent (see
        :meth:`CVSpace.mirrorPositions`) and lets every kernel be evaluated
        in one call. The images are summed per axis, since the kernels
        factorize over the axes.
        """
        space = self._cvSpace
        logKernels = self._logHeights
        for i, value in enumerate(point):
            terms = [
                self._shape.exponents(
                    space.axisDisplacement(i, self._positions[:, i], image)
                    / self._bandwidths[:, i]
                )
                for image in space.axisImages(i, value)
            ]
            logKernels = logKernels + np.logaddexp.reduce(terms, axis=0)
        return np.logaddexp.reduce(logKernels)

    #: Kernels contracted together by _logMixtureOnGrid, which bounds its
    #: temporary arrays to this many times the grid size.
    GRID_CHUNK_SIZE = 512

    def _logMixtureOnGrid(self, positions, bandwidths, logHeights):
        """Log of sum_k exp(logHeights[k]) G_k on the grid, in gridShape order.

        The kernels factorize over the axes, so their sum on the grid is a
        tensor contraction of per-axis profiles over the kernel index, done
        with BLAS instead of one grid evaluation per kernel. To stay in range,
        each axis's profiles are scaled by their maximum over the kernels at
        each node, and the heights by their maximum. Nodes where the scaled
        sum still underflows, far from every kernel, are summed in log space.
        """
        space = self._cvSpace
        if len(positions) == 0:
            return np.full(space.gridShape, -np.inf)
        logProfiles = [
            space.foldedAxisRows(
                i,
                self._shape.exponents(
                    space.axisGridDistances(i, positions[:, i])
                    / bandwidths[:, i, np.newaxis]
                ),
            )
            for i in range(space.numDimensions)
        ]
        peaks = [np.max(lp, axis=0) for lp in logProfiles]
        finitePeaks = [np.where(np.isfinite(m), m, 0.0) for m in peaks]
        scaled = [
            np.exp(lp - m[np.newaxis, :])
            for lp, m in zip(logProfiles, finitePeaks, strict=True)
        ]
        heightPeak = np.max(logHeights)
        heights = np.exp(logHeights - heightPeak)
        total = np.zeros(space.gridShape)
        for start in range(0, len(positions), self.GRID_CHUNK_SIZE):
            chunk = slice(start, start + self.GRID_CHUNK_SIZE)
            # Outer product over all axes but the last, then contract with
            # the last axis, which is the slowest in gridShape order.
            rest = functools.reduce(
                lambda acc, a: (a[:, :, np.newaxis] * acc[:, np.newaxis, :]).reshape(
                    a.shape[0], -1
                ),
                [a[chunk] for a in scaled[:-1]],
                np.ones((heights[chunk].size, 1)),
            )
            last = scaled[-1][chunk] * heights[chunk, np.newaxis]
            total += (last.T @ rest).reshape(space.gridShape)
        offset = heightPeak + functools.reduce(np.add.outer, reversed(peaks))
        with np.errstate(divide="ignore"):
            result = np.log(total) + offset
        underflow = (total == 0) & np.isfinite(offset)
        if np.any(underflow):
            result[underflow] = self._logMixtureAtNodes(
                np.argwhere(underflow), logProfiles, logHeights
            )
        return space.periodicCopies(result)

    @staticmethod
    def _logMixtureAtNodes(nodes, logProfiles, logHeights):
        """Exact log-space sums at selected grid nodes (gridShape indices)."""
        values = np.empty(len(nodes))
        for n, node in enumerate(nodes):
            terms = logHeights + sum(
                lp[:, j] for lp, j in zip(logProfiles, reversed(node), strict=True)
            )
            values[n] = np.logaddexp.reduce(terms)
        return values

    def _logMixtureAtPoints(self, points, positions, bandwidths, logHeights):
        """Log of sum_k exp(logHeights[k]) G_k at each of ``points``.

        Vectorized over chunks of kernels and summed per axis over the mirror
        images, as in :meth:`Kernel.evaluate`.
        """
        space = self._cvSpace
        result = np.full(len(points), -np.inf)
        for start in range(0, len(positions), self.CHUNK_SIZE):
            stop = start + self.CHUNK_SIZE
            logKernels = logHeights[start:stop, np.newaxis]
            for i in range(space.numDimensions):
                sigma = bandwidths[start:stop, i, np.newaxis]
                terms = [
                    self._shape.exponents(
                        space.axisDisplacement(i, image[:, np.newaxis], points[:, i])
                        / sigma
                    )
                    for image in space.axisImages(i, positions[start:stop, i])
                ]
                logKernels = logKernels + (
                    terms[0] if len(terms) == 1 else logsumexp(terms, axis=0)
                )
            result = np.logaddexp(result, logsumexp(logKernels, axis=0))
        return result

    #: Kernels evaluated together at every center by _logDensitiesAtCenters,
    #: which bounds its temporary arrays to this many times the kernel count.
    CHUNK_SIZE = 256

    def _logDensitiesAtCenters(self):
        """Log of the unnormalized mixture density at every kernel center."""
        centers = self._positions
        if len(centers) == 0:
            return np.empty(0)
        return self._logMixtureAtPoints(
            centers, centers, self._bandwidths, self._logHeights
        )

    def _maxLogDrop(self):
        with np.errstate(invalid="ignore"):
            return max(
                np.nanmax(self._peakPK - self._logPK, initial=0.0),
                np.nanmax(self._peakPG - self._logPG, initial=0.0),
            )

    def _rebuildCaches(self):
        """Recompute the densities on the grid and at the kernel centers.

        Subtracting a removed kernel loses precision where it held most of
        the density, and the loss compounds over successive removals. The
        rounding error of a cached value is bounded by machine epsilon times
        its peak since the last rebuild, so _pushKernel rebuilds once any
        value falls MAX_LOG_DROP below its peak.
        """
        self._logPG = self._logMixtureOnGrid(
            self._positions, self._bandwidths, self._logHeights
        )
        self._logPK = self._logDensitiesAtCenters()
        self._peakPK = self._logPK.copy()
        self._peakPG = self._logPG.copy()

    def _addKernel(
        self, position, bandwidth, logWeight, numSamples=1, adjustBandwidth=True
    ):
        if adjustBandwidth:
            bandwidth = bandwidth * self.bandwidthFactor(logWeight)
        bandwidth = np.maximum(bandwidth, self._minBandwidth)
        self._logSumW = np.logaddexp(self._logSumW, logWeight)
        self._logSumWSq = np.logaddexp(self._logSumWSq, 2 * logWeight)
        newKernel = Kernel(
            self._cvSpace, position, bandwidth, logWeight, numSamples, self._shape
        )
        if self._kernels:
            self._pushKernel(newKernel)
        else:
            self._setKernels([newKernel])
            self._rebuildCaches()

    def bandwidthFactor(self, logWeight) -> float:
        """Silverman shrink factor for a new kernel of the given log weight.

        Computed from this estimate's effective sample size with the new
        kernel's weight already counted, which is what :meth:`update` applies.
        The sample size also counts a prior weight w0 = 1, so kernels whose
        weights are small next to it barely shrink the bandwidth.
        """
        logSumW = np.logaddexp(0.0, np.logaddexp(self._logSumW, logWeight))
        logSumWSq = np.logaddexp(0.0, np.logaddexp(self._logSumWSq, 2 * logWeight))
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

    def getCorrectedLogPDF(
        self, pilotFactor: float = 2.0, iterations: int = 1, matchingScale: float = 5.0
    ):
        """Log density on the grid with the smoothing bias removed.

        Kernel smoothing makes a density estimate too low on steep walls and
        too high at the bottom of basins, to order h^2 in the bandwidth. This
        applies the multiplicative bias correction of Jones, Linton and
        Nielsen (Biometrika 82, 327 (1995)), which cuts the bias to order h^4
        while keeping the estimate positive:

        1. A pilot estimate P~ uses the same kernels, broadened by
           ``pilotFactor``, which keeps its noise low.
        2. Each kernel's weight is divided by P~ at its center, so the
           reweighted samples follow P / P~, which is nearly flat.
        3. Smoothing that ratio with the broad kernels costs little bias, and
           P~ times the smoothed ratio estimates P.

        ``iterations`` repeats steps 2 and 3 with the result as the pilot.
        The product does not conserve probability locally, which would shift
        free energy differences between basins. So the result is finally
        rescaled to carry, at the length scale of ``matchingScale`` pilot
        bandwidths, the same probability as the uncorrected estimate, which
        conserves it kernel by kernel.

        The correction needs the broadened kernels to stay narrow compared
        with the features of the density, say a third of a basin's width.
        Early in a run, when the kernels are still wide, it can do more harm
        than good.

        Only the reported density changes; the bias built from
        :meth:`getLogPDF` does not. Costs a few evaluations of the broadened
        kernels on the grid and at the kernel centers. NaN with no kernels.
        """
        standard = self.getLogPDF()
        if not self._kernels:
            return standard
        positions, logWeights = self._positions, self._logWeights()
        bandwidths = self._bandwidths * pilotFactor
        logNorm = self._d * self._shape.logNorm + np.sum(np.log(bandwidths), axis=1)
        logHeights = logWeights - logNorm
        logGrid = self._logMixtureOnGrid(positions, bandwidths, logHeights)
        logCenters = self._logMixtureAtPoints(
            positions, positions, bandwidths, logHeights
        )
        for _ in range(iterations):
            flattened = logHeights - logCenters
            logGrid = logGrid + self._logMixtureOnGrid(positions, bandwidths, flattened)
            logCenters = logCenters + self._logMixtureAtPoints(
                positions, positions, bandwidths, flattened
            )
        corrected = logGrid - self._logSumW
        weights = np.exp(logWeights - logWeights.max())
        scales = matchingScale * (weights @ bandwidths) / weights.sum()
        return self._cvSpace.periodicCopies(
            corrected + self._massRatio(standard, corrected, scales)
        )

    def _logWeights(self):
        return np.array([k.logWeight for k in self._kernels], dtype=float)

    def _massRatio(self, logTarget, logSource, scales):
        """Log of smooth(target) / smooth(source), Gaussian-smoothed per axis.

        Zero where either smoothed density vanishes. On a periodic axis the
        last node, which repeats the first, is left out of the smoothing and
        copied back.
        """
        space = self._cvSpace
        shifts = [np.nanmax(logTarget), np.nanmax(logSource)]
        smoothed = []
        for logDensity, shift in zip((logTarget, logSource), shifts, strict=True):
            density = np.nan_to_num(np.exp(logDensity - shift))
            for i, cv in enumerate(space.variables):
                axis = space.numDimensions - 1 - i
                nodes = np.linspace(cv.minValue, cv.maxValue, cv.gridWidth)
                if cv.periodic:
                    nodes = nodes[:-1]
                    density = np.take(density, range(len(nodes)), axis=axis)
                diff = nodes[:, np.newaxis] - nodes[np.newaxis, :]
                if cv.periodic:
                    length = cv.maxValue - cv.minValue
                    diff -= length * np.rint(diff / length)
                kernel = np.exp(-0.5 * (diff / scales[i]) ** 2)
                density = np.moveaxis(
                    np.tensordot(kernel, np.moveaxis(density, axis, 0), axes=1), 0, axis
                )
                if cv.periodic:
                    first = np.take(density, [0], axis=axis)
                    density = np.concatenate((density, first), axis=axis)
            smoothed.append(density)
        target, source = smoothed
        valid = (target > 0) & (source > 0)
        ratio = np.zeros_like(target)
        ratio[valid] = np.log(target[valid] / source[valid])
        return ratio + np.where(valid, shifts[0] - shifts[1], 0.0)

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

    def getNumSamples(self) -> int:
        """Number of samples deposited, however the kernels were merged."""
        return sum(kernel.numSamples for kernel in self._kernels)

    def getLogSumWeights(self) -> float:
        """Log of the total sample weight, ln(sum_k w_k); -inf with no samples."""
        return float(self._logSumW)

    def evaluate(self, point) -> float:
        """Log of the normalized density at a single point."""
        return self._logDensityAt(point) - self._logSumW

    def getState(self) -> dict:
        """Flat, npz-writable snapshot of this estimate.

        The CVSpace is deliberately excluded: all walkers share identical CV
        definitions, so a reader constructs its own and calls setState.
        """
        return {
            "positions": self._positions.copy(),
            "bandwidths": self._bandwidths.copy(),
            "logWeights": np.array([k.logWeight for k in self._kernels], dtype=float),
            "numSamples": np.array([k.numSamples for k in self._kernels], dtype=int),
            "logSumW": float(self._logSumW),
            "logSumWSq": float(self._logSumWSq),
        }

    def setState(self, state) -> None:
        """Restore from a :meth:`getState` snapshot, discarding current contents."""
        self._setKernels(
            [
                Kernel(
                    self._cvSpace, position, bandwidth, logWeight, int(n), self._shape
                )
                for position, bandwidth, logWeight, n in zip(
                    state["positions"],
                    state["bandwidths"],
                    state["logWeights"],
                    state["numSamples"],
                    strict=True,
                )
            ]
        )
        self._logSumW = float(state["logSumW"])
        self._logSumWSq = float(state["logSumWSq"])
        self._rebuildCaches()
