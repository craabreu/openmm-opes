"""Online kernel density estimation for OPES.

Pure NumPy/SciPy. This module must not import openmm.
"""

from __future__ import annotations

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
