"""
.. module:: opes
   :platform: Linux, MacOS
   :synopsis: On-the-fly Probability Enhanced Sampling with OpenMM

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import os
import re
import typing as t
from dataclasses import dataclass
from functools import reduce

import numpy as np
import openmm as mm
from openmm import app, unit
from openmm.app.metadynamics import _LoadedBias

LOG2PI = np.log(2 * np.pi)
DECAY_WINDOW: int = 10


@dataclass
class BiasData:
    """
    Data structure to store the bias information.

    Parameters
    ----------
    shape
        The shape of the bias grid.

    Attributes
    ----------
    logSumW
        The logarithm of the sum of the weights.
    logSumW2
        The logarithm of the sum of the squared weights.
    logAccInvDensity
        The logarithm of the accumulated inverse probability density.
    logAccGaussian
        The logarithm of the accumulated Gaussians on a grid.
    """

    logSumW: float
    logSumW2: float
    logAccInvDensity: float
    logAccGaussian: np.ndarray

    def __init__(self, shape: t.Tuple[int, ...]) -> None:
        self.logSumW = -np.inf
        self.logSumW2 = -np.inf
        self.logAccInvDensity = -np.inf
        self.logAccGaussian = np.full(shape, -np.inf)
        d = len(shape)
        self._prefactor = -2 / (d + 4)
        self._offset = self._prefactor * np.log((d + 2) / 4)

    def addWeight(self, logWeight: float) -> None:
        """
        Add a weight to the bias data.

        Parameters
        ----------
        logWeight
            The logarithm of the weight to be added.
        """
        self.logSumW = np.logaddexp(self.logSumW, logWeight)
        self.logSumW2 = np.logaddexp(self.logSumW2, 2 * logWeight)

    def selectBandwitdh(self, variances: np.ndarray) -> np.ndarray:
        """
        Select the bandwidth of the Gaussian kernel.

        Parameters
        ----------
        variances
            The variances of the Gaussian kernel.

        Returns
        -------
        np.ndarray
            The selected bandwidth.
        """
        logNeff = 2 * self.logSumW - self.logSumW2
        logSilvermanFactor = self._prefactor * logNeff + self._offset
        return np.exp(logSilvermanFactor) * variances

    def addKernel(
        self, logWeight: float, indices: t.Sequence[int], logGaussian: np.ndarray
    ) -> None:
        """
        Add a Gaussian kernel to the bias data.

        Parameters
        ----------
        logWeight
            The logarithm of the weight assigned to the kernel.
        indices
            The indices of the grid point that is nearest to the kernel center.
        logGaussian
            The logarithm of the Gaussian kernel on each grid point.
        """
        self.logAccGaussian = np.logaddexp(self.logAccGaussian, logWeight + logGaussian)
        logDensity = self.logAccGaussian[tuple(reversed(indices))] - self.logSumW
        # print(indices)
        # print(f"logDensity = {logDensity}")
        # import matplotlib.pyplot as plt
        # plt.plot(self.logAccGaussian - self.logSumW)
        # plt.show()
        self.logAccInvDensity = np.logaddexp(
            self.logAccInvDensity, logWeight - logDensity
        )

    def getLogPDF(self) -> np.ndarray:
        """Get the logarithm of the probability density function on a grid."""
        return self.logAccGaussian - self.logSumW

    def getLogScaledPDF(self) -> np.ndarray:
        """Get the logarithm of the scaled probability density function on a grid."""
        return self.logAccInvDensity - 2 * self.logSumW + self.logAccGaussian


class Kernel:
    """
    A multivariate Gaussian kernel with diagonal bandwidth matrix.

    Parameters
    ----------
    variables
        The collective variables that define the multidimensional domain of the kernel.
    position
        The point in space where the kernel is centered.
    bandwidth
        The bandwidth (standard deviation) of the kernel in each direction.
    logWeight
        The logarithm of the weight assigned to the kernel.

    Attributes
    ----------
    position : np.ndarray
        The point in space where the kernel is centered.
    bandwidth : np.ndarray
        The bandwidth (standard deviation) of the kernel in each direction.
    logWeight : float
        The logarithm of the weight assigned to the kernel.
    logHeight : float
        The logarithm of the kernel's height.
    """

    def __init__(
        self,
        variables: t.Sequence[app.BiasVariable],
        position: t.Sequence[float],
        bandwidth: t.Sequence[float],
        logWeight: float,
    ) -> None:
        ndims = len(variables)
        assert len(position) == len(bandwidth) == ndims
        self.position = np.asarray(position)
        self.bandwidth = np.asarray(bandwidth)
        self.logWeight = logWeight
        self._periodic = any(cv.periodic for cv in variables)
        if self._periodic:
            self._pdims = [i for i, cv in enumerate(variables) if cv.periodic]
            self._lbounds = np.array([variables[i].minValue for i in self._pdims])
            ubounds = np.array([variables[i].maxValue for i in self._pdims])
            self._lengths = ubounds - self._lbounds
        self.logHeight = self._computeLogHeight()

    def _computeLogHeight(self) -> float:
        if np.any(self.bandwidth == 0):
            return -np.inf
        ndims = len(self.bandwidth)
        log2pi = np.log(2 * np.pi)
        log_height = self.logWeight - ndims * log2pi / 2 - np.log(self.bandwidth).sum()
        return log_height

    def _squareMahalanobisDistances(self, points: np.ndarray) -> np.ndarray:
        return np.square(self.displacement(points) / self.bandwidth).sum(axis=-1)

    def displacement(self, endpoint: np.ndarray) -> np.ndarray:
        """
        Compute the displacement vector from the kernel's position to a given endpoint,
        taking periodicity into account.

        Parameters
        ----------
        endpoint
            The endpoint to which the displacement vector is computed.

        Returns
        -------
        np.ndarray
            The displacement vector from the kernel's position to the endpoint.
        """
        disp = endpoint - self.position
        if self._periodic:
            disp[..., self._pdims] -= self._lengths * np.round(
                disp[..., self._pdims] / self._lengths
            )
        return disp

    def endpoint(self, displacement: np.ndarray) -> np.ndarray:
        """
        Compute the endpoint of a displacement vector from the kernel's position

        Parameters
        ----------
        displacement
            The displacement vector from the kernel's position.

        Returns
        -------
        np.ndarray
            The endpoint of the displacement vector from the kernel's position.
        """
        end = self.position + displacement
        if self._periodic:
            end[..., self._pdims] = (
                self._lbounds + (end[..., self._pdims] - self._lbounds) % self._lengths
            )
        return end

    def findNearest(
        self, points: np.ndarray, ignore: t.Sequence[int] = ()
    ) -> t.Tuple[int, float]:
        """
        Given a list of points in space, return the index of the nearest one and the
        squared Mahalanobis distance to it. Optionally ignore some points.

        Parameters
        ----------
        points
            The list of points to compare against. The shape of this array must be
            :math:`(N, d)`, where :math:`N` is the number of points and :math:`d` is
            the dimensionality of the kernel.
        ignore
            The indices of points to ignore.

        Returns
        -------
        int
            The index of the point (or -1 if no points are given)
        float
            The squared Mahalanobis distance to the closest point (or infinity if
            no points are given)
        """
        if points.size == 0:
            return -1, np.inf
        sq_mahalanobis_distances = self._squareMahalanobisDistances(points)
        if ignore:
            sq_mahalanobis_distances[ignore] = np.inf
        index = np.argmin(sq_mahalanobis_distances)
        return index, sq_mahalanobis_distances[index]

    def merge(self, other: "Kernel") -> None:
        """
        Change this kernel by merging it with another one.

        Parameters
        ----------
        other
            The kernel to merge with.
        """
        log_sum_weights = np.logaddexp(self.logWeight, other.logWeight)
        w1 = np.exp(self.logWeight - log_sum_weights)
        w2 = np.exp(other.logWeight - log_sum_weights)

        displacement = self.displacement(other.position)
        mean_position = self.endpoint(w2 * displacement)
        mean_squared_bandwidth = w1 * self.bandwidth**2 + w2 * other.bandwidth**2

        self.logWeight = log_sum_weights
        self.position = mean_position
        self.bandwidth = np.sqrt(mean_squared_bandwidth + w1 * w2 * displacement**2)
        self.logHeight = self._computeLogHeight()

    def evaluate(self, points: np.ndarray) -> t.Union[float, np.ndarray]:
        """
        Compute the natural logarithm of the kernel evaluated at the given point or
        points.

        Parameters
        ----------
        point
            The point or points at which to evaluate the kernel. The shape of this
            array must be either :math:`(d,)` or :math:`(N, d)`, where :math:`d` is
            the dimensionality of the kernel and :math:`N` is the number of points.

        Returns
        -------
        float
            The logarithm of the kernel evaluated at the given point or points.
        """
        return self.logHeight - 0.5 * self._squareMahalanobisDistances(points)

    def evaluateOnGrid(
        self, gridMarks: t.Sequence[np.ndarray]
    ) -> t.Tuple[t.Tuple[int, ...], np.ndarray]:
        """
        Return the natural logarithms of the kernel evaluated on a rectilinear grid and
        the indices of the grid point that is closest to the kernel's position.

        Parameters
        ----------
        gridMarks
            The points in each dimension used to define the rectilinear grid. The length
            of this list must match the dimensionality :math:`d` of the kernel. The size
            :math:`N_i` of each array :math:`i` is arbitrary. For periodic dimensions,
            it is assumed that the grid spans the entire periodic length, i.e. that the
            last point differs from the first by the periodic length.

        Returns
        -------
        Tuple[int, ...]
            The indices of the grid point that is closest to the kernel's position.
        np.ndarray
            The logarithm of the kernel evaluated on the grid points. The shape of this
            array is :math:`(N_d, \\ldots, N_2, N_1)`, which makes it compatible with
            OpenMM's ``TabulatedFunction`` convention.
        """
        distances = [points - x for points, x in zip(gridMarks, self.position)]
        if self._periodic:
            for dim, length in zip(self._pdims, self._lengths):
                distances[dim] -= length * np.round(distances[dim] / length)
                distances[dim][-1] = distances[dim][0]
        exponents = [
            -0.5 * (distance / sigma) ** 2
            for distance, sigma in zip(distances, self.bandwidth)
        ]
        indices = tuple(map(np.argmax, reversed(exponents)))
        return indices, self.logHeight + reduce(np.add.outer, reversed(exponents))


class OPES(object):
    """Performs OPES."""

    def __init__(
        self,
        system,
        variables,
        temperature,
        barrier,
        frequency,
        exploreMode=False,
        adaptiveVariance=True,
        saveFrequency=None,
        biasDir=None,
    ):
        if not unit.is_quantity(temperature):
            temperature = temperature * unit.kelvin
        if not unit.is_quantity(barrier):
            barrier = barrier * unit.kilojoules_per_mole
        biasFactor = barrier / (unit.MOLAR_GAS_CONSTANT_R * temperature)
        if biasFactor <= 1.0:
            raise ValueError("biasFactor must be > 1")
        if (saveFrequency is None) != (biasDir is None):
            raise ValueError("Must specify both saveFrequency and biasDir")
        if saveFrequency and (saveFrequency % frequency != 0):
            raise ValueError("saveFrequency must be a multiple of frequency")

        self.variables = variables
        self.temperature = temperature
        self.frequency = frequency
        self.adaptiveVariance = adaptiveVariance
        self.exploreMode = exploreMode
        self.saveFrequency = saveFrequency
        self.biasDir = biasDir

        self._id = np.random.randint(0x7FFFFFFF)
        self._saveIndex = 0
        self._selfBias = np.zeros(tuple(v.gridWidth for v in reversed(variables)))
        # self._totalBias = np.zeros(tuple(v.gridWidth for v in reversed(variables)))
        widths = [v.gridWidth for v in variables]
        self._totalBias = np.full(np.prod(widths), -barrier / unit.kilojoules_per_mole)

        self._loadedBiases = {}
        self._syncWithDisk()
        varNames = ["cv%d" % i for i in range(len(variables))]
        self._force = mm.CustomCVForce("table(%s)" % ", ".join(varNames))
        for name, var in zip(varNames, variables):
            self._force.addCollectiveVariable(name, var.force)
        self._widths = widths if len(variables) > 1 else []
        self._limits = sum(([v.minValue, v.maxValue] for v in variables), [])
        numPeriodics = sum(v.periodic for v in variables)
        if numPeriodics not in [0, len(variables)]:
            raise ValueError(
                "Metadynamics cannot handle mixed periodic/non-periodic variables"
            )
        periodic = numPeriodics == len(variables)
        if len(variables) == 1:
            self._table = mm.Continuous1DFunction(
                self._totalBias.flatten(), *self._limits, periodic
            )
        elif len(variables) == 2:
            self._table = mm.Continuous2DFunction(
                *self._widths, self._totalBias.flatten(), *self._limits, periodic
            )
        elif len(variables) == 3:
            self._table = mm.Continuous3DFunction(
                *self._widths, self._totalBias.flatten(), *self._limits, periodic
            )
        else:
            raise ValueError("Metadynamics requires 1, 2, or 3 collective variables")
        self._force.addTabulatedFunction("table", self._table)
        freeGroups = set(range(32)) - set(
            force.getForceGroup() for force in system.getForces()
        )
        if len(freeGroups) == 0:
            raise RuntimeError(
                "Cannot assign a force group to the metadynamics force. "
                "The maximum number (32) of the force groups is already used."
            )
        self._force.setForceGroup(max(freeGroups))
        system.addForce(self._force)

        kbt = unit.MOLAR_GAS_CONSTANT_R * temperature
        self._kbt = kbt.in_units_of(unit.kilojoules_per_mole)
        self._grid = [
            np.linspace(cv.minValue, cv.maxValue, cv.gridWidth) for cv in variables
        ]
        self._logSumWeights = self._logSumWeightsSq = -np.inf
        self._logPGrid = np.full([cv.gridWidth for cv in reversed(variables)], -np.inf)

        self._bias_factor = biasFactor
        prefactor = (
            (biasFactor - 1) * kbt if exploreMode else (1 - 1 / biasFactor) * kbt
        )
        self._prefactor = prefactor.value_in_unit(unit.kilojoules_per_mole)
        self._logEpsilon = -barrier / prefactor

        self._tau = 10 * frequency
        self._counter = 0
        self._bwFactor = 1.0 if exploreMode else 1.0 / np.sqrt(biasFactor)

        self._log_acc_inv_density = -np.inf

        self._means = np.array([0.5*(cv.minValue + cv.maxValue) for cv in variables])
        self._variances = np.array([cv.biasWidth**2 for cv in variables])
        self._periodic = periodic
        self._lengths = np.array([cv.maxValue - cv.minValue for cv in variables])
        self._lbounds = np.array([cv.minValue for cv in variables])

    def _updateMovingKernel(self, values: t.Tuple[float, ...]) -> None:
        """
        Update the moving kernel used to estimate the bandwidth of the

        Parameters
        ----------
        values
            The current values of the collective variables.
        """
        delta = values - self._means
        if self._periodic:
            delta -= self._lengths * np.round(delta / self._lengths)
        x = 1 / self._tau
        self._means += x * delta
        if self._periodic:
            self._means = self._lbounds + (self._means - self._lbounds) % self._lengths
        self._variances = self._counter * self._variances + delta**2
        self._counter += 1
        self._variances /= self._counter

    def step(self, simulation, steps):
        """Advance the simulation by integrating a specified number of time steps.

        Parameters
        ----------
        simulation: Simulation
            the Simulation to advance
        steps: int
            the number of time steps to integrate
        """
        stepsToGo = steps
        forceGroup = self._force.getForceGroup()
        while stepsToGo > 0:
            nextSteps = stepsToGo
            nextSteps = min(
                nextSteps, self.frequency - simulation.currentStep % self.frequency
            )
            for _ in range(nextSteps):
                simulation.step(1)
                position = self._force.getCollectiveVariableValues(simulation.context)
                self._updateMovingKernel(position)
            if simulation.currentStep % self.frequency == 0:
                position = self._force.getCollectiveVariableValues(simulation.context)
                energy = simulation.context.getState(
                    getEnergy=True, groups={forceGroup}
                ).getPotentialEnergy()
                self._addGaussian(position, energy, simulation.context)
            if (
                self.saveFrequency is not None
                and simulation.currentStep % self.saveFrequency == 0
            ):
                self._syncWithDisk()
            stepsToGo -= nextSteps

    def getFreeEnergy(self):
        """Get the free energy of the system as a function of the collective variables.

        The result is returned as a N-dimensional NumPy array, where N is the number of
        collective
        variables.  The values are in kJ/mole.  The i'th position along an axis
        corresponds to
        minValue + i*(maxValue-minValue)/gridWidth.
        """
        free_energy = -self._kbt * (self._logPGrid - self._logSumWeights)
        if self.exploreMode:
            free_energy *= self._bias_factor
        return free_energy

    def getCollectiveVariables(self, simulation):
        """Get the current values of all collective variables in a Simulation."""
        return self._force.getCollectiveVariableValues(simulation.context)

    def _addGaussian(self, values, energy, context):
        """Add a Gaussian to the bias function."""
        # Compute a Gaussian along each axis.

        if self.exploreMode:
            log_weight = 0
        else:
            log_weight = energy / self._kbt

        self._logSumWeights = np.logaddexp(self._logSumWeights, log_weight)
        self._logSumWeightsSq = np.logaddexp(self._logSumWeightsSq, 2 * log_weight)
        neff = np.exp(2 * self._logSumWeights - self._logSumWeightsSq)
        d = len(self.variables)
        silverman = (neff * (d + 2) / 4) ** (-1 / (d + 4))

        # bandwidth = silverman * self._bwFactor * self._movingKernel.bandwidth
        bandwidth = silverman * self._bwFactor * np.sqrt(self._variances)
        new_kernel = Kernel(self.variables, values, bandwidth, log_weight)
        indices, log_gaussian = new_kernel.evaluateOnGrid(self._grid)

        self._logPGrid = np.logaddexp(self._logPGrid, log_gaussian)
        self._log_acc_inv_density = np.logaddexp(
            self._log_acc_inv_density,
            log_weight + self._logSumWeights - self._logPGrid[indices],
        )

        logP = self._logPGrid - self._logSumWeights
        logZ = self._logSumWeights - self._log_acc_inv_density
        self._table.setFunctionParameters(
            *self._widths,
            self._prefactor * np.logaddexp(logP - logZ, self._logEpsilon).flatten(),
            *self._limits,
        )
        self._force.updateParametersInContext(context)

    def _syncWithDisk(self):
        """
        Save biases to disk, and check for updated files created by other processes.
        """
        if self.biasDir is None:
            return

        # Use a safe save to write out the biases to disk, then delete the older file.

        oldName = os.path.join(
            self.biasDir, "bias_%d_%d.npy" % (self._id, self._saveIndex)
        )
        self._saveIndex += 1
        tempName = os.path.join(
            self.biasDir, "temp_%d_%d.npy" % (self._id, self._saveIndex)
        )
        fileName = os.path.join(
            self.biasDir, "bias_%d_%d.npy" % (self._id, self._saveIndex)
        )
        np.save(tempName, self._selfBias)
        os.rename(tempName, fileName)
        if os.path.exists(oldName):
            os.remove(oldName)

        # Check for any files updated by other processes.

        fileLoaded = False
        pattern = re.compile(r"bias_(.*)_(.*)\.npy")
        for filename in os.listdir(self.biasDir):
            match = pattern.match(filename)
            if match is not None:
                matchId = int(match.group(1))
                matchIndex = int(match.group(2))
                if matchId != self._id and (
                    matchId not in self._loadedBiases
                    or matchIndex > self._loadedBiases[matchId].index
                ):
                    try:
                        data = np.load(os.path.join(self.biasDir, filename))
                        self._loadedBiases[matchId] = _LoadedBias(
                            matchId, matchIndex, data
                        )
                        fileLoaded = True
                    except IOError:
                        # There's a tiny chance the file could get deleted by another
                        # process between when we check the directory and when we try
                        # to load it.  If so, just ignore the error and keep using
                        # whatever version of that process' biases we last loaded.
                        pass

        # If we loaded any files, recompute the total bias from all processes.

        if fileLoaded:
            self._totalBias = np.copy(self._selfBias)
            for bias in self._loadedBiases.values():
                self._totalBias += bias.bias
