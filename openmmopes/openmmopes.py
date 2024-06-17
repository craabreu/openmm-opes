"""
.. module:: opes
   :platform: Linux, MacOS
   :synopsis: On-the-fly Probability Enhanced Sampling with OpenMM

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import os
import re
from dataclasses import dataclass
from functools import reduce

import numpy as np
import openmm as mm
from openmm import unit
from openmm.app.metadynamics import _LoadedBias

LOG2PI = np.log(2 * np.pi)
DECAY_WINDOW: int = 10


@dataclass
class KernelDensityEstimate:
    """
    A kernel density estimate on a grid.

    Parameters
    ----------
    shape
        The shape of the grid.

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

    def __init__(self, shape):
        self.logSumW = -np.inf
        self.logSumW2 = -np.inf
        self.logAccInvDensity = -np.inf
        self.logAccGaussian = np.full(shape, -np.inf)
        self.d = len(shape)

    def _addWeight(self, weight):
        self.logSumW = np.logaddexp(self.logSumW, weight)
        self.logSumW2 = np.logaddexp(self.logSumW2, 2 * weight)

    def _getBandwidthFactor(self):
        neff = np.exp(2 * self.logSumW - self.logSumW2)
        return (neff * (self.d + 2) / 4) ** (-2 / (self.d + 4))

    def _addKernel(self, logWeight, logGaussian, indices):
        self.logAccGaussian = np.logaddexp(self.logAccGaussian, logGaussian)
        self.logAccInvDensity = np.logaddexp(
            self.logAccInvDensity,
            logWeight + self.logSumW - self.logAccGaussian[indices],
        )

    def getLogPDF(self):
        """
        Get the logarithm of the probability density function evaluated on the grid.
        """
        return self.logAccGaussian - self.logSumW

    def getBias(self, prefactor, logEpsilon):
        """
        Get the bias potential evaluated on the grid.

        Parameters
        ----------
        prefactor
            The prefactor of the bias potential.
        logEpsilon
            The logarithm of the minimum value of the bias potential.
        """
        return prefactor * np.logaddexp(
            self.logAccInvDensity - 2 * self.logSumW + self.logAccGaussian.ravel(),
            logEpsilon,
        )

    def update(self, logWeight, axisSquaredDistances, variances, indices):
        """
        Update the kernel density estimate with a new Gaussian kernel.

        Parameters
        ----------
        logWeight
            The logarithm of the weight assigned to the kernel.
        axisSquaredDistances
            The squared distances from the kernel center to the grid points along each
            axis.
        variances
            The variances of the variables along each axis.
        indices
            The indices of the grid point closest to the kernel center.
        """
        self._addWeight(logWeight)
        bandwidth = self._getBandwidthFactor() * variances
        exponents = [
            -0.5 * sqDistances / sqSigma
            for sqDistances, sqSigma in zip(axisSquaredDistances, bandwidth)
        ]
        logHeight = logWeight - 0.5 * (self.d * LOG2PI + np.log(bandwidth).sum())
        logGaussian = logHeight + reduce(np.add.outer, reversed(exponents))
        self._addKernel(logWeight, logGaussian, tuple(reversed(indices)))


class OPES:
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
        if (saveFrequency is None) != (biasDir is None):
            raise ValueError("Must specify both saveFrequency and biasDir")
        if saveFrequency and (saveFrequency % frequency != 0):
            raise ValueError("saveFrequency must be a multiple of frequency")

        d = len(variables)
        biasFactor = barrier / (unit.MOLAR_GAS_CONSTANT_R * temperature)
        numPeriodics = sum(v.periodic for v in variables)
        freeGroups = set(range(32)) - set(f.getForceGroup() for f in system.getForces())

        if biasFactor <= 1.0:
            raise ValueError("barrier must be greater than 1 kT")
        if numPeriodics not in [0, d]:
            raise ValueError("OPES cannot handle mixed periodic/non-periodic variables")
        if not 1 <= d <= 3:
            raise ValueError("OPES requires 1, 2, or 3 collective variables")
        if not freeGroups:
            raise RuntimeError("All 32 force groups are already in use.")

        self.variables = variables
        self.temperature = temperature
        self.frequency = frequency
        self.adaptiveVariance = adaptiveVariance
        self.exploreMode = exploreMode
        self.saveFrequency = saveFrequency
        self.biasDir = biasDir

        kbt = unit.MOLAR_GAS_CONSTANT_R * temperature
        prefactor = (1 - 1 / biasFactor) * kbt
        if exploreMode:
            prefactor *= biasFactor
        varNames = [f"cv{i}" for i in range(d)]

        self._d = d
        self._widths = np.array([v.gridWidth for v in variables])
        self._lengths = np.array([v.maxValue - v.minValue for v in variables])
        self._lbounds = np.array([v.minValue for v in variables])
        self._limits = sum(([v.minValue, v.maxValue] for v in variables), [])
        self._periodic = numPeriodics == d
        self._biasFactor = biasFactor
        self._kbt = kbt.in_units_of(unit.kilojoules_per_mole)
        self._prefactor = prefactor.value_in_unit(unit.kilojoules_per_mole)
        self._logEpsilon = -barrier / prefactor
        self._tau = DECAY_WINDOW * frequency
        self._sampleMean = None
        self._sampleVariance = np.array([v.biasWidth**2 for v in variables])
        self._id = np.random.randint(0x7FFFFFFF)
        self._saveIndex = 0
        # self._selfBias = np.zeros(tuple(v.gridWidth for v in reversed(variables)))
        # self._totalBias = np.zeros(tuple(v.gridWidth for v in reversed(variables)))
        self._loadedBiases = {}
        self._syncWithDisk()
        self._force = mm.CustomCVForce(f"table({', '.join(varNames)})")
        self._table = getattr(mm, f"Continuous{d}DFunction")(
            *(self._widths if d > 1 else []),
            np.full(np.prod(self._widths), -barrier / unit.kilojoules_per_mole),
            *self._limits,
            self._periodic,
        )
        self._grid = [
            np.linspace(v.minValue, v.maxValue, v.gridWidth) for v in variables
        ]
        self._kde = KernelDensityEstimate(self._widths)
        self._kdeRw = KernelDensityEstimate(self._widths) if exploreMode else self._kde

        for name, var in zip(varNames, variables):
            self._force.addCollectiveVariable(name, var.force)
        self._force.addTabulatedFunction("table", self._table)
        self._force.setForceGroup(max(freeGroups))
        system.addForce(self._force)

    def _updateSampleStats(self, values):
        """Update the sample mean and variance of the collective variables."""
        delta = values - self._sampleMean
        if self._periodic:
            delta -= self._lengths * np.rint(delta / self._lengths)
        self._sampleMean += delta / self._tau
        if self._periodic:
            self._sampleMean = (
                self._lbounds + (self._sampleMean - self._lbounds) % self._lengths
            )
        self._sampleVariance += (delta**2 - self._sampleVariance) / self._tau

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
        if self._sampleMean is None:
            self._sampleMean = np.array(self.getCollectiveVariables(simulation))
        while stepsToGo > 0:
            nextSteps = stepsToGo
            nextSteps = min(
                nextSteps, self.frequency - simulation.currentStep % self.frequency
            )
            if self.adaptiveVariance:
                for _ in range(nextSteps):
                    simulation.step(1)
                    position = self.getCollectiveVariables(simulation)
                    self._updateSampleStats(position)
            else:
                simulation.step(nextSteps)
            if simulation.currentStep % self.frequency == 0:
                position = self.getCollectiveVariables(simulation)
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

    def getFreeEnergy(self, reweighted=False):
        """Get the free energy of the system as a function of the collective variables.

        The result is returned as a N-dimensional NumPy array, where N is the number of
        collective
        variables.  The values are in kJ/mole.  The i'th position along an axis
        corresponds to
        minValue + i*(maxValue-minValue)/gridWidth.
        """
        kde = self._kdeRw if reweighted else self._kde
        freeEnergy = -self._kbt * kde.getLogPDF()
        if self.exploreMode and not reweighted:
            return freeEnergy * self._biasFactor
        return freeEnergy

    def getCollectiveVariables(self, simulation):
        """Get the current values of all collective variables in a Simulation."""
        return self._force.getCollectiveVariableValues(simulation.context)

    def _addGaussian(self, values, energy, context):
        """Add a Gaussian to the bias function."""
        values = np.array(values)
        if self._periodic:
            values = self._lbounds + (values - self._lbounds) % self._lengths
        scaled_values = (self._widths - 1) * (values - self._lbounds) / self._lengths
        indices = np.clip(np.rint(scaled_values).astype(int), 0, self._widths - 1)

        axisSquaredDistances = []
        for value, nodes, length in zip(values, self._grid, self._lengths):
            distances = nodes - value
            if self._periodic:
                distances -= length * np.rint(distances / length)
                distances[-1] = distances[0]
            axisSquaredDistances.append(distances**2)

        self._kdeRw.update(
            energy / self._kbt,
            axisSquaredDistances,
            self._sampleVariance / self._biasFactor,
            indices,
        )

        if self.exploreMode:
            self._kde.update(0, axisSquaredDistances, self._sampleVariance, indices)

        self._table.setFunctionParameters(
            *(self._widths if self._d > 1 else []),
            self._kde.getBias(self._prefactor, self._logEpsilon),
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

        oldName = os.path.join(self.biasDir, f"bias_{self._id}_{self._saveIndex}.npy")
        self._saveIndex += 1
        tempName = os.path.join(self.biasDir, f"temp_{self._id}_{self._saveIndex}.npy")
        fileName = os.path.join(self.biasDir, f"bias_{self._id}_{self._saveIndex}.npy")
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
