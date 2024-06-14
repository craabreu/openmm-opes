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
STATS_DECAY_WINDOW: int = 10


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
        self.logAccInvDensity = np.logaddexp(
            self.logAccInvDensity,
            logWeight + self.logSumW - self.logAccGaussian[tuple(indices)],
        )

    def getLogPDF(self) -> np.ndarray:
        """Get the logarithm of the probability density function on a grid."""
        return self.logAccGaussian - self.logSumW

    def getLogScaledPDF(self) -> np.ndarray:
        """Get the logarithm of the scaled probability density function on a grid."""
        return self.logAccInvDensity - 2 * self.logSumW + self.logAccGaussian


class OPES:  # pylint: disable=too-many-instance-attributes
    """Performs metadynamics.

    This class implements well-tempered metadynamics, as described in Barducci et al.,
    "Well-Tempered Metadynamics: A Smoothly Converging and Tunable Free-Energy Method"
    (https://doi.org/10.1103/PhysRevLett.100.020603).  You specify from one to three
    collective variables whose sampling should be accelerated.  A biasing force that
    depends on the collective variables is added to the simulation.  Initially the bias
    is zero.  As the simulation runs, Gaussian bumps are periodically added to the bias
    at the current location of the simulation.  This pushes the simulation away from
    areas
    it has already explored, encouraging it to sample other regions.  At the end of the
    simulation, the bias function can be used to calculate the system's free energy as a
    function of the collective variables.

    To use the class you create a Metadynamics object, passing to it the System you want
    to simulate and a list of BiasVariable objects defining the collective variables.
    It creates a biasing force and adds it to the System.  You then run the simulation
    as usual, but call step() on the Metadynamics object instead of on the Simulation.

    You can optionally specify a directory on disk where the current bias function
    should
    periodically be written.  In addition, it loads biases from any other files in the
    same directory and includes them in the simulation.  It loads files when the
    Metqdynamics object is first created, and also checks for any new files every time
    it
    updates its own bias on disk.

    This serves two important functions.  First, it lets you stop a metadynamics run and
    resume it later.  When you begin the new simulation, it will load the biases
    computed
    in the earlier simulation and continue adding to them.  Second, it provides an easy
    way to parallelize metadynamics sampling across many computers.  Just point all of
    them to a shared directory on disk.  Each process will save its biases to that
    directory, and also load in and apply the biases added by other processes.
    """

    def __init__(
        self,
        system: mm.System,
        variables: t.Sequence[app.BiasVariable],
        temperature: unit.Quantity,
        barrier: unit.Quantity,
        frequency: int,
        adaptiveVariance: bool = True,
        exploreMode: bool = False,
        saveFrequency: t.Optional[int] = None,
        biasDir: t.Union[os.PathLike, str, None] = None,
    ):
        """Create a Metadynamics object.

        Parameters
        ----------
        system: System
            the System to simulate.  A CustomCVForce implementing the bias is created
            and added to the System.
        variables: list of BiasVariables
            the collective variables to sample
        temperature: temperature
            the temperature at which the simulation is being run.  This is used in
            computing the free energy.
        biasFactor: float
            used in scaling the height of the Gaussians added to the bias.  The
            collective variables are sampled as if the effective temperature of the
            simulation were temperature*biasFactor.
        height: energy
            the initial height of the Gaussians to add
        frequency: int
            the interval in time steps at which Gaussians should be added to the bias
            potential
        saveFrequency: int (optional)
            the interval in time steps at which to write out the current biases to disk.
            At the same time it writes biases, it also checks for updated biases written
            by other processes and loads them in.  This must be a multiple of frequency.
        biasDir: str (optional)
            the directory to which biases should be written, and from which biases
            written by other processes should be loaded
        """
        if not unit.is_quantity(temperature):
            temperature = temperature * unit.kelvin
        if not unit.is_quantity(barrier):
            barrier = barrier * unit.kilojoules_per_mole
        self.variables = variables
        self.temperature = temperature
        self.barrier = barrier
        self.frequency = frequency
        self.adaptiveVariance = adaptiveVariance
        self.exploreMode = exploreMode
        self.biasDir = biasDir
        self.saveFrequency = saveFrequency

        d = len(variables)
        numPeriodics = sum(cv.periodic for cv in variables)
        biasFactor = barrier / (unit.MOLAR_GAS_CONSTANT_R * temperature)
        freeGroups = set(range(32)) - set(f.getForceGroup() for f in system.getForces())
        self._validate(d, numPeriodics, biasFactor, freeGroups)

        self._id = np.random.randint(0x7FFFFFFF)
        self._saveIndex = 0
        self._loadedBiases = {}
        self._syncWithDisk()

        widths = [cv.gridWidth for cv in reversed(variables)]
        self._kT = unit.MOLAR_GAS_CONSTANT_R * temperature
        self._biasFactor = biasFactor
        self._prefactor = self._kT * (
            (biasFactor - 1) if exploreMode else (1 - 1 / biasFactor)
        )
        self._logEpsilon = -barrier / self._prefactor
        self._scaledGrid = [np.linspace(0, 1, cv.gridWidth) for cv in variables]
        self._means = np.array([(cv.minValue + cv.maxValue) / 2 for cv in variables])
        self._variances = np.array([cv.biasWidth**2 for cv in variables])
        self._tau = STATS_DECAY_WINDOW * self.frequency
        self._lengths = np.array([cv.maxValue - cv.minValue for cv in variables])
        self._lbounds = np.array([cv.minValue for cv in variables])
        self._widths = widths if d > 1 else []
        self._limits = sum(([cv.minValue, cv.maxValue] for cv in variables), [])
        self._periodic = numPeriodics == d

        self._totalReweight = BiasData(widths)
        self._totalBias = BiasData(widths) if exploreMode else self._totalReweight
        if saveFrequency:
            self._selfBias = BiasData(widths)

        varNames = [f"cv{i}" for i in range(d)]
        self._force = mm.CustomCVForce(f"table({',  '.join(varNames)})")
        for name, cv in zip(varNames, variables):
            self._force.addCollectiveVariable(name, cv.force)
        table = np.full(np.prod(widths), self._logEpsilon)
        self._table = getattr(mm, f"Continuous{d}DFunction")(
            *self._widths, table, *self._limits, self._periodic
        )
        self._force.addTabulatedFunction("table", self._table)
        self._force.setForceGroup(max(freeGroups))
        system.addForce(self._force)

    def _validate(
        self, d: int, numPeriodics: int, biasFactor: float, freeGroups: set
    ) -> None:
        if not 1 <= d <= 3:
            raise ValueError("OPES requires 1, 2, or 3 collective variables")
        if numPeriodics not in [0, d]:
            raise ValueError("OPES cannot handle mixed periodic/non-periodic variables")
        if biasFactor <= 1.0:
            raise ValueError("barrier must be greater than 1 kT")
        if (self.saveFrequency is None) != (self.biasDir is None):
            raise ValueError("Must specify both saveFrequency and biasDir")
        if self.saveFrequency and (self.saveFrequency % self.frequency != 0):
            raise ValueError("saveFrequency must be a multiple of frequency")
        if len(freeGroups) == 0:
            raise RuntimeError("All 32 force groups are already in use.")

    def _updateSampleVariances(
        self, currentStep: int, position: t.Tuple[float, ...]
    ) -> None:
        delta = np.array(position) - self._means
        if self._periodic:
            delta -= self._lengths * np.round(delta / self._lengths)
        x = 1 / min(currentStep, self._tau)
        self._means += x * delta
        if self._periodic:
            self._means = self._lbounds + (self._means - self._lbounds) % self._lengths
        self._variances += x * ((1 - x) * delta**2 - self._variances)

    def _addGaussian(
        self, position: t.Tuple[float, ...], biasEnergy: float, context: mm.Context
    ) -> None:
        """Add a Gaussian to the bias function."""

        # Compute square distances along each axis.

        d = len(self.variables)
        indices = []
        axisSquaredDistances = []
        for i in reversed(range(d)):
            cv = self.variables[i]
            x = (position[i] - cv.minValue) / self._lengths[i]
            if cv.periodic:
                x = x % 1.0
            indices.append(round(x * (cv.gridWidth - 1)))
            dist = self._scaledGrid[i] - x
            if cv.periodic:
                dist -= np.round(dist)
                dist[-1] = dist[0]
            dist *= self._lengths[i]
            axisSquaredDistances.append(dist**2)

        # Add the Gaussian to the unbiased PDF estimate.

        logWeight = biasEnergy / self._kT
        self._totalReweight.addWeight(logWeight)
        logGaussian = self._logGaussian(
            axisSquaredDistances,
            self._totalReweight.selectBandwitdh(self._variances / self._biasFactor),
        )
        self._totalReweight.addKernel(logWeight, indices, logGaussian)

        # If in explore mode, add the Gaussian to the biased PDF estimate.

        if self.exploreMode:
            self._totalBias.addWeight(0)
            logGaussian = self._logGaussian(
                axisSquaredDistances,
                self._totalBias.selectBandwitdh(self._variances),
            )
            self._totalBias.addKernel(0, indices, logGaussian)

        potential = self._prefactor * np.logaddexp(
            self._totalBias.getLogScaledPDF().ravel(), self._logEpsilon
        )
        self._table.setFunctionParameters(*self._widths, potential, *self._limits)
        self._force.updateParametersInContext(context)

    def _logGaussian(
        self, axisSquaredDistances: t.List[np.ndarray], variances: np.ndarray
    ) -> np.ndarray:
        axisExponents = [
            -0.5 * squaredDistances / variance
            for squaredDistances, variance in zip(axisSquaredDistances, variances)
        ]
        d = len(self.variables)
        logHeight = -0.5 * (d * LOG2PI + np.log(variances).sum())
        if d == 1:
            return axisExponents[0] + logHeight
        return reduce(np.add.outer, axisExponents) + logHeight

    def _syncWithDisk(self) -> None:
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

    def step(self, simulation: app.SimulatedTempering, steps: int) -> None:
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
        context = simulation.context
        while stepsToGo > 0:
            nextSteps = min(
                stepsToGo, self.frequency - simulation.currentStep % self.frequency
            )
            if self.adaptiveVariance:
                for _ in range(nextSteps):
                    simulation.step(1)
                    position = self._force.getCollectiveVariableValues(context)
                    self._updateSampleVariances(simulation.currentStep, position)
            else:
                simulation.step(nextSteps)
            if simulation.currentStep % self.frequency == 0:
                state = context.getState(getEnergy=True, groups={forceGroup})
                energy = state.getPotentialEnergy()
                position = self._force.getCollectiveVariableValues(context)
                self._addGaussian(position, energy, context)
            if self.saveFrequency and simulation.currentStep % self.saveFrequency == 0:
                self._syncWithDisk()
            stepsToGo -= nextSteps

    def getFreeEnergy(self, reweighted: bool = True) -> np.ndarray:
        """Get the free energy of the system as a function of the collective variables.

        The result is returned as a N-dimensional NumPy array, where N is the number of
        collective variables.  The values are in kJ/mole.  The i'th position along an
        axis corresponds to minValue + i*(maxValue-minValue)/gridWidth.
        """
        if reweighted:
            logPDF = self._totalReweight.getLogPDF()
        else:
            logPDF = self._totalBias.getLogPDF() / self._biasFactor
        return -self._kT * logPDF

    def getCollectiveVariables(self, simulation: app.Simulation) -> t.Tuple[float, ...]:
        """Get the current values of all collective variables in a Simulation."""
        return self._force.getCollectiveVariableValues(simulation.context)
