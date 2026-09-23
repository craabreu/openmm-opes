"""The OPES bias, integrated with OpenMM as a CustomCVForce.

Closely mirrors ``openmm.app.Metadynamics``: same constructor shape, same
unit-coercion idiom, same force-group selection, same ``step`` contract.
"""

from __future__ import annotations

import warnings
from copy import copy

import numpy as np
import openmm as mm
from openmm import unit

from .io import BiasSharer
from .kde import CVSpace, OnlineKDE


class RunningAverage:
    """Accumulates a running mean of per-CV squared deviations."""

    def __init__(self, numDimensions: int = 1):
        self._num = 0
        self._total = np.zeros(numDimensions)

    def __iadd__(self, other):
        self._total = self._total + other._total
        self._num += other._num
        return self

    def copy(self):
        """An independent copy. The array is copied, not aliased."""
        new = self.__class__(self._total.shape[0])
        new._num = self._num
        # The source aliased this array, so merging peers into a copy silently
        # corrupted the original. See spec 7.4.1.
        new._total = self._total.copy()
        return new

    def update(self, sample) -> None:
        """Fold one sample into the average."""
        self._num += 1
        self._total = self._total + sample

    def get(self):
        """The running average."""
        return self._total / max(1, self._num)

    def getState(self) -> dict:
        """Flat, npz-writable snapshot."""
        return {"num": float(self._num), "total": self._total.copy()}

    def setState(self, state) -> None:
        """Restore from a :meth:`getState` snapshot."""
        self._num = int(state["num"])
        self._total = np.asarray(state["total"]).copy()


class OPES:
    """On-the-fly Probability Enhanced Sampling.

    Implements OPES and its exploratory variant, following Invernizzi and
    Parrinello, https://doi.org/10.1021/acs.jpclett.0c00497 and
    https://doi.org/10.1021/acs.jctc.2c00152.

    Parameters
    ----------
    system: System
        The System to simulate. A CustomCVForce implementing the bias is
        created and added to it.
    variables: list of BiasVariable
        The collective variables whose sampling should be enhanced.
    temperature: temperature
        The temperature at which the simulation is run.
    barrier: energy
        The free energy barrier the simulation should overcome.
    frequency: int
        Interval in time steps at which to deposit a kernel.
    varianceFrequency: int or None
        Interval in time steps at which to update the CV variance. When None,
        the bandwidth is fixed and taken from each variable's ``biasWidth``,
        which is interpreted as the standard deviation of the UNBIASED
        distribution, exactly as both papers define sigma^(0).
    biasFactor: float, optional
        Defaults to ``barrier / kT``.
    exploreMode: bool
        Whether to apply the OPES-explore variant.
    bounded: bool
        Whether non-periodic CVs have reflective boundaries.
    saveFrequency: int, optional
        Interval in time steps at which to share the bias on disk. Must be a
        multiple of ``frequency``.
    biasDir: str, optional
        Directory shared with other walkers.
    walkerId: int, optional
        This walker's identifier within ``biasDir``. Random when omitted; pass
        a stable value across restarts so a resumed walker reclaims its own
        file rather than leaving the previous run's behind as a phantom peer.
    warmupSteps: int, optional
        When given, run this many steps without depositing kernels while
        measuring the CV variance, then freeze it. See the spec, section 7.7.
    compressionThreshold: float
        Mahalanobis distance below which kernels are merged.
    useExistingBandwidths: bool
        Whether the merge search scales by existing kernels' bandwidths.
    kernelShape: str
        ``"gaussian"`` or ``"compact"``.
    statsWindowSize: int
        Window, in deposition strides, of the running CV-mean estimate. With
        an adaptive variance, the first kernel also waits this many strides,
        so that it is sized from a measured variance.
    """

    def __init__(
        self,
        system,
        variables,
        temperature,
        barrier,
        frequency,
        varianceFrequency,
        biasFactor=None,
        exploreMode: bool = False,
        bounded: bool = False,
        saveFrequency=None,
        biasDir=None,
        walkerId=None,
        warmupSteps=None,
        compressionThreshold: float = 1.0,
        useExistingBandwidths: bool = True,
        kernelShape: str = "gaussian",
        statsWindowSize: int = 10,
    ):
        if not unit.is_quantity(temperature):
            temperature = temperature * unit.kelvin
        if not unit.is_quantity(barrier):
            barrier = barrier * unit.kilojoules_per_mole

        self.variables = variables
        self.temperature = temperature
        self.barrier = barrier
        self.frequency = frequency
        self.varianceFrequency = varianceFrequency
        self.exploreMode = exploreMode
        self.bounded = bounded
        self.saveFrequency = saveFrequency
        self.biasDir = biasDir
        self.walkerId = walkerId
        self.warmupSteps = warmupSteps
        self.statsWindowSize = statsWindowSize

        d = len(variables)
        kbt = unit.MOLAR_GAS_CONSTANT_R * temperature
        userSuppliedBiasFactor = biasFactor is not None
        if not userSuppliedBiasFactor:
            biasFactor = barrier / kbt
        self.biasFactor = biasFactor
        numPeriodics = sum(v.periodic for v in variables)
        freeGroups = set(range(32)) - {f.getForceGroup() for f in system.getForces()}
        self._validate(d, biasFactor, numPeriodics, freeGroups, userSuppliedBiasFactor)

        prefactor = (1 - 1 / biasFactor) * kbt
        if exploreMode:
            prefactor *= biasFactor
        self._kbt = kbt.in_units_of(unit.kilojoules_per_mole)
        self._biasFactor = biasFactor
        self._prefactor = prefactor
        self._logEpsilon = -barrier / prefactor

        self._cvSpace = CVSpace(variables, bounded)
        self._cases = ("total",) + ("self",) * bool(saveFrequency)
        # Retained so every KDE this sampler rebuilds later -- in setState and
        # in _syncWithDisk -- is configured identically. Rebuilding with the
        # defaults silently reverted kernelShape/compressionThreshold on
        # restore, changing the bias and every subsequent deposition.
        self._compressionThreshold = compressionThreshold
        self._useExistingBandwidths = useExistingBandwidths
        self._kernelShape = kernelShape
        self._kde = {}
        for case in self._cases:
            self._kde[case] = self._newKDE()
            self._kde[f"{case}.rw"] = self._newKDE()

        self._adaptiveVariance = varianceFrequency is not None
        self._interval = varianceFrequency or frequency
        self._variance = {case: RunningAverage(d) for case in self._cases}
        self._warmupComplete = warmupSteps is None
        if self._adaptiveVariance:
            self._tau = statsWindowSize * frequency // varianceFrequency
            self._counter = 0
            self._sampleMean = np.zeros(d)
        else:
            # Spec 7.5: biasWidth is the unbiased sigma^(0), while _variance
            # holds a sampled-distribution variance, which is gamma times wider.
            self._setFixedVariance(
                biasFactor * np.array([v.biasWidth**2 for v in variables])
            )

        self._sharer = BiasSharer(biasDir, walkerId) if biasDir is not None else None

        gridWidths = [v.gridWidth for v in variables]
        self._widths = [] if d == 1 else gridWidths
        self._limits = [limit for v in variables for limit in (v.minValue, v.maxValue)]
        periodic = numPeriodics == d
        initial = np.full(int(np.prod(gridWidths)), -barrier / unit.kilojoules_per_mole)

        energyFunction = "table(" + ",".join(f"cv{i}" for i in range(d)) + ")"
        self._force = mm.CustomCVForce(energyFunction)
        for i, variable in enumerate(variables):
            self._force.addCollectiveVariable(f"cv{i}", variable.force)
        # Explicit branch rather than a dynamic getattr, so ty can check it.
        if d == 1:
            table = mm.Continuous1DFunction(initial, *self._limits, periodic)
        elif d == 2:
            table = mm.Continuous2DFunction(
                *self._widths, initial, *self._limits, periodic
            )
        else:
            table = mm.Continuous3DFunction(
                *self._widths, initial, *self._limits, periodic
            )
        self._force.addTabulatedFunction("table", table)
        self._force.setForceGroup(max(freeGroups))
        system.addForce(self._force)

    def _validate(
        self, d, biasFactor, numPeriodics, freeGroups, userSuppliedBiasFactor
    ):
        if self.frequency <= 0:
            raise ValueError("frequency must be positive")
        if self.statsWindowSize <= 0:
            raise ValueError("statsWindowSize must be positive")
        if self.varianceFrequency is not None and self.varianceFrequency <= 0:
            raise ValueError("varianceFrequency must be positive or None")
        if self.varianceFrequency and self.frequency % self.varianceFrequency != 0:
            raise ValueError("varianceFrequency must be a divisor of frequency")
        if (self.saveFrequency is None) != (self.biasDir is None):
            raise ValueError("Must specify both saveFrequency and biasDir")
        if self.saveFrequency is not None:
            if self.saveFrequency <= 0:
                raise ValueError("saveFrequency must be positive")
            if self.saveFrequency % self.frequency != 0:
                raise ValueError("saveFrequency must be a multiple of frequency")
        if self.warmupSteps is not None:
            if self.varianceFrequency is None:
                raise ValueError("warmupSteps requires varianceFrequency")
            if self.warmupSteps <= 0:
                raise ValueError("warmupSteps must be positive")
            if self.warmupSteps % self.varianceFrequency != 0:
                raise ValueError("warmupSteps must be a multiple of varianceFrequency")
            if self.warmupSteps // self.varianceFrequency < 2:
                raise ValueError(
                    "warmupSteps must span at least two varianceFrequency intervals"
                )
        # Checked on its own: with an explicit biasFactor, the barrier / kT
        # test below never sees the barrier, and a non-positive one gives
        # epsilon >= 1, which swamps P/Z and flattens the bias.
        if self.barrier <= 0 * unit.kilojoules_per_mole:
            raise ValueError("barrier must be positive")
        if biasFactor <= 1.0:
            raise ValueError(
                "biasFactor must be greater than 1"
                if userSuppliedBiasFactor
                else "barrier must be greater than 1 kT"
            )
        if numPeriodics not in [0, d]:
            raise ValueError("OPES cannot handle mixed periodic/non-periodic variables")
        if not 1 <= d <= 3:
            raise ValueError("OPES requires 1, 2, or 3 collective variables")
        if not freeGroups:
            raise RuntimeError("OPES requires a free force group, but all are in use.")

    def _setFixedVariance(self, value) -> None:
        """Replace every accumulator with a frozen value (spec 7.5 and 7.7)."""
        for case in self._cases:
            average = RunningAverage(len(self.variables))
            average.update(np.asarray(value, dtype=float))
            self._variance[case] = average
        self._adaptiveVariance = False

    def getVariance(self):
        """Variance currently used to size newly deposited kernels."""
        return self._variance["total"].get()

    def getNumKernels(self) -> int:
        """Number of kernels in the estimate that defines the bias."""
        return self._kde["total" if self.exploreMode else "total.rw"].getNumKernels()

    def getBias(self):
        """The OPES bias potential on the grid.

        With no kernels deposited yet the estimate is empty, and both the log
        PDF and the log mean density are -inf, whose difference is NaN. The
        bias is well defined in that limit, though: the regularization term
        dominates, leaving the flat -barrier floor the tabulated function is
        initialized to. Returning it explicitly keeps NaN out of the forces.
        """
        kde = self._kde["total" if self.exploreMode else "total.rw"]
        if kde.getNumKernels() == 0:
            return self._prefactor * np.full(self._cvSpace.gridShape, self._logEpsilon)
        return self._prefactor * np.logaddexp(
            kde.getLogPDF() - kde.getLogMeanDensity(), self._logEpsilon
        )

    def getFreeEnergy(self):
        """Free energy as a function of the collective variables.

        Returned as an N-dimensional array in kJ/mole. The i'th position along
        an axis corresponds to
        ``minValue + i*(maxValue-minValue)/(gridWidth-1)``, matching the
        ``numpy.linspace(minValue, maxValue, gridWidth)`` axis the estimate is
        built on. Always the importance-sampling estimate, which converges
        better than the direct one in explore mode.

        Every entry is NaN until the first kernel is deposited: with no
        samples there is no density to take a logarithm of. Unlike
        :meth:`getBias`, this has no defined limit to fall back on, and it
        never reaches the forces.
        """
        return -self._kbt * self._kde["total.rw"].getLogPDF()

    def getAverageDensity(self):
        """Z_n, the mean density over the explored CV space.

        NaN until the first kernel is deposited, like :meth:`getFreeEnergy`.
        """
        kde = self._kde["total" if self.exploreMode else "total.rw"]
        return np.exp(kde.getLogMeanDensity())

    def getCollectiveVariables(self, simulation):
        """Current values of all collective variables in a Simulation."""
        return self._force.getCollectiveVariableValues(simulation.context)

    def updateContext(self, context) -> None:
        """Push the current bias into a Context."""
        bias = self.getBias().value_in_unit(unit.kilojoules_per_mole)
        self._force.getTabulatedFunction(0).setFunctionParameters(
            *self._widths, bias.ravel(), *self._limits
        )
        self._force.updateParametersInContext(context)

    def addKernel(self, values, biasEnergy, variance=None) -> bool:
        """Deposit a kernel into the probability estimates.

        Returns whether a kernel was actually deposited: a non-positive
        variance estimate is skipped rather than deposited, since a zero
        bandwidth gives the kernel -inf log height and poisons the estimate.

        This does not refresh any Context; call :meth:`updateContext`
        afterwards if a simulation is running.
        """
        if not unit.is_quantity(biasEnergy):
            biasEnergy = biasEnergy * unit.kilojoules_per_mole
        if variance is None:
            variance = self._variance["total"].get()
        # A plain list survives the positivity check below (np.asarray there
        # is local to that check) but then fails the /= self._biasFactor
        # division that follows, so it is coerced once, up front, for both.
        variance = np.asarray(variance, dtype=float)
        if np.any(variance <= 0):
            # Spec 7.4.4: a zero bandwidth poisons the estimate with NaN.
            warnings.warn(
                "Skipping kernel deposition: the CV variance estimate is not "
                "yet positive. Use a varianceFrequency smaller than frequency, "
                "or set warmupSteps.",
                stacklevel=2,
            )
            return False
        logWeight = biasEnergy / self._kbt
        # Sized once, by the shared "total" estimates, and deposited
        # identically into "self". _syncWithDisk rebuilds "total" from every
        # walker's "self", so sizing "self" by the walker's own sample size
        # widened every kernel by numWalkers^(1/(d+4)) and made the bias
        # jump at each sync.
        factor = self._kde["total"].bandwidthFactor(0.0)
        factorRW = self._kde["total.rw"].bandwidthFactor(logWeight)
        for case in self._cases:
            self._kde[case].update(values, 0.0, variance, factor)
            self._kde[f"{case}.rw"].update(
                values, logWeight, variance / self._biasFactor, factorRW
            )
        return True

    def _updateSampleStats(self, values) -> None:
        self._counter += 1
        delta = self._cvSpace.displacement(self._sampleMean, values)
        x = 1 / min(self._tau, self._counter)
        self._sampleMean = self._cvSpace.endpoint(self._sampleMean, x * delta)
        sqdev = delta * self._cvSpace.displacement(self._sampleMean, values)
        for case in self._cases:
            self._variance[case].update(sqdev)

    def _syncWithDisk(self) -> None:
        # Only ever called when saveFrequency is set, which _validate ties to
        # biasDir being set, which is what constructs self._sharer.
        assert self._sharer is not None
        self._sharer.save(self._getSharedState())
        if not self._sharer.load():
            return
        self._kde["total"] = copy(self._kde["self"])
        self._kde["total.rw"] = copy(self._kde["self.rw"])
        self._variance["total"] = self._variance["self"].copy()
        for state in self._sharer.getLoadedStates().values():
            self._kde["total"] += self._kdeFromState(state, "kde")
            self._kde["total.rw"] += self._kdeFromState(state, "kdeRW")
            peer = RunningAverage(len(self.variables))
            peer.setState({"num": state["var_num"], "total": state["var_total"]})
            self._variance["total"] += peer

    def _getSharedState(self) -> dict:
        state = {}
        for prefix, key in (("kde", "self"), ("kdeRW", "self.rw")):
            for name, value in self._kde[key].getState().items():
                state[f"{prefix}_{name}"] = value
        variance = self._variance["self"].getState()
        state["var_num"] = variance["num"]
        state["var_total"] = variance["total"]
        return state

    def _newKDE(self) -> OnlineKDE:
        """An empty KDE carrying this sampler's configured options."""
        return OnlineKDE(
            self._cvSpace,
            compressionThreshold=self._compressionThreshold,
            useExistingBandwidths=self._useExistingBandwidths,
            kernelShape=self._kernelShape,
        )

    def _kdeFromState(self, state, prefix):
        kde = self._newKDE()
        strip = len(prefix) + 1
        kde.setState(
            {
                key[strip:]: value
                for key, value in state.items()
                if key.startswith(f"{prefix}_")
            }
        )
        return kde

    def step(self, simulation, steps) -> None:
        """Advance the simulation by a number of time steps.

        Parameters
        ----------
        simulation: Simulation
            The Simulation to advance.
        steps: int
            The number of time steps to integrate.
        """
        stepsToGo = steps
        while stepsToGo > 0:
            nextSteps = min(
                stepsToGo,
                self._interval - simulation.currentStep % self._interval,
            )
            simulation.step(nextSteps)
            if simulation.currentStep % self._interval == 0:
                self._onInterval(simulation)
            stepsToGo -= nextSteps

    def _onInterval(self, simulation) -> None:
        position = self.getCollectiveVariables(simulation)
        if not self._warmupComplete:
            self._updateSampleStats(position)
            # Not simulation.currentStep >= warmupSteps: currentStep is the
            # simulation's absolute clock, which already includes any steps
            # run before this OPES existed (equilibration, or steps from a
            # previous segment across a restart). _counter instead counts
            # only the variance samples THIS sampler has taken, which is
            # what warmupSteps is meant to bound and which getState/setState
            # carry across a restart.
            if self._counter >= self.warmupSteps // self.varianceFrequency:
                self._finishWarmup()
            return
        if self._adaptiveVariance:
            self._updateSampleStats(position)
            # Hold the first kernel until a full stats window of variance
            # samples is in: after a single stride the estimate rests on a
            # few correlated samples, and the too-narrow kernels it yields
            # survive compression. An estimate holding kernels (restored,
            # or loaded from peers) is past this point.
            if self._counter < self._tau and not self._kde["total"]:
                return
        if simulation.currentStep % self.frequency == 0:
            groups = {self._force.getForceGroup()}
            energy = simulation.context.getState(
                getEnergy=True, groups=groups
            ).getPotentialEnergy()
            # Only refresh the context when the estimate actually changed;
            # a skipped deposition leaves the bias exactly as it was.
            if self.addKernel(position, energy):
                self.updateContext(simulation.context)
            if (
                self.saveFrequency is not None
                and simulation.currentStep % self.saveFrequency == 0
            ):
                self._syncWithDisk()

    def _finishWarmup(self) -> None:
        """Freeze sigma^(0) measured from the unbiased warm-up segment.

        Nothing was deposited during warm-up, so the trajectory was unbiased
        and this measurement is the unbiased variance. _variance holds a
        sampled-distribution variance by convention, which is gamma times
        wider, so that is what gets stored. This is the same transformation
        the fixed-bandwidth path applies to biasWidth. See spec 7.5 and 7.7.
        """
        measured = self._variance["total"].get()
        if np.any(measured <= 0):
            raise ValueError(
                "Warm-up produced a non-positive CV variance. Increase "
                "warmupSteps or decrease varianceFrequency."
            )
        self._setFixedVariance(self._biasFactor * measured)
        self._warmupComplete = True

    def getState(self) -> dict:
        """Flat, npz-writable snapshot of this sampler's accumulated state."""
        state = self._getSharedState() if "self" in self._cases else {}
        for prefix, key in (("total", "total"), ("totalRW", "total.rw")):
            for name, value in self._kde[key].getState().items():
                state[f"{prefix}_{name}"] = value
        variance = self._variance["total"].getState()
        state["totalVar_num"] = variance["num"]
        state["totalVar_total"] = variance["total"]
        # Persisted explicitly rather than re-derived from the step counter:
        # re-deriving would risk rescaling an already-frozen variance by gamma
        # a second time on reload. See spec 7.7.
        state["warmupComplete"] = float(self._warmupComplete)
        # Only present when varianceFrequency was given (see __init__). Not
        # persisting these left a restored sampler's running CV mean and its
        # windowing counter starting from zero, instead of continuing where
        # the saved run left off.
        if hasattr(self, "_counter"):
            state["counter"] = float(self._counter)
            state["sampleMean"] = self._sampleMean.copy()
        return state

    def setState(self, state) -> None:
        """Restore from a :meth:`getState` snapshot."""
        for prefix, key in (("total", "total"), ("totalRW", "total.rw")):
            self._kde[key] = self._kdeFromState(state, prefix)
        average = RunningAverage(len(self.variables))
        average.setState(
            {"num": state["totalVar_num"], "total": state["totalVar_total"]}
        )
        self._variance["total"] = average
        # The own-contribution accumulators must come back too. _syncWithDisk
        # rebuilds "total" from "self" plus peers, so leaving "self" empty
        # threw the restored history away at the next sync and republished an
        # empty state to the other walkers.
        if "self" in self._cases and "kde_logWeights" in state:
            for prefix, key in (("kde", "self"), ("kdeRW", "self.rw")):
                self._kde[key] = self._kdeFromState(state, prefix)
            own = RunningAverage(len(self.variables))
            own.setState({"num": state["var_num"], "total": state["var_total"]})
            self._variance["self"] = own
        self._warmupComplete = bool(float(state["warmupComplete"]))
        if self._warmupComplete and self.warmupSteps is not None:
            self._adaptiveVariance = False
        if hasattr(self, "_counter") and "counter" in state:
            self._counter = int(state["counter"])
            self._sampleMean = np.asarray(state["sampleMean"]).copy()
