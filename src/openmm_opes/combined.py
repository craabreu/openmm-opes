"""OPES and OPES-explore combined in a single sampler (OPES_COM)."""

from __future__ import annotations

import os
import warnings
from copy import deepcopy

import numpy as np
import openmm as mm
from openmm import app, unit

from .opes import OPES, ExtraBias, currentCt, runningCt, tabulatedFunction

_COMPONENTS = ("opes", "explore")


class CombinedOPES:
    """OPES and OPES-explore biasing the same collective variables together.

    Implements the combined algorithm of Ray and Rizzi,
    https://doi.org/10.1021/acs.jctc.4c01231, often called OPES_COM. Both
    components deposit a kernel at the same steps, at the same CV values:

    * the OPES component estimates its marginal distribution with weights
      exp(beta V_opes) from its own bias only (their Eq. 11), so that
      distribution is still biased by the explore component (Eq. 12);
    * the explore component estimates its target distribution from the
      unweighted samples (Eq. 8), which feel both biases.

    The total bias is V_opes + V_explore. Neither component estimates the
    unbiased distribution by itself, so a third estimate reweights every sample
    by exp(beta (V_opes + V_explore)), the paper's Eq. 14, and
    :meth:`getFreeEnergy` returns it. Its kernels are sized as those of the
    explore component's reweighted estimate.

    Both components share one bias force, so the CVs are computed once per
    step. To weight OPES kernels by the OPES bias alone, the explore part of
    the force is switched off for a single energy evaluation at each
    deposition step.

    Parameters
    ----------
    system: System
        The System to simulate. A CustomCVForce implementing the total bias is
        created and added to it.
    variables: list of BiasVariable
        The collective variables both components bias.
    temperature: temperature
        The temperature at which the simulation is run.
    barrier: energy
        The barrier of the OPES component, Delta E_OPES.
    exploreBarrier: energy
        The barrier of the OPES-explore component, Delta E_EXPLORE. The paper
        finds systematic errors when it exceeds ``barrier``, and a warning is
        issued in that case.
    frequency: int
        Interval in time steps at which both components deposit a kernel.
    varianceFrequency: int or None
        Interval in time steps at which to update the CV variance, or None for
        a fixed bandwidth taken from each variable's ``biasWidth``. See
        :class:`OPES`.
    biasFactor: float, optional
        The OPES bias factor. Defaults to ``barrier / kT``.
    exploreBiasFactor: float, optional
        The OPES-explore bias factor. Defaults to ``exploreBarrier / kT``.
    saveFrequency: int, optional
        Interval in time steps at which to share the bias on disk.
    biasDir: str, optional
        Directory shared with other walkers. Each component uses its own
        subdirectory.
    walkerId: int, optional
        This walker's identifier within ``biasDir``. See :class:`OPES`.
    extraBias: list, optional
        Other biases acting on the simulation, as in :class:`OPES`. They join
        the weights of the OPES kernels and of the free energy estimate.

    The remaining parameters, ``bounded``, ``warmupSteps``,
    ``compressionThreshold``, ``useExistingBandwidths``, ``kernelShape``,
    ``statsWindowSize`` and ``minBandwidth``, apply to both components and are
    described in :class:`OPES`.
    """

    def __init__(
        self,
        system,
        variables,
        temperature,
        barrier,
        exploreBarrier,
        frequency,
        varianceFrequency,
        biasFactor=None,
        exploreBiasFactor=None,
        bounded: bool = False,
        saveFrequency=None,
        biasDir=None,
        walkerId=None,
        warmupSteps=None,
        compressionThreshold: float = 1.0,
        useExistingBandwidths: bool = True,
        kernelShape: str = "gaussian",
        statsWindowSize: int = 10,
        minBandwidth=None,
        extraBias=None,
    ):
        if not unit.is_quantity(barrier):
            barrier = barrier * unit.kilojoules_per_mole
        if not unit.is_quantity(exploreBarrier):
            exploreBarrier = exploreBarrier * unit.kilojoules_per_mole
        if exploreBarrier > barrier:
            warnings.warn(
                "exploreBarrier exceeds barrier. Ray and Rizzi found systematic "
                "free energy errors in that regime.",
                stacklevel=2,
            )
        if biasDir is not None and walkerId is None:
            walkerId = int(np.random.default_rng().integers(0x7FFFFFFF))
        settings = {
            "opes": (barrier, biasFactor, False),
            "explore": (exploreBarrier, exploreBiasFactor, True),
        }
        # Each component is a full OPES estimator. Its own bias force goes into
        # a private, never simulated System, built from copies of the variables
        # since a CustomCVForce takes ownership of them. Only this class's
        # force acts on the simulation.
        self._components = {
            name: OPES(
                mm.System(),
                [_copyVariable(v) for v in variables],
                temperature,
                componentBarrier,
                frequency,
                varianceFrequency,
                componentBiasFactor,
                exploreMode=exploreMode,
                bounded=bounded,
                saveFrequency=saveFrequency,
                biasDir=None if biasDir is None else os.path.join(biasDir, name),
                walkerId=walkerId,
                warmupSteps=warmupSteps,
                compressionThreshold=compressionThreshold,
                useExistingBandwidths=useExistingBandwidths,
                kernelShape=kernelShape,
                statsWindowSize=statsWindowSize,
                minBandwidth=minBandwidth,
            )
            for name, (componentBarrier, componentBiasFactor, exploreMode) in (
                settings.items()
            )
        }
        opes = self._components["opes"]
        self.variables = variables
        self.temperature = opes.temperature
        self.barrier = barrier
        self.exploreBarrier = exploreBarrier
        self.frequency = frequency
        self.varianceFrequency = varianceFrequency
        self.saveFrequency = saveFrequency
        self.biasDir = biasDir
        self.walkerId = walkerId
        self._interval = opes._interval
        self._contextStale = False

        freeGroups = set(range(32)) - {f.getForceGroup() for f in system.getForces()}
        if not freeGroups:
            raise RuntimeError(
                "CombinedOPES requires a free force group, but all are in use."
            )
        self._scale = _uniqueGlobalParameter(system, "opesExploreScale")
        d = len(variables)
        args = ",".join(f"cv{i}" for i in range(d))
        self._force = mm.CustomCVForce(
            f"opesTable({args}) + {self._scale}*exploreTable({args})"
        )
        self._force.addGlobalParameter(self._scale, 1.0)
        for i, variable in enumerate(variables):
            self._force.addCollectiveVariable(f"cv{i}", variable.force)
        initial = np.zeros(int(np.prod([v.gridWidth for v in variables])))
        periodic = all(v.periodic for v in variables)
        for name in _COMPONENTS:
            self._force.addTabulatedFunction(
                f"{name}Table",
                tabulatedFunction(opes._widths, initial, opes._limits, periodic),
            )
        self._force.setForceGroup(max(freeGroups))
        system.addForce(self._force)
        self._extraBias = ExtraBias(extraBias, self._force.getForceGroup())

    def step(self, simulation, steps) -> None:
        """Advance the simulation by a number of time steps.

        A CombinedOPES can also be passed to :meth:`OPES.stepTogether` along
        with other samplers.
        """
        OPES.stepTogether(simulation, [self], steps)

    def _onInterval(self, simulation) -> None:
        position = self.getCollectiveVariables(simulation)
        step = simulation.currentStep
        # Both components see the same samples, so they agree on readiness.
        ready = [c._observe(position) for c in self._components.values()]
        if not all(ready) or step % self.frequency != 0:
            return
        context = simulation.context
        opesEnergy, exploreEnergy = self.getComponentBiasEnergies(context)
        if self._extraBias:
            opesEnergy += self._extraBias.energy(context)
        opes, explore = self._components["opes"], self._components["explore"]
        # Eq. 11: OPES kernels are weighted by the OPES bias alone (plus any
        # extra bias). The explore component's target estimate is unweighted;
        # its reweighted estimate, unused by its bias, is the unbiased
        # estimate of Eq. 14.
        changed = opes.addKernel(position, opesEnergy)
        changed = explore.addKernel(position, opesEnergy + exploreEnergy) or changed
        shared = [c._share(step) for c in self._components.values()]
        if changed or any(shared):
            self.updateContext(simulation.context)

    def updateContext(self, context) -> None:
        """Push the current biases into a Context.

        Each table stores its component's bias shifted up by that component's
        barrier, so the unexplored floor sits at 0, as in :class:`OPES`.
        """
        for index, component in enumerate(self._components.values()):
            bias = component.getBias() + component.barrier
            self._force.getTabulatedFunction(index).setFunctionParameters(
                *component._widths,
                bias.value_in_unit(unit.kilojoules_per_mole).ravel(),
                *component._limits,
            )
        self._force.updateParametersInContext(context)
        self._contextStale = False

    def getCollectiveVariables(self, simulation):
        """Current values of all collective variables in a Simulation."""
        return self._force.getCollectiveVariableValues(simulation.context)

    def getComponentBiasEnergies(self, context):
        """The OPES and OPES-explore bias energies applied in a Context.

        The explore part of the force is switched off for one evaluation of
        the bias force group, then restored.
        """
        groups = {self._force.getForceGroup()}
        total = context.getState(getEnergy=True, groups=groups).getPotentialEnergy()
        context.setParameter(self._scale, 0.0)
        try:
            shiftedOpes = context.getState(
                getEnergy=True, groups=groups
            ).getPotentialEnergy()
        finally:
            context.setParameter(self._scale, 1.0)
        return (
            shiftedOpes - self.barrier,
            total - shiftedOpes - self.exploreBarrier,
        )

    def getBiasEnergy(self, context):
        """The total bias energy V_opes(s) + V_explore(s) applied in a Context."""
        groups = {self._force.getForceGroup()}
        total = context.getState(getEnergy=True, groups=groups).getPotentialEnergy()
        return total - self.barrier - self.exploreBarrier

    def getBias(self):
        """The total bias potential on the grid, V_opes + V_explore."""
        opes, explore = self.getComponentBiases()
        return opes + explore

    def getComponentBiases(self):
        """The OPES and OPES-explore bias potentials on the grid."""
        return tuple(c.getBias() for c in self._components.values())

    def getFreeEnergy(self):
        """Free energy as a function of the collective variables.

        The estimate reweights every sample by the total bias, Eq. 14 of Ray and
        Rizzi. The grid follows :meth:`OPES.getFreeEnergy`.
        """
        return self._components["explore"].getFreeEnergy()

    def getCt(self, estimate="current"):
        """c(t) of the total bias, V_opes + V_explore.

        ``"current"`` averages exp(-beta V) over the free energy estimate of
        :meth:`getFreeEnergy`; ``"running"`` is kT ln of the mean weight of its
        samples, with an initial sample under the flat bias
        -(barrier + exploreBarrier). See :meth:`OPES.getCt`.
        """
        explore = self._components["explore"]
        if estimate == "running":
            initial = -(self.barrier + self.exploreBarrier)
            return runningCt(explore._kde["total.rw"], initial, explore._kbt)
        if estimate == "current":
            return currentCt(
                explore._kde["total.rw"].getLogPDF(),
                self.getBias(),
                explore._kbt,
                all(v.periodic for v in self.variables),
            )
        raise ValueError('estimate must be "current" or "running"')

    def getNumKernels(self) -> tuple[int, int]:
        """Numbers of kernels in the OPES and OPES-explore bias estimates."""
        return (
            self._components["opes"].getNumKernels(),
            self._components["explore"].getNumKernels(),
        )

    def getState(self) -> dict:
        """Flat, npz-writable snapshot of both components' accumulated state."""
        return {
            f"{name}_{key}": value
            for name, component in self._components.items()
            for key, value in component.getState().items()
        }

    def setState(self, state) -> None:
        """Restore from a :meth:`getState` snapshot.

        The restored bias reaches the simulation at the start of the next
        :meth:`step`, or sooner through :meth:`updateContext`.
        """
        for name, component in self._components.items():
            prefix = f"{name}_"
            component.setState(
                {
                    key[len(prefix) :]: value
                    for key, value in state.items()
                    if key.startswith(prefix)
                }
            )
        self._contextStale = True


def _copyVariable(variable):
    """A BiasVariable identical to another but owning a copy of its force."""
    return app.BiasVariable(
        deepcopy(variable.force),
        variable.minValue,
        variable.maxValue,
        variable.biasWidth,
        variable.periodic,
        variable.gridWidth,
    )


def _uniqueGlobalParameter(system, base):
    """A global parameter name that no force in the System uses yet."""
    taken = set()
    for force in system.getForces():
        if hasattr(force, "getNumGlobalParameters"):
            taken.update(
                force.getGlobalParameterName(i)
                for i in range(force.getNumGlobalParameters())
            )
    name, index = base, 1
    while name in taken:
        index += 1
        name = f"{base}{index}"
    return name
