# Quickstart

A single particle on a two-dimensional Müller-Brown potential, biased along
`x` and `y`, recovering the free energy surface.

```python
import numpy as np
import openmm as mm
from openmm import app, unit

from openmm_opes import OPES

KB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
    unit.kilojoules_per_mole / unit.kelvin
)

muller_brown = (
    "-200*exp(-(x-1)^2-10*y^2)"
    " -100*exp(-x^2-10*(y-0.5)^2)"
    " -170*exp(-6.5*(0.5+x)^2+11*(x+0.5)*(y-1.5)-6.5*(y-1.5)^2)"
    " +15*exp(0.7*(1+x)^2+0.6*(x+1)*(y-1)+0.7*(y-1)^2)"
)

system = mm.System()
system.addParticle(1.0)
potential = mm.CustomExternalForce(muller_brown)
potential.addParticle(0, [])
system.addForce(potential)

variables = []
for expression, lo, hi in (("x", -1.5, 1.2), ("y", -0.2, 2.0)):
    force = mm.CustomExternalForce(expression)
    force.addParticle(0, [])
    variables.append(app.BiasVariable(force, lo, hi, 0.1, False, 101))

temperature = 1.0 / KB * unit.kelvin
sampler = OPES(
    system,
    variables,
    temperature,
    barrier=20 * unit.kilojoules_per_mole,
    frequency=500,
    varianceFrequency=50,
)

topology = app.Topology()
topology.addAtom("P", None, topology.addResidue("MOL", topology.addChain()))
integrator = mm.LangevinMiddleIntegrator(
    temperature, 10.0 / unit.picosecond, 0.005 * unit.picoseconds
)
simulation = app.Simulation(
    topology, system, integrator, mm.Platform.getPlatformByName("Reference")
)
simulation.context.setPositions([mm.Vec3(-0.5, 1.4, 0.0)])
simulation.context.setVelocitiesToTemperature(temperature)

sampler.step(simulation, 2_000_000)

fes = sampler.getFreeEnergy().value_in_unit(unit.kilojoules_per_mole)
fes -= fes.min()
print(f"{sampler.getNumKernels()} kernels, barrier {fes.max():.1f} kJ/mol")
```

`getFreeEnergy()` returns an array whose shape is the CV grid with the **last**
variable varying fastest, so for two CVs it has shape `(ny, nx)` and can be
handed straight to `matplotlib.pyplot.contourf`.

## Choosing the bandwidth

Both papers prescribe measuring the initial bandwidth from a short unbiased
run. You can let the sampler do that for you:

```python
sampler = OPES(..., varianceFrequency=50, warmupSteps=50_000)
```

Nothing is deposited during the first 50,000 steps; the CV variance measured
there becomes the initial bandwidth and is then held fixed.

To fix the bandwidth yourself instead, pass `varianceFrequency=None` and set
each variable's `biasWidth`. It is read as $\sigma^{(0)}$, the standard
deviation of the **unbiased** distribution, in both modes. OPES-explore
kernels estimate the sampled distribution, which is $\sqrt{\gamma}$ times
wider, so they are deposited $\sqrt{\gamma}$ times wider than `biasWidth`.

## Other biases and c(t)

When other biases act on the simulation, such as a restraint you want removed
from the free energy or another sampler, pass them as `extraBias`, as with
PLUMED's EXTRA_BIAS. Each sample is then weighted by all of them, so
`getFreeEnergy()` removes them too. Items can be force groups, `Force` objects
or other samplers:

```python
restraint.setForceGroup(5)  # give it a group of its own
sampler = OPES(system, variables, temperature, barrier, 500, 50,
               extraBias=[restraint])
```

`getCt()` returns c(t), the constant that makes reweighting with a bias that
still changes exact: a sample taken at time t gets weight
exp(β[V_t(s) − c(t)]). The default estimate uses the current bias and density
estimate. `getCt("running")` is PLUMED's `rct`. It lags behind the current
bias, so treat it, as PLUMED does, as a convergence diagnostic that levels off
once the bias converges, not as a reweighting constant.

`extraBias` also works in explore mode, where it changes only the reweighted
free energy. PLUMED offers EXTRA_BIAS for OPES_METAD only.

## OPES combined with OPES-explore

`CombinedOPES` runs OPES and OPES-explore together on the same collective
variables, the combined algorithm of Ray & Rizzi, J. Chem. Theory Comput. 21,
58 (2025), often called OPES_COM. The explore bias keeps pushing the system out
of its minima, which helps when the CVs are suboptimal, while the OPES bias
converges the free energy. It takes one barrier per component; the paper
recommends an explore barrier below the OPES one.

```python
from openmm_opes import CombinedOPES

sampler = CombinedOPES(system, variables, temperature, barrier=40,
                       exploreBarrier=20, frequency=500, varianceFrequency=50)
sampler.step(simulation, 1_000_000)
fes = sampler.getFreeEnergy()
```

Neither component alone estimates the unbiased distribution: each one's
samples also feel the other's bias. `getFreeEnergy()` therefore returns a
separate estimate that reweights every sample by the total bias.
`getComponentBiases()` and `getComponentBiasEnergies(context)` give the two
parts separately.

## Correcting the smoothing bias

Kernel smoothing makes steep basin walls too low and basin floors too high. To
remove that bias from the reported surface, ask for it explicitly:

```python
fes = sampler.getFreeEnergy(correctSmoothing=True)
```

It also lowers the noise of a single run, but it needs kernels narrow compared
with the basins, so use it on well-converged runs. See the theory page.

## Multiple walkers

Several simulations can share one bias. Give each its own sampler with the
same settings, point them all at the same directory, and give each a stable
walker id:

```python
sampler = OPES(system, variables, temperature, barrier, 500, 50,
               saveFrequency=5000, biasDir="walkers", walkerId=rank)
```

Every `saveFrequency` steps, which must be a multiple of the deposition
`frequency`, each walker writes its own kernels to the directory and folds in
everyone else's. The walkers can run as separate processes or on separate
machines that share a file system. A stable `walkerId` lets a restarted
walker take back its own file, instead of leaving the old one behind as a
phantom extra walker. `CombinedOPES` takes the same options and keeps each
component's files in its own subdirectory.

## Several samplers at once

More generally, several samplers can bias the same simulation, each depositing
kernels on its own schedule and weighting them by its own bias only, as several
OPES actions do in a single PLUMED input. Create them all on the same `System`
and advance the simulation with `OPES.stepTogether`. For example, an OPES
sampler on two CVs alongside an OPES-explore sampler on the first one:

```python
opes = OPES(system, [cv1, cv2], temperature, 55, 500, 50)
explore = OPES(system, [cv1Copy], temperature, 30, 500, 50, exploreMode=True)
simulation = app.Simulation(topology, system, integrator)
...
OPES.stepTogether(simulation, [opes, explore], 1_000_000)
```

Each variable's force is copied into its sampler's bias force, so pass every
sampler its own `BiasVariable` objects. Each sampler's `getFreeEnergy()` is
then biased by the others. To reweight, add up the biases, which
`getBiasEnergy(simulation.context)` returns for each sampler.
