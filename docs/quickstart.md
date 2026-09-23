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
