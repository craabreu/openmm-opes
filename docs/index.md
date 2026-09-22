# openmm-opes

An [OpenMM](https://openmm.org) implementation of **On-the-fly Probability
Enhanced Sampling** (OPES) and its exploratory variant, OPES-explore.

OPES builds a bias potential from an on-the-fly estimate of the probability
distribution along a set of collective variables, rather than accumulating
repulsive hills the way metadynamics does. It needs only three parameters:
the deposition pace, the initial kernel bandwidth, and the approximate height
of the barrier to overcome.

## Installation

OpenMM is distributed through conda-forge, so that is the supported route:

```bash
mamba install -c conda-forge openmm
pip install openmm-opes
```

## At a glance

```python
from openmm import app
from openmm_opes import OPES

sampler = OPES(
    system, [phi, psi], 300 * unit.kelvin, 40 * unit.kilojoules_per_mole,
    frequency=500, varianceFrequency=50,
)
sampler.step(simulation, 1_000_000)
fes = sampler.getFreeEnergy()
```

See the [quickstart](quickstart.md) for a complete runnable example, and the
[theory](theory.md) page for how the code maps onto the published equations.

## References

1. Invernizzi & Parrinello, *Rethinking Metadynamics: From Bias Potentials to
   Probability Distributions*, J. Phys. Chem. Lett. 2020.
   [doi:10.1021/acs.jpclett.0c00497](https://doi.org/10.1021/acs.jpclett.0c00497)
2. Invernizzi & Parrinello, *Exploration vs Convergence Speed in Adaptive-Bias
   Enhanced Sampling*, J. Chem. Theory Comput. 2022.
   [doi:10.1021/acs.jctc.2c00152](https://doi.org/10.1021/acs.jctc.2c00152)
