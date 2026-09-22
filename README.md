# openmm-opes

[![CI](https://github.com/craabreu/openmm-opes/actions/workflows/ci.yml/badge.svg)](https://github.com/craabreu/openmm-opes/actions/workflows/ci.yml)

An [OpenMM](https://openmm.org) implementation of On-the-fly Probability
Enhanced Sampling (OPES) and OPES-explore.

## Installation

```bash
mamba install -c conda-forge openmm
pip install openmm-opes
```

## Usage

```python
from openmm_opes import OPES

sampler = OPES(system, variables, temperature, barrier, frequency=500,
               varianceFrequency=50)
sampler.step(simulation, 1_000_000)
fes = sampler.getFreeEnergy()
```

Full documentation: <https://craabreu.github.io/openmm-opes>

## References

- Invernizzi & Parrinello, J. Phys. Chem. Lett. 2020,
  [doi:10.1021/acs.jpclett.0c00497](https://doi.org/10.1021/acs.jpclett.0c00497)
- Invernizzi & Parrinello, J. Chem. Theory Comput. 2022,
  [doi:10.1021/acs.jctc.2c00152](https://doi.org/10.1021/acs.jctc.2c00152)

## License

MIT
