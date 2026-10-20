# openmm-opes

On-the-fly Probability Enhanced Sampling (OPES) for [OpenMM](https://openmm.org).

OPES is an enhanced sampling method that builds its bias from an on-the-fly
estimate of the probability distribution along a few collective variables,
instead of accumulating repulsive hills as metadynamics does. The bias soon
becomes nearly static, so free energies converge quickly and reweighting is
simple. It needs only an expected barrier height, a deposition stride and a
kernel bandwidth. This package implements OPES in pure Python on top of
OpenMM, in the spirit of `openmm.app.Metadynamics`, and is validated against
PLUMED's `OPES_METAD` and `OPES_METAD_EXPLORE`.

## Features

- **OPES and OPES-explore,** with fixed, adaptive or warm-up bandwidths and
  on-the-fly kernel compression.
- **OPES and OPES-explore combined** (OPES_COM), as a single `CombinedOPES`
  sampler.
- **Multiple walkers** sharing one bias through a directory.
- **Several samplers on one simulation,** with extra biases folded into the
  reweighting.
- **Reweighting tools:** the reweighting constant c(t), and PLUMED's `rct` as a
  convergence check.
- **Smoothing-corrected free energy surfaces.**
- **Periodic, unbounded or reflective collective variables,** one to three of
  them.
- **Checkpoints** through `getState` and `setState`.

## Installation

```bash
pip install openmm-opes
```

It needs Python 3.11 or later and OpenMM 8.1 or later. OpenMM can also come
from conda-forge:

```bash
conda install -c conda-forge openmm
pip install openmm-opes
```

## At a glance

```python
from openmm import unit
from openmm_opes import OPES

sampler = OPES(
    system,
    [phi, psi],  # openmm.app.BiasVariable objects
    temperature=300 * unit.kelvin,
    barrier=40 * unit.kilojoules_per_mole,
    frequency=500,
    varianceFrequency=50,
)
sampler.step(simulation, 1_000_000)
fes = sampler.getFreeEnergy(correctSmoothing=True)
```

The [quickstart](quickstart.md) has a complete runnable example and the main
options. The [theory](theory.md) page shows how the code maps onto the
published equations and where it departs from them. The
[benchmarks](https://github.com/craabreu/openmm-opes/tree/main/benchmarks)
compare openmm-opes with PLUMED's OPES and with OpenMM's metadynamics, against
a reference surface for alanine dipeptide.

## Origin

This library packages and hardens the prototype implementation developed in
[craabreu/opes-simulations](https://github.com/craabreu/opes-simulations),
porting its `OPES` and `OnlineKDE` classes into a tested, documented package.

## References

1. M. Invernizzi and M. Parrinello, *Rethinking Metadynamics: From Bias
   Potentials to Probability Distributions*, J. Phys. Chem. Lett. 11, 2731
   (2020). [doi:10.1021/acs.jpclett.0c00497](https://doi.org/10.1021/acs.jpclett.0c00497)
2. M. Invernizzi and M. Parrinello, *Exploration vs Convergence Speed in
   Adaptive-Bias Enhanced Sampling*, J. Chem. Theory Comput. 18, 3988 (2022).
   [doi:10.1021/acs.jctc.2c00152](https://doi.org/10.1021/acs.jctc.2c00152)
3. D. Ray and V. Rizzi, *Enhanced Sampling with Suboptimal Collective
   Variables: Reconciling Accuracy and Convergence Speed*, J. Chem. Theory
   Comput. 21, 58 (2025).
   [doi:10.1021/acs.jctc.4c01231](https://doi.org/10.1021/acs.jctc.4c01231)
4. M. C. Jones, O. Linton and J. P. Nielsen, *A Simple Bias Reduction Method
   for Density Estimation*, Biometrika 82, 327 (1995).
