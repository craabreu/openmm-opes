# Alanine dipeptide: OPES and metadynamics against a reference surface

This benchmark compares how fast openmm-opes and OpenMM's well-tempered
metadynamics converge the free energy surface of alanine dipeptide in vacuum,
against an accurate reference surface computed here.

## Reference free energy surface of alanine dipeptide (Amber ff14SB, vacuum, 300 K)

`alanine_dipeptide_fes.dat` is a reference Ramachandran free energy surface,
F(φ, ψ), of alanine dipeptide (Ace-Ala-Nme) in vacuum with the Amber ff14SB
force field at 300 K. It is a plain-text table on a 100 × 100 grid of the
backbone dihedrals φ and ψ, in radians, with the free energy and its standard
error in kcal/mol. Use it as ground truth when testing enhanced sampling
methods such as OPES, metadynamics, umbrella sampling or adaptive biasing
force, or as a reference potential of mean force for the alanine dipeptide
Ramachandran plot. We found no published reference for this force field and
these conditions.

| Property | Value |
|---|---|
| Force field | Amber ff14SB, OpenMM `amber14-all.xml` |
| Environment | Vacuum, no cutoff |
| Temperature | 300 K, Langevin middle integrator, friction 1/ps |
| Time step | 4 fs, bonds to hydrogen constrained, hydrogen mass 4 amu |
| Collective variables | φ = C–N–Cα–C and ψ = N–Cα–C–N |
| Sampling | 48 runs × 200 ns under a fixed bias, 9.6 million samples |
| Grid | 100 × 100, from −π in steps of 2π/100 |
| ΔF, φ > 0 relative to φ < 0 | 2.0734 ± 0.0018 kcal/mol |
| Median standard error where F < 5 kcal/mol | 0.013 kcal/mol |

**How it was computed.** The bias is the converged well-tempered
metadynamics bias, averaged over four 500 ns runs and then frozen. Under a
fixed bias, reweighting each sample by e^{βV} is exact, and the bias makes
both basins and the barriers between them easy to visit. The density is
estimated with product von Mises kernels at two widths and extrapolated to
zero width, which removes the kernels' smoothing bias to leading order. A
three-point extrapolation differs from it by only 0.002 kcal/mol RMS. The
standard errors come from a jackknife over the 48 runs. The ΔF between basins
uses the weighted samples directly, with no kernel. The file's header repeats
all of this.

`reference.py` reproduces the surface:

```bash
python reference.py run METAD_DIR SEED static/run_SEED.npy --ns 200   # once per seed
python reference.py kde static reference.npz
python reference.py table reference.npz alanine_dipeptide_fes.dat
```

METAD_DIR holds long metadynamics runs from `run.py metad`, whose averaged
final bias becomes the fixed bias.

## Running the benchmark

```bash
python run.py opes SEED runs/opes_SEED.npz      # 16 seeds
python run.py metad SEED runs/metad_SEED.npz    # 16 seeds
python analyze.py runs --truth reference.npz
python figure.py runs reference.npz figure
```

Settings follow the alanine dipeptide inputs of Invernizzi and Parrinello,
J. Phys. Chem. Lett. 11, 2731 (2020), PLUMED-NEST plumID:19.068, with
hydrogen mass repartitioning for a 4 fs step. Each 50 ns run takes about
10 minutes on one CPU core with OpenMM's Reference platform.
