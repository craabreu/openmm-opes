# Benchmarks against PLUMED's OPES

These scripts compare openmm-opes with the reference implementation of OPES in
[PLUMED](https://www.plumed.org) (`OPES_METAD` and `OPES_METAD_EXPLORE`).

| Script | Question |
|---|---|
| `offline_parity.py` | Given the same CV samples, do the two codes build the same bias? |
| `dynamics.py`, `analyze_dynamics.py` | Do they converge alike in a simulation with a known free energy? |
| `merge_accuracy.py` | Which kernel merge rule better preserves the density estimate? |

`plumed_merge.py` holds PLUMED's merge rule, so that openmm-opes can be run
with it. `common.py` holds shared constants.

## Setup

**PLUMED with the opes module.** The conda-forge build of PLUMED leaves the
opes module out, so build PLUMED yourself:

```bash
curl -L -o plumed.tgz https://github.com/plumed/plumed2/archive/refs/tags/v2.10.1.tar.gz
tar xzf plumed.tgz && cd plumed2-2.10.1
./configure --prefix=$HOME/plumed-opes --enable-modules=all --disable-python --disable-doc
make -j 16 && make install
export PLUMED=$HOME/plumed-opes
```

**Python environment.** OpenMM, openmm-plumed and this package:

```bash
conda create -n opes-bench -c conda-forge python=3.13 openmm openmm-plumed numpy scipy
conda activate opes-bench
pip install -e /path/to/openmm-opes
```

**Using your PLUMED from openmm-plumed.** The conda-forge openmm-plumed plugin
links the conda PLUMED library through a fixed relative path, which neither
`PLUMED_KERNEL` nor `LD_LIBRARY_PATH` overrides. Preloading your build does:

```bash
export LD_PRELOAD="$PLUMED/lib/libplumedKernel.so $PLUMED/lib/libplumed.so"
```

Set this only for PLUMED runs of `dynamics.py`. `offline_parity.py` calls the
`plumed` executable directly and needs only `LD_LIBRARY_PATH=$PLUMED/lib`.

Run every script from this directory.

## Offline parity

```bash
LD_LIBRARY_PATH=$PLUMED/lib python offline_parity.py $PLUMED/bin/plumed parity-runs
```

The script generates a CV trajectory from overdamped dynamics on a double well
at 900 K. PLUMED reads it with `plumed driver`. openmm-opes replays it through
`OPES.stepTogether` with a stand-in for a Simulation, with the same order of
bias evaluation and deposition. Both use adaptive bandwidths unless a case
fixes them, and PLUMED's default kernel cutoff. All six cases take a few
minutes.

## Convergence in dynamics

```bash
for impl in plumed openmm-opes; do
  for mode in opes explore com; do
    for seed in $(seq 1 24); do
      python dynamics.py $impl $mode $seed runs/${impl}_${mode}_${seed}.dat --ns 20
    done
  done
done
python analyze_dynamics.py runs
```

Remember the `LD_PRELOAD` for the PLUMED runs. Each run takes 10–20 minutes on
one CPU core, so run them in parallel, for example with `xargs -P`. `com` runs
OPES and OPES-explore together. PLUMED's log, about 1 GB per run, is
discarded.

## Merge accuracy

```bash
python merge_accuracy.py
```

## Results

These results come from PLUMED 2.10.1 and openmm-opes at 97fe927, which
includes #17–#19, with the scripts here as they are.

**Offline parity.** Without `--plumed-merge`, the biases differ by 0.7–1.2
kJ/mol RMS on the grid. The exception is 2D OPES-explore, at 8.9 kJ/mol,
where explore's large prefactor amplifies small density differences. With
`--plumed-merge`, openmm-opes reproduces PLUMED in all six cases: the same
kernel counts, and biases within 0.01 kJ/mol RMS on the grid and along the
trajectory. The merge rule is therefore the only algorithmic difference.

**Dynamics.** These are 24 seeds per case, 20 ns each, with exact
ΔF = 5.90 kJ/mol. The table shows the mean |ΔF − exact| ± standard error, in
kJ/mol.

| mode | time | PLUMED | openmm-opes | openmm-opes, PLUMED merge |
|---|---|---|---|---|
| OPES | 0.5 ns | 4.41 ± 0.50 | 4.49 ± 0.81 | 3.28 ± 0.53 |
| OPES | 20 ns | 0.48 ± 0.08 | 0.46 ± 0.08 | 0.44 ± 0.07 |
| OPES-explore | 0.5 ns | 5.90 ± 1.05 | 5.39 ± 0.95 | 6.00 ± 1.09 |
| OPES-explore | 20 ns | 1.14 ± 0.42 | 1.12 ± 0.36 | 1.18 ± 0.41 |
| OPES + explore | 0.5 ns | 6.56 ± 1.07 | 5.75 ± 0.67 | 6.00 ± 0.76 |
| OPES + explore | 20 ns | 0.98 ± 0.24 | 0.59 ± 0.13 | 0.76 ± 0.13 |

No difference between implementations exceeds 1.5 combined standard errors,
and all three make about the same number of transitions between basins.

**Merge accuracy.** Against the uncompressed estimate, openmm-opes's
mass-preserving merge has a mean absolute log-density error of 0.03–0.05.
PLUMED's height-weighted merge has 0.17–0.25 and inflates the total mass by
17–27%. Both keep about the same number of kernels.
