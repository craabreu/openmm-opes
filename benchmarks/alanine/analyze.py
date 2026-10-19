"""
Convergence of the alanine dipeptide free energy surface from run.py outputs.

Usage: python analyze.py RUNS_DIR

Files are named LABEL_SEED.npz, and runs sharing a LABEL form one set. Each
set is judged against itself: its target is the surface obtained by averaging
the probabilities of the final surfaces of the other runs in the set, so
that no run is compared with an average that includes it. This measures how
each method converges to its own long-time answer; how the methods' answers
compare with one another is a separate question. For each set, prints over
time, as the mean +/- standard error over seeds:

* |dF - dF_target|, with dF = F(phi > 0) - F(phi < 0) the free energy of the
  alpha_L / C7ax region relative to the C7eq / C5 region;
* the RMS deviation from the target surface where it is within CUTOFF of its
  minimum, after removing the mean offset over that region.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

KB = 0.0083144626  # kJ/mol/K
KT = KB * 300.0
CUTOFF = 20.0  # kJ/mol
TIMES = [1, 2, 5, 10, 20, 50]


def interior(fes):
    """Drop the last row and column, which repeat the first at phi, psi = pi."""
    return fes[..., :-1, :-1]


def combine(surfaces):
    """The surface of the averaged probabilities, shifted to a zero minimum."""
    fes = -KT * np.log(np.mean([np.exp(-f / KT) for f in surfaces], axis=0))
    return fes - fes.min()


def delta_f(fes, phi):
    """F(phi > 0) - F(phi < 0); fes has psi along rows and phi along columns."""
    weights = np.exp(-(fes - fes.min()) / KT)
    return -KT * np.log(weights[:, phi > 0].sum() / weights[:, phi < 0].sum())


def rms_deviation(fes, target, mask):
    diff = (fes - target)[mask]
    return np.sqrt(np.mean((diff - diff.mean()) ** 2))


METRICS = ("|dF - target|", "FES rmsd")


def metric(name, fes, target, target_df, phi):
    """One of METRICS for a surface, given its target surface and dF."""
    if name == "|dF - target|":
        return abs(delta_f(fes, phi) - target_df)
    return rms_deviation(fes, target, target < CUTOFF)


def load(directory):
    sets = defaultdict(list)
    for path in sorted(Path(directory).glob("*_*.npz")):
        label = path.stem.rsplit("_", 1)[0]
        data = np.load(path)
        sets[label].append((data["times"], interior(data["fes"]), data["grid"][:-1]))
    return sets


def main():
    sets = load(sys.argv[1])
    print("mean +/- standard error over seeds, kJ/mol, each run against the")
    print("final average of the other runs in its set")
    for label, runs in sorted(sets.items()):
        phi = runs[0][2]
        finals = [fes[-1] for _, fes, _ in runs]
        targets = [combine(finals[:i] + finals[i + 1 :]) for i in range(len(runs))]
        target_df = [delta_f(t, phi) for t in targets]
        for name in METRICS:
            cells = []
            for t in TIMES:
                values = []
                for i, (times, fes, _) in enumerate(runs):
                    index = np.searchsorted(times, t - 1e-9)
                    if index < len(times):
                        values.append(
                            metric(name, fes[index], targets[i], target_df[i], phi)
                        )
                if len(values) > 1:
                    mean = np.mean(values)
                    error = np.std(values, ddof=1) / np.sqrt(len(values))
                    cells.append(f"{t:>3g} ns {mean:5.2f}+/-{error:4.2f}")
            print(f"{label:17s} {name:13s} n={len(runs):2d}  " + "  ".join(cells))
        full = combine(finals)
        print(
            f"{label:17s} final dF of the set average: {delta_f(full, phi):.3f} kJ/mol"
        )


if __name__ == "__main__":
    main()
