"""
Convergence of the alanine dipeptide free energy surface from run.py outputs.

Usage: python analyze.py RUNS_DIR [--truth TRUTH.npz]

Files are named LABEL_SEED.npz, and runs sharing a LABEL form one set; OPES
files also carry smoothing-corrected surfaces, which form the set
LABEL-corrected. Each
set is judged against itself: its target is the surface obtained by averaging
the probabilities of the final surfaces of the other runs in the set, so
that no run is compared with an average that includes it. This measures how
each method converges to its own long-time answer; how the methods' answers
compare with one another is a separate question. For each set, prints over
time, as the mean +/- standard error over seeds:

* |dF - dF_target|, with dF = F(phi > 0) - F(phi < 0) the free energy of the
  alpha_L / C7ax region relative to the C7eq / C5 region;
* the RMS deviation from the target surface where it is within CUTOFF,
  5 kcal/mol, of its minimum, after removing the mean offset over that region.

With --truth, a ground-truth surface from reference.py (key "fes", on the
shared grid without its periodic copies), each run is also judged against it,
which measures accuracy rather than self-consistency.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

KB = 0.0083144626  # kJ/mol/K
KT = KB * 300.0  # kJ/mol, the unit of the saved surfaces
KCAL = 4.184  # kJ per kcal
CUTOFF = 5.0 * KCAL  # 5 kcal/mol
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
    """Runs by set label; corrected OPES surfaces form a set of their own."""
    sets = defaultdict(list)
    for path in sorted(Path(directory).glob("*_*.npz")):
        label = path.stem.rsplit("_", 1)[0]
        data = np.load(path)
        times, grid = data["times"], data["grid"][:-1]
        sets[label].append((times, interior(data["fes"]), grid))
        if "fes_corrected" in data:
            corrected = interior(data["fes_corrected"])
            sets[f"{label}-corrected"].append((times, corrected, grid))
    return sets


def report(label, runs, targets, target_df, phi, kind):
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
                mean = np.mean(values) / KCAL
                error = np.std(values, ddof=1) / np.sqrt(len(values)) / KCAL
                cells.append(f"{t:>3g} ns {mean:5.2f}+/-{error:4.2f}")
        print(f"{label:17s} {kind:5s} {name:13s} n={len(runs):2d}  " + "  ".join(cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--truth", type=Path)
    args = parser.parse_args()
    sets = load(args.runs_dir)
    truth = None if args.truth is None else np.load(args.truth)["fes"]
    print("mean +/- standard error over seeds, kcal/mol; LOO: against the final")
    print("average of the other runs in the set; truth: against the ground truth")
    for label, runs in sorted(sets.items()):
        phi = runs[0][2]
        finals = [fes[-1] for _, fes, _ in runs]
        targets = [combine(finals[:i] + finals[i + 1 :]) for i in range(len(runs))]
        target_df = [delta_f(t, phi) for t in targets]
        report(label, runs, targets, target_df, phi, "LOO")
        if truth is not None:
            report(
                label,
                runs,
                [truth] * len(runs),
                [delta_f(truth, phi)] * len(runs),
                phi,
                "truth",
            )
        full = combine(finals)
        print(
            f"{label:17s} final dF of the set average:"
            f" {delta_f(full, phi) / KCAL:.3f} kcal/mol"
        )


if __name__ == "__main__":
    main()
