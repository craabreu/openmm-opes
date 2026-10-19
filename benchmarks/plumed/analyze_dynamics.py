"""
Summarize dynamics.py runs: the error of the basin free energy difference over
time, and the number of basin-to-basin transitions.

Usage: python analyze_dynamics.py RUNS_DIR

Run files are named IMPLEMENTATION_MODE_SEED.dat. Each run is reweighted with
the sum of its biases, as is standard when several OPES biases act together.
For each implementation and mode, prints the mean |dF - exact| over seeds,
with its standard error, at several times, plus the mean signed error at the
last time and the mean number of transitions. A transition is counted each
time x enters one basin's core, |x| > 0.7, after having been in the other's.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from common import KB
from dynamics import TEMPERATURE, exact_delta_f

CHECKPOINTS_NS = [0.5, 1, 2, 5, 10, 20]


def delta_f(x, logw):
    """F(x > 0) - F(x < 0) from reweighted samples, in kJ/mol."""
    kT = KB * TEMPERATURE
    right, left = logw[x > 0], logw[x < 0]
    if len(right) == 0 or len(left) == 0:
        return np.nan
    return -kT * (np.logaddexp.reduce(right) - np.logaddexp.reduce(left))


def transitions(x, core=0.7):
    state, count = 0, 0
    for value in x:
        new = 1 if value > core else -1 if value < -core else 0
        if new and new != state:
            count += state != 0
            state = new
    return count


def mean_and_error(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    return values.mean(), values.std(ddof=1) / np.sqrt(len(values))


def main():
    runs = Path(sys.argv[1])
    exact = exact_delta_f()
    kT = KB * TEMPERATURE
    results = defaultdict(list)
    for path in sorted(runs.glob("*_*_*.dat")):
        implementation, mode, _ = path.stem.rsplit("_", 2)
        data = np.loadtxt(path, ndmin=2)
        if len(data) < 10:
            continue
        time_ns = data[:, 0] / 1000
        x = data[:, 1]
        logw = data[:, 2:].sum(axis=1) / kT
        errors = [
            delta_f(x[time_ns <= t], logw[time_ns <= t]) - exact
            if np.sum(time_ns <= t) > 10
            else np.nan
            for t in CHECKPOINTS_NS
        ]
        results[(mode, implementation)].append((errors, transitions(x)))
    print(f"exact dF = {exact:.3f} kJ/mol")
    print("mean |dF - exact| +/- standard error over seeds, in kJ/mol")
    print(
        f"{'mode':8s} {'implementation':24s} {'seeds':>5s}  "
        + "  ".join(f"{f'{t:g} ns':>11s}" for t in CHECKPOINTS_NS)
        + f"  {'signed, last':>12s}  {'transitions':>11s}"
    )
    for (mode, implementation), entries in sorted(results.items()):
        errors = np.array([entry[0] for entry in entries])
        cells = []
        for column in errors.T:
            mean, error = mean_and_error(np.abs(column))
            cells.append(f"{mean:5.2f}+/-{error:4.2f}")
        signed, signed_error = mean_and_error(errors[:, -1])
        trans = np.mean([entry[1] for entry in entries])
        print(
            f"{mode:8s} {implementation:24s} {len(entries):5d}  "
            + "  ".join(cells)
            + f"  {signed:+5.2f}+/-{signed_error:4.2f}  {trans:11.1f}"
        )


if __name__ == "__main__":
    main()
