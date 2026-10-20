"""
README figure for the alanine dipeptide benchmark.

Usage: python figure.py RUNS_DIR TRUTH.npz OUTFILE

RUNS_DIR holds the run.py outputs, opes_SEED.npz and metad_SEED.npz, and
TRUTH.npz the reference surface from reference.py. Draws, against the
reference and over time, the basin free energy error, the surface error of
single runs, and the surface error of the average of all runs. Surface errors
are RMS deviations where the reference is below 5 kcal/mol, after removing
the mean offset. OUTFILE gets .png and .svg versions.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze import CUTOFF, KCAL, combine, delta_f, load, rms_deviation

SHOW = {
    "opes": ("OPES", "-", "#0072B2"),
    "opes-corrected": ("OPES, smoothing-corrected", "-.", "#009E73"),
    "metad": ("Well-tempered MetaD", "--", "#D55E00"),
}


def curves(runs, truth, phi):
    """Per-run basin dF error and surface RMSD, and the RMSD of the average."""
    mask, target = truth < CUTOFF, delta_f(truth, phi)
    perRun = np.array(
        [
            [
                (abs(delta_f(f, phi) - target), rms_deviation(f, truth, mask))
                for f in fes
            ]
            for _, fes, _ in runs
        ]
    )
    average = np.array(
        [
            rms_deviation(combine([fes[i] for _, fes, _ in runs]), truth, mask)
            for i in range(len(runs[0][0]))
        ]
    )
    return perRun / KCAL, average / KCAL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("outfile", type=Path)
    args = parser.parse_args()
    sets = load(args.runs_dir)
    truth = np.load(args.truth)["fes"]

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "lines.linewidth": 1.4})
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), layout="constrained")
    phi = sets["metad"][0][2]

    for key, (name, style, color) in SHOW.items():
        runs = sets[key]
        times = runs[0][0]
        perRun, average = curves(runs, truth, phi)
        for k, ax in enumerate(axs[:2]):
            mean = perRun[:, :, k].mean(0)
            error = perRun[:, :, k].std(0, ddof=1) / np.sqrt(len(runs))
            ax.plot(times, mean, style, color=color, label=f"{name}, {len(runs)} runs")
            ax.fill_between(
                times, mean - error, mean + error, color=color, alpha=0.25, lw=0
            )
        axs[2].plot(times, average, style, color=color)
        print(
            f"{key:15s} at {times[-1]:g} ns: dF error {perRun[:, -1, 0].mean():.3f}, "
            f"single-run RMSD {perRun[:, -1, 1].mean():.3f}, "
            f"RMSD of average {average[-1]:.3f} kcal/mol"
        )
    axs[0].set(
        ylabel=r"$|\Delta F - \Delta F_\mathrm{ref}|$ (kcal mol$^{-1}$)",
        title=r"Basin $\Delta F$",
    )
    axs[1].set(ylabel=r"RMSD (kcal mol$^{-1}$)", title="Single-run surface")
    axs[2].set(ylabel=r"RMSD (kcal mol$^{-1}$)", title="Average surface")
    for ax in axs:
        ax.set(xlabel="Simulation time (ns)", xlim=(0, times[-1]))
        ax.set_ylim(bottom=0)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncols=3)
    for ax, label in zip(axs, "abc", strict=True):
        ax.text(
            -0.16, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=10
        )
    for suffix in (".png", ".svg"):
        fig.savefig(args.outfile.with_suffix(suffix), dpi=200)


if __name__ == "__main__":
    main()
