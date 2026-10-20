"""
README figure for the alanine dipeptide benchmark.

Usage: python figure.py RUNS_DIR TRUTH.npz OUTFILE

RUNS_DIR holds the run.py outputs, opes_SEED.npz and metad_SEED.npz, and
TRUTH.npz the reference surface from reference.py. Draws the reference
surface and, against it and over time, the basin free energy error, the
surface error of single runs, and the surface error of the average of all
runs. OUTFILE gets .png and .svg versions.
"""

import argparse
from pathlib import Path

import cmcrameri.cm as cmc
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
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6), layout="constrained")
    ax = axs[0, 0]
    phi = sets["metad"][0][2]
    closed = np.append(phi, np.pi)
    surface = np.pad(truth / KCAL, ((0, 1), (0, 1)), mode="wrap")
    levels = np.arange(0, 12.5, 1)
    filled = ax.contourf(
        closed, closed, surface, levels=levels, cmap=cmc.devon, extend="max"
    )
    ax.contour(closed, closed, surface, levels=levels, colors="k", linewidths=0.3)
    fig.colorbar(filled, ax=ax, label=r"$F$ (kcal mol$^{-1}$)", ticks=range(0, 13, 2))
    ax.set(xlabel=r"$\phi$ (rad)", ylabel=r"$\psi$ (rad)", title="Reference surface")
    for name, (x, y) in {"C7eq": (-1.5, 1.2), "C7ax": (1.2, -1.0)}.items():
        ax.annotate(name, (x, y), color="w", ha="center", fontsize="small")

    for key, (name, style, color) in SHOW.items():
        runs = sets[key]
        times = runs[0][0]
        perRun, average = curves(runs, truth, phi)
        for k, ax in enumerate((axs[0, 1], axs[1, 0])):
            mean = perRun[:, :, k].mean(0)
            error = perRun[:, :, k].std(0, ddof=1) / np.sqrt(len(runs))
            ax.plot(times, mean, style, color=color, label=f"{name}, {len(runs)} runs")
            ax.fill_between(
                times, mean - error, mean + error, color=color, alpha=0.25, lw=0
            )
        axs[1, 1].plot(times, average, style, color=color)
        print(
            f"{key:15s} at {times[-1]:g} ns: dF error {perRun[:, -1, 0].mean():.3f}, "
            f"single-run RMSD {perRun[:, -1, 1].mean():.3f}, "
            f"RMSD of average {average[-1]:.3f} kcal/mol"
        )
    axs[0, 1].set(
        ylabel=r"$|\Delta F - \Delta F_\mathrm{ref}|$ (kcal mol$^{-1}$)",
        title=r"Basin $\Delta F$ ($\phi>0$ vs $\phi<0$)",
    )
    axs[1, 0].set(
        ylabel=r"RMSD (kcal mol$^{-1}$)",
        title=r"Single-run surface, $F_\mathrm{ref} < 5$ kcal mol$^{-1}$",
    )
    axs[1, 1].set(
        ylabel=r"RMSD (kcal mol$^{-1}$)",
        title=r"Surface averaged over runs, $F_\mathrm{ref} < 5$ kcal mol$^{-1}$",
    )
    for ax in (axs[0, 1], axs[1, 0], axs[1, 1]):
        ax.set(xlabel="Simulation time (ns)", xlim=(0, None))
        ax.set_ylim(bottom=0)
    axs[0, 1].legend(fontsize="small")
    for ax, label in zip(axs.ravel(), "abcd", strict=True):
        ax.set_title(label, loc="left", fontweight="bold")
    for suffix in (".png", ".svg"):
        fig.savefig(args.outfile.with_suffix(suffix), dpi=200)


if __name__ == "__main__":
    main()
