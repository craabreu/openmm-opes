"""
Error maps for the alanine dipeptide benchmark README.

Usage: python differences.py RUNS_DIR TRUTH.npz SEED OUTFILE

Draws, at the end of the runs, the difference from the reference surface of
the surfaces reported by OPES, by OPES with the smoothing correction, and by
metadynamics: for one seed (top row) and for the pool of all runs, whose
probabilities are averaged (bottom row). Pooling averages out most of the
noise and leaves the systematic error. Differences are shown where the
reference is below 11 kcal/mol, which includes the transition region between
the basins, after removing the mean offset where it is below 5 kcal/mol. The
color scale stops at 0.4 kcal/mol; larger differences, which uncorrected OPES
reaches near the transition region, are clipped. OUTFILE gets .png and .svg
versions.
"""

import argparse
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
from analyze import CUTOFF, KCAL, combine, interior

PANELS = {
    "opes": ("OPES", "opes", "fes"),
    "opes-corrected": ("OPES, corrected", "opes", "fes_corrected"),
    "metad": ("MetaD", "metad", "fes"),
}
SHOWN = 11.0 * KCAL
LIMIT = 0.4  # kcal/mol


def final(path, field):
    data = np.load(path)
    return interior(data[field][-1]), data["grid"][:-1], data["times"][-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("seed", type=int)
    parser.add_argument("outfile", type=Path)
    args = parser.parse_args()
    truth = np.load(args.truth)["fes"]
    fit, shown = truth < CUTOFF, truth < SHOWN

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8})
    fig, axs = plt.subplots(
        2, 3, figsize=(7.0, 4.6), sharex=True, sharey=True, layout="constrained"
    )
    for col, (key, (name, method, field)) in enumerate(PANELS.items()):
        single, phi, end = final(args.runs_dir / f"{method}_{args.seed}.npz", field)
        paths = sorted(args.runs_dir.glob(f"{method}_*.npz"))
        pooled = combine([final(path, field)[0] for path in paths])
        rowNames = (f"Seed {args.seed}", f"{len(paths)} runs pooled")
        for row, surface in enumerate((single, pooled)):
            diff = surface - truth
            diff -= diff[fit].mean()
            rmsd = np.sqrt(np.mean(diff[fit] ** 2)) / KCAL
            ax = axs[row, col]
            ax.set_facecolor("0.85")
            mesh = ax.pcolormesh(
                phi,
                phi,
                np.where(shown, diff / KCAL, np.nan),
                cmap=cmc.vik,
                vmin=-LIMIT,
                vmax=LIMIT,
                shading="nearest",
                rasterized=True,
            )
            ax.contour(
                phi,
                phi,
                truth / KCAL,
                levels=np.arange(1, 12),
                colors="k",
                linewidths=0.3,
            )
            heading = f"{name}\n" if row == 0 else ""
            ax.set_title(f"{heading}RMSD {rmsd:.3f} kcal mol$^{{-1}}$")
            print(
                f"{key:15s} {rowNames[row]:15s} at {end:g} ns: RMSD {rmsd:.3f} kcal/mol"
            )
    fig.colorbar(
        mesh, ax=axs, extend="both", label=r"$F - F_\mathrm{ref}$ (kcal mol$^{-1}$)"
    )
    for ax in axs[1]:
        ax.set_xlabel(r"$\phi$ (rad)")
    for ax, rowName in zip(axs[:, 0], rowNames, strict=True):
        ax.set_ylabel(f"{rowName}\n" + r"$\psi$ (rad)")
    for ax, label in zip(axs.ravel(), "abcdef", strict=True):
        ax.text(
            -0.1, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=10
        )
    for suffix in (".png", ".svg"):
        fig.savefig(args.outfile.with_suffix(suffix), dpi=200)


if __name__ == "__main__":
    main()
