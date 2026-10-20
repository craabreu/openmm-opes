"""
Single-run surfaces for the alanine dipeptide benchmark README.

Usage: python surfaces.py RUNS_DIR SEED OUTFILE

Draws, for one seed at the end of its run, the surfaces reported by OPES,
by OPES with the smoothing correction, and by metadynamics. differences.py
draws their errors. OUTFILE gets .png and .svg versions.
"""

import argparse
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
from analyze import KCAL, interior

PANELS = {
    "opes": ("OPES", "opes", "fes"),
    "opes-corrected": ("OPES, corrected", "opes", "fes_corrected"),
    "metad": ("MetaD", "metad", "fes"),
}


def closed(values):
    """Append the periodic copies at phi = pi and psi = pi, for drawing."""
    return np.pad(values, ((0, 1), (0, 1)), mode="wrap")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("seed", type=int)
    parser.add_argument("outfile", type=Path)
    args = parser.parse_args()

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8})
    fig, axs = plt.subplots(
        1, 3, figsize=(7.0, 2.4), sharex=True, sharey=True, layout="constrained"
    )
    levels = np.arange(0, 12.5, 1)
    for ax, (name, method, field) in zip(axs, PANELS.values(), strict=True):
        data = np.load(args.runs_dir / f"{method}_{args.seed}.npz")
        phi = data["grid"][:-1]
        surface = closed(interior(data[field][-1]) / KCAL)
        axis = np.append(phi, np.pi)
        filled = ax.contourf(
            axis, axis, surface, levels=levels, cmap=cmc.devon, extend="max"
        )
        ax.contour(axis, axis, surface, levels=levels, colors="k", linewidths=0.3)
        ax.set(title=name, xlabel=r"$\phi$ (rad)")
    axs[0].set_ylabel(r"$\psi$ (rad)")
    fig.colorbar(filled, ax=axs, label=r"$F$ (kcal mol$^{-1}$)", ticks=range(0, 13, 2))
    for ax, label in zip(axs, "abc", strict=True):
        ax.text(
            -0.1, 1.03, label, transform=ax.transAxes, fontweight="bold", fontsize=10
        )
    for suffix in (".png", ".svg"):
        fig.savefig(args.outfile.with_suffix(suffix), dpi=200)


if __name__ == "__main__":
    main()
