"""
Single-run surfaces for the alanine dipeptide benchmark README.

Usage: python surfaces.py RUNS_DIR TRUTH.npz SEED OUTFILE

Draws, for one seed at the end of its run, the OPES surface before and after
the smoothing correction and the metadynamics surface (top row), and their
differences from the reference surface (bottom row). Differences are shown
where the reference is below 8 kcal/mol, after removing the mean offset where
it is below 5 kcal/mol. OUTFILE gets .png and .svg versions.
"""

import argparse
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
from analyze import CUTOFF, KCAL, interior

PANELS = {
    "opes": "OPES",
    "opes-corrected": "OPES, corrected",
    "metad": "MetaD",
}
FILES = {
    "opes": ("opes", "fes"),
    "opes-corrected": ("opes", "fes_corrected"),
    "metad": ("metad", "fes"),
}
SHOWN = 8.0 * KCAL


def closed(values):
    """Append the periodic copies at phi = pi and psi = pi, for drawing."""
    return np.pad(values, ((0, 1), (0, 1)), mode="wrap")


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
    levels = np.arange(0, 12.5, 1)
    for col, (key, name) in enumerate(PANELS.items()):
        method, field = FILES[key]
        data = np.load(args.runs_dir / f"{method}_{args.seed}.npz")
        times, phi = data["times"], data["grid"][:-1]
        surface = interior(data[field][-1])
        axis = np.append(phi, np.pi)
        top = axs[0, col]
        filled = top.contourf(
            axis,
            axis,
            closed(surface / KCAL),
            levels=levels,
            cmap=cmc.devon,
            extend="max",
        )
        top.contour(
            axis,
            axis,
            closed(surface / KCAL),
            levels=levels,
            colors="k",
            linewidths=0.3,
        )
        top.set_title(name)
        diff = surface - truth
        diff -= diff[fit].mean()
        rmsd = np.sqrt(np.mean(diff[fit] ** 2)) / KCAL
        bottom = axs[1, col]
        bottom.set_facecolor("0.85")
        mesh = bottom.pcolormesh(
            phi,
            phi,
            np.where(shown, diff / KCAL, np.nan),
            cmap=cmc.vik,
            vmin=-0.4,
            vmax=0.4,
            shading="nearest",
            rasterized=True,
        )
        bottom.contour(
            phi,
            phi,
            truth / KCAL,
            levels=np.arange(1, 9),
            colors="k",
            linewidths=0.3,
        )
        bottom.set_title(f"RMSD {rmsd:.3f} kcal mol$^{{-1}}$")
        print(
            f"{key:15s} seed {args.seed} at {times[-1]:g} ns: RMSD {rmsd:.3f} kcal/mol"
        )
    fig.colorbar(
        filled, ax=axs[0], label=r"$F$ (kcal mol$^{-1}$)", ticks=range(0, 13, 2)
    )
    fig.colorbar(
        mesh, ax=axs[1], extend="both", label=r"$F - F_\mathrm{ref}$ (kcal mol$^{-1}$)"
    )
    for ax in axs[1]:
        ax.set_xlabel(r"$\phi$ (rad)")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\psi$ (rad)")
    for ax, label in zip(axs.ravel(), "abcdef", strict=True):
        ax.text(
            -0.1, 1.03, label, transform=ax.transAxes, fontweight="bold", fontsize=10
        )
    for suffix in (".png", ".svg"):
        fig.savefig(args.outfile.with_suffix(suffix), dpi=200)


if __name__ == "__main__":
    main()
