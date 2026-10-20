"""
Reference and single-run surfaces for the main README.

Usage: python overview.py RUNS_DIR TRUTH.npz SEED TIME OUTFILE [--dark]

Draws the reference surface next to the surface one OPES run reports after
TIME ns, before and after the smoothing correction, with each surface's RMSD
from the reference where it is below 5 kcal/mol. With --dark, the figure has
light text on a transparent background, for dark mode. OUTFILE gets .png and
.svg versions.
"""

import argparse
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
from analyze import CUTOFF, KCAL, interior

DARK_TEXT = "#E6EDF3"  # GitHub's dark-mode foreground


def closed(values):
    """Append the periodic copies at phi = pi and psi = pi, for drawing."""
    return np.pad(values, ((0, 1), (0, 1)), mode="wrap")


def rmsd(surface, truth):
    mask = truth < CUTOFF
    diff = (surface - truth)[mask]
    return np.sqrt(np.mean((diff - diff.mean()) ** 2)) / KCAL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("seed", type=int)
    parser.add_argument("time", type=float, help="ns")
    parser.add_argument("outfile", type=Path)
    parser.add_argument("--dark", action="store_true")
    args = parser.parse_args()
    truth = np.load(args.truth)["fes"]
    data = np.load(args.runs_dir / f"opes_{args.seed}.npz")
    index = int(np.argmin(np.abs(data["times"] - args.time)))
    time = data["times"][index]
    raw = interior(data["fes"][index])
    corrected = interior(data["fes_corrected"][index])
    phi = data["grid"][:-1]

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8})
    if args.dark:
        plt.rcParams.update(
            dict.fromkeys(
                (
                    "text.color",
                    "axes.labelcolor",
                    "axes.edgecolor",
                    "xtick.color",
                    "ytick.color",
                ),
                DARK_TEXT,
            )
        )
    fig, axs = plt.subplots(
        1, 3, figsize=(7.2, 2.5), sharex=True, sharey=True, layout="constrained"
    )
    panels = (
        (truth, "Reference\n48 runs × 200 ns"),
        (raw, f"OPES, one {time:g} ns run\nRMSD {rmsd(raw, truth):.2f} kcal/mol"),
        (
            corrected,
            "Same run, smoothing-corrected\n"
            f"RMSD {rmsd(corrected, truth):.2f} kcal/mol",
        ),
    )
    axis = np.append(phi, np.pi)
    levels = np.arange(0, 12.5, 1)
    for ax, (surface, title) in zip(axs, panels, strict=True):
        shifted = closed((surface - surface.min()) / KCAL)
        filled = ax.contourf(
            axis, axis, shifted, levels=levels, cmap=cmc.devon, extend="max"
        )
        ax.contour(axis, axis, shifted, levels=levels, colors="k", linewidths=0.3)
        ax.set(title=title, xlabel=r"$\phi$ (rad)")
    axs[0].set_ylabel(r"$\psi$ (rad)")
    fig.colorbar(filled, ax=axs, label=r"$F$ (kcal mol$^{-1}$)", ticks=range(0, 13, 2))
    print(
        f"seed {args.seed} at {time:g} ns: RMSD raw {rmsd(raw, truth):.3f}, "
        f"corrected {rmsd(corrected, truth):.3f} kcal/mol"
    )
    for suffix in (".png", ".svg"):
        fig.savefig(args.outfile.with_suffix(suffix), dpi=200, transparent=args.dark)


if __name__ == "__main__":
    main()
