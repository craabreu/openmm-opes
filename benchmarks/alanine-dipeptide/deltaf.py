"""
Basin free energy difference over time, for the alanine dipeptide README.

Usage: python deltaf.py RUNS_DIR TRUTH.npz SEED OUTFILE

Plots dF = F(phi > 0) - F(phi < 0) against simulation time, for one seed
(left) and for every seed in RUNS_DIR (right), with the reference value from
TRUTH.npz and a band of +/- 0.1 kcal/mol around it. OUTFILE gets .png and
.svg versions.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze import KCAL, delta_f, interior

SHOW = {
    "opes": ("OPES", "opes", "fes", "-", "#0072B2"),
    "opes-corrected": (
        "OPES, smoothing-corrected",
        "opes",
        "fes_corrected",
        "-.",
        "#009E73",
    ),
    "metad": ("Well-tempered MetaD", "metad", "fes", "--", "#D55E00"),
}
BAND = 0.1  # kcal/mol


def series(path, field):
    data = np.load(path)
    phi = data["grid"][:-1]
    return data["times"], np.array(
        [delta_f(f, phi) for f in interior(data[field])]
    ) / KCAL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("seed", type=int)
    parser.add_argument("outfile", type=Path)
    args = parser.parse_args()
    reference = float(np.load(args.truth)["delta_f"]) / KCAL

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "lines.linewidth": 1.4})
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True, layout="constrained")
    for key, (name, method, field, style, color) in SHOW.items():
        times, values = series(args.runs_dir / f"{method}_{args.seed}.npz", field)
        axs[0].plot(times, values, style, color=color, label=name)
        if key == "opes-corrected":
            continue  # same basin weights as OPES by construction
        paths = sorted(args.runs_dir.glob(f"{method}_*.npz"))
        for k, path in enumerate(paths):
            times, values = series(path, field)
            axs[1].plot(
                times,
                values,
                "-",
                color=color,
                lw=0.5,
                alpha=0.6,
                label=f"{name}, {len(paths)} runs" if k == 0 else None,
            )
    for ax in axs:
        ax.axhline(reference, color="k", lw=0.8)
        ax.axhspan(reference - BAND, reference + BAND, color="0.85", lw=0, zorder=0)
        ax.set(xlabel="Simulation time (ns)", xlim=(0, None))
        ax.legend(fontsize="small")
    axs[0].set(
        ylabel=r"$\Delta F$, $\phi>0$ vs $\phi<0$ (kcal mol$^{-1}$)",
        title=f"Seed {args.seed}",
    )
    axs[1].set(title="All runs")
    for suffix in (".png", ".svg"):
        fig.savefig(args.outfile.with_suffix(suffix), dpi=200)


if __name__ == "__main__":
    main()
