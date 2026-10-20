"""
Accuracy of kernel compression: openmm-opes's mass-preserving merge versus
PLUMED's height-weighted merge, against the uncompressed estimate.

The same weighted samples, with the same bandwidths, go into three OnlineKDE
instances: one without compression (the reference), one with openmm-opes's
merge, which preserves total weight, mean and second moment, and one with
PLUMED's rule (see plumed_merge.py). Their log densities are compared on the
grid points where the reference is within 10 nats of its maximum.

Usage: python merge_accuracy.py

Prints, for each case, the kernel counts, the reference-weighted mean and the
maximum of |log P - log P_ref|, and the ratio of total probability mass.
"""

from __future__ import annotations

import numpy as np
from offline_parity import CASES, trajectory
from plumed_merge import plumed_merge

from openmm_opes import kde

MASS_PRESERVING = kde.Kernel.merge


def reference_estimate(space, samples, logWeights, variance):
    """The uncompressed estimate and the bandwidth factor of each sample."""
    reference = kde.OnlineKDE(space, compressionThreshold=0.0)
    factors = []
    for x, logWeight in zip(samples, logWeights, strict=True):
        factors.append(reference.bandwidthFactor(logWeight))
        reference.update(x, logWeight, variance, factors[-1])
    return reference, factors


def compressed_estimate(space, samples, logWeights, variance, factors, merge):
    """A compressed estimate built with the given merge rule."""
    kde.Kernel.merge = merge
    try:
        estimate = kde.OnlineKDE(space)
        for x, logWeight, factor in zip(samples, logWeights, factors, strict=True):
            estimate.update(x, logWeight, variance, factor)
    finally:
        kde.Kernel.merge = MASS_PRESERVING
    return estimate


def main():
    rng = np.random.default_rng(11)
    for name in ["1d-opes", "2d-opes"]:
        case = CASES[name]
        samples = trajectory(case)[::10]
        grids = [(-2.0, 2.0, 401, False), (-2.0, 2.0, 201, False)][: case.dims]
        space = kde.CVSpace([kde.CVSpace.CV(*grid) for grid in grids])
        variance = samples.var(axis=0) / 4
        for weighting in ["uniform", "opes-like"]:
            if weighting == "uniform":
                logWeights = np.zeros(len(samples))
            else:
                logWeights = rng.uniform(-8, 0, len(samples))
            reference, factors = reference_estimate(
                space, samples, logWeights, variance
            )
            logRef = reference.getLogPDF()
            mask = logRef > logRef.max() - 10
            p = np.exp(logRef[mask])
            p /= p.sum()
            for label, merge in [
                ("mass-preserving", MASS_PRESERVING),
                ("height-weighted", plumed_merge),
            ]:
                estimate = compressed_estimate(
                    space, samples, logWeights, variance, factors, merge
                )
                logEst = estimate.getLogPDF()[mask]
                diff = logEst - logRef[mask]
                mass = np.exp(
                    np.logaddexp.reduce(logEst) - np.logaddexp.reduce(logRef[mask])
                )
                print(
                    f"{name:8s} {weighting:9s} {label:16s}"
                    f" kernels {estimate.getNumKernels():4d}"
                    f" of {reference.getNumKernels()}"
                    f" | weighted mean |dlogP| {np.sum(p * np.abs(diff)):.4f}"
                    f" max {np.abs(diff).max():.3f}"
                    f" | mass ratio {mass:.4f}"
                )


if __name__ == "__main__":
    main()
