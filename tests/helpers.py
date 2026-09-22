"""Test helpers that build CV spaces without requiring OpenMM."""

from openmm_opes.kde import CVSpace


def makeSpace(*specs, bounded=False):
    """Build a CVSpace from (minValue, maxValue, gridWidth, periodic) tuples."""
    return CVSpace([CVSpace.CV(*spec) for spec in specs], bounded=bounded)
