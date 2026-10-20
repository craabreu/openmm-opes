"""OpenMM implementation of On-the-fly Probability Enhanced Sampling (OPES)."""

__version__ = "0.2.0"

__all__ = ["OPES", "CombinedOPES", "__version__"]


def __getattr__(name):
    """Import the samplers lazily (PEP 562).

    An eager `from .opes import OPES` would also run on
    `import openmm_opes.kde`, dragging openmm in and breaking the guarantee
    that kde and io stay OpenMM-free.
    """
    if name == "OPES":
        from .opes import OPES

        return OPES
    if name == "CombinedOPES":
        from .combined import CombinedOPES

        return CombinedOPES
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
