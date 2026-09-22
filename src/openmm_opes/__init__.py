"""OpenMM implementation of On-the-fly Probability Enhanced Sampling (OPES)."""

__version__ = "0.1.0"

__all__ = ["OPES", "__version__"]


def __getattr__(name):
    """Import OPES lazily (PEP 562).

    An eager ``from .opes import OPES`` would also run on
    ``import openmm_opes.kde``, dragging openmm in and breaking the guarantee
    that kde and io stay OpenMM-free. Task 1's test enforces that.
    """
    if name == "OPES":
        from .opes import OPES

        return OPES
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
