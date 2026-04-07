"""Data loader module namespace.

This package uses lazy imports so `from dataset import data_loader` is fast and
doesn't require optional/heavy dependencies for every loader up front.
"""

import importlib

_SUBMODULES = (
    "BaseLoader",
    "BaseSingleLoader",
    "COHFACELoader",
    "UBFCrPPGLoader",
    "PURELoader",
    "iBVPLoader",
    "SCAMPSLoader",
    "MMPDLoader",
    "BP4DPlusLoader",
    "BP4DPlusBigSmallLoader",
    "UBFCPHYSLoader",
    "UBFCrPPGSingleLoader",
)


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f"dataset.data_loader.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
