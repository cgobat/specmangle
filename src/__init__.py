"""Photometry-constrained mangling of astronomical spectra."""

from .bandpasses import Bandpass
from .mangling import MangleResult, mangle
from .photometry import SyntheticPhotometry, synthetic_ab_magnitude

__version__ = "0.1.0-beta"

__all__ = [
    "Bandpass",
    "MangleResult",
    "SyntheticPhotometry",
    "mangle",
    "synthetic_ab_magnitude",
]
