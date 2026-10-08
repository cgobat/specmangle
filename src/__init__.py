"""Photometry-constrained mangling of astronomical spectra."""

from .bandpasses import Bandpass
from .mangling import MangleResult, mangle
from .photometry import SyntheticPhotometry, synthetic_ab_magnitude

__all__ = [
    "Bandpass",
    "MangleResult",
    "SyntheticPhotometry",
    "mangle",
    "synthetic_ab_magnitude",
]
