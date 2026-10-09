"""Photometry-constrained mangling of astronomical spectra."""

from .bandpasses import Bandpass
from .mangling import MangleResult, mangle
from .photometry import (
    InsufficientCoverageError,
    SyntheticPhotometry,
    spectrum_to_magnitude,
    synthetic_ab_magnitude,
)

__version__ = "0.1.0-beta"

__all__ = [
    "Bandpass",
    "InsufficientCoverageError",
    "MangleResult",
    "SyntheticPhotometry",
    "mangle",
    "spectrum_to_magnitude",
    "synthetic_ab_magnitude",
]
