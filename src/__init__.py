"""Photometry-constrained mangling of astronomical spectra."""

from .bandpasses import Bandpass
from .mangling import MangleResult, mangle
from .photometry import (
    InsufficientCoverageError,
    MagnitudeSystem,
    SyntheticPhotometry,
    spectrum_to_magnitude,
    synthetic_ab_magnitude,
    synthetic_magnitude,
)

__version__ = "0.1.0b2"

__all__ = [
    "Bandpass",
    "InsufficientCoverageError",
    "MagnitudeSystem",
    "MangleResult",
    "SyntheticPhotometry",
    "mangle",
    "spectrum_to_magnitude",
    "synthetic_ab_magnitude",
    "synthetic_magnitude",
]
