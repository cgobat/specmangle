from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import astropy.units as u
import numpy as np
from scipy.integrate import trapezoid


DetectorType = Literal["photon", "energy"]


@dataclass(frozen=True)
class Bandpass:
    """A bandpass response curve."""

    filter_id: str
    wavelength: u.Quantity
    transmission: np.ndarray
    detector_type: DetectorType

    def __post_init__(self) -> None:
        transmission = np.asarray(self.transmission, dtype=float)
        object.__setattr__(self, "transmission", transmission)

        if not self.filter_id:
            raise ValueError("filter_id must not be empty")
        if self.wavelength.ndim != 1 or transmission.ndim != 1:
            raise ValueError("bandpass wavelength and transmission must be one-dimensional")
        if len(self.wavelength) != len(transmission):
            raise ValueError("bandpass wavelength and transmission must have the same length")
        if len(self.wavelength) < 2:
            raise ValueError("a bandpass must contain at least two samples")
        if not self.wavelength.unit.is_equivalent(u.AA):
            raise u.UnitConversionError("bandpass wavelength must have units of length")
        if self.detector_type not in ("photon", "energy"):
            raise ValueError("detector_type must be either \"photon\" or \"energy\"")
        if not np.all(np.isfinite(transmission)):
            raise ValueError("bandpass transmission must be finite")
        if np.any(transmission < 0.0):
            raise ValueError("bandpass transmission must not be negative")
        if not np.any(transmission > 0.0):
            raise ValueError("bandpass transmission must contain a positive value")

        wavelength = self.wavelength.to_value(u.AA)
        if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
            raise ValueError("bandpass wavelengths must be positive and finite")
        if np.any(np.diff(wavelength) <= 0.0):
            raise ValueError("bandpass wavelengths must be strictly increasing")

    @classmethod
    def from_svo(
        cls,
        filter_id: str,
        *,
        detector_type: DetectorType | None = None,
    ) -> Bandpass:
        """
        Retrieve a bandpass from the SVO Filter Profile Service.

        Astroquery's own query caching is used; specmangle does not implement
        an additional cache.
        """
        from astroquery.svo_fps import SvoFps

        table = SvoFps.get_transmission_data(filter_id)
        wavelength = u.Quantity(table["Wavelength"])
        transmission = np.asarray(table["Transmission"], dtype=float)

        if detector_type is None:
            detector_type = _get_svo_detector_type(filter_id)

        order = np.argsort(wavelength.to_value(u.AA))
        return cls(
            filter_id=filter_id,
            wavelength=wavelength[order],
            transmission=transmission[order],
            detector_type=detector_type,
        )

    @property
    def pivot_wavelength(self) -> u.Quantity:
        """Return the pivot wavelength of the response curve."""
        wavelength = self.wavelength.to_value(u.AA)

        if self.detector_type == "photon":
            numerator = trapezoid(self.transmission * wavelength, wavelength)
            denominator = trapezoid(self.transmission / wavelength, wavelength)
        else:
            numerator = trapezoid(self.transmission, wavelength)
            denominator = trapezoid(self.transmission / wavelength**2, wavelength)

        return np.sqrt(numerator / denominator) * u.AA


def _get_svo_detector_type(filter_id: str) -> DetectorType:
    from astroquery.svo_fps import SvoFps

    get_filter_metadata = getattr(SvoFps, "get_filter_metadata", None)
    if get_filter_metadata is not None:
        metadata = get_filter_metadata(filter_id)
        if "DetectorType" in metadata:
            return _normalize_svo_detector_type(metadata["DetectorType"])

    if "/" not in filter_id:
        raise ValueError(
            f"cannot infer SVO facility and instrument from filter ID {filter_id!r}"
        )

    facility, filter_name = filter_id.split("/", 1)
    instrument = filter_name.split(".", 1)[0] if "." in filter_name else None

    instrument_options = [instrument]
    if instrument is not None:
        instrument_options.append(None)

    for instrument_option in instrument_options:
        table = SvoFps.get_filter_list(
            facility,
            instrument=instrument_option,
        )
        identifiers = np.asarray(table["filterID"]).astype(str)
        matches = np.flatnonzero(identifiers == filter_id)
        if len(matches) == 1:
            return _normalize_svo_detector_type(table["DetectorType"][matches[0]])

    raise ValueError(
        f"could not determine DetectorType metadata for SVO filter {filter_id}"
    )


def _normalize_svo_detector_type(value: object) -> DetectorType:
    text = str(value).strip().lower()
    if text in ("photon", "photon counter", "photon-counter"):
        return "photon"
    if text in ("energy", "energy counter", "energy-counter"):
        return "energy"

    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = None

    # IVOA Photometry Data Model convention: 0 = energy counter, 1 = photon counter.
    if numeric == 0:
        return "energy"
    if numeric == 1:
        return "photon"

    raise ValueError(f"unrecognized SVO DetectorType value: {value!r}")
