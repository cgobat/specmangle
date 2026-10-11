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
    components: tuple[str, ...] = ()

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

        metadata = SvoFps.get_filter_metadata(filter_id)
        components = _parse_svo_components(metadata.get("components"))
        if detector_type is None:
            detector_type = _normalize_svo_detector_type(metadata["DetectorType"])

        order = np.argsort(wavelength.to_value(u.AA))
        return cls(
            filter_id=filter_id,
            wavelength=wavelength[order],
            transmission=transmission[order],
            detector_type=detector_type,
            components=components,
        )

    @property
    def includes_atmosphere(self) -> bool:
        """Whether the response includes an atmospheric component."""
        return any("atmospher" in component.casefold() for component in self.components)

    def with_atmosphere(
        self,
        wavelength: u.Quantity,
        transmission: np.ndarray,
    ) -> Bandpass:
        """Return a copy of the bandpass including atmospheric transmission.

        The output grid is the union of the bandpass and atmospheric wavelength
        samples so that structure in either response curve is retained.
        """
        if self.includes_atmosphere:
            raise ValueError(f"bandpass {self.filter_id} already includes atmosphere")

        atmosphere_wavelength = u.Quantity(wavelength)
        atmosphere_transmission = np.asarray(transmission, dtype=float)

        if atmosphere_wavelength.ndim != 1 or atmosphere_transmission.ndim != 1:
            raise ValueError(
                "atmospheric wavelength and transmission must be one-dimensional"
            )
        if len(atmosphere_wavelength) != len(atmosphere_transmission):
            raise ValueError(
                "atmospheric wavelength and transmission must have the same length"
            )
        if len(atmosphere_wavelength) < 2:
            raise ValueError("atmospheric transmission must contain at least two samples")
        if not atmosphere_wavelength.unit.is_equivalent(u.AA):
            raise u.UnitConversionError("atmospheric wavelength must have units of length")
        if not np.all(np.isfinite(atmosphere_transmission)):
            raise ValueError("atmospheric transmission must be finite")
        if np.any(atmosphere_transmission < 0.0):
            raise ValueError("atmospheric transmission must not be negative")

        unit = self.wavelength.unit
        band_wavelength = self.wavelength.to_value(unit)
        atmosphere_wavelength_value = atmosphere_wavelength.to_value(unit)
        if (
            not np.all(np.isfinite(atmosphere_wavelength_value))
            or np.any(atmosphere_wavelength_value <= 0.0)
        ):
            raise ValueError("atmospheric wavelengths must be positive and finite")
        if np.any(np.diff(atmosphere_wavelength_value) <= 0.0):
            raise ValueError("atmospheric wavelengths must be strictly increasing")

        positive = np.flatnonzero(self.transmission > 0.0)
        lower_index = max(positive[0] - 1, 0)
        upper_index = min(positive[-1] + 1, len(band_wavelength) - 1)
        if (
            atmosphere_wavelength_value[0] > band_wavelength[lower_index]
            or atmosphere_wavelength_value[-1] < band_wavelength[upper_index]
        ):
            raise ValueError(
                "atmospheric transmission must cover the nonzero bandpass response"
            )

        atmosphere_samples = atmosphere_wavelength_value[
            (atmosphere_wavelength_value >= band_wavelength[0])
            & (atmosphere_wavelength_value <= band_wavelength[-1])
        ]
        combined_wavelength = np.unique(
            np.concatenate((band_wavelength, atmosphere_samples))
        )
        combined_transmission = np.interp(
            combined_wavelength,
            band_wavelength,
            self.transmission,
        ) * np.interp(
            combined_wavelength,
            atmosphere_wavelength_value,
            atmosphere_transmission,
        )

        return Bandpass(
            filter_id=self.filter_id,
            wavelength=combined_wavelength * unit,
            transmission=combined_transmission,
            detector_type=self.detector_type,
            components=(*self.components, "Atmosphere"),
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


def _parse_svo_components(value: object | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(
        component.strip()
        for component in str(value).split("+")
        if component.strip()
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
