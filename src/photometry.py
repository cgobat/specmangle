from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Literal

import astropy.units as u
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.utils.data import download_file
import numpy as np
from scipy.integrate import trapezoid
from specutils import Spectrum

from .bandpasses import Bandpass, DetectorType


FLAM = u.erg / (u.s * u.cm**2 * u.AA)
AB_ZERO_FNU = (0. * u.ABmag).to(u.Jy)
MagnitudeSystem = Literal["AB", "Vega"]
_VEGA_SPECTRUM_URL = (
    "https://ssb.stsci.edu/cdbs/calspec/alpha_lyr_stis_012.fits"
)


class InsufficientCoverageError(ValueError):
    """Raised when a bandpass lacks sufficient coverage for some operation."""

    def __init__(self, message: str, coverage: float) -> None:
        super().__init__(message)
        self.coverage = coverage


@dataclass(frozen=True)
class SyntheticPhotometry:
    """Synthetic magnitude and associated diagnostics."""

    magnitude: float
    uncertainty: float | None
    coverage: float


def spectrum_to_magnitude(
    spectrum: Spectrum,
    band: str | Bandpass,
    *,
    system: MagnitudeSystem = "AB",
    min_coverage: float = 0.98,
) -> u.Magnitude | tuple[u.Magnitude, u.Magnitude]:
    """Compute synthetic photometry for a spectrum in an SVO bandpass.

    Parameters
    ----------
    spectrum
        Input spectrum. If it includes an uncertainty, the propagated
        magnitude uncertainty is returned as well.
    band
        SVO FPS filter identifier or an already loaded `Bandpass`.
    system
        Magnitude system, either ``"AB"`` or ``"Vega"``. Vega magnitudes use
        the CALSPEC Vega reference spectrum.
    min_coverage
        Minimum fraction of the bandpass reference signal that must overlap
        the valid spectrum.

    Returns
    -------
    magnitude
        Synthetic magnitude. AB magnitudes carry ``u.ABmag``; Vega magnitudes
        carry ``u.mag``. If the input spectrum has an uncertainty, returns
        ``(magnitude, magnitude_uncertainty)`` instead.
    """
    if isinstance(band, str):
        bandpass = Bandpass.from_svo(band)
    elif isinstance(band, Bandpass):
        bandpass = band
    else:
        raise TypeError("band must be an SVO FPS filter ID or Bandpass")

    normalized_system = _normalize_magnitude_system(system)
    synthetic = synthetic_magnitude(
        spectrum,
        bandpass,
        system=normalized_system,
        min_coverage=min_coverage,
    )
    magnitude_unit = u.ABmag if normalized_system == "AB" else u.mag
    magnitude = synthetic.magnitude * magnitude_unit
    if synthetic.uncertainty is None:
        return magnitude

    return magnitude, synthetic.uncertainty * u.mag


def synthetic_magnitude(
    spectrum: Spectrum,
    bandpass: Bandpass,
    *,
    system: MagnitudeSystem = "AB",
    min_coverage: float = 0.98,
) -> SyntheticPhotometry:
    """
    Compute synthetic photometry for a spectrum through a bandpass.

    The calculation compares the integrated detector signal from the spectrum
    with that from the zero-magnitude reference spectrum for ``system``.
    Photon-counting response curves include the expected additional factor of
    wavelength in the signal integral; energy-counting response curves do not.

    For Vega magnitudes, the reference is the versioned CALSPEC
    ``alpha_lyr_stis_012.fits`` spectrum. It is downloaded on first use and
    cached by Astropy. The native CALSPEC sampling is included in the
    reference integral so stellar spectral structure is retained.

    Masked spectral samples are omitted. Spectral uncertainties, if present,
    are converted to standard deviations and assumed to be independent between
    input wavelength samples. Wavelength cells around masked samples are
    excluded from the coverage and signal integrals.
    """
    system = _normalize_magnitude_system(system)
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be in the interval (0, 1]")

    wavelength, flux, uncertainty, masked_intervals = _spectrum_samples(spectrum)
    wave_value = wavelength.to_value(u.AA)

    band_wave = bandpass.wavelength.to_value(u.AA)
    response = bandpass.transmission

    lower = max(wave_value[0], band_wave[0])
    upper = min(wave_value[-1], band_wave[-1])
    if lower >= upper:
        raise InsufficientCoverageError(
            f"spectrum does not overlap passband {bandpass.filter_id}",
            coverage=0.0,
        )

    integration_segments = []
    segment_start = lower
    for gap_start, gap_end in masked_intervals:
        if gap_end <= lower or gap_start >= upper:
            continue
        gap_start = max(gap_start, lower)
        gap_end = min(gap_end, upper)
        if segment_start < gap_start:
            integration_segments.append((segment_start, gap_start))
        segment_start = max(segment_start, gap_end)
    if segment_start < upper:
        integration_segments.append((segment_start, upper))

    integration_knots = np.unique(
        np.concatenate((band_wave, wave_value, np.ravel(integration_segments)))
    )
    integration_knots = integration_knots[
        (integration_knots >= band_wave[0]) & (integration_knots <= band_wave[-1])
    ]

    reference_knots = np.unique(
        np.concatenate(
            (
                integration_knots,
                _reference_wavelengths(system),
            )
        )
    )
    reference_knots = reference_knots[
        (reference_knots >= band_wave[0]) & (reference_knots <= band_wave[-1])
    ]
    full_reference_signal = _reference_signal(
        reference_knots,
        np.interp(reference_knots, band_wave, response),
        bandpass.detector_type,
        system,
    )

    overlap_reference_signal = 0.0
    source_signal = 0.0
    coefficients = np.zeros(len(wavelength), dtype=float)
    flux_values = flux.to_value(FLAM)

    # Retain the existing f_nu-relative interpolation for the source signal.
    # This is independent of the requested magnitude zero point.
    source_reference_flux_values = _ab_reference_flux(wave_value)
    source_ratio = flux_values / source_reference_flux_values
    for segment_start, segment_end in integration_segments:
        interior = integration_knots[
            (integration_knots > segment_start) & (integration_knots < segment_end)
        ]
        integration_wave = np.unique(
            np.concatenate(([segment_start], interior, [segment_end]))
        )
        integration_response = np.interp(integration_wave, band_wave, response)

        reference_interior = reference_knots[
            (reference_knots > segment_start) & (reference_knots < segment_end)
        ]
        reference_wave = np.unique(
            np.concatenate(([segment_start], reference_interior, [segment_end]))
        )
        overlap_reference_signal += _reference_signal(
            reference_wave,
            np.interp(reference_wave, band_wave, response),
            bandpass.detector_type,
            system,
        )

        source_reference_flux = _ab_reference_flux(integration_wave)
        interpolated_ratio = np.interp(
            integration_wave,
            wave_value,
            source_ratio,
        )
        detector_weight = _detector_weight(
            integration_wave,
            bandpass.detector_type,
        )
        source_signal += trapezoid(
            interpolated_ratio
            * source_reference_flux
            * integration_response
            * detector_weight,
            integration_wave,
        )
        if uncertainty is not None:
            coefficients += _linear_interpolation_integral_coefficients(
                wave_value,
                integration_wave,
                source_reference_flux * integration_response * detector_weight,
            )

    coverage = overlap_reference_signal / full_reference_signal
    if coverage < min_coverage:
        raise InsufficientCoverageError(
            f"spectrum covers only {coverage:.3f} of passband "
            f"{bandpass.filter_id}; required coverage is {min_coverage:.3f}",
            coverage=float(coverage),
        )

    if not np.isfinite(source_signal) or source_signal <= 0.0:
        raise ValueError(
            f"integrated flux through passband {bandpass.filter_id} is not positive"
        )

    magnitude = -2.5 * np.log10(source_signal / overlap_reference_signal)

    magnitude_uncertainty = None
    if uncertainty is not None:
        signal_uncertainty = np.sqrt(
            np.sum(
                (
                    coefficients
                    / source_reference_flux_values
                    * uncertainty.to_value(FLAM)
                )
                ** 2
            )
        )
        magnitude_uncertainty = (
            2.5 / np.log(10.0) * signal_uncertainty / source_signal
        )

    return SyntheticPhotometry(
        magnitude=float(magnitude),
        uncertainty=(
            None if magnitude_uncertainty is None else float(magnitude_uncertainty)
        ),
        coverage=float(coverage),
    )


def synthetic_ab_magnitude(
    spectrum: Spectrum,
    bandpass: Bandpass,
    *,
    min_coverage: float = 0.98,
) -> SyntheticPhotometry:
    """Compute synthetic AB photometry for a spectrum through a bandpass."""
    return synthetic_magnitude(
        spectrum,
        bandpass,
        system="AB",
        min_coverage=min_coverage,
    )


def _normalize_magnitude_system(system: object) -> MagnitudeSystem:
    text = str(system).strip().casefold()
    if text == "ab":
        return "AB"
    if text in ("vega", "vegamag"):
        return "Vega"
    raise ValueError('magnitude system must be either "AB" or "Vega"')


def _spectrum_samples(
    spectrum: Spectrum,
) -> tuple[u.Quantity, u.Quantity, u.Quantity | None, list[tuple[float, float]]]:
    if not isinstance(spectrum, Spectrum):
        raise TypeError("spectrum must be a specutils.Spectrum")
    if spectrum.flux.ndim != 1:
        raise ValueError("only one-dimensional Spectrum objects are supported")
    if len(spectrum.flux) < 2:
        raise ValueError("a spectrum must contain at least two samples")

    wavelength = spectrum.wavelength.to(u.AA)
    if len(wavelength) != len(spectrum.flux):
        raise ValueError("spectral axis must specify one coordinate per flux sample")

    flux = spectrum.flux.to(FLAM, equivalencies=u.spectral_density(wavelength))
    uncertainty = _standard_deviation(spectrum, wavelength)

    valid = np.ones(len(flux), dtype=bool)
    if spectrum.mask is not None:
        mask = np.asarray(spectrum.mask, dtype=bool)
        if mask.shape != flux.shape:
            raise ValueError("spectrum mask must match the one-dimensional flux shape")
        valid &= ~mask

    wave_values = wavelength.to_value(u.AA)
    flux_values = flux.to_value(FLAM)
    if np.any(~np.isfinite(wave_values)) or np.any(wave_values <= 0.0):
        raise ValueError("spectrum wavelengths must be positive and finite")
    if np.any(~np.isfinite(flux_values[valid])):
        raise ValueError("unmasked spectrum flux must be finite")

    if uncertainty is not None:
        uncertainty_values = uncertainty.to_value(FLAM)
        if np.any(~np.isfinite(uncertainty_values[valid])):
            raise ValueError("unmasked spectrum uncertainty must be finite")
        if np.any(uncertainty_values[valid] < 0.0):
            raise ValueError("spectrum uncertainty must not be negative")

    if np.count_nonzero(valid) < 2:
        raise ValueError("spectrum must contain at least two valid, unmasked samples")

    order = np.argsort(wave_values)
    wave_values = wave_values[order]
    sorted_mask = ~valid[order]
    if np.any(np.diff(wave_values) <= 0.0):
        raise ValueError("spectrum wavelengths must be unique")

    masked_intervals = []
    for index in np.flatnonzero(sorted_mask):
        left = wave_values[index - 1] if index > 0 else wave_values[index]
        right = (
            wave_values[index + 1]
            if index + 1 < len(wave_values)
            else wave_values[index]
        )
        masked_intervals.append(
            (
                0.5 * (left + wave_values[index]),
                0.5 * (wave_values[index] + right),
            )
        )

    wavelength = wavelength[valid]
    flux = flux[valid]
    if uncertainty is not None:
        uncertainty = uncertainty[valid]

    valid_order = np.argsort(wavelength.to_value(u.AA))
    wavelength = wavelength[valid_order]
    flux = flux[valid_order]
    if uncertainty is not None:
        uncertainty = uncertainty[valid_order]

    return wavelength, flux, uncertainty, masked_intervals


def _standard_deviation(
    spectrum: Spectrum,
    wavelength: u.Quantity,
) -> u.Quantity | None:
    if spectrum.uncertainty is None:
        return None

    try:
        uncertainty = spectrum.uncertainty.represent_as(StdDevUncertainty)
    except (AttributeError, TypeError) as exc:
        raise TypeError(
            "spectrum uncertainty must support conversion to StdDevUncertainty"
        ) from exc

    if uncertainty.unit is None:
        quantity = u.Quantity(uncertainty.array, spectrum.flux.unit)
    else:
        quantity = uncertainty.quantity

    return quantity.to(FLAM, equivalencies=u.spectral_density(wavelength))


def _detector_weight(
    wavelength_angstrom: np.ndarray,
    detector_type: DetectorType,
) -> np.ndarray:
    if detector_type == "photon":
        return wavelength_angstrom
    return np.ones_like(wavelength_angstrom)


def _reference_signal(
    wavelength_angstrom: np.ndarray,
    response: np.ndarray,
    detector_type: DetectorType,
    system: MagnitudeSystem,
) -> float:
    reference_flam = _reference_flux(wavelength_angstrom, system)
    detector_weight = _detector_weight(wavelength_angstrom, detector_type)
    return float(
        trapezoid(
            reference_flam * response * detector_weight,
            wavelength_angstrom,
        )
    )


def _reference_flux(
    wavelength_angstrom: np.ndarray,
    system: MagnitudeSystem,
) -> np.ndarray:
    if system == "AB":
        return _ab_reference_flux(wavelength_angstrom)
    return _vega_reference_flux(wavelength_angstrom)


def _reference_wavelengths(system: MagnitudeSystem) -> np.ndarray:
    if system == "AB":
        return np.empty(0, dtype=float)
    wavelength, _ = _vega_reference_spectrum()
    return wavelength


def _ab_reference_flux(wavelength_angstrom: np.ndarray) -> np.ndarray:
    wavelength = u.Quantity(wavelength_angstrom, u.AA)
    return AB_ZERO_FNU.to(
        FLAM,
        equivalencies=u.spectral_density(wavelength),
    ).value


@cache
def _vega_reference_spectrum() -> tuple[np.ndarray, np.ndarray]:
    filename = download_file(
        _VEGA_SPECTRUM_URL,
        cache=True,
        pkgname="specmangle",
    )
    data = fits.getdata(filename, ext=1)
    wavelength = np.asarray(data["WAVELENGTH"], dtype=float)
    flux = np.asarray(data["FLUX"], dtype=float)

    if wavelength.ndim != 1 or flux.ndim != 1 or len(wavelength) != len(flux):
        raise ValueError("CALSPEC Vega spectrum has invalid wavelength or flux arrays")
    if len(wavelength) < 2 or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("CALSPEC Vega wavelengths must be strictly increasing")
    if (
        np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(~np.isfinite(flux))
        or np.any(flux < 0.0)
        or not np.any(flux > 0.0)
    ):
        raise ValueError("CALSPEC Vega spectrum must contain non-negative finite flux")

    return wavelength, flux


def _vega_reference_flux(wavelength_angstrom: np.ndarray) -> np.ndarray:
    wavelength_angstrom = np.asarray(wavelength_angstrom, dtype=float)
    vega_wavelength, vega_flux = _vega_reference_spectrum()
    if (
        np.min(wavelength_angstrom) < vega_wavelength[0]
        or np.max(wavelength_angstrom) > vega_wavelength[-1]
    ):
        raise ValueError(
            "CALSPEC Vega spectrum does not cover the requested wavelength range"
        )

    # Interpolate relative to a smooth flat-f_nu reference, matching the source
    # interpolation used above while retaining the native Vega spectral samples
    # in the integration grid.
    vega_ratio = vega_flux / _ab_reference_flux(vega_wavelength)
    return (
        np.interp(wavelength_angstrom, vega_wavelength, vega_ratio)
        * _ab_reference_flux(wavelength_angstrom)
    )


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    weights = np.empty_like(x, dtype=float)
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    if len(x) > 2:
        weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    return weights


def _linear_interpolation_integral_coefficients(
    source_x: np.ndarray,
    integration_x: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    integration_weights = _trapezoid_weights(integration_x) * kernel

    left = np.searchsorted(source_x, integration_x, side="right") - 1
    left = np.clip(left, 0, len(source_x) - 2)
    right = left + 1

    fraction = (
        (integration_x - source_x[left])
        / (source_x[right] - source_x[left])
    )

    coefficients = np.zeros(len(source_x), dtype=float)
    np.add.at(coefficients, left, integration_weights * (1.0 - fraction))
    np.add.at(coefficients, right, integration_weights * fraction)
    return coefficients
