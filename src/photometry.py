from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
from astropy.nddata import StdDevUncertainty
import numpy as np
from scipy.integrate import trapezoid
from specutils import Spectrum

from .bandpasses import Bandpass, DetectorType


FLAM = u.erg / (u.s * u.cm**2 * u.AA)
AB_ZERO_FNU = (0. * u.ABmag).to(u.Jy)


class _InsufficientCoverageError(ValueError):
    pass


@dataclass(frozen=True)
class SyntheticPhotometry:
    """Synthetic AB magnitude and associated diagnostics."""

    magnitude: float
    uncertainty: float | None
    coverage: float


def spectrum_to_magnitude(
    spectrum: Spectrum,
    band: str | Bandpass,
    *,
    min_coverage: float = 0.98,
) -> u.Magnitude | tuple[u.Magnitude, u.Magnitude]:
    """Compute synthetic AB photometry for a spectrum in an SVO bandpass.

    Parameters
    ----------
    spectrum
        Input spectrum. If it includes an uncertainty, the propagated
        magnitude uncertainty is returned as well.
    band
        SVO FPS filter identifier or an already loaded `Bandpass`.
    min_coverage
        Minimum fraction of the bandpass reference signal that must overlap
        the valid spectrum.

    Returns
    -------
    magnitude
        Synthetic AB magnitude. If the input spectrum has an uncertainty,
        returns ``(magnitude, magnitude_uncertainty)`` instead.
    """
    if isinstance(band, str):
        bandpass = Bandpass.from_svo(band)
    elif isinstance(band, Bandpass):
        bandpass = band
    else:
        raise TypeError("band must be an SVO FPS filter ID or Bandpass")

    synthetic = synthetic_ab_magnitude(
        spectrum,
        bandpass,
        min_coverage=min_coverage,
    )
    magnitude = synthetic.magnitude * u.ABmag
    if synthetic.uncertainty is None:
        return magnitude

    return magnitude, synthetic.uncertainty * u.mag


def synthetic_ab_magnitude(
    spectrum: Spectrum,
    bandpass: Bandpass,
    *,
    min_coverage: float = 0.98,
) -> SyntheticPhotometry:
    """
    Compute synthetic AB photometry for a spectrum through a bandpass.

    The calculation compares the integrated detector signal from the spectrum
    with that from a zero-magnitude AB reference spectrum. Photon-counting
    response curves include the expected additional factor of wavelength in
    the signal integral; energy-counting response curves do not.

    Masked spectral samples are omitted. Spectral uncertainties, if present,
    are converted to standard deviations and assumed to be independent between
    input wavelength samples. Wavelength cells around masked samples are
    excluded from the coverage and signal integrals.
    """
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be in the interval (0, 1]")

    wavelength, flux, uncertainty, masked_intervals = _spectrum_samples(spectrum)
    wave_value = wavelength.to_value(u.AA)

    band_wave = bandpass.wavelength.to_value(u.AA)
    response = bandpass.transmission
    full_reference_signal = _reference_signal(
        band_wave,
        response,
        bandpass.detector_type,
    )

    lower = max(wave_value[0], band_wave[0])
    upper = min(wave_value[-1], band_wave[-1])
    if lower >= upper:
        raise _InsufficientCoverageError(
            f"spectrum does not overlap passband {bandpass.filter_id}"
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

    overlap_reference_signal = 0.0
    source_signal = 0.0
    coefficients = np.zeros(len(wavelength), dtype=float)
    flux_values = flux.to_value(FLAM)
    for segment_start, segment_end in integration_segments:
        interior = band_wave[
            (band_wave > segment_start) & (band_wave < segment_end)
        ]
        integration_wave = np.unique(
            np.concatenate(([segment_start], interior, [segment_end]))
        )
        integration_response = np.interp(integration_wave, band_wave, response)

        overlap_reference_signal += _reference_signal(
            integration_wave,
            integration_response,
            bandpass.detector_type,
        )

        interpolated_flux = np.interp(
            integration_wave,
            wave_value,
            flux_values,
        )
        detector_weight = _detector_weight(
            integration_wave,
            bandpass.detector_type,
        )
        source_signal += trapezoid(
            interpolated_flux * integration_response * detector_weight,
            integration_wave,
        )
        if uncertainty is not None:
            coefficients += _linear_interpolation_integral_coefficients(
                wave_value,
                integration_wave,
                integration_response * detector_weight,
            )

    coverage = overlap_reference_signal / full_reference_signal
    if coverage < min_coverage:
        raise _InsufficientCoverageError(
            f"spectrum covers only {coverage:.3f} of passband "
            f"{bandpass.filter_id}; required coverage is {min_coverage:.3f}"
        )

    if not np.isfinite(source_signal) or source_signal <= 0.0:
        raise ValueError(
            f"integrated flux through passband {bandpass.filter_id} is not positive"
        )

    magnitude = -2.5 * np.log10(source_signal / overlap_reference_signal)

    magnitude_uncertainty = None
    if uncertainty is not None:
        signal_uncertainty = np.sqrt(
            np.sum((coefficients * uncertainty.to_value(FLAM)) ** 2)
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
) -> float:
    wavelength = wavelength_angstrom * u.AA
    reference_flam = AB_ZERO_FNU.to(
        FLAM,
        equivalencies=u.spectral_density(wavelength),
    ).value
    detector_weight = _detector_weight(wavelength_angstrom, detector_type)
    return float(
        trapezoid(
            reference_flam * response * detector_weight,
            wavelength_angstrom,
        )
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
