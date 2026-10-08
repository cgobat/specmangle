from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import astropy.units as u
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import OptimizeResult, least_squares


DetectorType = Literal["photon", "energy"]
FLAM = u.erg / (u.s * u.cm**2 * u.AA)
AB_ZERO_FNU = u.Quantity(0.0, u.ABmag).to(u.Jy)


@dataclass(frozen=True)
class Spectrum:
    """One-dimensional spectrum and optional 1-sigma uncertainty."""

    wavelength: u.Quantity
    flux: u.Quantity
    uncertainty: u.Quantity | None = None

    def __post_init__(self) -> None:
        if self.wavelength.ndim != 1 or self.flux.ndim != 1:
            raise ValueError("wavelength and flux must be one-dimensional")
        if len(self.wavelength) != len(self.flux):
            raise ValueError("wavelength and flux must have the same length")
        if len(self.wavelength) < 2:
            raise ValueError("a spectrum must contain at least two samples")
        if not self.wavelength.unit.is_equivalent(u.AA):
            raise u.UnitConversionError("wavelength must have units of length")
        if self.uncertainty is not None:
            if self.uncertainty.ndim != 1 or len(self.uncertainty) != len(self.flux):
                raise ValueError("uncertainty must match the shape of flux")


@dataclass(frozen=True)
class PhotometryPoint:
    """Contemporaneous AB photometry for one SVO passband."""

    filter_id: str
    magnitude: float
    uncertainty: float | None = None

    def __post_init__(self) -> None:
        if not self.filter_id:
            raise ValueError("filter_id must not be empty")
        if not np.isfinite(self.magnitude):
            raise ValueError("magnitude must be finite")
        if self.uncertainty is not None:
            if not np.isfinite(self.uncertainty) or self.uncertainty <= 0.0:
                raise ValueError("photometric uncertainty must be positive and finite")


@dataclass(frozen=True)
class Bandpass:
    """A passband response curve."""

    filter_id: str
    wavelength: u.Quantity
    transmission: np.ndarray
    detector_type: DetectorType

    def __post_init__(self) -> None:
        transmission = np.asarray(self.transmission, dtype=float)
        object.__setattr__(self, "transmission", transmission)

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
        Retrieve a passband from the SVO Filter Profile Service.

        Astroquery's own HTTP cache is used; this class does not add another
        caching layer.
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
        """Pivot wavelength computed from this response curve."""
        wavelength = self.wavelength.to_value(u.AA)
        response = self.transmission

        if self.detector_type == "photon":
            numerator = np.trapezoid(response * wavelength, wavelength)
            denominator = np.trapezoid(response / wavelength, wavelength)
        else:
            numerator = np.trapezoid(response, wavelength)
            denominator = np.trapezoid(response / wavelength**2, wavelength)

        return np.sqrt(numerator / denominator) * u.AA


@dataclass(frozen=True)
class SyntheticPhotometry:
    magnitude: float
    uncertainty: float | None
    coverage: float


@dataclass(frozen=True)
class ManglingResult:
    spectrum: Spectrum
    correction: np.ndarray
    correction_uncertainty: np.ndarray | None
    anchor_wavelengths: u.Quantity
    log_correction_parameters: np.ndarray
    parameter_covariance: np.ndarray | None
    synthetic_magnitudes: np.ndarray
    synthetic_uncertainties: np.ndarray
    photometric_residuals: np.ndarray
    bandpasses: dict[str, Bandpass]
    optimizer: OptimizeResult


def synthetic_ab_magnitude(
    spectrum: Spectrum,
    bandpass: Bandpass,
    *,
    min_coverage: float = 0.98,
) -> SyntheticPhotometry:
    """
    Compute synthetic AB photometry for a spectrum through a passband.

    The calculation compares the integrated detector signal from the spectrum
    with that from a zero-magnitude AB reference spectrum. SVO's detector type
    convention is respected: photon counters include an extra factor of
    wavelength in the signal integral, while energy counters do not.

    Parameters
    ----------
    spectrum
        Input spectrum. Flux may be any spectral-flux-density unit understood
        by Astropy's spectral-density equivalency.
    bandpass
        Passband response curve.
    min_coverage
        Minimum fraction of the AB-reference passband signal that must lie
        within the wavelength range of the spectrum.
    """
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be in the interval (0, 1]")

    spectrum = _sorted_spectrum(spectrum)
    wave = spectrum.wavelength.to(u.AA)
    wave_value = wave.value
    flux = spectrum.flux.to(FLAM, equivalencies=u.spectral_density(wave))

    uncertainty = None
    if spectrum.uncertainty is not None:
        uncertainty = spectrum.uncertainty.to(
            FLAM,
            equivalencies=u.spectral_density(wave),
        )

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
        raise ValueError(
            f"spectrum does not overlap passband {bandpass.filter_id}"
        )

    interior = band_wave[(band_wave > lower) & (band_wave < upper)]
    integration_wave = np.unique(np.concatenate(([lower], interior, [upper])))
    integration_response = np.interp(integration_wave, band_wave, response)

    overlap_reference_signal = _reference_signal(
        integration_wave,
        integration_response,
        bandpass.detector_type,
    )
    coverage = overlap_reference_signal / full_reference_signal
    if coverage < min_coverage:
        raise ValueError(
            f"spectrum covers only {coverage:.3f} of passband "
            f"{bandpass.filter_id}; required coverage is {min_coverage:.3f}"
        )

    interpolated_flux = np.interp(
        integration_wave,
        wave_value,
        flux.value,
    )
    detector_weight = _detector_weight(
        integration_wave,
        bandpass.detector_type,
    )
    source_signal = np.trapezoid(
        interpolated_flux * integration_response * detector_weight,
        integration_wave,
    )

    if not np.isfinite(source_signal) or source_signal <= 0.0:
        raise ValueError(
            f"integrated flux through passband {bandpass.filter_id} is not positive"
        )

    magnitude = -2.5 * np.log10(source_signal / overlap_reference_signal)

    magnitude_uncertainty = None
    if uncertainty is not None:
        coefficients = _linear_interpolation_integral_coefficients(
            wave_value,
            integration_wave,
            integration_response * detector_weight,
        )
        signal_uncertainty = np.sqrt(
            np.sum((coefficients * uncertainty.value) ** 2)
        )
        magnitude_uncertainty = (
            2.5 / np.log(10.0) * signal_uncertainty / source_signal
        )

    return SyntheticPhotometry(
        magnitude=float(magnitude),
        uncertainty=(
            None
            if magnitude_uncertainty is None
            else float(magnitude_uncertainty)
        ),
        coverage=float(coverage),
    )


def mangle_spectrum(
    spectrum: Spectrum,
    photometry: Sequence[PhotometryPoint],
    *,
    bandpasses: Mapping[str, Bandpass] | None = None,
    min_coverage: float = 0.98,
    smoothness: float = 0.0,
    max_nfev: int | None = None,
) -> ManglingResult:
    """
    Mangle a spectrum to match contemporaneous AB photometry.

    A positive multiplicative correction is represented by a natural cubic
    spline in log(wavelength) and log(flux scale), with one free anchor per
    unique passband. Outside the bluest and reddest anchors the correction is
    held constant. The free anchor values are optimized so that synthetic AB
    magnitudes of the corrected spectrum match the supplied photometry.

    Repeated measurements in the same passband are supported. Photometric and
    spectral uncertainties, when supplied, are combined in quadrature when
    weighting magnitude residuals. Spectral uncertainties are assumed to be
    independent between input wavelength samples.

    The uncertainty attached to ``result.spectrum`` is the input spectral
    uncertainty scaled by the best-fit correction. Uncertainty in the mangling
    correction itself is returned separately as ``correction_uncertainty`` and
    is not folded into the spectral uncertainty because it is wavelength
    correlated.

    Parameters
    ----------
    spectrum
        Spectrum to mangle.
    photometry
        One or more contemporaneous AB photometric measurements.
    bandpasses
        Optional mapping from filter ID to pre-loaded Bandpass objects. Any
        missing passbands are retrieved from SVO via Astroquery.
    min_coverage
        Minimum fractional passband coverage required for every measurement.
    smoothness
        Optional regularization strength on changes in slope of the log
        correction between adjacent anchors. Zero reproduces the photometric
        constraints without an explicit smoothness penalty.
    max_nfev
        Optional maximum number of least-squares function evaluations.
    """
    if not photometry:
        raise ValueError("at least one photometry point is required")
    if smoothness < 0.0 or not np.isfinite(smoothness):
        raise ValueError("smoothness must be finite and non-negative")

    spectrum = _sorted_spectrum(spectrum)
    supplied_bandpasses = {} if bandpasses is None else dict(bandpasses)

    filter_ids = list(dict.fromkeys(point.filter_id for point in photometry))
    loaded_bandpasses: dict[str, Bandpass] = {}
    for filter_id in filter_ids:
        if filter_id in supplied_bandpasses:
            loaded_bandpasses[filter_id] = supplied_bandpasses[filter_id]
        else:
            loaded_bandpasses[filter_id] = Bandpass.from_svo(filter_id)

    anchor_pairs = sorted(
        (
            loaded_bandpasses[filter_id].pivot_wavelength.to_value(u.AA),
            filter_id,
        )
        for filter_id in filter_ids
    )
    anchor_wavelength_values = np.array([pair[0] for pair in anchor_pairs])
    ordered_filter_ids = [pair[1] for pair in anchor_pairs]

    if np.any(np.diff(anchor_wavelength_values) <= 0.0):
        raise ValueError(
            "two passbands have identical pivot wavelengths; independent mangling "
            "anchors require distinct pivot wavelengths"
        )

    parameter_index = {
        filter_id: index for index, filter_id in enumerate(ordered_filter_ids)
    }

    original_synthetic = {
        filter_id: synthetic_ab_magnitude(
            spectrum,
            loaded_bandpasses[filter_id],
            min_coverage=min_coverage,
        )
        for filter_id in filter_ids
    }

    initial_parameters = np.zeros(len(ordered_filter_ids), dtype=float)
    for filter_id in ordered_filter_ids:
        estimates = []
        weights = []
        synthetic = original_synthetic[filter_id]
        for point in photometry:
            if point.filter_id != filter_id:
                continue
            estimates.append(
                0.4 * np.log(10.0) * (synthetic.magnitude - point.magnitude)
            )
            sigma = _combined_magnitude_uncertainty(point, synthetic)
            weights.append(1.0 if sigma is None else 1.0 / sigma**2)
        initial_parameters[parameter_index[filter_id]] = np.average(
            estimates,
            weights=weights,
        )

    spectrum_wave = spectrum.wavelength.to_value(u.AA)

    def residual_function(parameters: np.ndarray) -> np.ndarray:
        correction = np.exp(
            _evaluate_log_correction(
                spectrum_wave,
                anchor_wavelength_values,
                parameters,
            )
        )
        corrected_spectrum = _apply_correction(spectrum, correction)
        synthetic_by_filter = {
            filter_id: synthetic_ab_magnitude(
                corrected_spectrum,
                loaded_bandpasses[filter_id],
                min_coverage=min_coverage,
            )
            for filter_id in filter_ids
        }

        residuals = []
        for point in photometry:
            synthetic = synthetic_by_filter[point.filter_id]
            sigma = _combined_magnitude_uncertainty(point, synthetic)
            scale = 1.0 if sigma is None else sigma
            residuals.append((synthetic.magnitude - point.magnitude) / scale)

        if smoothness > 0.0 and len(parameters) >= 3:
            log_anchor = np.log(anchor_wavelength_values)
            slopes = np.diff(parameters) / np.diff(log_anchor)
            slope_changes = np.diff(slopes)
            residuals.extend(np.sqrt(smoothness) * slope_changes)

        return np.asarray(residuals, dtype=float)

    optimizer = least_squares(
        residual_function,
        initial_parameters,
        max_nfev=max_nfev,
    )
    if not optimizer.success:
        raise RuntimeError(f"spectral mangling fit failed: {optimizer.message}")

    best_log_correction = _evaluate_log_correction(
        spectrum_wave,
        anchor_wavelength_values,
        optimizer.x,
    )
    correction = np.exp(best_log_correction)
    corrected_spectrum = _apply_correction(spectrum, correction)

    final_synthetic_by_filter = {
        filter_id: synthetic_ab_magnitude(
            corrected_spectrum,
            loaded_bandpasses[filter_id],
            min_coverage=min_coverage,
        )
        for filter_id in filter_ids
    }
    synthetic_magnitudes = np.array(
        [final_synthetic_by_filter[point.filter_id].magnitude for point in photometry]
    )
    synthetic_uncertainties = np.array(
        [
            np.nan
            if final_synthetic_by_filter[point.filter_id].uncertainty is None
            else final_synthetic_by_filter[point.filter_id].uncertainty
            for point in photometry
        ],
        dtype=float,
    )
    photometric_residuals = synthetic_magnitudes - np.array(
        [point.magnitude for point in photometry],
        dtype=float,
    )

    parameter_covariance = _parameter_covariance(
        optimizer,
        spectrum,
        photometry,
    )
    correction_uncertainty = None
    if parameter_covariance is not None:
        basis = _correction_basis(
            spectrum_wave,
            anchor_wavelength_values,
        )
        log_variance = np.einsum(
            "ij,jk,ik->i",
            basis,
            parameter_covariance,
            basis,
        )
        correction_uncertainty = correction * np.sqrt(
            np.clip(log_variance, 0.0, None)
        )

    return ManglingResult(
        spectrum=corrected_spectrum,
        correction=correction,
        correction_uncertainty=correction_uncertainty,
        anchor_wavelengths=anchor_wavelength_values * u.AA,
        log_correction_parameters=np.array(optimizer.x, copy=True),
        parameter_covariance=parameter_covariance,
        synthetic_magnitudes=synthetic_magnitudes,
        synthetic_uncertainties=synthetic_uncertainties,
        photometric_residuals=photometric_residuals,
        bandpasses=loaded_bandpasses,
        optimizer=optimizer,
    )


def _sorted_spectrum(spectrum: Spectrum) -> Spectrum:
    wavelength = spectrum.wavelength.to_value(u.AA)
    if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("spectrum wavelengths must be positive and finite")

    flux_values = np.asarray(spectrum.flux.value)
    if not np.all(np.isfinite(flux_values)):
        raise ValueError("spectrum flux must be finite")

    if spectrum.uncertainty is not None:
        uncertainty_values = np.asarray(spectrum.uncertainty.value)
        if not np.all(np.isfinite(uncertainty_values)):
            raise ValueError("spectrum uncertainty must be finite")
        if np.any(uncertainty_values < 0.0):
            raise ValueError("spectrum uncertainty must not be negative")

    order = np.argsort(wavelength)
    sorted_wavelength = spectrum.wavelength[order]
    sorted_flux = spectrum.flux[order]
    sorted_uncertainty = (
        None if spectrum.uncertainty is None else spectrum.uncertainty[order]
    )

    if np.any(np.diff(sorted_wavelength.to_value(u.AA)) <= 0.0):
        raise ValueError("spectrum wavelengths must be unique")

    return Spectrum(
        wavelength=sorted_wavelength,
        flux=sorted_flux,
        uncertainty=sorted_uncertainty,
    )


def _apply_correction(spectrum: Spectrum, correction: np.ndarray) -> Spectrum:
    uncertainty = None
    if spectrum.uncertainty is not None:
        uncertainty = spectrum.uncertainty * correction
    return Spectrum(
        wavelength=spectrum.wavelength,
        flux=spectrum.flux * correction,
        uncertainty=uncertainty,
    )


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
        np.trapezoid(
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


def _evaluate_log_correction(
    wavelength_angstrom: np.ndarray,
    anchor_wavelength_angstrom: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    if len(parameters) == 1:
        return np.full_like(wavelength_angstrom, parameters[0], dtype=float)

    log_anchor = np.log(anchor_wavelength_angstrom)
    log_wavelength = np.log(wavelength_angstrom)
    spline = CubicSpline(log_anchor, parameters, bc_type="natural")
    result = spline(np.clip(log_wavelength, log_anchor[0], log_anchor[-1]))
    result[log_wavelength < log_anchor[0]] = parameters[0]
    result[log_wavelength > log_anchor[-1]] = parameters[-1]
    return result


def _correction_basis(
    wavelength_angstrom: np.ndarray,
    anchor_wavelength_angstrom: np.ndarray,
) -> np.ndarray:
    basis = np.empty(
        (len(wavelength_angstrom), len(anchor_wavelength_angstrom)),
        dtype=float,
    )
    for index in range(len(anchor_wavelength_angstrom)):
        parameter = np.zeros(len(anchor_wavelength_angstrom), dtype=float)
        parameter[index] = 1.0
        basis[:, index] = _evaluate_log_correction(
            wavelength_angstrom,
            anchor_wavelength_angstrom,
            parameter,
        )
    return basis


def _combined_magnitude_uncertainty(
    point: PhotometryPoint,
    synthetic: SyntheticPhotometry,
) -> float | None:
    variance = 0.0
    has_uncertainty = False

    if point.uncertainty is not None:
        variance += point.uncertainty**2
        has_uncertainty = True
    if synthetic.uncertainty is not None:
        variance += synthetic.uncertainty**2
        has_uncertainty = True

    if not has_uncertainty:
        return None
    return float(np.sqrt(variance))


def _parameter_covariance(
    optimizer: OptimizeResult,
    spectrum: Spectrum,
    photometry: Sequence[PhotometryPoint],
) -> np.ndarray | None:
    has_absolute_uncertainties = all(
        point.uncertainty is not None or spectrum.uncertainty is not None
        for point in photometry
    )
    if not has_absolute_uncertainties:
        return None

    jacobian = np.asarray(optimizer.jac, dtype=float)
    information = jacobian.T @ jacobian
    if np.linalg.matrix_rank(information) < information.shape[0]:
        return None
    return np.linalg.inv(information)


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
            return _normalize_svo_detector_type(
                table["DetectorType"][matches[0]]
            )

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

    # SVO Photometry Data Model convention: 0 = photon counter, 1 = energy counter.
    if numeric == 0:
        return "photon"
    if numeric == 1:
        return "energy"

    raise ValueError(f"unrecognized SVO DetectorType value: {value!r}")
