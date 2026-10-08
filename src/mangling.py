from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping

import astropy.units as u
from astropy.nddata import InverseVariance, StdDevUncertainty, VarianceUncertainty
from astropy.table import Table
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import OptimizeResult, least_squares
from specutils import Spectrum

from .bandpasses import Bandpass
from .photometry import SyntheticPhotometry, synthetic_ab_magnitude


@dataclass(frozen=True)
class MangleResult:
    """Result of a photometry-constrained spectral mangling fit."""

    spectrum: Spectrum
    correction: np.ndarray
    correction_uncertainty: np.ndarray | None
    anchor_wavelengths: u.Quantity
    log_correction_parameters: np.ndarray
    parameter_covariance: np.ndarray | None
    photometry: Table
    bandpasses: dict[str, Bandpass]
    optimizer: OptimizeResult


@dataclass(frozen=True)
class _PhotometryRow:
    band: str
    mag: float
    mag_err: float | None


def mangle(
    spectrum: Spectrum,
    photometry: Table,
    *,
    bandpasses: Mapping[str, Bandpass] | None = None,
    min_coverage: float = 0.98,
    smoothness: float = 0.0,
    max_nfev: int | None = None,
) -> MangleResult:
    """
    Mangle a spectrum to match contemporaneous AB photometry.

    ``photometry`` must contain ``band`` and ``mag`` columns and may contain a
    ``mag_err`` column. Each ``band`` value is interpreted as an SVO filter ID
    unless a matching :class:`~specmangle.Bandpass` is supplied through
    ``bandpasses``. All magnitudes are interpreted as AB magnitudes.

    The positive multiplicative correction is represented by a natural cubic
    spline in log wavelength and log flux scale, with one free anchor per
    unique passband. Outside the bluest and reddest anchors the correction is
    held constant. Repeated measurements in a passband are supported.

    Photometric and synthetic-photometry uncertainties, when available, are
    combined in quadrature when weighting residuals. The uncertainty attached
    to ``result.spectrum`` contains only the input spectral uncertainty scaled
    by the best-fit correction. Uncertainty in the mangling function itself is
    returned separately in ``result.correction_uncertainty`` because it is
    correlated with wavelength.
    """
    rows = _photometry_rows(photometry)
    if smoothness < 0.0 or not np.isfinite(smoothness):
        raise ValueError("smoothness must be finite and non-negative")
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be in the interval (0, 1]")
    if not isinstance(spectrum, Spectrum):
        raise TypeError("spectrum must be a specutils.Spectrum")
    if spectrum.flux.ndim != 1:
        raise ValueError("only one-dimensional Spectrum objects are supported")

    supplied_bandpasses = {} if bandpasses is None else dict(bandpasses)
    filter_ids = list(dict.fromkeys(row.band for row in rows))

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
        for row in rows:
            if row.band != filter_id:
                continue
            estimates.append(0.4 * np.log(10.0) * (synthetic.magnitude - row.mag))
            sigma = _combined_magnitude_uncertainty(row, synthetic)
            weights.append(1.0 if sigma is None else 1.0 / sigma**2)
        initial_parameters[parameter_index[filter_id]] = np.average(
            estimates,
            weights=weights,
        )

    spectrum_wave = spectrum.wavelength.to_value(u.AA)
    if len(spectrum_wave) != len(spectrum.flux):
        raise ValueError("spectral axis must specify one coordinate per flux sample")
    if np.any(~np.isfinite(spectrum_wave)) or np.any(spectrum_wave <= 0.0):
        raise ValueError("spectrum wavelengths must be positive and finite")

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
        for row in rows:
            synthetic = synthetic_by_filter[row.band]
            sigma = _combined_magnitude_uncertainty(row, synthetic)
            scale = 1.0 if sigma is None else sigma
            residuals.append((synthetic.magnitude - row.mag) / scale)

        if smoothness > 0.0 and len(parameters) >= 3:
            log_anchor = np.log(anchor_wavelength_values)
            slopes = np.diff(parameters) / np.diff(log_anchor)
            residuals.extend(np.sqrt(smoothness) * np.diff(slopes))

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

    parameter_covariance = _parameter_covariance(optimizer, spectrum, rows)
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

    diagnostics = photometry.copy(copy_data=True)
    diagnostics["synthetic_mag"] = [
        final_synthetic_by_filter[row.band].magnitude for row in rows
    ]
    diagnostics["synthetic_mag_err"] = [
        np.nan
        if final_synthetic_by_filter[row.band].uncertainty is None
        else final_synthetic_by_filter[row.band].uncertainty
        for row in rows
    ]
    diagnostics["residual"] = diagnostics["synthetic_mag"] - np.array(
        [row.mag for row in rows]
    )

    return MangleResult(
        spectrum=corrected_spectrum,
        correction=correction,
        correction_uncertainty=correction_uncertainty,
        anchor_wavelengths=anchor_wavelength_values * u.AA,
        log_correction_parameters=np.array(optimizer.x, copy=True),
        parameter_covariance=parameter_covariance,
        photometry=diagnostics,
        bandpasses=loaded_bandpasses,
        optimizer=optimizer,
    )


def _photometry_rows(photometry: Table) -> list[_PhotometryRow]:
    if not isinstance(photometry, Table):
        raise TypeError("photometry must be an astropy.table.Table")
    if len(photometry) == 0:
        raise ValueError("at least one photometry measurement is required")

    missing = {"band", "mag"} - set(photometry.colnames)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"photometry is missing required column(s): {missing_text}")

    rows = []
    for index in range(len(photometry)):
        band_value = photometry["band"][index]
        if np.ma.is_masked(band_value):
            raise ValueError(f"photometry band is masked at row {index}")
        band = str(band_value).strip()
        if not band:
            raise ValueError(f"photometry band is empty at row {index}")

        mag_value = photometry["mag"][index]
        if np.ma.is_masked(mag_value):
            raise ValueError(f"photometry mag is masked at row {index}")
        if isinstance(mag_value, u.Quantity):
            mag_value = mag_value.value
        mag = float(mag_value)
        if not np.isfinite(mag):
            raise ValueError(f"photometry mag must be finite at row {index}")

        mag_err = None
        if "mag_err" in photometry.colnames:
            mag_err_value = photometry["mag_err"][index]
            if not np.ma.is_masked(mag_err_value):
                if isinstance(mag_err_value, u.Quantity):
                    mag_err_value = mag_err_value.value
                mag_err = float(mag_err_value)
                if not np.isfinite(mag_err) or mag_err <= 0.0:
                    raise ValueError(
                        f"photometry mag_err must be positive and finite at row {index}"
                    )

        rows.append(_PhotometryRow(band=band, mag=mag, mag_err=mag_err))

    return rows


def _apply_correction(spectrum: Spectrum, correction: np.ndarray) -> Spectrum:
    if correction.shape != spectrum.flux.shape:
        raise ValueError("correction must match the one-dimensional spectrum shape")

    uncertainty = _scale_uncertainty(spectrum.uncertainty, correction)
    mask = None if spectrum.mask is None else np.array(spectrum.mask, copy=True)

    return Spectrum(
        spectral_axis=spectrum.spectral_axis,
        flux=spectrum.flux * correction,
        uncertainty=uncertainty,
        mask=mask,
        meta=deepcopy(spectrum.meta),
    )


def _scale_uncertainty(uncertainty, correction: np.ndarray):
    if uncertainty is None:
        return None

    unit = uncertainty.unit
    if isinstance(uncertainty, StdDevUncertainty):
        return StdDevUncertainty(uncertainty.array * correction, unit=unit)
    if isinstance(uncertainty, VarianceUncertainty):
        return VarianceUncertainty(uncertainty.array * correction**2, unit=unit)
    if isinstance(uncertainty, InverseVariance):
        return InverseVariance(uncertainty.array / correction**2, unit=unit)

    raise TypeError(
        "spectrum uncertainty must be StdDevUncertainty, VarianceUncertainty, "
        "or InverseVariance"
    )


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
    row: _PhotometryRow,
    synthetic: SyntheticPhotometry,
) -> float | None:
    variance = 0.0
    has_uncertainty = False

    if row.mag_err is not None:
        variance += row.mag_err**2
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
    rows: list[_PhotometryRow],
) -> np.ndarray | None:
    has_absolute_uncertainties = all(
        row.mag_err is not None or spectrum.uncertainty is not None
        for row in rows
    )
    if not has_absolute_uncertainties:
        return None

    jacobian = np.asarray(optimizer.jac, dtype=float)
    information = jacobian.T @ jacobian
    if np.linalg.matrix_rank(information) < information.shape[0]:
        return None
    return np.linalg.inv(information)
