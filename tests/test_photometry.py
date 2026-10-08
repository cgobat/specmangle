import numpy as np
import pytest
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from scipy.integrate import trapezoid
from specutils import Spectrum

from specmangle import Bandpass, synthetic_ab_magnitude


WAVELENGTH = np.array([400.0, 450.0, 500.0, 550.0, 600.0]) * u.nm
TRANSMISSION = np.array([0.0, 0.5, 1.0, 0.5, 0.0])


def make_bandpass(detector_type):
    return Bandpass(
        filter_id=f"test/{detector_type}",
        wavelength=WAVELENGTH,
        transmission=TRANSMISSION,
        detector_type=detector_type,
    )


@pytest.mark.parametrize("detector_type", ["photon", "energy"])
@pytest.mark.parametrize("scale", [1.0, 0.1])
def test_scaled_constant_ab_reference_matches_known_magnitude(
    detector_type,
    scale,
):
    spectrum = Spectrum(
        spectral_axis=WAVELENGTH,
        flux=np.full(len(WAVELENGTH), 3631.0 * scale) * u.Jy,
    )

    result = synthetic_ab_magnitude(spectrum, make_bandpass(detector_type))

    ab_zero_jy = (0.0 * u.ABmag).to_value(u.Jy)
    expected = -2.5 * np.log10(3631.0 * scale / ab_zero_jy)
    assert result.magnitude == pytest.approx(expected, abs=1e-12)
    assert result.uncertainty is None
    assert result.coverage == pytest.approx(1.0)


@pytest.mark.parametrize("detector_type", ["photon", "energy"])
def test_ab_magnitude_matches_independent_detector_weighted_reference(detector_type):
    flux_density = np.array([1.0, 1.1, 0.9, 1.2, 1.0]) * 3631.0 * u.Jy
    spectrum = Spectrum(spectral_axis=WAVELENGTH, flux=flux_density)
    bandpass = make_bandpass(detector_type)

    result = synthetic_ab_magnitude(spectrum, bandpass)

    wavelength_angstrom = WAVELENGTH.to_value(u.AA)
    flux_flam = flux_density.to(
        u.erg / (u.s * u.cm**2 * u.AA),
        equivalencies=u.spectral_density(WAVELENGTH),
    ).value
    reference_flam = (0.0 * u.ABmag).to(
        u.erg / (u.s * u.cm**2 * u.AA),
        equivalencies=u.spectral_density(WAVELENGTH),
    ).value
    detector_weight = (
        wavelength_angstrom
        if detector_type == "photon"
        else np.ones_like(wavelength_angstrom)
    )
    expected = -2.5 * np.log10(
        trapezoid(flux_flam * TRANSMISSION * detector_weight, wavelength_angstrom)
        / trapezoid(reference_flam * TRANSMISSION * detector_weight, wavelength_angstrom)
    )

    assert result.magnitude == pytest.approx(expected, abs=1e-12)


def test_synthetic_magnitude_uncertainty_propagates_converted_flux_uncertainty():
    wavelength = np.array([400.0, 500.0, 600.0]) * u.nm
    transmission = np.array([0.0, 1.0, 0.0])
    bandpass = Bandpass("test/uncertainty", wavelength, transmission, "energy")
    spectrum = Spectrum(
        spectral_axis=wavelength,
        flux=np.full(3, 3631.0) * u.Jy,
        uncertainty=StdDevUncertainty(np.full(3, 36.31), unit=u.Jy),
    )

    result = synthetic_ab_magnitude(spectrum, bandpass)

    expected_uncertainty = 2.5 / np.log(10.0) * 0.01
    assert result.magnitude == pytest.approx(0.0, abs=1e-4)
    assert result.uncertainty == pytest.approx(expected_uncertainty, abs=1e-12)
