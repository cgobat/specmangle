import numpy as np
import pytest
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from scipy.integrate import trapezoid
from specutils import Spectrum

from specmangle import Bandpass, InsufficientCoverageError, synthetic_ab_magnitude


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


def test_isolated_masked_sample_reduces_coverage():
    spectrum = Spectrum(
        spectral_axis=np.array([4000.0, 4500.0, 5000.0, 5500.0, 6000.0]) * u.AA,
        flux=np.full(5, 3631.0) * u.Jy,
        mask=[False, False, True, False, False],
    )

    result = synthetic_ab_magnitude(
        spectrum,
        make_bandpass("energy"),
        min_coverage=0.5,
    )

    assert result.coverage == pytest.approx(0.567, abs=0.001)


def test_substantial_masked_internal_gap_fails_coverage_threshold():
    spectrum = Spectrum(
        spectral_axis=np.array([4000.0, 4500.0, 5000.0, 5500.0, 6000.0]) * u.AA,
        flux=np.full(5, 3631.0) * u.Jy,
        mask=[False, True, True, True, False],
    )

    with pytest.raises(ValueError, match="covers only 0.066 of passband"):
        synthetic_ab_magnitude(spectrum, make_bandpass("energy"))


COARSE_WAVELENGTH = np.array([4000.0, 4100.0, 4900.0, 5000.0]) * u.AA
COARSE_TRANSMISSION = np.array([0.0, 1.0, 1.0, 0.0])


def make_coarse_bandpass(detector_type):
    return Bandpass(
        "test/coarse", COARSE_WAVELENGTH, COARSE_TRANSMISSION, detector_type
    )


@pytest.mark.parametrize("detector_type", ["photon", "energy"])
@pytest.mark.parametrize("feature_amplitude", [10.0, -0.9])
def test_coarse_passband_resolves_narrow_spectral_features(
    detector_type,
    feature_amplitude,
):
    wavelength = np.arange(4000.0, 5001.0) * u.AA
    bandpass = make_coarse_bandpass(detector_type)

    def profile(wave):
        return 1.0 + feature_amplitude * np.exp(-0.5 * ((wave - 4500.0) / 20.0) ** 2)

    spectrum = Spectrum(
        spectral_axis=wavelength,
        flux=profile(wavelength.value) * (0.0 * u.ABmag).to_value(u.Jy) * u.Jy,
    )
    result = synthetic_ab_magnitude(spectrum, bandpass)

    fine_wave = np.linspace(4000.0, 5000.0, 10001)
    transmission = np.interp(fine_wave, COARSE_WAVELENGTH.value, COARSE_TRANSMISSION)
    # In f_lambda, a flat f_nu reference contributes lambda^-2; a photon
    # detector introduces one additional factor of lambda.
    reference_kernel = transmission / fine_wave ** (
        1 if detector_type == "photon" else 2
    )
    expected = -2.5 * np.log10(
        trapezoid(profile(fine_wave) * reference_kernel, fine_wave)
        / trapezoid(reference_kernel, fine_wave)
    )

    assert result.magnitude == pytest.approx(expected, abs=1e-4)
    assert result.coverage == pytest.approx(1.0)


@pytest.mark.parametrize("detector_type", ["photon", "energy"])
@pytest.mark.parametrize("slope", [0.0, 0.8])
def test_coarse_passband_smooth_and_ab_reference(detector_type, slope):
    wavelength = np.arange(4000.0, 5001.0, 10.0) * u.AA
    bandpass = make_coarse_bandpass(detector_type)
    spectrum = Spectrum(
        spectral_axis=wavelength,
        flux=(
            (wavelength.value / 4500.0) ** slope
            * (0.0 * u.ABmag).to_value(u.Jy)
            * u.Jy
        ),
    )

    result = synthetic_ab_magnitude(spectrum, bandpass)

    fine_wave = np.linspace(4000.0, 5000.0, 10001)
    transmission = np.interp(fine_wave, COARSE_WAVELENGTH.value, COARSE_TRANSMISSION)
    reference_kernel = transmission / fine_wave ** (
        1 if detector_type == "photon" else 2
    )
    expected = -2.5 * np.log10(
        trapezoid((fine_wave / 4500.0) ** slope * reference_kernel, fine_wave)
        / trapezoid(reference_kernel, fine_wave)
    )
    assert result.magnitude == pytest.approx(expected, abs=1e-4)
    assert result.coverage == pytest.approx(1.0)


def test_coarse_bandpass_masked_feature_preserves_coverage():
    wavelength = np.arange(4000.0, 5001.0) * u.AA
    bandpass = make_coarse_bandpass("photon")
    spectrum = Spectrum(
        spectral_axis=wavelength,
        flux=np.full(len(wavelength), 3631.0) * u.Jy,
        mask=(wavelength.value >= 4450.0) & (wavelength.value <= 4550.0),
    )

    with pytest.raises(InsufficientCoverageError) as exc:
        synthetic_ab_magnitude(spectrum, bandpass)

    fine_wave = np.linspace(4000.0, 5000.0, 10001)
    transmission = np.interp(
        fine_wave, bandpass.wavelength.value, bandpass.transmission
    )
    reference = trapezoid(transmission / fine_wave, fine_wave)
    gap_wave = np.linspace(4449.5, 4550.5, 1011)
    missing = trapezoid(1.0 / gap_wave, gap_wave)
    assert exc.value.coverage == pytest.approx(1.0 - missing / reference, abs=1e-4)


def test_coarse_bandpass_uncertainty_uses_spectral_samples():
    wavelength = np.arange(4000.0, 5001.0, 10.0) * u.AA
    bandpass = make_coarse_bandpass("energy")
    zero_jy = (0.0 * u.ABmag).to_value(u.Jy)
    profile = 1.0 + 10.0 * np.exp(-0.5 * ((wavelength.value - 4500.0) / 20.0) ** 2)
    spectrum = Spectrum(
        spectral_axis=wavelength,
        flux=zero_jy * profile * u.Jy,
        uncertainty=StdDevUncertainty(
            np.full(len(wavelength), 0.02 * zero_jy), unit=u.Jy
        ),
    )

    result = synthetic_ab_magnitude(spectrum, bandpass)

    wave = wavelength.value
    transmission = np.interp(wave, bandpass.wavelength.value, bandpass.transmission)
    integration_weights = np.empty(len(wave))
    integration_weights[0] = (wave[1] - wave[0]) / 2
    integration_weights[-1] = (wave[-1] - wave[-2]) / 2
    integration_weights[1:-1] = (wave[2:] - wave[:-2]) / 2
    coefficients = integration_weights * transmission / wave**2
    expected_uncertainty = (
        2.5 / np.log(10.0)
        * np.sqrt(np.sum((coefficients * 0.02) ** 2))
        / np.sum(coefficients * profile)
    )
    assert result.uncertainty == pytest.approx(expected_uncertainty, rel=1e-12)
