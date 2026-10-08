import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table
from specutils import Spectrum

from specmangle import Bandpass, mangle


def test_mangle_recovers_constant_flux_scale_from_two_bandpasses():
    wavelength = np.arange(400.0, 801.0, 50.0) * u.nm
    spectrum = Spectrum(
        spectral_axis=wavelength,
        flux=np.full(len(wavelength), 3631.0) * u.Jy,
    )
    bandpasses = {
        "blue": Bandpass(
            "test/blue",
            np.array([400.0, 450.0, 500.0, 550.0, 600.0]) * u.nm,
            np.array([0.0, 1.0, 1.0, 1.0, 0.0]),
            "photon",
        ),
        "red": Bandpass(
            "test/red",
            np.array([600.0, 650.0, 700.0, 750.0, 800.0]) * u.nm,
            np.array([0.0, 1.0, 1.0, 1.0, 0.0]),
            "photon",
        ),
    }
    target_magnitude = -0.5

    result = mangle(
        spectrum,
        Table({"band": ["blue", "red"], "mag": [target_magnitude] * 2}),
        bandpasses=bandpasses,
    )

    ab_zero_jy = (0.0 * u.ABmag).to_value(u.Jy)
    initial_magnitude = -2.5 * np.log10(3631.0 / ab_zero_jy)
    expected_scale = 10.0 ** (0.4 * (initial_magnitude - target_magnitude))
    assert np.allclose(result.correction, expected_scale, rtol=1e-9, atol=1e-12)
    assert np.allclose(
        result.log_correction_parameters,
        np.log(expected_scale),
        rtol=1e-9,
        atol=1e-12,
    )
    assert np.allclose(result.photometry["synthetic_mag"], target_magnitude, atol=1e-10)
    assert np.allclose(
        result.spectrum.flux.to_value(u.Jy),
        3631.0 * expected_scale,
        rtol=1e-9,
    )


def test_mangle_skips_passbands_without_sufficient_coverage():
    wavelength = np.arange(400.0, 801.0, 50.0) * u.nm
    spectrum = Spectrum(
        spectral_axis=wavelength,
        flux=np.full(len(wavelength), 3631.0) * u.Jy,
    )
    bandpasses = {
        "covered": Bandpass(
            "test/covered",
            np.array([400.0, 450.0, 500.0, 550.0, 600.0]) * u.nm,
            np.array([0.0, 1.0, 1.0, 1.0, 0.0]),
            "photon",
        ),
        "partial": Bandpass(
            "test/partial",
            np.array([300.0, 350.0, 400.0, 450.0, 500.0]) * u.nm,
            np.array([0.0, 1.0, 1.0, 1.0, 0.0]),
            "photon",
        ),
    }

    result = mangle(
        spectrum,
        Table({"band": ["partial", "covered"], "mag": [0.0, 0.0]}),
        bandpasses=bandpasses,
        min_coverage=0.95,
    )

    assert set(result.bandpasses) == {"covered"}
    assert np.isnan(result.photometry["synthetic_mag"][0])
    assert np.isnan(result.photometry["residual"][0])
    assert np.isfinite(result.photometry["synthetic_mag"][1])
    assert np.isfinite(result.photometry["residual"][1])


def test_mangle_raises_if_no_passband_has_sufficient_coverage():
    wavelength = np.arange(400.0, 801.0, 50.0) * u.nm
    spectrum = Spectrum(
        spectral_axis=wavelength,
        flux=np.full(len(wavelength), 3631.0) * u.Jy,
    )
    bandpasses = {
        "uv": Bandpass(
            "test/uv",
            np.array([200.0, 250.0, 300.0, 350.0]) * u.nm,
            np.array([0.0, 1.0, 1.0, 0.0]),
            "photon",
        ),
    }

    with pytest.raises(
        ValueError,
        match="no photometric passbands have sufficient spectral coverage",
    ):
        mangle(
            spectrum,
            Table({"band": ["uv"], "mag": [0.0]}),
            bandpasses=bandpasses,
        )
