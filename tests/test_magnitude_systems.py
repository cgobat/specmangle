import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table
from specutils import Spectrum

import specmangle.photometry as photometry_module
from specmangle import Bandpass, mangle, spectrum_to_magnitude, synthetic_magnitude


WAVELENGTH = np.array([4000.0, 4500.0, 5000.0, 5500.0, 6000.0]) * u.AA
TRANSMISSION = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
VEGA_FLUX = np.array([3.5, 3.1, 2.8, 2.5, 2.3]) * 1e-9


def make_bandpass(detector_type="photon"):
    return Bandpass(
        filter_id="test/filter",
        wavelength=WAVELENGTH,
        transmission=TRANSMISSION,
        detector_type=detector_type,
    )


@pytest.fixture
def mock_vega(monkeypatch):
    monkeypatch.setattr(
        photometry_module,
        "_vega_reference_spectrum",
        lambda: (WAVELENGTH.to_value(u.AA), VEGA_FLUX),
    )


@pytest.mark.parametrize("detector_type", ["photon", "energy"])
@pytest.mark.parametrize("scale", [1.0, 0.1])
def test_scaled_vega_reference_matches_known_magnitude(
    mock_vega,
    detector_type,
    scale,
):
    spectrum = Spectrum(
        spectral_axis=WAVELENGTH,
        flux=scale * VEGA_FLUX * photometry_module.FLAM,
    )

    result = synthetic_magnitude(
        spectrum,
        make_bandpass(detector_type),
        system="Vega",
    )

    assert result.magnitude == pytest.approx(-2.5 * np.log10(scale), abs=1e-12)
    assert result.coverage == pytest.approx(1.0)


def test_spectrum_to_magnitude_returns_vega_magnitude_unit(mock_vega):
    spectrum = Spectrum(
        spectral_axis=WAVELENGTH,
        flux=VEGA_FLUX * photometry_module.FLAM,
    )

    magnitude = spectrum_to_magnitude(
        spectrum,
        make_bandpass(),
        system="Vega",
    )

    assert magnitude.to_value(u.mag) == pytest.approx(0.0, abs=1e-12)


def test_mangle_defaults_missing_system_values_to_ab():
    spectrum = Spectrum(
        spectral_axis=WAVELENGTH,
        flux=np.full(len(WAVELENGTH), (0.0 * u.ABmag).to_value(u.Jy)) * u.Jy,
    )
    photometry = Table(
        {
            "band": ["test", "test", "test"],
            "mag": [0.0, 0.0, 0.0],
            "system": np.array([np.nan, "nan", "ab"], dtype=object),
        }
    )

    result = mangle(
        spectrum,
        photometry,
        bandpasses={"test": make_bandpass()},
    )

    assert list(result.photometry["system"]) == ["AB", "AB", "AB"]
    assert np.allclose(result.correction, 1.0, atol=1e-12)


def test_mangle_supports_mixed_systems_in_same_bandpass(mock_vega):
    spectrum = Spectrum(
        spectral_axis=WAVELENGTH,
        flux=VEGA_FLUX * photometry_module.FLAM,
    )
    bandpass = make_bandpass()
    ab_magnitude = synthetic_magnitude(spectrum, bandpass, system="AB").magnitude
    vega_magnitude = synthetic_magnitude(spectrum, bandpass, system="Vega").magnitude

    result = mangle(
        spectrum,
        Table(
            {
                "band": ["test", "test"],
                "mag": [ab_magnitude, vega_magnitude],
                "system": ["AB", "Vega"],
            }
        ),
        bandpasses={"test": bandpass},
    )

    assert np.allclose(result.correction, 1.0, atol=1e-12)
    assert np.allclose(
        result.photometry["synthetic_mag"],
        [ab_magnitude, vega_magnitude],
        atol=1e-12,
    )


def test_mangle_rejects_unknown_magnitude_system():
    spectrum = Spectrum(
        spectral_axis=WAVELENGTH,
        flux=np.full(len(WAVELENGTH), 3631.0) * u.Jy,
    )

    with pytest.raises(ValueError, match='system must be "AB" or "Vega"'):
        mangle(
            spectrum,
            Table({"band": ["test"], "mag": [0.0], "system": ["ST"]}),
            bandpasses={"test": make_bandpass()},
        )
