import numpy as np
import pytest
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from specutils import Spectrum

from specmangle import Bandpass, spectrum_to_magnitude
from specmangle.bandpasses import _normalize_svo_detector_type


def make_bandpass():
    return Bandpass(
        filter_id="test/filter",
        wavelength=np.array([400.0, 500.0, 600.0]) * u.nm,
        transmission=np.array([0.0, 1.0, 0.0]),
        detector_type="energy",
    )


def test_spectrum_to_magnitude_accepts_svo_id(monkeypatch):
    bandpass = make_bandpass()
    monkeypatch.setattr(Bandpass, "from_svo", lambda filter_id: bandpass)
    spectrum = Spectrum(
        spectral_axis=bandpass.wavelength,
        flux=np.full(3, (0.0 * u.ABmag).to_value(u.Jy)) * u.Jy,
    )

    magnitude = spectrum_to_magnitude(spectrum, "test/filter")

    assert magnitude.to_value(u.ABmag) == pytest.approx(0.0, abs=1e-12)


def test_spectrum_to_magnitude_returns_uncertainty():
    bandpass = make_bandpass()
    spectrum = Spectrum(
        spectral_axis=bandpass.wavelength,
        flux=np.full(3, (0.0 * u.ABmag).to_value(u.Jy)) * u.Jy,
        uncertainty=StdDevUncertainty(np.full(3, 36.31), unit=u.Jy),
    )

    magnitude, magnitude_uncertainty = spectrum_to_magnitude(spectrum, bandpass)

    assert magnitude.to_value(u.ABmag) == pytest.approx(0.0, abs=1e-12)
    assert magnitude_uncertainty.to_value(u.mag) > 0.0


def test_svo_numeric_detector_type_convention():
    assert _normalize_svo_detector_type(0) == "energy"
    assert _normalize_svo_detector_type(1) == "photon"
