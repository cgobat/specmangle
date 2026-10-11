import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

from specmangle import Bandpass


def make_bandpass(*, components=()):
    return Bandpass(
        filter_id="test/filter",
        wavelength=np.array([400.0, 500.0, 600.0]) * u.nm,
        transmission=np.array([0.0, 1.0, 0.0]),
        detector_type="photon",
        components=components,
    )


def test_includes_atmosphere_from_components():
    assert not make_bandpass(components=("Filter", "Instrument")).includes_atmosphere
    assert make_bandpass(components=("Filter", "Atmosphere")).includes_atmosphere


def test_with_atmosphere_combines_response_on_union_grid():
    bandpass = make_bandpass()
    atmosphere_wavelength = np.array([400.0, 450.0, 500.0, 550.0, 600.0]) * u.nm
    atmosphere_transmission = np.array([1.0, 0.8, 0.5, 0.8, 1.0])

    combined = bandpass.with_atmosphere(
        atmosphere_wavelength,
        atmosphere_transmission,
    )

    assert combined.filter_id == bandpass.filter_id
    assert combined.detector_type == bandpass.detector_type
    assert combined.includes_atmosphere
    assert combined.components == ("Atmosphere",)
    np.testing.assert_allclose(
        combined.wavelength.to_value(u.nm),
        atmosphere_wavelength.value,
    )
    np.testing.assert_allclose(
        combined.transmission,
        np.array([0.0, 0.4, 0.5, 0.4, 0.0]),
    )


def test_with_atmosphere_preserves_existing_components():
    combined = make_bandpass(components=("Filter", "Instrument")).with_atmosphere(
        np.array([400.0, 500.0, 600.0]) * u.nm,
        np.array([0.8, 0.8, 0.8]),
    )

    assert combined.components == ("Filter", "Instrument", "Atmosphere")


def test_with_atmosphere_rejects_existing_atmosphere():
    bandpass = make_bandpass(components=("Filter", "Atmosphere"))

    with pytest.raises(ValueError, match="already includes atmosphere"):
        bandpass.with_atmosphere(
            np.array([400.0, 500.0, 600.0]) * u.nm,
            np.ones(3),
        )


def test_with_atmosphere_requires_full_bandpass_support():
    bandpass = make_bandpass()

    with pytest.raises(ValueError, match="cover the nonzero bandpass response"):
        bandpass.with_atmosphere(
            np.array([450.0, 500.0, 550.0]) * u.nm,
            np.ones(3),
        )


def test_from_svo_records_components(monkeypatch):
    from astroquery.svo_fps import SvoFps

    table = Table(
        {
            "Wavelength": np.array([4000.0, 5000.0, 6000.0]) * u.AA,
            "Transmission": [0.0, 1.0, 0.0],
        }
    )
    monkeypatch.setattr(SvoFps, "get_transmission_data", lambda filter_id: table)
    monkeypatch.setattr(
        SvoFps,
        "get_filter_metadata",
        lambda filter_id: {
            "DetectorType": 1,
            "components": "Filter + Instrument + Atmosphere",
        },
    )

    bandpass = Bandpass.from_svo("TEST/Test.g")

    assert bandpass.detector_type == "photon"
    assert bandpass.components == ("Filter", "Instrument", "Atmosphere")
    assert bandpass.includes_atmosphere
