Synthetic photometry
====================

AB and Vega magnitudes from an SVO passband
-------------------------------------------

The public :func:`specmangle.spectrum_to_magnitude` helper computes synthetic
photometry directly from a :class:`specutils.Spectrum` and an SVO FPS filter
identifier. AB is the default magnitude system:

.. code-block:: python

    from specmangle import spectrum_to_magnitude

    ab_mag = spectrum_to_magnitude(spectrum, "SLOAN/SDSS.r")
    vega_mag = spectrum_to_magnitude(
        spectrum,
        "Generic/Johnson.V",
        system="Vega",
    )

If the input spectrum includes uncertainty, the function returns
``(mag, mag_err)``. AB magnitudes carry ``u.ABmag``; Vega magnitudes and all
magnitude uncertainties carry ``u.mag``.

For lower-level access to the calculated coverage and propagated uncertainty,
use :func:`specmangle.synthetic_magnitude` with a
:class:`~specmangle.Bandpass`:

.. code-block:: python

    from specmangle import Bandpass, synthetic_magnitude

    bandpass = Bandpass.from_svo("SLOAN/SDSS.r")
    synthetic = synthetic_magnitude(spectrum, bandpass, system="AB")

    print(synthetic.magnitude)
    print(synthetic.uncertainty)
    print(synthetic.coverage)

:func:`specmangle.synthetic_ab_magnitude` remains available as an AB-only
convenience wrapper/backwards-compatibility shim.

Vega reference spectrum
-----------------------

Vega magnitudes are defined by integrating the CALSPEC Vega spectrum through
the same response curve as the source and assigning that reference signal a
magnitude of zero. ``specmangle`` uses the versioned CALSPEC
``alpha_lyr_stis_012.fits`` spectrum. The file is downloaded on first use and
cached using Astropy's data cache, so no additional package dependency is
required.

The native CALSPEC wavelength samples are included in the reference integration
rather than reducing Vega to a precomputed scalar zero point. This is important
for custom response curves and for passbands modified with additional
components such as atmospheric transmission.

SVO response curves
-------------------

:meth:`specmangle.Bandpass.from_svo` retrieves the response profile with
:class:`astroquery.svo_fps.SvoFps`. Astroquery's own query caching is used.
``specmangle`` also reads the SVO ``DetectorType`` metadata so that photon- and
energy-counting systems receive the appropriate wavelength weighting in the
synthetic-photometry integral.

The same treatment applies when an SVO profile represents an effective-area
curve rather than a dimensionless filter transmission: for both AB and Vega
magnitudes the absolute response normalization cancels between the source and
reference integrals.

Custom bandpasses
-----------------

A custom response curve can be supplied directly:

.. code-block:: python

    from specmangle import Bandpass

    bandpass = Bandpass(
        filter_id="my/filter",
        wavelength=filter_wavelength * u.AA,
        transmission=filter_response,
        detector_type="photon",
    )

Pass custom bandpasses to :func:`specmangle.mangle` using a mapping whose keys
match the values in the photometry table's ``band`` column:

.. code-block:: python

    result = mangle(
        spectrum,
        photometry,
        bandpasses={"my/filter": bandpass},
    )

Coverage errors
---------------

Direct synthetic-photometry calls are strict. If the spectrum does not satisfy
``min_coverage``, :class:`specmangle.InsufficientCoverageError` is raised. This
exception is a :class:`ValueError` subclass and exposes the calculated
``coverage`` as an attribute. Coverage is weighted by the zero-magnitude
reference spectrum for the requested magnitude system.
