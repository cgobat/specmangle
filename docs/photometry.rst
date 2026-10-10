Synthetic photometry
====================

AB magnitudes from an SVO passband
----------------------------------

The public :func:`specmangle.spectrum_to_magnitude` helper computes synthetic AB
photometry directly from a :class:`specutils.Spectrum` and an SVO FPS filter
identifier:

.. code-block:: python

    from specmangle import spectrum_to_magnitude

    mag = spectrum_to_magnitude(spectrum, "SLOAN/SDSS.r")

If the input spectrum includes uncertainty, the function returns
``(mag, mag_err)``. The magnitude carries ``u.ABmag`` and the uncertainty carries
``u.mag``.

For lower-level access to the calculated coverage and propagated uncertainty,
use :func:`specmangle.synthetic_ab_magnitude` with a
:class:`~specmangle.Bandpass`:

.. code-block:: python

    from specmangle import Bandpass, synthetic_ab_magnitude

    bandpass = Bandpass.from_svo("SLOAN/SDSS.r")
    synthetic = synthetic_ab_magnitude(spectrum, bandpass)

    print(synthetic.magnitude)
    print(synthetic.uncertainty)
    print(synthetic.coverage)

SVO response curves
-------------------

:meth:`specmangle.Bandpass.from_svo` retrieves the response profile with
:class:`astroquery.svo_fps.SvoFps`. Astroquery's own query caching is used.
``specmangle`` also reads the SVO ``DetectorType`` metadata so that photon- and
energy-counting systems receive the appropriate wavelength weighting in the
synthetic-photometry integral.

The same treatment applies when an SVO profile represents an effective-area
curve rather than a dimensionless filter transmission: for AB magnitudes the
absolute response normalization cancels between the source and AB-reference
integrals.

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
``coverage`` as an attribute.
