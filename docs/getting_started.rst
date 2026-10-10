Getting started
===============

Installation
------------

Install the package from a local checkout with::

    python -m pip install .

For development, including the documentation dependencies, use::

    python -m pip install -e ".[test,docs]"

A basic mangling fit
--------------------

The spectrum is supplied as a :class:`specutils.Spectrum`. Flux uncertainty can
be attached with the standard Astropy uncertainty classes, such as
:class:`astropy.nddata.StdDevUncertainty`.

.. code-block:: python

    import astropy.units as u
    from astropy.nddata import StdDevUncertainty
    from astropy.table import Table
    from specutils import Spectrum

    from specmangle import mangle

    flux_unit = u.erg / u.s / u.cm**2 / u.AA
    spectrum = Spectrum(
        spectral_axis=wavelength * u.AA,
        flux=flux * flux_unit,
        uncertainty=StdDevUncertainty(flux_error, unit=flux_unit),
    )

Photometric constraints are supplied in an :class:`astropy.table.Table`. The
required columns are ``band`` and ``mag``; an optional ``mag_err`` column gives
1-sigma magnitude uncertainties. All magnitudes are interpreted as AB
magnitudes.

.. code-block:: python

    photometry = Table(
        rows=[
            ("SLOAN/SDSS.g", 18.42, 0.03),
            ("SLOAN/SDSS.r", 18.17, 0.02),
            ("SLOAN/SDSS.i", 18.05, 0.04),
        ],
        names=("band", "mag", "mag_err"),
    )

    result = mangle(spectrum, photometry)

    mangled = result.spectrum
    correction = result.correction
    print(result.photometry)

The returned :class:`~specmangle.MangleResult` contains the corrected spectrum,
the fitted correction and its parameters, the bandpasses used in the fit, and a
copy of the input photometry with fit diagnostics appended.
