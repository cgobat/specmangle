Mangling model
==============

Correction function
-------------------

The mangling correction is positive and multiplicative. ``specmangle`` models
its logarithm with a natural cubic spline in log wavelength, using one free
anchor at the pivot wavelength of each usable unique passband. Beyond the
bluest and reddest anchors the correction is held constant rather than
extrapolating the spline.

The fitted correction is returned at the original spectral samples as
``result.correction`` and can be evaluated on any other wavelength grid with
:meth:`specmangle.MangleResult.correction_at`:

.. code-block:: python

    new_wave = [3500, 4500, 5500, 6500, 7500] * u.AA
    correction = result.correction_at(new_wave)

Photometric constraints
-----------------------

For every usable passband, synthetic AB photometry is recomputed from the
corrected spectrum during the fit. If photometric or spectral uncertainties are
available, their corresponding magnitude uncertainties are combined in
quadrature when weighting residuals.

Repeated photometric measurements in the same passband are supported. They
constrain the same spline anchor rather than adding independent anchors at the
same wavelength.

Coverage and skipped bands
--------------------------

A passband is only used when enough of its response is supported by valid
spectral samples. The default threshold is ``min_coverage=0.98``. Both missing
wavelength range and masked internal gaps reduce the calculated coverage.
Passbands below the threshold are skipped rather than making the whole fit
fail, provided at least one usable passband remains.

The returned photometry table includes diagnostic columns:

``used``
    Whether the measurement participated in the fit.
``coverage``
    Fraction of the passband reference signal covered by valid spectral data.
``skip_reason``
    Explanation for a band excluded because of insufficient coverage.
``synthetic_mag`` and ``synthetic_mag_err``
    Synthetic photometry of the final mangled spectrum.
``residual``
    Final synthetic magnitude minus the supplied magnitude.

Regularization
--------------

By default no additional regularization is imposed. A positive ``smoothness``
penalizes changes in slope between neighboring spline anchors:

.. code-block:: python

    result = mangle(spectrum, photometry, smoothness=0.1)

Uncertainties
-------------

Input spectral uncertainty is propagated through synthetic photometry and
scaled by the best-fit multiplicative correction in ``result.spectrum``.
Supported representations include
:class:`astropy.nddata.StdDevUncertainty`,
:class:`astropy.nddata.VarianceUncertainty`, and
:class:`astropy.nddata.InverseVariance`.

Uncertainty in the fitted correction itself is returned separately as
``result.correction_uncertainty`` because it is correlated with wavelength.
When identifiable, ``result.parameter_covariance`` contains the covariance of
the spline parameters.
