# specmangle

`specmangle` applies a smooth, wavelength-dependent multiplicative correction to
an observed astronomical spectrum such that synthetic broadband photometry based
on that spectrum matches contemporaneous observations.

The package is deliberately narrow in scope. It uses Astropy-ecosystem data
structures rather than defining parallel containers:

- spectra are `specutils.Spectrum` objects
- photometric measurements are stored in `astropy.table.Table` objects
- passband profiles are retrieved from the [SVO Filter Profile Service](https://svo2.cab.inta-csic.es/theory/fps/index.php) via `astroquery` (or user-defined)

## Installation

From the project directory:

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e .
```

## Basic use

```python
import astropy.units as u
from astropy.nddata import StdDevUncertainty
from astropy.table import Table
from specutils import Spectrum

from specmangle import mangle

spectrum = Spectrum(
    spectral_axis=wavelength * u.AA,
    flux=flux * u.erg / u.s / u.cm**2 / u.AA,
    uncertainty=StdDevUncertainty(
        flux_error,
        unit=u.erg / u.s / u.cm**2 / u.AA,
    ),
)

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
```

The input photometry table must contain:

- `band`: passband identifier, normally an SVO FPS filter ID
- `mag`: AB magnitude

It may additionally contain:

- `mag_err`: 1-sigma magnitude uncertainty. Masked values are treated as missing
uncertainties.

Additional columns are preserved in `result.photometry`, which adds `used`,
`coverage`, `skip_reason`, `synthetic_mag`, `synthetic_mag_err`, and `residual`
columns. Bands that do not meet `min_coverage` are excluded from the fit and
reported explicitly in these diagnostic columns.

## Mangling model

The correction is positive and multiplicative. It is represented by a natural
cubic spline in log wavelength and log flux scale, with one free anchor at the
pivot wavelength of each unique passband. The correction is held constant
beyond the bluest and reddest anchors rather than extrapolating the spline.

The fit minimizes the difference between the mangled spectrum's synthetic AB
magnitudes and the supplied photometry. When uncertainties are available,
photometric uncertainties and synthetic-photometry uncertainties propagated from
the spectrum are combined in quadrature.

A nonzero `smoothness` can be supplied to penalize changes in slope between
neighboring spline anchors:

```python
result = mangle(spectrum, photometry, smoothness=0.1)
```

The default is `smoothness=0.0`, so no additional regularization is imposed.

## History & heritage

The term "spectral mangling" is commonly used for smooth wavelength-dependent
corrections that are applied to a spectrum to make it reproduce contemporaneous
broadband photometry. The approach has roots in earlier spectral "warping"
methods (e.g.
[Tonry *et al.* 2003](https://scixplorer.org/abs/2003ApJ...594....1T)), was
formalized under the "mangling" nomenclature by
[Hsiao *et al.* (2007)](https://scixplorer.org/abs/2007ApJ...663.1187H) and
[Conley *et al.* (2008)](https://scixplorer.org/abs/2008ApJ...681..482C), and
was later applied directly to spectrophotometric calibration of observed
core-collapse supernova spectra by
[Vincenzi *et al.* (2019)](https://scixplorer.org/abs/2019MNRAS.489.5802V).

## Spectral uncertainties

`specmangle` supports the same uncertainty representations supported by
`specutils` for arithmetic propagation:

- `astropy.nddata.StdDevUncertainty`;
- `astropy.nddata.VarianceUncertainty`;
- `astropy.nddata.InverseVariance`.

For synthetic photometry, uncertainties are converted to standard deviations and
spectral samples are assumed to be statistically independent. Masked spectral
samples are omitted.

The uncertainty stored on `result.spectrum` is the input spectral uncertainty
scaled by the best-fit mangling correction. Uncertainty in the correction
itself is returned separately as `result.correction_uncertainty`, because errors
in the fitted correction are correlated across wavelength.

## Bandpasses

By default, values in the `band` column are passed to the SVO Filter Profile
Service:

```python
from specmangle import Bandpass

bandpass = Bandpass.from_svo("SLOAN/SDSS.g")
```

Custom response curves can be supplied instead:

```python
import astropy.units as u
from specmangle import Bandpass, mangle

custom = Bandpass(
    filter_id="my/filter",
    wavelength=filter_wavelength * u.AA,
    transmission=filter_transmission,
    detector_type="photon",
)

result = mangle(
    spectrum,
    photometry,
    bandpasses={"my/filter": custom},
)
```

The synthetic-photometry calculation distinguishes photon-counting and
energy-counting response curves. SVO's `DetectorType` metadata is used
automatically for SVO filters.

## Passband coverage

By default, at least 98% of the passband's AB-reference signal must fall within
the wavelength range covered by the valid spectrum samples. This prevents a
synthetic magnitude from silently relying on a substantially incomplete
bandpass.

The threshold can be changed with `min_coverage`:

```python
result = mangle(spectrum, photometry, min_coverage=0.95)
```

## Direct synthetic photometry

For the common/general case of computing an AB magnitude directly from an SVO
FPS identifier:

```python
from specmangle import spectrum_to_magnitude

mag = spectrum_to_magnitude(spectrum, "SLOAN/SDSS.r")
```

If the input `Spectrum` includes an uncertainty, the function returns
`(mag, mag_err)`. Both values carry Astropy magnitude units. An already loaded
`Bandpass` can be supplied instead of an SVO identifier.

The lower-level synthetic-photometry result is also exposed when coverage and
other diagnostics are useful:

```python
from specmangle import Bandpass, synthetic_ab_magnitude

bandpass = Bandpass.from_svo("SLOAN/SDSS.r")
synthetic = synthetic_ab_magnitude(spectrum, bandpass)

print(synthetic.magnitude)
print(synthetic.uncertainty)
print(synthetic.coverage)
```

## Result object

`mangle` returns a `MangleResult` containing:

- `spectrum`: the mangled `specutils.Spectrum`
- `correction`: multiplicative correction evaluated at each input spectral sample
- `correction_uncertainty`: estimated 1-sigma uncertainty in that correction, when identifiable
- `anchor_wavelengths`: spline-anchor pivot wavelengths
- `log_correction_parameters`: fitted log-correction values at the anchors
- `parameter_covariance`: fitted parameter covariance when absolute uncertainties are available
- `photometry`: the input table with synthetic-photometry diagnostics appended
- `bandpasses`: the bandpasses used in the fit
- `optimizer`: the underlying `scipy.optimize.OptimizeResult`
