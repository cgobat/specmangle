
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from specmangle import MangleResult


colors = {"u": "C4", "g": "C2", "r": "C3", "i": "gold", "z": "C5", "U": "C4", "B": "C0", "V": "limegreen", "R": "C3", "I": "C5",
          "UVW1": "slateblue", "UVM2": "mediumslateblue", "UVW2": "darkslateblue", "orange": "C1"}
markers = {"Swift/UVOT": "s", "Liverpool/IOO": "^", "Palomar/ZTF": "o"}


def plot_result(result: MangleResult) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axs = plt.subplots(figsize=(8, 8), ncols=1, nrows=2, sharex=True, squeeze=True, height_ratios=(0.7, 0.3))
    wl_A = result.spectrum.spectral_axis.quantity.to_value(u.AA)
    orig_flux = result.spectrum.flux / result.correction
    axs[0].plot(wl_A, orig_flux.to_value(1e-15*u.erg/u.s/u.cm**2/u.AA), lw=0.6, color="cyan", label="original flux")
    axs[0].plot(wl_A, result.spectrum.flux.to_value(1e-15*u.erg/u.s/u.cm**2/u.AA), lw=0.6, alpha=0.8, color="k", label="corrected flux")
    band_ax = axs[0].twinx()
    for name, bandpass in result.bandpasses.items():
        band_ax.fill_between(bandpass.wavelength.to_value(u.AA), 0, bandpass.transmission/np.max(bandpass.transmission),
                            color=colors.get(name.split(".")[-1].removeprefix("SDSS-").removesuffix("_trn"), "grey"),
                            alpha=0.5, label=name.removesuffix("_trn"))
    band_ax.set_ylim(0, 5)
    band_ax.set_yticks([])
    axs[1].plot(wl_A, result.correction)
    axs[1].plot(result.anchor_wavelengths, result.correction_at(result.anchor_wavelengths), "o")
    # resid_ax = axs[1].twinx()
    axs[0].set_xlim(wl_A.min(), wl_A.max())
    axs[0].set_ylabel(r"$f_\lambda$ [$10^{-15}$ erg/s/cm$^2$/$\mathrm{\AA}$]")
    axs[1].set_xlabel(r"Observed wavelength [$\mathrm{\AA}$]")
    axs[1].set_ylabel("Applied correction factor")
    axs[0].legend(*map(sum, zip(axs[0].get_legend_handles_labels(), band_ax.get_legend_handles_labels()), [[], []]))
    return fig, axs
