##########
specmangle
##########

**Photometry-constrained mangling of astronomical spectra**

``specmangle`` applies a smooth, wavelength-dependent multiplicative correction
so that synthetic broadband photometry from an observed spectrum matches
contemporaneous observations. It is designed as a small, general-purpose tool
that interoperates with the Astropy ecosystem: spectra are represented by
:class:`specutils.Spectrum`, photometric constraints by
:class:`astropy.table.Table`, and passband data can be retrieved from the SVO
Filter Profile Service through Astroquery.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   mangling
   photometry
   api

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: getting_started
      :link-type: doc

      Install ``specmangle``, construct the input spectrum and photometry table,
      and perform a basic mangling fit.

   .. grid-item-card:: Mangling model
      :link: mangling
      :link-type: doc

      Understand the spline correction, wavelength coverage, uncertainties,
      regularization, and fit diagnostics.

   .. grid-item-card:: Synthetic photometry
      :link: photometry
      :link-type: doc

      Compute AB magnitudes directly and work with SVO or custom response
      curves.

   .. grid-item-card:: API reference
      :link: api
      :link-type: doc

      Reference documentation for the public classes and functions.
