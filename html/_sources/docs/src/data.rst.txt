.. module:: data


Data
================================================================================
A handful of UVFITS files are supplied with the package, collected in the the
**uvfits** folder.  Currently, these include:

* M87_230GHz.uvfits -- A simulated ngEHT_ data set generated from a GRMHD
  simulation image located at M87, viewed at 230 GHz.  Courtesy of Dom Pesece.
* M87_345GHz.uvfits -- A simulated ngEHT_ data set generated from a GRMHD
  simulation image located at M87, viewed at 345 GHz.  Courtesy of Dom Pesece.
* circ_gauss_x2_F0=3.0_FWHM=2.0_sep=5.0_angle=0_230GHz_ngeht_ref1.uvfits -- A
  simulated ngEHT_ data set generated for a binary Gaussian as described in the
  `ngEHT first analysis challenge <https://challenge.ngeht.org/challenge1/>`_. Courtesy of Freek Roelofs.

Beyond loading these, there are many other ways to generate and/or modify an
:class:`ehtim.obsdata.Obsdata` object.  Here are some functions for doing so,
some of which are implemented in ngEHTforecast (and its dependencies), while
others depend on external libraries designed to generate simulation data sets
for ngEHT_.



Processing
--------------------------------------------------------------------------------
.. automodule:: data.processing
   :members:


Generation Interfaces
--------------------------------------------------------------------------------




.. _ngEHT: https://www.ngeht.org/
