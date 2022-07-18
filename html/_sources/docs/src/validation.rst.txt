.. module:: validation


Validation
============================================================
ngEHTforecast has been extensively validated against analytical expectations and
the existing extensively validated posterior exploration methods, Themis and
DMC. In all instances, ngEHTforecast quantitatively matches the relevant
comparison.  Here we collect a subset of these validation tests.  The relevant
test scripts are contained within **validation** folder.


Analytical Tests
------------------------------------------------------------
A collection of analytical validation experiments were performed using a point
source model, for which the computation of the expected uncertainty on the
total flux is analytically tractable, even in the presence of station gains.  The
FisherForecast model was effectively generated using the symmetric Gaussian
with a small "true" FWHM (0.001 uas) and a tight prior imposed on the FWHM
(0.001 uas).  The test data set was given a uniform uncertainty of 1 mJy and
all stations were reset to AA or AP (ALMA/APEX), and thus at most two
complex gains were possible.

The analytical estimates for the uncertainties were obtained from standard
error analysis, with expressions that may be found in
**validation/analytical_tests.py**.  The results are summarized in the table
below, where excellent quantitative agreement is found across the board.

+--------------------------------+--------------+--------------+-------------+
| Point Source Test              | Marginalized :math:`\sigma_F ({\rm Jy})`  |
+                                +--------------+--------------+-------------+
|                                | Fisher Est.  | Analytical   | Frac. Err.  |
+================================+==============+==============+=============+
|                  Without gains |  1.2e-05     |  1.2e-05     | 5.37e-08    |
+--------------------------------+--------------+--------------+-------------+
|         Single gain, one epoch |    0.100     |    0.100     | 2.27e-09    |
+--------------------------------+--------------+--------------+-------------+
|        Single gain, two epochs |    0.071     |    0.071     | 2.14e-09    |
+--------------------------------+--------------+--------------+-------------+
|          Two gains, two epochs |    0.100     |    0.100     | 3.13e-03    |
+--------------------------------+--------------+--------------+-------------+


..
   +--------------------------------+--------------+--------------+-------------+
   | Point Source Test              | Marginalized :math:`\sigma_F ({\rm Jy})`  |
   +                                +--------------+--------------+-------------+
   |                                | Fisher Est.  | Analytical   | Frac. Err.  |
   +================================+==============+==============+=============+
   |                  Without gains |  1.2e-05     |  1.2e-05     | 5.37e-08    |
   +--------------------------------+--------------+--------------+-------------+
   |         Single gain, one epoch |      0.1     |      0.1     | 2.27e-09    |
   +--------------------------------+--------------+--------------+-------------+
   |           Two gains, one epoch |     0.16     |     0.14     | 1.47e-01    |
   +--------------------------------+--------------+--------------+-------------+
   |        Single gain, two epochs |    0.071     |    0.071     | 2.14e-09    |
   +--------------------------------+--------------+--------------+-------------+
   |          Two gains, two epochs |      0.1     |      0.1     | 3.13e-03    |
   +--------------------------------+--------------+--------------+-------------+

A single test was performed for a symmetric Gaussian of finite size, for which
an analytical estimate was not readily available.  Nevertheless, the dimension
of the parameter space is sufficiently small that the exact posterior may be
estimated directly via a grid search.  The result is show below, and is again
in excellent quantitative agreement.

.. figure:: ./validation_figures/gaussian_validation_2d.png
   :scale: 25%
   
   Comparison between the posterior obtained from an ngEHTforecast fisher
   analysis (red) and a grid search of the exact likelihood (blue).  This figure
   can be reproduced using **validation/analytical_tests.py**.


Themis Comparison
------------------------------------------------------------
Both a Themis posterior estimation and an ngEHTforecast fisher-matrix analysis
was performed on a binary model consisting of two symmetric Gaussians.  The six
free parameters are the fluxes, FWHMs, and displacements between the two
components.  In both, complex gains were incorporated, with 10% log-normal priors
on the gain amplitudes.  The "truth" parameters were
:math:`\{I_1,{\rm FWHM}_1,I_2,{\rm FWHM}_2,x,y\}`
= {1.5 Jy, 2 uas, 1.5 Jy, 2 uas, 5 uas, 0 uas}.  A comparison of the
joint posteriors is shown below.

.. figure:: ./validation_figures/themis_comparison.png
   :scale: 25%

   Comparison between Themis analysis (blue) and ngEHTforecast.fisher forecast
   (red) for all of the parameters in the Gaussian binary model.  This figure
   can be reproduced using **validation/themis_comparison.py**
   (requires ThemisPy).

All posteriors are well recovered by the ngEHTforecast analysis, including
the strong correlations between the fluxes of the two components.  Small
differences are expected due to the presence of thermal noise in the dataset
being fit by Themis, while implicitly marginalized over in the fisher matrix
analysis.


DMC Comparison
------------------------------------------------------------
Coming soon!

