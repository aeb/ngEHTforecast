.. module:: tutorials


Tutorials
================================================================================
Having trouble knowing where to start?  These tutorials cover key functionality
in bitesized chuncks.  You can jump to any specific tutorial in the navigation
bar or go through them all!  As functionality is added, new tutorials will
appear (on a best effort basis!).  If you would like to help generate tutorials,
especially for new features that you have written, please contact us!  For help
on getting ngEHTforecast, see the :ref:`Installation Guide`.



Creating an Obsdata object
--------------------------------------------------------------------------------
Data within ngEHTforecast is handled using the :class:`ehtim.obsdata.Obsdata`
objects.  These encapsulate the :math:`(u,v)`-positions, scan times, station
names, among many other elements (see the ehtim_ documentation for full
details!).  They also come with convenient functions for generating, loading,
and manipulating data sets.

In this tutorial we will read data from disk, scan average it, flag a subset of
stations, and add a systematic error.  The result will be an
:class:`ehtim.obsdata.Obsdata` object suitable for paassing to subsequent
ngEHTforecast analyses.

We begin by importing ehtim_, and loading a data file.  The most common saved
data format is UVFITS files.  Some examples are contained in the `uvfits/`
directory.

.. code-block:: python
   
   import ehtim as eh

   obs = eh.obsdata.load_uvfits('../uvfits/M87_230GHz.uvfits')

This :class:`ehtim.obsdata.Obsdata` object can already be passed to subsequent
ngEHTforecast analyses.  However, we might want to perform some typical
preprocessing steps.  First, we will average the data over observation scans,
reducing the data volume.

.. code-block:: python
   
   obs.add_scans()
   obs = obs.avg_coherent(0,scan_avg=True)

Note that we have over-written our :class:`ehtim.obsdata.Obsdata` object.  We
could give this another name, keeping the original should we desire to do so.

We can see which stations are in the data set by looking at the **obs.data**
entries directly.  In order to select only unique entries we will make use
of Numpy_ functions.

.. code-block:: python
   
   import numpy as np
   
   print( "Unique stations:",np.unique((obs.data['t1'],obs.data['t2'])) )
   print( "Unique scan times:",np.unique(obs.data['time']) )

which produces

::

   Unique stations: ['ALMA' 'APEX' 'BAJA' 'CAT' 'CNI' 'GAM' 'GARS' 'GLT' 'HAY' 'JCMT' 'KP' 'LMT' 'NZ' 'OVRO' 'PDB' 'PV' 'SGO' 'SMA' 'SMT']
   Unique scan times: [ 0.66666667  0.83333337  1.00000003  ... 21.66666698 21.83333302 22.00000048]

   
While it is possible to get a list of stations from the telescope array
specification in **obs.tarr**, ngEHTforecast functions make use of the stations
that appear explicitly in the **obs.data** object.  Thus, the above is more
closely related to what an ngEHTforercast analysis will find in the data.
   
A common characterization of non-closing systematic errors is an additional
fractional error contribution.  For example, many EHT_ analyses assume a 1%
systematic uncertainty.  This may be included via

.. code-block:: python

   obs = obs.add_fractional_noise(0.01)

Finally, we will flag a subset of stations, creating a second observation
associated with a smaller array.  This is easily done using the
:meth:`ehtim.obsdata.Obsdata.flag_sites` function:

.. code-block:: python

   obs2 = obs.flag_sites(['ALMA','JCMT','SMT','SPT','PV','PDB'])

   print( "Unique stations after flagging:",np.unique((obs2.data['t1'],obs2.data['t2'])) )


where we have removed ALMA, JCMT, SMA, SPT, PV, and PDB.  We have now saved
this to a second :class:`ehtim.obsdata.Obsdata` object, and thus can make
predictions for both.  The list of sites is now those except the ones we
removed:

::

   Unique stations after flagging: ['APEX' 'BAJA' 'CAT' 'CNI' 'GAM' 'GARS' 'GLT' 'HAY' 'KP' 'LMT' 'NZ' 'OVRO' 'SGO' 'SMA']
   
You are ready to begin forecasting!  All of the above may be found in the
Python_ script **tutorials/obsdata.py**.

   
Creating a FisherForecast object
--------------------------------------------------------------------------------
Fisher-matrix based analyses begin with the specification of an underlying model
that we imagine will be fit the simulate data set.  In ngEHTforecast, we specify
this model via the creation of a :class:`fisher.fisher_forecast.FisherForecast`
object.  This encapsulates both the model definition and provides a number of
useful forecasting functions.

In this tutorial we will create a binary consisting of a symmetric and
asymmetric Gaussian, and incorporate complex station gains.  The result will
be a :class:`fisher.fisher_forecast.FisherForecast` object suitable for making
forecasts given a data set contained in an :class:`ehtim.obsdata.Obsdata` object.

We begin by importing the :ref:`fisher` functionality from ngEHTforecast
and generating :class:`fisher.fisher_forecast.FisherForecast` objects for each
of the two components.

.. code-block:: python

   import ngEHTforecast.fisher as fp

   ff1 = fp.FF_symmetric_gaussian()
   ff2 = fp.FF_asymmetric_gaussian()

   print("Primary parameters:",ff1.parameter_labels())
   print("Secondary parameters:",ff2.parameter_labels())

Both **ff1** and **ff2** are :class:`fisher.fisher_forecast.FisherForecast`
objects, either of which could be used to forecast ngEHT_ science capabilities.
We also have printed the names of the parameters of each object:

::

   Primary parameters: ['$\\delta F~({\\rm Jy})$', '$\\delta FWHM~(\\mu{\\rm as})$']
   Secondary parameters: ['$\\delta F~({\\rm Jy})$', '$\\delta FWHM~(\\mu{\\rm as})$', '$\\delta A$', '$\\delta {\\rm PA}~({\\rm rad})$']

from which it is evident that the primary has two parameters and the secondary
has four parameters.

A binary, consisting of the primary and secondary separated by some
displacement, may be constructed using :class:`fisher.ff_metamodels.FF_sum`, which takes a
list of the :class:`fisher.fisher_forecast.FisherForecast` to be summed.

.. code-block:: python

   ff = fp.FF_sum([ff1,ff2])

   print("Binary parameters:",ff.parameter_labels())

Again we print the names of the parameters,

::

   Binary parameters: ['$\\delta F~({\\rm Jy})$', '$\\delta FWHM~(\\mu{\\rm as})$', '$\\delta F~({\\rm Jy})$', '$\\delta FWHM~(\\mu{\\rm as})$', '$\\delta A$', '$\\delta {\\rm PA}~({\\rm rad})$', '$\\delta\\Delta x~(\\mu{\\rm as})$', '$\\delta\\Delta y~(\\mu{\\rm as})$']

from which we note that there are now eight parameters: two from the primary,
four from the secondary, and the two that specify the displacement.  Again,
**ff** is a :class:`fisher.fisher_forecast.FisherForecast` object, and may itself
be used to forecast ngEHT_ science capabilities.

Finally, we will incorporate the complex station gains, set the gain solution
intervals (gain epochs), and define a prior on the gain amplitudes.  We do this
using :class:`fisher.ff_complex_gains.FF_complex_gains`, which takes a
:class:`fisher.fisher_forecast.FisherForecast` object and constructs another
:class:`fisher.fisher_forecast.FisherForecast` object that marginalizes over the
desired complex station gains.

.. code-block:: python
   
   ffg = fp.FF_complex_gains(ff)

   ffg.set_gain_epochs(scans=True)
   ffg.set_gain_amplitude_prior(0.1)
   
   print("Binary w/ gains parameters:",ffg.parameter_labels())

The gain solution interval is set to observation scans; other available options
are described in the :class:`fisher.ff_complex_gains.FF_complex_gains`
documentation.  The priors on the complex gain amplitudes are log-normal and set
to 0.1, corresponding to a 10% uncertainty, typcial of current EHT_ operation.  In
the absence of specifying gain amplitude priors, they will be unconstrained (as
are the complex gain phases).

Again, we have output the parameter labels,

::

   Binary w/ gains parameters: ['$\\delta F~({\\rm Jy})$', '$\\delta FWHM~(\\mu{\\rm as})$', '$\\delta F~({\\rm Jy})$', '$\\delta FWHM~(\\mu{\\rm as})$', '$\\delta A$', '$\\delta {\\rm PA}~({\\rm rad})$', '$\\delta\\Delta x~(\\mu{\\rm as})$', '$\\delta\\Delta y~(\\mu{\\rm as})$']

Note that the number and names of the parameters have not changed.  This is
because the complex station gains will be marginalized over, i.e., we will not
retain access to the gains themselves.

Given an :class:`ehtim.obsdata.Obsdata` object, you are ready to start
forcasting the capability of ngEHT_ to constrain your binary model!  All of the
above may be found in the Python_ script **tutorials/binary_ff.py**.



Forecasting Uncertainties
--------------------------------------------------------------------------------
With the above two tutorials, :ref:`Creating an Obsdata object` and
:ref:`Creating a FisherForecast object`, we now can start estimating the
capabilities of ngEHT_!

In this tutorial we will estimatate the precision with which ngEHT_ should be
able to constrain the parameters of our binary model for the different data
sets.  Specifically, we will:

1. Generate estimate for uncertainties on a handful of parameters after
   marginalizing over all others.
2. Plot the one-dimensional marginalized posteriors for both of our data sets.
3. Plot joint two-dimensional marginalized posteriors for both of our data sets.
4. Generate a triangle plot for our binary model.

This code in this tutorial can be found in **tutorials/forecasting.py**,
which includes the code from the previous two tutorials at the top (without the
print statements).

We must first specify the parameters of the underlying "truth" values, i.e.,
what are the parameter values of the true brightness distribution on he sky in
our simulated exercise.  These should be specified via a list in the same order
as the parameter labels.  For concreteness, we will assume that:

* the primary has total flux 0.5 Jy and a FWHM of 10 uas,
* the secondary has a total flux of 1 Jy, a symmetrized-FWHM of 20 uas, an
  asymmery parameter of 0.5, and a position angle of 0.3 radians E of N,
* the two components are separated by 20 uas in RA and 5 uas in Dec.

This corresponds to the following parameter list.
  
::

   p = [0.5,10, 1.0,20,0.5,0.3, 20,5]

   
To check that this looks the way we expect it to, we can plot the image with the
:meth:`fisher.fisher_forecast.FisherForecast.display_image` function.

.. code-block:: python
   :emphasize-lines: 3
   
   import matplotlib.pyplot as plt
   
   ffg.display_image(p)
   plt.savefig('tutorial_image.png',dpi=300)

where we have now imported Matplotlib_ to use its plotting functionality.  The
resulting plot is shown below.

.. figure:: ./tutorials_figures/tutorial_image.png
   :scale: 25%

   Image of the brightness map for the binary model made using the
   :meth:`fisher.fisher_forecast.FisherForecast.display_image` function.


We begin our science forecasts with computing the uncertainties on the fluxes
and separations of the two components.  To do this we make use of the
:meth:`fisher.fisher_forecast.FisherForecast.marginalized_uncertainties`
function, which computes the uncertainties for each parameter after
marginalizing over all others.  We specify the observation for which to
compute the uncertainties and the indices of the parameters for which we
would like uncertainty estimates.

.. code-block:: python

   Sigma_obs = ffg.marginalized_uncertainties(obs,p,ilist=[0,2,6,7])
   Sigma_obs2 = ffg.marginalized_uncertainties(obs2,p,ilist=[0,2,6,7])

   print("Sigma's for obs:",Sigma_obs)
   print("Sigma's for obs2:",Sigma_obs2)

which generates the output,

::

   Sigma's for obs: [0.00300763 0.00600743 0.00101485 0.00205108]
   Sigma's for obs2: [0.00377924 0.00753871 0.00203864 0.00333373]

From this we might conclude that the reduced array is similarly capable of
constraining the fluxes of the two components (differing by about 10%-15%), but
is considerably worse at constraining their relative location (though still
pretty great!).

We can generate plots comparing the ability of the two arrays to constrain the
RA offsets with the
:meth:`fisher.fisher_forecast.FisherForecast.plot_1d_forecast` function.  We
must select the observations to include (i.e., the :math:`(u,v)`-coverage), the
index of the parameter that we wish to plot (the RA offset is the seventh
parameter, and therefore index 6 due to the zero-offset indexing), and may
optionally set some clarifying labels to indicate which observation details the
two different curves correspond.

.. code-block:: python
   :emphasize-lines: 1

   ffg.plot_1d_forecast([obs2,obs],p,6,labels=['ngEHT','Reduced ngEHT'])
   plt.savefig('tutorial_1d.png',dpi=300)

Note that we have ensured that the typically more constraining case is plotted
second by setting the order of the observations in the list.

.. figure:: ./tutorials_figures/tutorial_1d.png
   :scale: 25%

   Marginalized posterior on the shift in RA between the two binary components
   made using :meth:`fisher.fisher_forecast.FisherForecast.plot_1d_forecast`
   function.  The two plots correspond to the different observations generated
   in :ref:`Creating an Obsdata object`.

Similarly, we can generate plots of the two-dimensional joint posterior,
marginalized over all other parameters using the
:meth:`fisher.fisher_forecast.FisherForecast.plot_2d_forecast` function.  The
syntax is very similar to the
:meth:`fisher.fisher_forecast.FisherForecast.plot_1d_forecast` function, with
the exception that now we must specify two parameter indices.

.. code-block:: python
   :emphasize-lines: 1

   ffg.plot_2d_forecast([obs2,obs],p,6,7,labels=['ngEHT','Reduced ngEHT'])
   plt.savefig('tutorial_2d.png',dpi=300)

.. figure:: ./tutorials_figures/tutorial_2d.png
   :scale: 25%

   Marginalized joint posterior on the shift in RA and Dec between the two
   binary components made using
   :meth:`fisher.fisher_forecast.FisherForecast.plot_2d_forecast`
   function.  The two sets of contours correspond to the different observations
   generated in :ref:`Creating an Obsdata object`.

A triangle plot, which is simply a collection of marginalized joint and
one-dimesional posteriors, may be generated via the
:meth:`fisher.fisher_forecast.FisherForecast.plot_triangle_forecast` function.
Again, the syntax is very similar, the only difference being the manner in which
indices are specified.  By default, all parameters are included.  It is helpful
to also include some guidance on the location relative to the figure to ensure
labels are visible.

.. code-block:: python
   :emphasize-lines: 1

   ffg.plot_triangle_forecast([obs2,obs],p,labels=['ngEHT','Reduced ngEHT'],axis_location=[0.075,0.075,0.9,0.9])
   plt.savefig('tutorial_tri.png',dpi=300)

.. figure:: ./tutorials_figures/tutorial_tri.png
   :scale: 25%

   Triangle plot for the binary model made using
   :meth:`fisher.fisher_forecast.FisherForecast.plot_triangle_forecast`
   function.  The two sets of contours correspond to the different observations
   generated in :ref:`Creating an Obsdata object`.

All of the above may be found in the Python_ script **tutorials/forecasting.py**.


Splined Raster Models
--------------------------------------------------------------------------------
A splined-raster model, i.e., "themage", is available in the
:class:`ff_models.FF_splined_raster` class.  This provides a flexible image
model that can assess imaging performance and hybrid imaging-modeling
performance.  However, the large number of parameters can make it difficult to
sensibly specify an image.  Therefore, a number of special ways to construct a
parameter list are available.

In this tutorial we will construct a splined-raster model and initialize it
using an existing FisherForecast object, the name of a FisherForecast child
class, a FITS file, and an :class:`ehtim.image.Image` object.

We begin by constructing a splined-raster model.  At initialization, we must
specify the size of the raster, i.e., the number of control points in each
direction, and the field of view of the raster.  In this case, we choose 20
control points in each direction and a field of view of 60 uas:

.. code-block:: python

   import ngEHTforecast.fisher as fp

   ff = fp.FF_splined_raster(20,60.0)

This model has 400 control points, and thus 400 parameters: the log of the
intensities at each control point.  Even if the intensity is well defined,
initializing 400 parameters is a daunting task!  Fortunately, a number of
options are available with the
:meth:`fisher.ff_models.FF_splined_raster.generate_parameter_list` function.

The first we consider is initializing from an existing FisherForecast object. We
begin by creating another
:class:`fisher.ff_models.FF_smoothed_delta_ring` object.  This is then passed to
the :meth:`fisher.ff_models.FF_splined_raster.generate_parameter_list` function
along with the parameter list (here for a total flux of 1 Jy, diameter of 40 uas,
and width of 10 uas) as a key-word argument. We pass an additional argument,
**limits**, which specifies the extent of the image created by the
:meth:`fisher.fisher_forecast.display_image` function.  

.. code-block:: python

   ffinit = fp.FF_smoothed_delta_ring()
   pinit = [1.0, 40.0, 10.0]
   ffinit.display_image(pinit,limits=[-50,50,-50,50])

   p = ff.generate_parameter_list(ffinit,p=pinit,limits=100)
   ff.display_image(p,limits=[-50,50,-50,50])

To compare the in splined-raster model and original smoothed delta-ring, we
display both:

.. figure:: ./tutorials_figures/tutorial_smdr.png
   :scale: 25%

   Smoothed delta-ring model (left) and the splined raster model initialized from it (right).

The second is an initialization from a FisherForecast model without actually
constructing an instantiation.  This avoids the overhead of creating the model,
but still leverages the other FisherForecast models.  In this case, we use an
asymmetric Gaussian model with total flux 1 Jy, symmerized FWHM of 20 uas,
asymmetry parameter of 0.5, and PA of 1.0 rad:

.. code-block:: python
   
   p = ff.generate_parameter_list(fp.FF_asymmetric_gaussian,p=[1,20,0.5,1.0])
   ff.display_image(p,limits=50)

In the third example, we initialize from an :class:`ehtim.image.Image` object,
which we create from a FITS file.

.. code-block:: python

   img = eh.image.load_fits('M87_230GHz.fits')
   img = img.blur_circ(np.sqrt(ff.dxcp*ff.dycp))

   p = ff.generate_parameter_list(img)
   ff.display_image(p,limits=80)

Prior to generating the splined raster parameter list, we blur the image to
the raster resolution to give a better impression of the results of a splined
raster fit.  The result is shown below.

.. figure:: ./tutorials_figures/tutorial_img.png
   :scale: 25%

   Image from ehtim (left) and the splined raster model initialized from it (right).   

Finally, we can initialize the parameter list from a FITS file directly.  This
is identical to initializing from an :class:`ehtim.image.Image` object, which it
creates internally, including blurring to the raster resolution.

All of the above may be found in the Python_ script
**tutorials/splined_raster_initialization.py**.   



Exploring & Parallelization
--------------------------------------------------------------------------------
A key question for many science cases will be how the uncertainties depends on
specific model prameters.  For example, how well the binary separation can be
determined as a function of the flux of the secondary.  In this tutorial, we
will make a plot that shows how a chosen parameter uncertainty varies with
respect to the source model parameters.  Because large parameter space
explorations can quickly become computationally intensive, we will also
demonstrate how this can be parallelized to take advantage of multiple cores.

Exploring parameter dependence can easily addressed via an appropriate loop over
the model paramters.  Again, we will assume that the code from the tutorials
:ref:`Creating an Obsdata object` and :ref:`Creating a FisherForecast object` is
included.  The only ngEHTforecast function that we are using is the same
:meth:`fisher.fisher_forecast.FisherForecast.marginalized_uncertainties`
described in :ref:`Forecasting Uncertainties`.  However, now it is embedded
in a loop which varies the "truth" parameters:
      
.. code-block:: python
   :emphasize-lines: 1-5

   secondary_flux = np.logspace(-3,0,16)
   Sigma_list = 0*secondary_flux
   for i in range(len(secondary_flux)) :
       p = [0.5,10, secondary_flux[i],20,0.5,0.3, 20,5]
       Sigma_list[i] = ffg.marginalized_uncertainties(obs,p,ilist=6)

   import matplotlib.pyplot as plt

   plt.plot(secondary_flux,Sigma_list,'-ob')
   plt.xscale('log')
   plt.yscale('log')
   plt.xlabel(r'Flux of Secondary (Jy)')
   plt.ylabel(r'$\sigma_{\rm RA}~(\mu{\rm as})$')
   plt.grid(True,alpha=0.25)

   plt.savefig('tutorial_sep',dpi=300)

The remainder of the code makes a Matplotlib_ plot, specifies the scales, adds
labels and other accoutrements, and saves the following plot to a file.

.. figure:: ./tutorials_figures/tutorial_sep.png
   :scale: 25%

   Uncertainty on the separation in RA as a function of the total flux of the
   secondary in the binary model.

All of the above may be found in the Python_ script **tutorials/binary_separation.py**.

While the above code completes in approximately 1 minute on a single modern core,
increasing the number of parameters being surveyed rapidly grows the
computational expense.  Beacause the loop is trivially parallel, the evaluation
of the uncertainty at each parameter set is independent of all others, this
problem lends itself to parallelization.

There are many packages that enable parallelization in Python_.  We will make
use of two: the the Joblib_ package and multiprocessing_ library.  We begin with
the former, Joblib_.

All that changes from above is the syntax surrounding the computation of the
elements of **Sigma_list**.  From Joblib_ we import the functions
:class:`joblib.parallel.Parallel` and :meth:`joblib.parallel.delayed` (see the Joblib_
documentation for why the latter is useful).  Joblib_ will handle the
distribution of individual computations to the cores (here set to 4), we must
only define a single function to return the desired marginalized uncertainty at
each new point in the parameter space.  

.. code-block:: python
   :emphasize-lines: 1,3-5,8

   from joblib import Parallel, delayed

   def get_sigma(flux) :
       p = [0.5,10, flux,20,0.5,0.3, 20,5]
       return ffg.marginalized_uncertainties(obs,p,ilist=6)

   secondary_flux = np.logspace(-3,0,16)
   Sigma_list = Parallel(n_jobs=4)(delayed(get_sigma)(flux) for flux in secondary_flux)

   import matplotlib.pyplot as plt
   ...   
   plt.savefig('tutorial_sep_joblib.png',dpi=300)

We do so by defining a small function that, when given a flux for the secondary,
sets the parameter list and returns the marginalized uncertaint on the RA
offset.  This function is then passed to :class:`joblib.parallel.Parallel` using
:meth:`joblib.parallel.delayed` as specified in the Joblib_ documentation.

The advantage of using Joblib_ is that the entirety of the modification due to
parallelization is to define as a function the elements of the computation that
we wish to parallelize and some minor syntax changes.  This version of the binary
separation may be found in the Python_ script
**tutorials/binary_separation_joblib.py**.

Alternatively, we can make use of the Python_ multiprocessing_ library.  Again,
the primary difference is that part we wish to parallelize is most conveniently
contained in a single function.  To parallelize across 4 processes:

.. code-block:: python
   :emphasize-lines: 1,3-5,7,10-11
		     
   import multiprocessing as mp
   
   def get_sigma(flux) :
       p = [0.5,10, flux,20,0.5,0.3, 20,5]
       return ffg.marginalized_uncertainties(obs,p,ilist=6)

   if __name__ == "__main__" :
       secondary_flux = np.logspace(-3,0,16)
    
       with mp.Pool(4) as mpp :
           Sigma_list = mpp.map(get_sigma,secondary_flux)

       plt.plot(secondary_flux,Sigma_list,'-ob')
       ...
       plt.savefig('tutorial_sep_multiproc.png',dpi=300)

To avoid repeating the initialization steps (loading the UVFITS file, creating
the :class:`fisher.fisher_forecast.FisherForecast`), per the multiprocessing_
library documentation, the portion of the code that will ultimately be
parallelized is contained in the `__name__ == "__main__"` guards.  This version
of the binary separation may be found in the Python_ script
**tutorials/binary_separation_multiprocessing.py**.

.. _ehtim: https://achael.github.io/eht-imaging/
.. _Numpy: https://numpy.org
.. _EHT: https://www.eht.org
.. _Python: https://www.python.org/
.. _ngEHT: https://www.ngeht.org/
.. _Matplotlib: https://matplotlib.org
.. _Joblib: https://joblib.readthedocs.io
.. _multiprocessing: https://docs.python.org/3/library/multiprocessing.html
