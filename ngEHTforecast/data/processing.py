import numpy as np
import ehtim as eh
import os

import ngEHTforecast.fisher.fisher_forecast as ff

def preprocess_obsdata(obs,ff=None,p=None,snr_cut=None,sys_err=None,const_err=None,avg_time=None,verbosity=0,**kwargs) :
    """
    Applies default preprocessing and creates and returns a new
    :class:`ehtim.obsdata.Obsdata` object.  Flagging and averaging is performed 
    using ehtim functinality.  See the source code for the order in which 
    operations are applied.

    Args:
      obs (eht.obsdata.Obsdata,str): An ehtim Obsdata object containing the desired observing profile or a uvfits file name to read.
      ff (fisher.fisher_forecast.FisherForecast): A :class:`fisher.fisher_forecast.FisherForecast` child class describing the model for which we wish to apply the processing. Required for applying the SNR cut. Default: None.
      p (list): Parameter list appropriate for ff. Requires ff to be specified. Default: None.
      snr_cut (float): SNR below which to exclude baselines. A proxy for a detectiton threshold. Rquires ff and p to be specified. Default: None.
      sys_err (float): Percent fractional systematic error to add. A proxy for non-closing errors. Default: None.
      const_err (float): A constant fractional error in mJy to add in quadrature. Default: None.
      avg_time (float,str): Coherently average data set. If a float, specifies the averaging time in seconds.  If 'scan', will create and average on scans. Default: None.
      verbosity (int): Verbosity level. Default: 0.

    Returns:
      (ehtim.obsdata.Obsdata): A new ehtime Obsdata object with the desired processing applied.
    """

    if (isinstance(obs,str)) :
        _,ext = os.path.splitext(obs)
        if (ext.lower()=='.uvfits') :
            obs = eh.obsdata.load_uvfits(obs)
        else :
            raise(RuntimeError("Only UVFITS files can be read at this time."))

    obs_new = obs.copy()

    if (not avg_time is None) :
        if (avg_time=='scan') :
            obs_new = obs_new.avg_coherent(0,scan_avg=True)
        else :
            obs_new = obs_new.avg_coherent(avg_time)

    if (not ff is None ) :
        if (p is None) :
            raise(RuntimeError("Parameter list must be provided if FisherForecast object is not None."))
        obs_new.data['vis'] = ff.visibilities(obs,p)

    if (not avg_time is None) :
        if (ff is None) :
            raise(RuntimeError("FisherForecast object must be provided to apply SNR cut."))
        obs_new = obs_new.flag_snr(snr_cut)

    if (not snr_cut is None) :
        obs_new = obs_new.flag_low_snr(snr_cut)
        
    if (not sys_err is None) :
        obs_new = obs_new.add_fractional_noise(frac_noise=0.01*sys_err)

    if (not const_err is None) :
        obs_new.data['sigma'] = np.sqrt(obs_new.data['sigma']**2 + (0.001*const_err)**2)


    return obs_new




