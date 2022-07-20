import numpy as np
import ehtim as eh
import os
import matplotlib.pyplot as plt

import ngEHTforecast.fisher.fisher_forecast as ff

def preprocess_obsdata(obs,ff=None,p=None,snr_cut=None,sys_err=None,const_err=None,avg_time=None,verbosity=0,**kwargs) :
    """
    Applies default preprocessing and creates and returns a new
    :class:`ehtim.obsdata.Obsdata` object.  Flagging and averaging is performed 
    using ehtim functinality.  See the source code for the order in which 
    operations are applied.

    Args:
      obs (ehtim.obsdata.Obsdata,str): An ehtim Obsdata object containing the desired observing profile or a uvfits file name to read.
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

    if (not snr_cut is None) :
        if (ff is None) :
            raise(Warning("SNR cut is being applied without FisherForecast object."))
        obs_new = obs_new.flag_low_snr(snr_cut)
        
    if (not sys_err is None) :
        if (ff is None) :
            raise(Warning("Systematic error is being applied without FisherForecast object."))
        obs_new = obs_new.add_fractional_noise(frac_noise=0.01*sys_err)

    if (not const_err is None) :
        obs_new.data['sigma'] = np.sqrt(obs_new.data['sigma']**2 + (0.001*const_err)**2)


    return obs_new



def display_visibilities(obs,figsize=None,fig=None,axs=None,**kwargs) :
    """
    Plots the visilibility amplitudes, phases, and uncertainties as a function
    of baseline length for a given :class:`ehtim.obsdata.Obsdata` object.

    Args:
      obs (ehtim.obsdata.Obsdata): Observation for which to plot the visibilities.
      figsize (tuple): Figure size in inches.
      fig (matplotlib.figure.Figure): Figure object on which to generate plots.
      axs (list): List of :class:`matplotlib.axes.Axes` objects on which to generate plots.
      **kwargs (dict): Key word arguments understood by errorbar and plot to be applied.

    Returns:
      (matplotlib.figure.Figure,list): Figure object and list of Axes objects as produced by :meth:`matplotlib.figure.Figure.subplots`.
    """

    if (figsize is None) :
        figsize = (6,8)

    if (axs is None) :
        fig,axs = plt.subplots(nrows=3,ncols=1,figsize=figsize,sharex=True)
        plt.subplots_adjust(left=0.15,bottom=0.1,right=0.975,top=0.975)
        xmax = 0
    elif (fig is None) :
        fig = plt.gcf()
        xmax = list(axs[0].get_xlim())[1]
    else :
        plt.scf(fig)
        xmax = list(axs[0].get_xlim())[1]
        
    if (not 'alpha' in kwargs.keys()) :
        kwargs['alpha'] = 0.25

    if (not 'marker' in kwargs.keys()) :
        kwargs['marker'] = '.'

    if (not 'color' in kwargs.keys()) :
        kwargs['color'] = 'b'

    if ( (not 'linestyle' in kwargs.keys()) and (not 'ls' in kwargs.keys()) ) :
        kwargs['linestyle'] = ''
        

    uv = np.sqrt( obs.data['u']**2 + obs.data['v']**2 )/1e9
    amp = np.abs(obs.data['vis'])
    phs = np.angle(obs.data['vis'])
    sig = obs.data['sigma']

    xlim = (-0.1,max(xmax,1.1*np.max(uv)))
    axs[0].errorbar(uv,amp,yerr=sig,**kwargs)
    axs[0].set_xlim(xlim)
    axs[0].set_yscale('log')
    axs[0].set_ylabel(r'$|V|~({\rm Jy})$')
    
    axs[1].errorbar(uv,phs,yerr=sig/amp,**kwargs)    
    axs[1].set_xlim(xlim)
    axs[1].set_ylabel(r'${\rm arg}(V)~({\rm rad})$')

    axs[2].plot(uv,1e3*sig,**kwargs)
    axs[2].set_xlim(xlim)
    if (np.max(sig)/np.min(sig)>10) :
        axs[2].set_yscale('log')
    axs[2].set_ylabel(r'$\sigma~({\rm mJy})$')
    axs[2].set_xlabel(r'$|u|~({\rm G}\lambda)$')
    
    for i in range(3) :
        axs[i].grid(visible=True,alpha=0.25)
        
    return fig,axs
    



def display_baselines(obs,figsize=None,fig=None,axes=None,**kwargs) :
    """
    Plots the baseline map for a given :class:`ehtim.obsdata.Obsdata` object.

    Args:
      obs (ehtim.obsdata.Obsdata): Observation for which to plot the visibilities.
      figsize (tuple): Figure size in inches.
      fig (matplotlib.figure.Figure): Figure object on which to generate plot.
      axes (matplotlib.axes.Axes): Axes object on which to generate plot.
      **kwargs (dict): Key word arguments understood by errorbar and plot to be applied.

    Returns:
      (matplotlib.figure.Figure,matplotlib.axes.Axes): Figure and Axes object of plots.
    """

    if (figsize is None) :
        figsize = (5,5)

    if (axes is None) :
        fig = plt.figure(figsize=figsize)
        axes = plt.axes([0.15,0.15,0.8,0.8])
    elif (fig is None) :
        fig = plt.gcf()
    else :
        fig = plt.scf(fig)
        
    if (not 'alpha' in kwargs.keys()) :
        kwargs['alpha'] = 0.25

    if (not 'marker' in kwargs.keys()) :
        kwargs['marker'] = '.'

    if (not 'color' in kwargs.keys()) :
        kwargs['color'] = 'b'

    if ( (not 'linestyle' in kwargs.keys()) and (not 'ls' in kwargs.keys()) ) :
        kwargs['linestyle'] = ''
    
    uvlim = 1.1*max(np.max(np.abs(obs.data['u']))/1e9,np.max(np.abs(obs.data['v']))/1e9)
    axes.plot(obs.data['u']/1e9,obs.data['v']/1e9,**kwargs)
    axes.plot(-obs.data['u']/1e9,-obs.data['v']/1e9,**kwargs)
    xlim = list(axes.get_xlim())
    ylim = list(axes.get_ylim())
    xlim[1] = min(xlim[1],-uvlim)
    xlim[0] = max(xlim[0],uvlim)
    ylim[0] = min(ylim[0],-uvlim)
    ylim[1] = max(ylim[1],uvlim)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_xlabel(r'$u~({\rm G}\lambda)$')
    axes.set_ylabel(r'$v~({\rm G}\lambda)$')
    
    axes.grid(visible=True,alpha=0.25)

        
    return fig,axes


def display_trivial_cphases(obs,utriv=1e-2,return_data=False,print_outliers=False) :
    """
    Plots trivial closure phases, defined by those on triangles with a side
    shorter than **utriv** (by default :math:`0.01\\,{\\rm G}\\lambda`), and the 
    distribution of their normalized residuals.  Optionally, will return the 
    trivial closure phase data and/or print any outliers to the screen.

    Args:
      obs (ehtim.obsdata.Obsdata): Observation for which to plot the visibilities.
      utriv (float): Baseline length cutoff in :math:`{\\rm G}\\lambda`, below which a baseline will be considered "trivial". Default: 0.01.
      return_data (bool): If True, will return the trivial closure phases.
      print_outliers (bool): If True, will print the most discrepant trivial closure phases, in order of decreasing discrepancy.

    Returns:
      (np.recarray): If return_data=True, returns the closure phase data for the trivial triangles.
    """

    
    utriv *= 1e9 # From Glambda to lambda

    # Make a local copy so that we don't change the underlying obsdata object
    obs = obs.copy()
    
    # Closure phases
    obs.add_cphase(count='max')
    trivial_mask = ( (np.sqrt(obs.cphase['u1']**2+obs.cphase['v1']**2)<utriv) +
                     (np.sqrt(obs.cphase['u2']**2+obs.cphase['v2']**2)<utriv) +
                     (np.sqrt(obs.cphase['u3']**2+obs.cphase['v3']**2)<utriv) )
    cpd_trivials = obs.cphase[trivial_mask]
    cpd_normed = cpd_trivials['cphase']/cpd_trivials['sigmacp']
    mean_cpd = np.mean(cpd_normed)
    std_cpd = np.std(cpd_normed)

    plt.figure(figsize=(6,8))

    plt.axes([0.15,0.6,0.8,0.375])
    plt.errorbar(cpd_trivials['time'],cpd_trivials['cphase'],yerr=cpd_trivials['sigmacp'],fmt='.',color='cornflowerblue')
    plt.xlabel(r'UTC (hr)')
    plt.ylabel(r'Trivial Closure Phase (deg)')
    ylim = list(plt.ylim())
    plt.ylim((max(-180,ylim[0]),min(180,ylim[1])))
    plt.grid(True,alpha=0.25)

    plt.axes([0.15,0.1,0.8,0.375])
    plt.hist(cpd_trivials['cphase'],bins=33,color='cornflowerblue',density=True)
    plt.xlabel(r'Normalized Closure Phase')
    plt.ylabel(r'Frequency')
    plt.grid(True,alpha=0.25)

    # print("Cphase:",mean_cpd,std_cpd)

    is_outlier = (np.abs(cpd_normed)>2)
    cpd_trivial_outliers = cpd_trivials[is_outlier]

    if (print_outliers) :
        ilist = np.flipud(np.argsort(np.abs(cpd_trivial_outliers['cphase']/cpd_trivial_outliers['sigmacp']),))
        if (len(ilist)>20) :
            print("%i 2-sigma outliers found.  Printing only worst 20."%(len(ilist)))
            ilist = ilist[:20]
            print(cpd_trivial_outliers[ilist])
            print("  ...")
        else :
            print("%i 2-sigma outliers found."%(len(ilist)))
            print(cpd_trivial_outliers[ilist])
        print("-------------------------------------------------------------------")
            
        
    if (return_data) :
        return cpd_trivials
    
