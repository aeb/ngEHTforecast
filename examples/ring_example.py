"""
Analysis of precision with which the properties of a ring can be characterized
in light of a diffuse background and in the presence of station gain uncertainties.
Three figures are generated, comparing various ring parameters for data sets
appropriate for ngEHT at 230 GHz, 345 GHz, and combining both frequencies:

1. A single-parameter forecast of the diameter precision for each data set.
2. A two-parameter forecast of the diameter and width precisions for each data set.
3. A triangle plot of all ring parameters for reach data set.
"""

import ngEHTforecast.fisher as fp
import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__" :

    # Make Fisher forecast object
    ff1 = fp.FF_splined_raster(5,100)
    ff2 = fp.FF_smoothed_delta_ring()
    ffr = fp.FF_sum([ff1,ff2])
    ff = fp.FF_complex_gains(ffr)
    ff.set_gain_epochs(scans=True)
    ff.set_gain_amplitude_prior(0.1)

    # Read in some data
    obs_230 = eh.obsdata.load_uvfits('../uvfits/M87_230GHz.uvfits')
    obs_230.add_scans()
    obs_230 = obs_230.avg_coherent(0,scan_avg=True)
    #
    obs_345 = eh.obsdata.load_uvfits('../uvfits/M87_345GHz.uvfits')
    obs_345.add_scans()
    obs_345 = obs_345.avg_coherent(0,scan_avg=True)

    # Construct a combined frequency data set
    obs_345.data['time'] += 1e-8 # HACK to make new scans as far as ehtim is concerned
    obs = eh.obsdata.merge_obs([obs_230,obs_345],force_merge=True)

    # Set a list of observations for which to compare forecasts
    obslist = [obs_230,obs_345,obs]
    labels = ['ngEHT 230 GHz','ngEHT 345 GHz','ngEHT 230+345 GHz']

    # Choose a default image
    p = np.zeros(ff.size)
    rad2uas = 3600e6*180/np.pi
    for j in range(ff1.npx) :
        for i in range(ff1.npx) :
            p[i+ff1.npx*j] =  -((ff1.xcp[i,j]-5.0/rad2uas)**2+ff1.ycp[i,j]**2)*rad2uas**2/(2.0*25.0**2) + np.log( 0.4/( 2.0*np.pi * (25.0/rad2uas)**2 ) )
    p[-5] = 0.2
    p[-4] = 40
    p[-3] = 1
    p[-2] = 0
    p[-1] = 0
    p = np.array(p)

    # 1D Diameter
    ff.plot_1d_forecast(obslist,p,ff.size-4,labels=labels)
    plt.savefig('ring_forecast_1d.png',dpi=300)

    # 2D diameter vs width
    ff.plot_2d_forecast(obslist,p,ff.size-4,ff.size-3,labels=labels)
    plt.savefig('ring_forecast_2d.png',dpi=300)

    # Triangle
    plist = np.arange(len(p)-5,len(p))
    ff.plot_triangle_forecast(obslist,p,ilist=plist,labels=labels,axis_location=[0.1,0.1,0.88,0.88])
    plt.savefig('ring_forecast_tri.png',dpi=300)

    # # Diameter vs size
    # plt.figure(figsize=(5,4))
    # plt.axes([0.15,0.15,0.8,0.65])

    # dlist = np.logspace(-1,2,256)
    # Sigudlist = 0.0*dlist
    # Sigmdlist = 0.0*dlist
    # q = np.copy(p)
    # Sigdlist = 0.0*dlist
    # for i,d in enumerate(dlist) :
    #     q[-4] = d
    #     Sigu,Sigm = ff.uncertainties(obs_ngeht,q)
    #     Sigudlist[i] = Sigu[-4]
    #     Sigmdlist[i] = Sigm[-4]
    # plt.plot(dlist,Sigudlist,'-r',lw=0.5)
    # plt.plot(dlist[dlist>5],Sigmdlist[dlist>5],'-r',lw=2,label=labels[0])
    # Sigdlist = 0.0*dlist
    # for i,d in enumerate(dlist) :
    #     q[-4] = d
    #     Sigu,Sigm = ff.uncertainties(obs_ngeht2,q)
    #     Sigudlist[i] = Sigu[-4]
    #     Sigmdlist[i] = Sigm[-4]
    # plt.plot(dlist,Sigudlist,'-b',lw=0.5)
    # plt.plot(dlist[dlist>4.5],Sigmdlist[dlist>4.5],'-b',lw=2,label=labels[1])
    # Sigdlist = 0.0*dlist
    # for i,d in enumerate(dlist) :
    #     q[-4] = d
    #     Sigu,Sigm = ff.uncertainties(obs_ngeht_multifreq,q)
    #     Sigudlist[i] = Sigu[-4]
    #     Sigmdlist[i] = Sigm[-4]
    # plt.plot(dlist,Sigudlist,'-g',lw=0.5)
    # plt.plot(dlist[dlist>4],Sigmdlist[dlist>4],'-g',lw=2,label=labels[2])

    # plt.plot(dlist,dlist,':',color='grey')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.grid(True,alpha=0.25)
    # plt.xlabel(r'$d~({\rm \mu as})$')
    # plt.ylabel(r'$\sigma_d~({\rm \mu as})$')
    # plt.xlim((0.3,100))
    # plt.ylim((1e-3,150))
    # # Nominal super-resolution
    # plt.axhline(10,color='g',ls=':')
    # # Location of M87
    # plt.axvline(43.0,color='k',ls='--')
    # plt.text(3,3.5,'d')
    # plt.text(35,1,'M87',rotation=90)
    # # Volume axis
    # xlim=plt.xlim()
    # newxlim = (43.0/np.array(xlim))**3 
    # axt = plt.twiny()
    # axt.set_xlim(newxlim)
    # axt.set_xscale('log')
    # axt.set_xlabel(r'Volume Factor')
    # plt.savefig('ring_forecast_prec.png',dpi=300,bbox_inches='tight')
