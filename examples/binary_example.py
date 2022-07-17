import ngEHTforecast.fisher as fp
import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt

# Read in some data and perform some preprocessing
obs1 = eh.obsdata.load_uvfits('../uvfits/M87_230GHz.uvfits')
obs1.add_scans()
obs1 = obs1.avg_coherent(0,scan_avg=True)

# Create a second data set for comparisons
obs2 = obs1.flag_sites(['ALMA','APEX','JCMT','SMA','SMT','LMT','SPT','PV','PDB','KP'])

# Set a list of observations for which to compare forecasts
obslist = [obs2,obs1]
labels = ['ngEHT w/o EHT','Full ngEHT']

# Construct a FisherForecast model
ff1 = fp.FF_symmetric_gaussian()
ff2 = fp.FF_symmetric_gaussian()
ff = fp.FF_sum([ff1,ff2])

# Add gain mitigation and set solution interval and amplitude priors. 
ffg = fp.FF_complex_gains(ff)
ffg.set_gain_epochs(scans=True)
ffg.set_gain_amplitude_prior(0.1)

# Choose a set of "truth" parameters
p = [0.1,20,0.05,20,10,15]

# Make and save a triangle
ff.plot_triangle_forecast(obslist,p,labels=labels,axis_location=[0.15,0.15,0.8,0.8])
plt.savefig('binary_forecast_tri.png',dpi=300)

# Plot of ngeht separation precision as functions of total flux and flux ratio
plt.figure(figsize=(5,4))
plt.axes([0.15,0.15,0.8,0.8])
lslist = ['--','-']
lbllist = ['10 mJy','1 mJy']
for j,ftot in enumerate([1e-2, 1]) :
    qlist = np.logspace(-4,0,16)
    Sigmalist1 = 0.0*qlist
    Sigmalist2 = 0.0*qlist
    for i,q in enumerate(qlist) :
        f1 = ftot/(1+q)
        f2 = ftot*q/(1+q)
        p = [f1, 20, f2, 20, 0, 20]
        Sigmalist1[i] = ffg.marginalized_uncertainties(obs1,p,ilist=5)
        Sigmalist2[i] = ffg.marginalized_uncertainties(obs2,p,ilist=5)
    plt.plot(qlist,Sigmalist1,'b',ls=lslist[j],label='Full ngEHT '+lbllist[j])
    plt.plot(qlist,Sigmalist2,'r',ls=lslist[j],label='ngEHT w/o EHT '+lbllist[j])

plt.legend()

plt.grid(True,alpha=0.25)
plt.xlabel(r'Flux Ratio')
plt.ylabel(r'Separation Uncertainty $(\mu{\rm as})$')
plt.xscale('log')
plt.yscale('log')

plt.axhline(p[-1],ls='--',color='k')
plt.xlim((1e-4,1))
plt.ylim(top=1e2)

plt.savefig('binary_forecast_prec.png',dpi=300)



    
