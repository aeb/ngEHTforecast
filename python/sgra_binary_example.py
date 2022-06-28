import fisher_package as fp
import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt

# Read in some data
obs_ngeht = eh.obsdata.load_uvfits('../data/M87_230GHz_40uas.uvfits')
obs_ngeht.add_scans()
obs_ngeht = obs_ngeht.avg_coherent(0,scan_avg=True)

obs = obs_ngeht.flag_sites(['ALMA','APEX','JCMT','SMA','SMT','LMT','SPT','PV','PDB','KP'])

obslist = [obs,obs_ngeht]
labels = ['ngEHT w/o EHT','Full ngEHT']

# Make a model object for which to generate forecasts
ff1 = fp.FF_splined_raster(5,50)
ff2 = fp.FF_symmetric_gaussian()
ffr = fp.FF_sum([ff1,ff2])
ff = fp.FF_complex_gains(ffr) # Marginalized over complex gains

ff.set_gain_epochs(scans=True)
ff.set_gain_amplitude_prior(0.1)


# Choose the "truth" model
p = np.zeros(ff.size)
rad2uas = 3600e6*180/np.pi  # radians to microarcseconds

# 1 Jy Gaussian with std dev of 25 uas.
for j in range(ff1.npx) :
    for i in range(ff1.npx) :
        p[i+ff1.npx*j] =  -(ff1.xcp[i,j]**2+ff1.ycp[i,j]**2)*rad2uas**2/(2.0*25.0**2) + np.log( 2.0/( 2.0*np.pi * (25.0/rad2uas)**2 ) )

# Symmetric Gaussian (masing star)
p[-4] = 0.001 # 0.1 Jy
p[-3] = 1 # 1 uas std dev.
p[-2] = 5000.0 # dRA in uas
p[-1] = 0.0 # dDec in uas


# Set systematic errors
for o in obslist :
    o.data['vis'] = ff.visibilities(o,p)
    o = o.add_fractional_noise(0.01)
    # o = o.flag_low_snr()
    

# Make a plot!
plist = [25,26,27,28] # Gaussiann parameters to plot
fp.plot_triangle_forecast(ff,p,plist,obslist,labels=labels)

plt.savefig('sgra_binary_forecast.png',dpi=300)
plt.show()




