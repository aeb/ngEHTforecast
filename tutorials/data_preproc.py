import ngEHTforecast.data as fd
import ngEHTforecast.fisher as fp
import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt

# Read in some data
obs = eh.obsdata.load_uvfits('../uvfits/M87_230GHz.uvfits')

# Choose a FisherForecast object
ff = fp.FF_smoothed_delta_ring()

# Make one preprocessed data set
obs1 = fd.preprocess_obsdata(obs,ff=ff,p=[1.0,40,5],avg_time='scan')

# Make a second preprocessed data set
obs2 = fd.preprocess_obsdata(obs,ff=ff,p=[1.0,40,20],avg_time='scan',snr_cut=7,sys_err=0.5,const_err=1)

# Make a plot of the baseline maps
_,ax = fd.display_baselines(obs1)
fd.display_baselines(obs2,axes=ax,color='r')
plt.savefig('tutorial_data_bls.png',dpi=300)

# Make a plot of the visibilities
_,axs = fd.display_visibilities(obs1)
fd.display_visibilities(obs2,axs=axs,color='r')
plt.savefig('tutorial_data_vis.png',dpi=300)

# Make a plot of the trivial closure phases
fd.display_trivial_cphases(obs,print_outliers=True)
plt.savefig('tutorial_data_triv.png')






