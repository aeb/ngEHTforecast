################################################
## Create observation objects
##

import ehtim as eh
import numpy as np
import ngEHTforecast.fisher as fp

obs = eh.obsdata.load_uvfits('../uvfits/M87_230GHz.uvfits')

obs.add_scans()
obs = obs.avg_coherent(0,scan_avg=True)
   
obs = obs.add_fractional_noise(0.01)
obs2 = obs.flag_sites(['ALMA','JCMT','SMT','SPT','PV','PDB'])


################################################
## Create FisherForecast objects
##

ff1 = fp.FF_symmetric_gaussian()
ff2 = fp.FF_asymmetric_gaussian()
ff = fp.FF_sum([ff1,ff2])

ffg = fp.FF_complex_gains(ff)
ffg.set_gain_epochs(scans=True)
ffg.set_gain_amplitude_prior(0.1)


################################################
## Create Forecast plots
##

p = [0.5,10, 1.0,20,0.5,0.3, 20,5]

import matplotlib.pyplot as plt

ffg.display_image(p)
plt.savefig('tutorial_image.png',dpi=300)


Sigma_obs = ffg.marginalized_uncertainties(obs,p,ilist=[0,2,6,7])
Sigma_obs2 = ffg.marginalized_uncertainties(obs2,p,ilist=[0,2,6,7])

print("Sigma's for obs:",Sigma_obs)
print("Sigma's for obs2:",Sigma_obs2)

ffg.plot_1d_forecast([obs2,obs],p,6,labels=['ngEHT','Reduced ngEHT'])
plt.savefig('tutorial_1d.png',dpi=300)

ffg.plot_2d_forecast([obs2,obs],p,6,7,labels=['ngEHT','Reduced ngEHT'])
plt.savefig('tutorial_2d.png',dpi=300)

ffg.plot_triangle_forecast([obs2,obs],p,labels=['ngEHT','Reduced ngEHT'],axis_location=[0.075,0.075,0.9,0.9])
plt.savefig('tutorial_tri.png',dpi=300)

