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

secondary_flux = np.logspace(-3,0,64)
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

plt.savefig('tutorial_sep.png',dpi=300)
