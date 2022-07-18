import ngEHTforecast.fisher as fp
import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt
import themispy as ty

########################
### Read in test data set
###
obs_ngeht = eh.obsdata.load_uvfits('../uvfits/circ_gauss_x2_F0=3.0_FWHM=2.0_sep=5.0_angle=0_230GHz_ngeht_ref1.uvfits')
obs_ngeht.add_scans()
obs_ngeht = obs_ngeht.avg_coherent(0,scan_avg=True)

########################
### Create binary FisherForecast object
###
ff1 = fp.FF_symmetric_gaussian()
ff2 = fp.FF_symmetric_gaussian()
ff = fp.FF_sum([ff1,ff2])
ffg = fp.FF_complex_gains(ff)
ffg.set_gain_epochs(scans=True)
ffg.set_gain_amplitude_prior(0.1)

########################
### Set the "truth" and generate triangle plot
###
p = [1.5,2.0,1.5,2.0,5.0,0.0]
plist = [0,1,2,3,4,5]
plt.figure(figsize=(15,15))
fig,axs = ffg.plot_triangle_forecast(obs_ngeht,p,plist,axis_location=[0.085,0.085,0.9,0.9],alphas=0.25)
#
# Keep limits to set later (if need be)
xlim_list = {}
ylim_list = {}
for k in list(axs.keys()) :
    xlim_list[k] = axs[k].get_xlim()
    ylim_list[k] = axs[k].get_ylim()

 
    
########################
### Themis triangle plot overlay
###
chain = ty.chain.sample_chain('themis_chain.dat',parameter_list=[0,1,4,5,6,7],samples=25000,burn_fraction=0)
#
# Rescale to same units/parameter definitions
rad2uas = 3600e6 * 180.0/np.pi
chain[:,1] = 2.355*chain[:,1] * rad2uas
chain[:,3] = 2.355*chain[:,3] * rad2uas
chain[:,4] = -chain[:,4] * rad2uas # Difference in definition
chain[:,5] = chain[:,5] * rad2uas
#
# Remove the means (so we see just the uncertainties)
for j in range(6) :
    chain[:,j] = chain[:,j] - np.mean(chain[:,j])
#
labels = [r'$\delta I_1~({\rm Jy})$', r'$\delta{\rm FWHM}_1~(\mu{\rm as})$', r'$\delta I_2~({\rm Jy})$', r'$\delta{\rm FWHM}_2~(\mu{\rm as})$', r'$\delta x~(\mu{\rm as})$', r'$\delta y~(\mu{\rm as})$']
ty.vis.kde_triangle_plot(chain,alpha=0.75,axes=axs,labels=labels,scott_factor=2)
    

plt.savefig('themis_comparison.png',dpi=300)
plt.close()


quit()
