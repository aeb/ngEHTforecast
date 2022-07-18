import ngEHTforecast.fisher as fp
import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt


##########################################
### Read in test data 
###
obs = eh.obsdata.load_uvfits('../uvfits/circ_gauss_x2_F0=3.0_FWHM=2.0_sep=5.0_angle=0_230GHz_ngeht_ref1.uvfits')
obs.add_scans()
obs = obs.avg_coherent(0,scan_avg=True)
obs.data['sigma'] = 0*obs.data['sigma'] + 0.001 # mJy stuff

##########################################
### Define Gaussian FisherForecast object
###
ff = fp.FF_symmetric_gaussian()

##########################################
### Fill obs data with "truth" for later
###
p = [1.0,0.001]
obs.data['vis'] = ff.visibilities(obs,p)
obs.source = 'Test'


# Keep up to this point
obs_orig = obs.copy() # All gains


# Restrict by default to just two stations
obs.data['t1'] = np.array( len(obs.data['t1'])*['AA'] )
obs.data['t2'] = np.array( len(obs.data['t2'])*['AP'] )


print("")


#print("\n=== Point Source Test ===========================================")
ff.add_gaussian_prior(1,0.001)
p = [1.0,0.001]
Sigma = ff.marginalized_uncertainties(obs,p,ilist=0)
Sigma_exp = np.sqrt(1.0/(np.sum(obs.data['sigma']**-2)))
print("| %30s | %8.2g | %8.2g | %8.2e |"%("Without gains",Sigma,Sigma_exp,(Sigma-Sigma_exp)/Sigma_exp))


#print("\n=== Point Source w/ Single Gain & Weak Amp Prior Test ===========")
ffgs = fp.FF_complex_gains_single_epoch(ff)
ga = 0.1
ffgs.set_gain_amplitude_prior(1e-10)
ffgs.set_gain_amplitude_prior(ga,station='AA')
ffgs.set_gain_phase_prior(1e-10)
p = [1.0,0.001]
Sigma = ffgs.marginalized_uncertainties(obs,p,ilist=0)
Sigma_F = np.sqrt(1.0/(np.sum(obs.data['sigma']**-2)))
Sigma_exp = p[0] * np.sqrt( (Sigma_F/p[0])**2 + ga**2 )
print("| %30s | %8.2g | %8.2g | %8.2e |"%("Single gain",Sigma,Sigma_exp,(Sigma-Sigma_exp)/Sigma_exp))



#print("\n=== Point Source w/ Two Gains & Weak Amp Prior Test ==============")
ffgs = fp.FF_complex_gains_single_epoch(ff)
ga = 0.1
ffgs.set_gain_amplitude_prior(1e-10)
ffgs.set_gain_amplitude_prior(ga,station='AA')
ffgs.set_gain_amplitude_prior(ga,station='AP')
ffgs.set_gain_phase_prior(1e-10)
p = [1.0,0.001]
Sigma = ffgs.marginalized_uncertainties(obs,p,ilist=0)
Sigma_F = np.sqrt(1.0/(np.sum(obs.data['sigma']**-2)))
# Sigma_F = obs.data['sigma'][0]/np.sqrt(len(obs.data['time']))
Sigma_exp = p[0] * np.sqrt( (Sigma_F/p[0])**2 + ga**2 + ga**2)
print("| %30s | %8.2g | %8.2g | %8.2e |"%("Two gains",Sigma,Sigma_exp,(Sigma-Sigma_exp)/Sigma_exp))


#print("\n=== Point Source w/ Single Double-epoch Gains & Weak Amp Prior Test ===========")
obs_double_time = obs.copy()
t1 = obs_double_time.data['time'][len(obs_double_time.data['time'])//2]
N1 = len(obs_double_time.data['time'][obs_double_time.data['time']<=t1])
N2 = len(obs_double_time.data['time'][obs_double_time.data['time']>t1])
obs_double_time.data['time'][:N1] = 0.0*obs_double_time.data['time'][:N1] + obs_double_time.data['time'][0]
obs_double_time.data['time'][N1:] = 0.0*obs_double_time.data['time'][N1:] + t1
ffg = fp.FF_complex_gains(ff)
ffg.set_gain_epochs(scans=True)
ga = 0.1
ffg.set_gain_amplitude_prior(1e-10,verbosity=0)
ffg.set_gain_amplitude_prior(ga,station='AA',verbosity=0)
ffg.set_gain_phase_prior(1e-10,verbosity=0)
p = [1.0,0.001]
Sigma = ffg.marginalized_uncertainties(obs_double_time,p,ilist=0)
Sigma_F1 = obs.data['sigma'][0]/np.sqrt(N1)
Sigma_F2 = obs.data['sigma'][0]/np.sqrt(N2)
Sigma_exp = np.sqrt( 1.0/( 1.0/(Sigma_F1**2+(ga*p[0])**2) + 1.0/(Sigma_F2**2+(ga*p[0])**2) ) )
print("| %30s | %8.2g | %8.2g | %8.2e |"%("Single gain, two epochs",Sigma,Sigma_exp,(Sigma-Sigma_exp)/Sigma_exp))


#print("\n=== Point Source w/ Two Double-epoch Gains & Weak Amp Prior Test & Weak Phases ===")
obs_double_time = obs.copy()
t1 = obs_double_time.data['time'][len(obs_double_time.data['time'])//2]
N1 = len(obs_double_time.data['time'][obs_double_time.data['time']<=t1])
N2 = len(obs_double_time.data['time'][obs_double_time.data['time']>t1])
obs_double_time.data['time'][:N1] = 0.0*obs_double_time.data['time'][:N1] + obs_double_time.data['time'][0]
obs_double_time.data['time'][N1:] = 0.0*obs_double_time.data['time'][N1:] + t1
ffg = fp.FF_complex_gains(ff)
ffg.set_gain_epochs(scans=True)
ga = 0.1
ffg.set_gain_amplitude_prior(1e-10,verbosity=0)
ffg.set_gain_amplitude_prior(ga,station='AA',verbosity=0)
ffg.set_gain_amplitude_prior(ga,station='AP',verbosity=0)
ffg.set_gain_phase_prior(100.0,verbosity=0)
p = [1.0,0.001]
Sigma = ffg.marginalized_uncertainties(obs_double_time,p,ilist=0)
Sigma_F1 = obs.data['sigma'][0]/np.sqrt(N1)
Sigma_F2 = obs.data['sigma'][0]/np.sqrt(N2)
Sigma_exp = np.sqrt( 1.0/( 1.0/(Sigma_F1**2+(ga*p[0])**2+(ga*p[0])**2) + 1.0/(Sigma_F2**2+(ga*p[0])**2+(ga*p[0])**2) ) )
print("| %30s | %8.2g | %8.2g | %8.2e |"%("Two gains, two epochs",Sigma,Sigma_exp,(Sigma-Sigma_exp)/Sigma_exp))



print("\n=== 2D Test ==================================================")
# 2D comparison Test
ff.add_gaussian_prior(1,None)
p = [1.0,20.0]
ff.plot_2d_forecast(obs,p,0,1,labels=['Fisher Est.'])
Vd = ff.visibilities(obs,p)

Sigma = ff.marginalized_uncertainties(obs,p)
sf = 5
x,y = np.meshgrid(np.linspace(-sf*Sigma[0],sf*Sigma[0],128),np.linspace(-sf*Sigma[1],sf*Sigma[1],128))
chi2 = 0*x
for i in range(x.shape[0]) :
    for j in range(x.shape[1]) :
        q = [p[0]+x[i,j],p[1]+y[i,j]]
        Vm = ff.visibilities(obs,q)
        chi2[i,j] = np.sum( (Vd.real-Vm.real)**2/obs.data['sigma']**2 ) + np.sum( (Vd.imag-Vm.imag)**2/obs.data['sigma']**2 )
plt.contour(x,y,np.sqrt(chi2),colors='b',linestyles='--',levels=[0,1,2,3])
plt.plot([],[],'--b',label='Grid search')
plt.gcf().set_size_inches(5.5,5)
plt.gca().set_position([0.225,0.15,0.7,0.8])
plt.legend()
plt.xlabel(r'$\delta I~({\rm Jy})$')
plt.ylabel(r'$\delta{\rm FWHM}~({\mu{\rm as}})$')
plt.savefig('gaussian_validation_2d.png',dpi=300)

print("\n=== Resolved Gaussian Marginalization Test ======================")
P = np.exp(-0.5*chi2)
P = P / np.sum(P)
sigx = np.sqrt(np.sum(P*x**2))
sigy = np.sqrt(np.sum(P*y**2))
print("  1D marginalized Sigmas, direct:  %8.3g %8.3g"%(Sigma[0],Sigma[1]))
print("  1D marginalized Sigmas, from 2D: %8.3g %8.3g"%(sigx,sigy))

print("")



# print("\n=== Prior Test w/ Many Gains ====================================")
# ff.add_gaussian_prior(0,None)
# ff.add_gaussian_prior(1,None)
# ffg.set_gain_epochs(scans=True)
# ga = 0.1
# ffg.set_gain_amplitude_prior(ga,verbosity=0)
# ffg.set_gain_phase_prior(100.0,verbosity=0)
# p = [1.0,20.0]
# Sig,Sigm = ffg.uncertainties(obs,p)
# print("  Uncertainties from FF -- marginalized: %15.8g %15.8g"%(Sigm[0],Sigm[1]))
# Sig_exp = np.sqrt(1.0/(np.sum(obs.data['sigma']**-2)))
# print("  Expected uncertainty from priors:      ???")
# fp.plot_2d_forecast(ffg,p,0,1,[obs],labels=['Fisher Est.'])
# plt.gcf().set_size_inches(5.5,5)
# plt.gca().set_position([0.225,0.15,0.7,0.8])
# plt.legend()
# plt.xlabel(r'$\delta I~({\rm Jy})$')
# plt.ylabel(r'$\delta\sigma~({\mu{\rm as}})$')
# plt.savefig('gaussian_validation_wgains_2d.png',dpi=300)
# obs_orig.data['vis'] = ff.visibilities(obs_orig,p)
# obs_orig.save_uvfits('symmetric_gaussian_large_sigma.uvfits',polrep_out='stokes')


# plt.show()


