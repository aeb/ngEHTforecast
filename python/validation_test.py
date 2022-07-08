import fisher_package as fp
import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt

# Read in some data
obs = eh.obsdata.load_uvfits('../data/circ_gauss_x2_F0=3.0_FWHM=2.0_sep=5.0_angle=0_230GHz_ngeht_ref1.uvfits')
obs.add_scans()
obs = obs.avg_coherent(0,scan_avg=True)
obs.data['sigma'] = 0*obs.data['sigma'] + 0.001 # mJy stuff
obs.data['t1'] = np.array( len(obs.data['t1'])*['AA'] )
obs.data['t2'] = np.array( len(obs.data['t2'])*['AP'] )

ff = fp.FF_symmetric_gaussian()


print("")

# Point source (only one parameter : total flux)
ff.add_gaussian_prior(1,0.001)
p = [1.0,0.001]
Sig,Sigm = ff.uncertainties(obs,p)
print("\n=== Point Source Test ===========================================")
print("  Uncertainties from FF -- marginalized / single: %15.8g / %15.8g"%(Sigm[0],Sig[0]))
Sig_exp = np.sqrt(1.0/(np.sum(obs.data['sigma']**-2)))
print("  Expected uncertainty from standard propagation: %15.8g"%(Sig_exp))

# Point source (only one parameter : total flux)
ffgs = fp.FF_complex_gains_single_epoch(ff)
ffgs.set_gain_amplitude_prior(1e-10)
ffgs.set_gain_phase_prior(1e-10)
p = [1.0,0.001]
Sig,Sigm = ffgs.uncertainties(obs,p)
print("\n=== Point Source w/ Single Gain & Strong Prior Test =============")
print("  Uncertainties from FF -- marginalized / single: %15.8g / %15.8g"%(Sigm[0],Sig[0]))
Sig_exp = np.sqrt(1.0/(np.sum(obs.data['sigma']**-2)))
print("  Expected uncertainty from standard propagation: %15.8g"%(Sig_exp))

# Point source (only one parameter : total flux)
print("\n=== Point Source w/ Single Gain & Weak Amp Prior Test ===========")
ffgs = fp.FF_complex_gains_single_epoch(ff)
ga = 0.01
ffgs.set_gain_amplitude_prior(1e-10)
ffgs.set_gain_amplitude_prior(ga,station='AA')
ffgs.set_gain_phase_prior(1e-10)
p = [1.0,0.001]
Sig,Sigm = ffgs.uncertainties(obs,p,verbosity=0)
print("  Uncertainties from FF -- marginalized / single: %15.8g / %15.8g"%(Sigm[0],Sig[0]))
# print("  Sig all: ",Sig)
# print("  Sigm all:",Sigm)
Sig_F = np.sqrt(1.0/(np.sum(obs.data['sigma']**-2)))
Sig_exp = p[0] * np.sqrt( (Sig_F/p[0])**2 + ga**2 )
print("  Expected uncertainty from standard propagation: %15.8g | "%(Sig_exp)) #,Sig_F,p[0],ga)
print("  Gain uncertainties from FF -- marginalized: %15.8g %15.8g %15.8g %15.8g"%tuple(Sigm[2:]))
print("  Expected uncertainties from priors:         %15.8g %15.8g %15.8g %15.8g"%tuple(ffgs.prior_sigma_list[2:]))


# Point source (only one parameter : total flux)
ffgs = fp.FF_complex_gains_single_epoch(ff)
ga = 0.01
ffgs.set_gain_amplitude_prior(1e-10)
ffgs.set_gain_amplitude_prior(ga,station='AA')
ffgs.set_gain_amplitude_prior(ga,station='AP')
ffgs.set_gain_phase_prior(1e-10)
p = [1.0,0.001]
Sig,Sigm = ffgs.uncertainties(obs,p,verbosity=0)
print("\n=== Point Source w/ Two Gain & Weak Amp Prior Test ==============")
print("  Uncertainties from FF -- marginalized / single: %15.8g / %15.8g"%(Sigm[0],Sig[0]))
Sig_F = np.sqrt(1.0/(np.sum(obs.data['sigma']**-2)))
Sig_exp = p[0] * np.sqrt( (Sig_F/p[0])**2 + ga**2 + ga**2)
print("  Expected uncertainty from standard propagation: %15.8g | "%(Sig_exp)) #,Sig_F,p[0],ga)


# ######### plotting log-likelihood after marginalization over gains
# N = len(obs.data['u'])
# sig = obs.data['sigma'][0]
# sigg = ga
# F,a = np.meshgrid(np.linspace(1-10*Sig[0],1+10*Sig[0],512),np.linspace(-0.001,0.001,512))
# # lL = -N*(F*np.exp(a)-1)**2/(2.0*sig**2) - a**2/(2.0*sigg**2)
# # lL = -N*(F*(1+a)-1)**2/(2.0*sig**2) - a**2/(2.0*sigg**2)
# lL = -N*(F+a-1)**2/(2.0*sig**2) - a**2/(2.0*sigg**2)
# P = np.exp(lL)
# P = P/np.sum(P)
# plt.pcolormesh(F,a,np.sqrt(-2.0*np.log(P)))
# plt.colorbar()
# # plt.contour(F,a,np.sqrt(-2.0*np.log(P)),colors='b',linestyles='-',levels=[0,1,2,3])
# plt.plot([],[],'-b',label='P(F,a)')
# plt.legend()
# plt.xlabel(r'$I~{\rm Jy}$')
# plt.ylabel(r'$a~{\mu{\rm as}}$')
# plt.savefig('gain_validation_2d.png',dpi=300)
# plt.show()


# General source prior check
ff.add_gaussian_prior(0,1e-10)
ff.add_gaussian_prior(1,1e-6)
p = [1.0,20.0]
Sig,Sigm = ff.uncertainties(obs,p)
print("\n=== Prior Test ==================================================")
print("  Uncertainties from FF -- marginalized: %15.8g %15.8g"%(Sigm[0],Sigm[1]))
Sig_exp = np.sqrt(1.0/(np.sum(obs.data['sigma']**-2)))
print("  Expected uncertainty from priors:      %15.8g %15.8g"%(ff.prior_sigma_list[0],ff.prior_sigma_list[1]))
ff.add_gaussian_prior(0,None)
ff.add_gaussian_prior(1,0.001)

# General source
ff.add_gaussian_prior(1,None)
p = [1.0,20.0]
fp.plot_2d_forecast(ff,p,0,1,[obs],labels=['Fisher Est.'])
Vd = ff.visibilities(obs,p)

Sig,Sigm = ff.uncertainties(obs,p)
sf = 5
x,y = np.meshgrid(np.linspace(-sf*Sig[0],sf*Sig[0],128),np.linspace(-sf*Sig[1],sf*Sig[1],128))
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
plt.ylabel(r'$\delta\sigma~({\mu{\rm as}})$')
plt.savefig('gaussian_validation_2d.png',dpi=300)

P = np.exp(-0.5*chi2)
P = P / np.sum(P)
sigx = np.sqrt(np.sum(P*x**2))
sigy = np.sqrt(np.sum(P*y**2))
print("\n=== Resolved Gaussian Marginalization Test ======================")
print("  1D marginalized Sigmas, direct:  %15.8g %15.8g"%(Sigm[0],Sigm[1]))
print("  1D marginalized Sigmas, from 2D: %15.8g %15.8g"%(sigx,sigy))

print("")

# plt.show()


