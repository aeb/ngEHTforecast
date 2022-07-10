import numpy as np
import matplotlib.pyplot as plt
import ehtim as eh
import fisher_package as fp
np.random.seed(12345)

#######################################
# specify which things to test

test_visibilities = False
test_gradients = False
test_single_epoch = True
test_multi_epoch = False

#######################################
# specify model parameters

F0 = 1.0
d = 40.0
alpha = d / 4.0
x0 = 0.0
y0 = 0.0

m = 1
beta_list = np.random.uniform(-0.5,0.5,size=1) + (1.0j)*np.random.uniform(-0.5,0.5,size=1)

mp = 2
bpol0 = np.random.uniform(-0.1,0.1) + (1.0j)*np.random.uniform(-0.1,0.1)
bpoln1 = np.random.uniform(-0.05,0.05) + (1.0j)*np.random.uniform(-0.05,0.05)
bpol1 = np.random.uniform(-0.05,0.05) + (1.0j)*np.random.uniform(-0.05,0.05)
bpoln2 = np.random.uniform(-0.1,0.1) + (1.0j)*np.random.uniform(-0.1,0.1)
bpol2 = np.random.uniform(-0.3,0.3) + (1.0j)*np.random.uniform(-0.3,0.3)
beta_list_pol = [bpoln2,bpoln1,bpol0,bpol1,bpol2]

mc = 0

#######################################
# generate model

mod = eh.model.Model()
mod = mod.add_thick_mring(F0=F0,
                          d=d*eh.RADPERUAS,
                          alpha=alpha*eh.RADPERUAS,
                          x0=x0*eh.RADPERUAS,
                          y0=y0*eh.RADPERUAS,
                          beta_list=beta_list,
                          beta_list_pol=beta_list_pol)

# parameter list
params = mod.params[0]
p = list()
p.append(params['F0'])
p.append(params['d']/eh.RADPERUAS)
p.append(params['alpha']/eh.RADPERUAS)
# p.append(params['x0']/eh.RADPERUAS)
# p.append(params['y0']/eh.RADPERUAS)
p.append(params['beta_list'])
p.append(params['beta_list_pol'])

#######################################
# read in some data

obs_ngeht = eh.obsdata.load_uvfits('../data/M87_230GHz_40uas.uvfits')
obs_ngeht.add_scans()
obs_ngeht = obs_ngeht.avg_coherent(0,scan_avg=True)
obs_ngeht = obs_ngeht.switch_polrep('circ')
# obs_ngeht = obs_ngeht.add_fractional_noise(0.01)

u = obs_ngeht.data['u']
v = obs_ngeht.data['v']

#######################################
# compute model visibilities

vis = eh.model.sample_1model_uv(u,v,'thick_mring',params,pol='I')

vis_RR = eh.model.sample_1model_uv(u,v,'thick_mring',params,pol='RR')
vis_LL = eh.model.sample_1model_uv(u,v,'thick_mring',params,pol='LL')
vis_RL = eh.model.sample_1model_uv(u,v,'thick_mring',params,pol='RL')
vis_LR = eh.model.sample_1model_uv(u,v,'thick_mring',params,pol='LR')

#######################################
# compute model gradients

grad = eh.model.sample_1model_grad_uv(u,v,'thick_mring',params,pol='I',fit_pol=True)

grad_RR = eh.model.sample_1model_grad_uv(u,v,'thick_mring',params,pol='RR',fit_pol=True,fit_cpol=True)
grad_LL = eh.model.sample_1model_grad_uv(u,v,'thick_mring',params,pol='LL',fit_pol=True,fit_cpol=True)
grad_RL = eh.model.sample_1model_grad_uv(u,v,'thick_mring',params,pol='RL',fit_pol=True,fit_cpol=True)
grad_LR = eh.model.sample_1model_grad_uv(u,v,'thick_mring',params,pol='LR',fit_pol=True,fit_cpol=True)

#######################################
# test out FF implementation

stokes = 'full'
ff0 = fp.FF_thick_mring(m,mp,mc,stokes=stokes)
# ff1 = fp.FF_sum([ff0])
ff1 = ff0

# set priors
# for i in range(8,ff1.size):
#     ff1.add_gaussian_prior(i,1e-10)

# build list of scalar parameters
pcheck = list()
for i in range(len(p)):
    if (type(p[i]) != np.ndarray):
        pcheck.append(p[i])
    else:
        for item in p[i]:
            pcheck.append(np.abs(item))
            pcheck.append(np.angle(item))

# check the wrapping of the parameter list
params2 = ff0.param_wrapper(pcheck)

# check visibilities
if test_visibilities:
    RR0,LL0,RL0,LR0 = ff0.visibilities(obs_ngeht,pcheck)
    RR1,LL1,RL1,LR1 = ff1.visibilities(obs_ngeht,pcheck)
    print('RR difference:',np.sum(RR0 - RR1))
    print('LL difference:',np.sum(LL0 - LL1))
    print('RL difference:',np.sum(RL0 - RL1))
    print('LR difference:',np.sum(LR0 - LR1))

# check gradients
if test_gradients:
    RR0,LL0,RL0,LR0 = ff0.visibility_gradients(obs_ngeht,pcheck)
    RR1,LL1,RL1,LR1 = ff1.visibility_gradients(obs_ngeht,pcheck)
    print('RR gradient difference:',np.sum(RR0 - RR1))
    print('LL gradient difference:',np.sum(LL0 - LL1))
    print('RL gradient difference:',np.sum(RL0 - RL1))
    print('LR gradient difference:',np.sum(LR0 - LR1))

# check gradient
h = 1.0e-4
if test_gradients:
    ff1.check_gradients(obs_ngeht,pcheck,h=h)

# check full-stokes covariance
fisher_covar0 = ff0.fisher_covar(obs_ngeht,pcheck)
fisher_covar = ff1.fisher_covar(obs_ngeht,pcheck)

covar = np.linalg.inv(fisher_covar)
diags = np.zeros(len(covar))
for i in range(len(covar)):
    diags[i] = np.sqrt(covar[i][i])
Sig_uni,Sig_marg = ff1.uncertainties(obs_ngeht,pcheck)

#######################################
# test out single-epoch gains

if test_single_epoch:

    print('-'*120)
    print('Testing single-epoch gains')
    print('-'*120)

    # tight gain priors
    print('computing tight gain priors...')
    ffgst = fp.FF_complex_gains_single_epoch(ff1)
    ffgst.set_gain_amplitude_prior(1e-10)
    ffgst.set_gain_phase_prior(1e-10)
    ffgst.set_gain_ratio_amplitude_prior(1e-10)
    ffgst.set_gain_ratio_phase_prior(1e-10)
    Sig_uni_gst,Sig_marg_gst = ffgst.uncertainties(obs_ngeht,pcheck)

    # loose gain priors
    print('computing loose gain priors...')
    ffgsl = fp.FF_complex_gains_single_epoch(ff1)
    ffgsl.set_gain_amplitude_prior(10.0)
    ffgsl.set_gain_phase_prior(1.0)
    ffgsl.set_gain_ratio_amplitude_prior(1e-10)
    ffgsl.set_gain_ratio_phase_prior(1e-10)
    Sig_uni_gsl,Sig_marg_gsl = ffgsl.uncertainties(obs_ngeht,pcheck)

    # loose gain ratio priors
    print('computing loose gain ratio priors...')
    ffgslr = fp.FF_complex_gains_single_epoch(ff1)
    ffgslr.set_gain_amplitude_prior(1e-10)
    ffgslr.set_gain_phase_prior(1e-10)
    ffgslr.set_gain_ratio_amplitude_prior(10.0)
    ffgslr.set_gain_ratio_phase_prior(1.0)
    Sig_uni_gslr,Sig_marg_gslr = ffgslr.uncertainties(obs_ngeht,pcheck)

    # loose gain + gain ratio priors
    print('computing loose gain+ratio priors...')
    ffgslrg = fp.FF_complex_gains_single_epoch(ff1)
    ffgslrg.set_gain_amplitude_prior(10.0)
    ffgslrg.set_gain_phase_prior(1.0)
    ffgslrg.set_gain_ratio_amplitude_prior(10.0)
    ffgslrg.set_gain_ratio_phase_prior(1.0)
    Sig_uni_gslrg,Sig_marg_gslrg = ffgslrg.uncertainties(obs_ngeht,pcheck)

    # print results
    print('='*120)
    header = 'parameter'.ljust(40)
    header += 'no G'.ljust(16)
    header += 'tight G+R'.ljust(16)
    header += 'loose G'.ljust(16)
    header += 'loose R'.ljust(16)
    header += 'loose G+R'
    print(header)
    param_labels = ffgslrg.parameter_labels()
    for i in range(len(param_labels)):
        strhere = param_labels[i].ljust(40)
        try:
            strhere += str(np.round(Sig_marg[i],10)).ljust(16)
        except:
            strhere += '...'.ljust(16)
        strhere += str(np.round(Sig_marg_gst[i],10)).ljust(16)
        strhere += str(np.round(Sig_marg_gsl[i],10)).ljust(16)
        strhere += str(np.round(Sig_marg_gslr[i],10)).ljust(16)
        strhere += str(np.round(Sig_marg_gslrg[i],10))
        print(strhere)
    print('='*120)

#######################################
# test out multi-epoch gains

if test_multi_epoch:

    print('-'*120)
    print('Testing multi-epoch gains')
    print('-'*120)

    # tight gain priors
    print('computing tight gain priors...')
    ffgst = fp.FF_complex_gains(ff1)
    ffgst.set_gain_epochs(scans=True)
    ffgst.set_gain_amplitude_prior(1e-10)
    ffgst.set_gain_phase_prior(1e-10)
    ffgst.set_gain_ratio_amplitude_prior(1e-10)
    ffgst.set_gain_ratio_phase_prior(1e-10)
    Sig_uni_gst,Sig_marg_gst = ffgst.uncertainties(obs_ngeht,pcheck)

    # loose gain priors
    print('computing loose gain priors...')
    ffgsl = fp.FF_complex_gains(ff1)
    ffgsl.set_gain_epochs(scans=True)
    ffgsl.set_gain_amplitude_prior(10.0)
    ffgsl.set_gain_phase_prior(1.0)
    ffgsl.set_gain_ratio_amplitude_prior(1e-10)
    ffgsl.set_gain_ratio_phase_prior(1e-10)
    Sig_uni_gsl,Sig_marg_gsl = ffgsl.uncertainties(obs_ngeht,pcheck)

    # loose gain ratio priors
    print('computing loose gain ratio priors...')
    ffgslr = fp.FF_complex_gains(ff1)
    ffgslr.set_gain_epochs(scans=True)
    ffgslr.set_gain_amplitude_prior(1e-10)
    ffgslr.set_gain_phase_prior(1e-10)
    ffgslr.set_gain_ratio_amplitude_prior(10.0)
    ffgslr.set_gain_ratio_phase_prior(1.0)
    Sig_uni_gslr,Sig_marg_gslr = ffgslr.uncertainties(obs_ngeht,pcheck)

    # loose gain + gain ratio priors
    print('computing loose gain+ratio priors...')
    ffgslrg = fp.FF_complex_gains(ff1)
    ffgslrg.set_gain_epochs(scans=True)
    ffgslrg.set_gain_amplitude_prior(10.0)
    ffgslrg.set_gain_phase_prior(1.0)
    ffgslrg.set_gain_ratio_amplitude_prior(10.0)
    ffgslrg.set_gain_ratio_phase_prior(1.0)
    Sig_uni_gslrg,Sig_marg_gslrg = ffgslrg.uncertainties(obs_ngeht,pcheck)

    # print results
    print('='*120)
    header = 'parameter'.ljust(40)
    header += 'no G'.ljust(16)
    header += 'tight G+R'.ljust(16)
    header += 'loose G'.ljust(16)
    header += 'loose R'.ljust(16)
    header += 'loose G+R'
    print(header)
    param_labels = ffgslrg.parameter_labels()
    for i in range(len(param_labels)):
        strhere = param_labels[i].ljust(40)
        try:
            strhere += str(np.round(Sig_marg[i],10)).ljust(16)
        except:
            strhere += '...'.ljust(16)
        strhere += str(np.round(Sig_marg_gst[i],10)).ljust(16)
        strhere += str(np.round(Sig_marg_gsl[i],10)).ljust(16)
        strhere += str(np.round(Sig_marg_gslr[i],10)).ljust(16)
        strhere += str(np.round(Sig_marg_gslrg[i],10))
        print(strhere)
    print('='*120)

