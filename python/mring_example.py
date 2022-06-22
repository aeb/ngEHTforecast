import fisher_package as fp
import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12345)

#######################################

# make Fisher forecast object
ff = fp.FF_thick_mring(1,2,0)

# Read in some data
obs_ngeht = eh.obsdata.load_uvfits('../data/M87_230GHz_40uas.uvfits')
obs_ngeht.add_scans()
obs_ngeht = obs_ngeht.avg_coherent(0,scan_avg=True)
obs_ngeht = obs_ngeht.switch_polrep('circ')
#
obs_ngeht2 = eh.obsdata.load_uvfits('../data/M87_345GHz_40uas.uvfits')
obs_ngeht2.add_scans()
obs_ngeht2 = obs_ngeht2.avg_coherent(0,scan_avg=True)
obs_ngeht2 = obs_ngeht2.switch_polrep('circ')
#
obs_ngeht2.data['time'] += 1e-8 # HACK to make new scans as far as ehtim is concerned
obs_ngeht_multifreq = eh.obsdata.merge_obs([obs_ngeht,obs_ngeht2],force_merge=True)

obslist = [obs_ngeht,obs_ngeht2,obs_ngeht_multifreq]
labels = ['ngEHT 230 GHz','ngEHT 345 GHz','ngEHT 230+345 GHz']

for o in obslist :
    print("Max |u|=%15.8g, Ndata=%15.8g"%(np.max(np.sqrt(o.data['u']**2+o.data['v']**2)),len(o.data)))

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

# set the model
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
pinit = list()
pinit.append(params['F0'])
pinit.append(params['d']/eh.RADPERUAS)
pinit.append(params['alpha']/eh.RADPERUAS)
pinit.append(params['x0']/eh.RADPERUAS)
pinit.append(params['y0']/eh.RADPERUAS)
pinit.append(params['beta_list'])
pinit.append(params['beta_list_pol'])

# build list of scalar parameters
p = list()
for i in range(len(pinit)):
    if (type(pinit[i]) != np.ndarray):
        p.append(pinit[i])
    else:
        for item in pinit[i]:
            p.append(np.abs(item))
            p.append(np.angle(item))

#######################################

# Set systematic errors
for o in obslist :
    RR,LL,RL,LR = ff.visibilities(o,p,stokes='full')
    o.data['rrvis'] = RR
    o.data['llvis'] = LL
    o.data['rlvis'] = RL
    o.data['lrvis'] = LR
    o = o.add_fractional_noise(0.01)

# 1D Diameter
fp.plot_1d_forecast(ff,p,1,obslist,stokes='full',labels=labels)
plt.savefig('mring_forecast_1d.png',dpi=300)
plt.close()

# 2D diameter vs width
fp.plot_2d_forecast(ff,p,1,2,obslist,stokes='full',labels=labels)
plt.savefig('mring_forecast_2d.png',dpi=300)
plt.close()

# Triangle
plist = np.array([0,1,2,5,6,7,8,9,10,11,12,13,14,15,16])
fp.plot_triangle_forecast(ff,p,plist,obslist,stokes='full',labels=labels)
plt.savefig('mring_forecast_tri.png',dpi=300)
plt.close()

# Diameter vs size
plt.figure(figsize=(5,4))
plt.axes([0.15,0.15,0.8,0.65])

dlist = np.logspace(-1,2,256)
Sigudlist = 0.0*dlist
Sigmdlist = 0.0*dlist
q = np.copy(p)
Sigdlist = 0.0*dlist
for i,d in enumerate(dlist) :
    q[1] = d
    Sigu,Sigm = ff.uncertainties(obs_ngeht,q,stokes='full')
    Sigudlist[i] = Sigu[1]
    Sigmdlist[i] = Sigm[1]
plt.plot(dlist,Sigudlist,'-r',lw=0.5)
plt.plot(dlist[dlist>5],Sigmdlist[dlist>5],'-r',lw=2,label=labels[0])
Sigdlist = 0.0*dlist
for i,d in enumerate(dlist) :
    q[1] = d
    Sigu,Sigm = ff.uncertainties(obs_ngeht2,q,stokes='full')
    Sigudlist[i] = Sigu[1]
    Sigmdlist[i] = Sigm[1]
plt.plot(dlist,Sigudlist,'-b',lw=0.5)
plt.plot(dlist[dlist>4.5],Sigmdlist[dlist>4.5],'-b',lw=2,label=labels[1])
Sigdlist = 0.0*dlist
for i,d in enumerate(dlist) :
    q[1] = d
    Sigu,Sigm = ff.uncertainties(obs_ngeht_multifreq,q,stokes='full')
    Sigudlist[i] = Sigu[1]
    Sigmdlist[i] = Sigm[1]
plt.plot(dlist,Sigudlist,'-g',lw=0.5)
plt.plot(dlist[dlist>4],Sigmdlist[dlist>4],'-g',lw=2,label=labels[2])

plt.plot(dlist,dlist,':',color='grey')
plt.xscale('log')
plt.yscale('log')
plt.grid(True,alpha=0.25)
plt.xlabel(r'$d~({\rm \mu as})$')
plt.ylabel(r'$\sigma_d~({\rm \mu as})$')
plt.xlim((0.3,100))
plt.ylim((1e-3,150))
# Nominal super-resolution
plt.axhline(10,color='g',ls=':')
# Location of M87
plt.axvline(43.0,color='k',ls='--')
plt.text(3,3.5,'d')
plt.text(35,1,'M87',rotation=90)
plt.savefig('mring_forecast_prec.png',dpi=300,bbox_inches='tight')
plt.close()
