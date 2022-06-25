import fisher_package as fp
import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12345)

#######################################

# make Fisher forecast object
stokes = 'I'
ff1 = fp.FF_splined_raster(5,100,stokes=stokes)
ff2 = fp.FF_thick_mring(1,0,0,stokes=stokes)
ffsum = fp.FF_sum([ff1,ff2])

# add gains
ff = fp.FF_complex_gains(ffsum)
ff.set_gain_epochs(scans=True)
ff.set_gain_amplitude_prior(0.1)

# Read in some data
obs_ngeht = eh.obsdata.load_uvfits('../data/M87_230GHz_40uas.uvfits')
obs_ngeht.add_scans()
obs_ngeht = obs_ngeht.avg_coherent(0,scan_avg=True)
obs_ngeht = obs_ngeht.switch_polrep('stokes')
#
obs_ngeht2 = eh.obsdata.load_uvfits('../data/M87_345GHz_40uas.uvfits')
obs_ngeht2.add_scans()
obs_ngeht2 = obs_ngeht2.avg_coherent(0,scan_avg=True)
obs_ngeht2 = obs_ngeht2.switch_polrep('stokes')
#
obs_ngeht2.data['time'] += 1e-8 # HACK to make new scans as far as ehtim is concerned
obs_ngeht_multifreq = eh.obsdata.merge_obs([obs_ngeht,obs_ngeht2],force_merge=True)

obslist = [obs_ngeht,obs_ngeht2,obs_ngeht_multifreq]
labels = ['ngEHT 230 GHz','ngEHT 345 GHz','ngEHT 230+345 GHz']

for o in obslist :
    print("Max |u|=%15.8g, Ndata=%15.8g"%(np.max(np.sqrt(o.data['u']**2+o.data['v']**2)),len(o.data)))

#######################################
# specify model parameters

p = list()

# add image parameters
ptemp = np.zeros(ff1.size)
rad2uas = 3600e6*180/np.pi
countI = 0
for j in range(ff1.npx) :
    for i in range(ff1.npx) :
        ptemp[i+ff1.npx*j] =  -((ff1.xcp[i,j]-5.0/rad2uas)**2+ff1.ycp[i,j]**2)*rad2uas**2/(2.0*25.0**2) + np.log( 0.4/( 2.0*np.pi * (25.0/rad2uas)**2 ) )
        countI += 1

for phere in ptemp:
    p.append(phere)

# add ring parameters
F0 = 0.2
d = 40.0
alpha = d / 3.0
x0 = 0.0
y0 = 0.0

m = 1
beta_list = np.random.uniform(-0.5,0.5,size=1) + (1.0j)*np.random.uniform(-0.5,0.5,size=1)

mp = 0
mc = 0

# set the model
mod = eh.model.Model()
mod = mod.add_thick_mring(F0=F0,
                          d=d*eh.RADPERUAS,
                          alpha=alpha*eh.RADPERUAS,
                          x0=x0*eh.RADPERUAS,
                          y0=y0*eh.RADPERUAS,
                          beta_list=beta_list)

# parameter list
params = mod.params[0]
pinit = list()
pinit.append(params['F0'])
pinit.append(params['d']/eh.RADPERUAS)
pinit.append(params['alpha']/eh.RADPERUAS)
pinit.append(params['beta_list'])

# build list of scalar parameters
for i in range(len(pinit)):
    if (type(pinit[i]) != np.ndarray):
        p.append(pinit[i])
    else:
        for item in pinit[i]:
            p.append(np.abs(item))
            p.append(np.angle(item))

# add shift parameters
p.append(0.0)
p.append(0.0)

#######################################

# Set systematic errors
for o in obslist :
    vis = ff.visibilities(o,p)
    o.data['vis'] = vis
    o = o.add_fractional_noise(0.01)

# 1D Diameter
fp.plot_1d_forecast(ff,p,len(p)-6,obslist,labels=labels)
plt.savefig('mring_StokesI_forecast_1d.png',dpi=300)
plt.close()

# 2D diameter vs width
fp.plot_2d_forecast(ff,p,len(p)-6,len(p)-5,obslist,labels=labels)
plt.savefig('mring_StokesI_forecast_2d.png',dpi=300)
plt.close()

# Triangle
plist = len(p)+np.array([-7,-6,-5,-4,-3])
fp.plot_triangle_forecast(ff,p,plist,obslist,labels=labels)
plt.savefig('mring_StokesI_forecast_tri.png',dpi=300)
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
    # q[0] = F0*((d/40.0)**2.0)
    q[-6] = d
    q[-5] = d/3.0
    Sigu,Sigm = ff.uncertainties(obs_ngeht,q)
    Sigudlist[i] = Sigu[-6]
    Sigmdlist[i] = Sigm[-6]
plt.plot(dlist[Sigudlist <= dlist],Sigudlist[Sigudlist <= dlist],'-r',lw=0.5)
indm = (Sigmdlist <= dlist) & (dlist > 1)
plt.plot(dlist[indm],Sigmdlist[indm],'-r',lw=2,label=labels[0])
Sigdlist = 0.0*dlist
for i,d in enumerate(dlist) :
    # q[0] = F0*((d/40.0)**2.0)
    q[-6] = d
    q[-5] = d/3.0
    Sigu,Sigm = ff.uncertainties(obs_ngeht2,q)
    Sigudlist[i] = Sigu[-6]
    Sigmdlist[i] = Sigm[-6]
plt.plot(dlist[Sigudlist <= dlist],Sigudlist[Sigudlist <= dlist],'-b',lw=0.5)
indm = (Sigmdlist <= dlist) & (dlist > 1)
plt.plot(dlist[indm],Sigmdlist[indm],'-b',lw=2,label=labels[1])
Sigdlist = 0.0*dlist
for i,d in enumerate(dlist) :
    # q[0] = F0*((d/40.0)**2.0)
    q[-6] = d
    q[-5] = d/3.0
    Sigu,Sigm = ff.uncertainties(obs_ngeht_multifreq,q)
    Sigudlist[i] = Sigu[-6]
    Sigmdlist[i] = Sigm[-6]
plt.plot(dlist[Sigudlist <= dlist],Sigudlist[Sigudlist <= dlist],'-g',lw=0.5)
indm = (Sigmdlist <= dlist) & (dlist > 1)
plt.plot(dlist[indm],Sigmdlist[indm],'-g',lw=2,label=labels[2])

plt.plot(dlist,dlist,':',color='grey')
plt.xscale('log')
plt.yscale('log')
plt.grid(True,alpha=0.25)
plt.xlabel(r'$d~({\rm \mu as})$')
plt.ylabel(r'$\sigma_d~({\rm \mu as})$')
plt.xlim((0.3,100))
plt.ylim((1e-3,150))
# Location of M87
plt.axvline(43.0,color='k',ls='--')
plt.text(3,4,'d')
plt.text(35,1,'M87',rotation=90)
plt.savefig('mring_StokesI_forecast_prec.png',dpi=300,bbox_inches='tight')
plt.close()
