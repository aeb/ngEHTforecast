import fisher_package as fp
import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt


# Make Fisher forecast object
ff1 = fp.FF_splined_raster(5,100)
ff2 = fp.FF_smoothed_delta_ring()
ff = fp.FF_sum([ff1,ff2])

# Read in some data
obs_ngeht = eh.obsdata.load_uvfits('../data/M87_230GHz_40uas.uvfits')
obs_ngeht.add_scans()
obs_ngeht = obs_ngeht.avg_coherent(0,scan_avg=True)
#
obs_ngeht2 = eh.obsdata.load_uvfits('../data/M87_345GHz_40uas.uvfits')
obs_ngeht2.add_scans()
obs_ngeht2 = obs_ngeht2.avg_coherent(0,scan_avg=True)
#
obs_ngeht2.data['time'] += 1e-8 # HACK to make new scans as far as ehtim is concerned
obs_ngeht_multifreq = eh.obsdata.merge_obs([obs_ngeht,obs_ngeht2],force_merge=True)


obslist = [obs_ngeht,obs_ngeht2,obs_ngeht_multifreq]
labels = ['ngEHT 230 GHz','ngEHT 345 GHz','ngEHT 230+345 GHz']

for o in obslist :
    print("Max |u|=%15.8g, Ndata=%15.8g"%(np.max(np.sqrt(o.data['u']**2+o.data['v']**2)),len(o.data)))


    

# Choose a default image
p = np.zeros(ff.size)
rad2uas = 3600e6*180/np.pi
for j in range(ff1.npx) :
    for i in range(ff1.npx) :
        p[i+ff1.npx*j] =  -((ff1.xcp[i,j]-5.0/rad2uas)**2+ff1.ycp[i,j]**2)*rad2uas**2/(2.0*25.0**2) + np.log( 0.4/( 2.0*np.pi * (25.0/rad2uas)**2 ) )
p[-5] = 0.2
p[-4] = 40
p[-3] = 1
p[-2] = 0
p[-1] = 0
p = np.array(p)

# Set systematic errors
for o in obslist :
    o.data['vis'] = ff.visibilities(o.data['u'],o.data['v'],p)
    o = o.add_fractional_noise(0.01)

# 1D Diameter
fp.plot_1d_forecast(ff,p,ff.size-4,obslist,labels=labels)
plt.savefig('ring_forecast_1d.png',dpi=300)

# 2D diameter vs width
fp.plot_2d_forecast(ff,p,ff.size-4,ff.size-3,obslist,labels=labels)
plt.savefig('ring_forecast_2d.png',dpi=300)

# Triangle
plist = np.arange(len(p)-5,len(p))
fp.plot_triangle_forecast(ff,p,plist,obslist,labels=labels)
plt.savefig('ring_forecast_tri.png',dpi=300)

# Diameter vs size
plt.figure(figsize=(5,4))
plt.axes([0.15,0.15,0.8,0.65])

dlist = np.logspace(-1,2,256)
Sigudlist = 0.0*dlist
Sigmdlist = 0.0*dlist
q = np.copy(p)
Sigdlist = 0.0*dlist
for i,d in enumerate(dlist) :
    q[-4] = d
    Sigu,Sigm = ff.uncertainties_from_obs(obs_ngeht,q)
    Sigudlist[i] = Sigu[-4]
    Sigmdlist[i] = Sigm[-4]
plt.plot(dlist,Sigudlist,'-r',lw=0.5)
plt.plot(dlist[dlist>5],Sigmdlist[dlist>5],'-r',lw=2,label=labels[0])
Sigdlist = 0.0*dlist
for i,d in enumerate(dlist) :
    q[-4] = d
    Sigu,Sigm = ff.uncertainties_from_obs(obs_ngeht2,q)
    Sigudlist[i] = Sigu[-4]
    Sigmdlist[i] = Sigm[-4]
plt.plot(dlist,Sigudlist,'-b',lw=0.5)
plt.plot(dlist[dlist>4.5],Sigmdlist[dlist>4.5],'-b',lw=2,label=labels[1])
Sigdlist = 0.0*dlist
for i,d in enumerate(dlist) :
    q[-4] = d
    Sigu,Sigm = ff.uncertainties_from_obs(obs_ngeht_multifreq,q)
    Sigudlist[i] = Sigu[-4]
    Sigmdlist[i] = Sigm[-4]
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
# Volume axis
xlim=plt.xlim()
newxlim = (43.0/np.array(xlim))**3 
axt = plt.twiny()
axt.set_xlim(newxlim)
axt.set_xscale('log')
axt.set_xlabel(r'Volume Factor')
plt.savefig('ring_forecast_prec.png',dpi=300,bbox_inches='tight')
