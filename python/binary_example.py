import fisher_package as fp
import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt

# Read in some data
obs_ngeht = eh.obsdata.load_uvfits('../data/M87_230GHz_40uas.uvfits')
obs_ngeht.add_scans()
obs_ngeht = obs_ngeht.avg_coherent(0,scan_avg=True)
# obs_ngeht = obs_ngeht.add_fractional_noise(0.01)

obs = obs_ngeht.flag_sites(['ALMA','APEX','JCMT','SMA','SMT','LMT','SPT','PV','PDB','KP'])


obslist = [obs,obs_ngeht]
labels = ['ngEHT w/o EHT','Full ngEHT']


ff1 = fp.FF_symmetric_gaussian()
ff2 = fp.FF_symmetric_gaussian()
ff = fp.FF_sum([ff1,ff2])

p = [0.1,20,0.05,20,10,15]

# Set systematic errors
for o in obslist :
    o.data['vis'] = ff.visibilities(o,p)
    o = o.add_fractional_noise(0.01)

    o = o.flag_low_snr()

    

# Triangle
plist = np.arange(len(p))
fp.plot_triangle_forecast(ff,p,plist,obslist,labels=labels)
plt.savefig('binary_forecast_tri.png',dpi=300)


## Plot of ngeht separation precision as functions of total flux and flux ratio
plt.figure(figsize=(5,4))
plt.axes([0.15,0.15,0.8,0.8])

lslist = [':','--','-']
lbllist = ['1 mJy','10 mJy','100 mJy']
for j,ftot in enumerate([1e-3, 1e-2, 1e-1]) :

    qlist = np.logspace(-4,0,32)
    Sigmlist = 0.0*qlist
    
    for i,q in enumerate(qlist) :
        f1 = ftot/(1+q)
        f2 = ftot*q/(1+q)
        p = [f1, 20, f2, 20, 0, 20]
        _,Sigm = ff.uncertainties(obs_ngeht,p)
        Sigmlist[i] = max(Sigm[-2],Sigm[-1])
    plt.plot(qlist,Sigmlist,'b',ls=lslist[j],label='Full ngEHT '+lbllist[j])

for j,ftot in enumerate([1e-3, 1e-2, 1e-1]) :

    qlist = np.logspace(-4,0,32)
    Sigmlist = 0.0*qlist

    for i,q in enumerate(qlist) :
        f1 = ftot/(1+q)
        f2 = ftot*q/(1+q)
        p = [f1, 20, f2, 20, 0, 20]
        _,Sigm = ff.uncertainties(obs,p)
        Sigmlist[i] = max(Sigm[-2],Sigm[-1])
    plt.plot(qlist,Sigmlist,'r',ls=lslist[j],label='ngEHT w/o EHT '+lbllist[j])


plt.legend()

plt.grid(True,alpha=0.25)
plt.xlabel(r'Flux Ratio')
plt.ylabel(r'Separation Uncertainty $(\mu{\rm as})$')
plt.xscale('log')
plt.yscale('log')

plt.axhline(p[-1],ls='--',color='k')
plt.xlim((1e-4,1))
plt.ylim(top=1e2)

plt.savefig('binary_forecast_prec.png',dpi=300)



## Plot of ngeht separation precision as functions of separation
if (False) : 
    plt.figure(figsize=(5,4))
    plt.axes([0.15,0.15,0.8,0.8])

    ffg = ff
    # ffg = fp.FF_complex_gains_single_epoch(ff)
    # ffg = fp.FF_complex_gains(ff)
    # ffg.set_gain_epochs(scans=True)
    # ffg.set_gain_amplitude_prior(0.1)

    seplist = np.linspace(1,40,32)
    lslist = [':','--','-']
    lbllist = ['1 mJy','10 mJy','100 mJy']
    # lslist = ['-']
    # lbllist = ['1 Jy']
    for j,ftot in enumerate([1e-3, 1e-2, 1e-1]) :
    # for j,ftot in enumerate([1.0]) :

        Sigmlist = 0.0*seplist

        q = 1

        print("Starting!")

        for i,s in enumerate(seplist) :
            f1 = ftot/(1+q)
            f2 = ftot*q/(1+q)
            p = [f1, 20, f2, 20, 0, s]
            _,Sigm = ffg.uncertainties(obs_ngeht,p)
            Sigmlist[i] = Sigm[5]
        plt.plot(seplist,Sigmlist,'b',ls=lslist[j],label='Full ngEHT '+lbllist[j])

        print("%15s %15s"%("Sep.","Sigma"))
        for i in range(len(seplist)) :
            print("%15.8g %15.8g"%(seplist[i],Sigmlist[i]))

        print("Full -------- %g %g ---------------"%(j,ftot))

    for j,ftot in enumerate([1e-3, 1e-2, 1e-1]) :

        Sigmlist = 0.0*seplist

        q = 1

        for i,s in enumerate(seplist) :
            f1 = ftot/(1+q)
            f2 = ftot*q/(1+q)
            p = [f1, 20, f2, 20, 0, s]
            _,Sigm = ffg.uncertainties(obs,p)
            Sigmlist[i] = Sigm[5]
        plt.plot(seplist,Sigmlist,'r',ls=lslist[j],label='ngEHT w/o EHT '+lbllist[j])

        print("Red. -------- %g %g ---------------"%(j,ftot))

    plt.legend()

    plt.grid(True,alpha=0.25)
    plt.xlabel(r'Separation $(\mu{\rm as})$')
    plt.ylabel(r'Separation Uncertainty $(\mu{\rm as})$')
    # plt.xscale('log')
    plt.yscale('log')

    plt.axhline(p[-1],ls='--',color='k')
    plt.xlim((0,40))
    #plt.ylim(top=1e2)

    plt.savefig('binary_forecast_separationo_prec.png',dpi=300)


## Make PA uncertainty plot for comparison with Freek's analysis
if (True) :
    plt.figure(figsize=(5,4))
    plt.axes([0.15,0.15,0.8,0.8])

    ffg = ff
    # ffg = fp.FF_complex_gains_single_epoch(ff)
    ffg = fp.FF_complex_gains(ff)
    ffg.set_gain_epochs(scans=True)
    ffg.set_gain_amplitude_prior(0.1)

    seplist = np.linspace(0,10,33)[1:]

    Sigmlist = 0.0*seplist

    for i,s in enumerate(seplist) :
        p = [1.5,2.0/2.35,1.5,2.0/2.35,s,0]
        # p = [0.75,20,0.75,20,s,0]
        _,Sigm = ffg.uncertainties(obs_ngeht,p)
        #Sigmlist[i] = Sigm[4]/s * 180.0/np.pi
        Sigmlist[i] = Sigm[5]/s * 180.0/np.pi * 180/np.pi #HACK?
    plt.plot(seplist,Sigmlist,'-b',label='Full ngEHT 230 GHz')

    plt.legend()

    plt.grid(True,alpha=0.25)
    plt.xlabel(r'Separation $(\mu{\rm as})$')
    plt.ylabel(r'PA Uncertainty (deg)')
    # plt.xscale('log')
    plt.yscale('log')

    plt.axhline(p[-1],ls='--',color='k')
    plt.xlim((0,10))
    # plt.ylim(top=1e2)
    plt.ylim((1e-3,2e1))

    plt.savefig('binary_forecast_PA_prec.png',dpi=300)


    
