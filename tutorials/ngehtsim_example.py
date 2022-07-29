#######################################################
# imports

import ngEHTforecast as nf

import ngehtsim.obs.obs_generator as og
import ngehtsim.obs.obs_plotter as op

import matplotlib.pyplot as plt

#######################################################
# generate a FisherForecast object

ff = nf.fisher.FF_symmetric_gaussian()
p = [0.05,20.0]

#######################################################
# generate an observation

# initialize the observation generator
settings = {}
settings['weather'] = 'poor'
settings['source'] = 'Test'
settings['RA'] = 12.4852 # 12h 29m 06.7s
settings['DEC'] = 2.0525 # +02° 03′ 09″
settings['frequency'] = '230' # GHz
settings['array'] = 'ngEHTphase1'

# generate the observation by passing the FisherForecast object and parameters
obsgen = og.obs_generator(settings)
obs_poor = obsgen.make_obs(ff,p=p,addnoise=False,addgains=False)

# Change a setting and make another data set
settings['weather'] = 'good'
obsgen = og.obs_generator(settings)
obs_good = obsgen.make_obs(ff,p=p,addnoise=False,addgains=False)


# display data with ngEHTforecast
_,ax = nf.data.display_baselines(obs_good,color='b')
nf.data.display_baselines(obs_poor,axes=ax,color='r')
plt.savefig('ngehtsim_uv.png',dpi=300)
plt.close()

# display data with ngEHTforecast
_,axs = nf.data.display_visibilities(obs_good,color='b')
nf.data.display_visibilities(obs_poor,axs=axs,color='r')
plt.savefig('ngehtsim_vis.png',dpi=300)
plt.close()

# make triangle plot
ff.plot_triangle_forecast([obs_good,obs_poor],p,axis_location=[0.2,0.2,0.75,0.75],labels=[r'Good weather',r'Poor weather'])
plt.savefig('ngehtsim_tri.png',dpi=300)
plt.close()

