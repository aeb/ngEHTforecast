import ehtim as eh
   
obs = eh.obsdata.load_uvfits('../uvfits/M87_230GHz.uvfits')

obs.add_scans()
obs = obs.avg_coherent(0,scan_avg=True)

import numpy as np
   
print( "Unique stations:",np.unique((obs.data['t1'],obs.data['t2'])) )
print( "Unique scan times:",np.unique(obs.data['time']) )

obs = obs.add_fractional_noise(0.01)
   
obs2 = obs.flag_sites(['ALMA','JCMT','SMT','SPT','PV','PDB'])

print( "Unique stations after flagging:",np.unique((obs2.data['t1'],obs2.data['t2'])) )
