import numpy as np
import copy

import ngEHTforecast.fisher.fisher_forecast as ff

# Some constants
uas2rad = np.pi/180./3600e6            
rad2uas = 1.0/(uas2rad)
sig2fwhm = np.sqrt(8.0*np.log(2.0))
fwhm2sig = 1.0/sig2fwhm

class FF_sum(ff.FisherForecast) :
    """
    FisherForecast object constructed from the sum of multiple 
    FisherForecast objects. For example, a binary might be generated from
    the sum of two Gaussians.  

    The parameter vector is constructed from the concatenation of the parameter
    vectors from each individual object, with all objects after the first
    gaining a pair of offset parameters.  That is:

    * p[0] ............ Obj1 p[0]
    * p[1] ............ Obj1 p[1]
    * ...
    * p[n1] ........... Obj2 p[0]
    * p[n1+1] ......... Obj2 p[0]
    * ...
    * p[n1+n2] ........ Obj2 dx
    * p[n1+n2+1] ...... Obj2 dy
    * p[n1+n2+2] ...... Obj3 p[0]
    * ...
    * p[n1+n2+n3+2] ... Obj3 dx
    * p[n1+n2+n3+3] ... Obj3 dy
    *  ...

    Prior lists are constructed similarly.
    

    Args:
      ff_list (list): List of FisherForecast objects to be summed. Additional objects can be added later. Default: None.
      stokes (str): Indicates if this is limited to Stokes I ('I') or include full polarization ('full'). Default: 'I'.
    """

    def __init__(self,ff_list=None,stokes='I') :
        super().__init__()
        self.ff_list = []
        self.stokes = stokes

        if (not ff_list is None) :
            self.ff_list = copy.copy(ff_list) # This is a SHALLOW copy

        self.size=0
        if (not ff_list is None) :
            self.size = ff_list[0].size
            for ff in self.ff_list[1:] :
                self.size += ff.size+2

        self.prior_sigma_list = []
        if (not ff_list is None) :
            if (len(ff_list[0].prior_sigma_list)==0) :
                self.prior_sigma_list.extend((ff_list[0].size)*[None])
            else :
                self.prior_sigma_list.extend(ff_list[0].prior_sigma_list)
            for ff in self.ff_list[1:] :
                if (len(ff.prior_sigma_list)==0) :
                    self.prior_sigma_list.extend((ff.size+2)*[None])
                else :
                    self.prior_sigma_list.extend(ff.prior_sigma_list+[None,None])

        if (not ff_list is None):
            if len(ff_list) > 0:
                self.stokes = ff_list[0].stokes
                if (len(ff_list) > 1):
                    for ff in self.ff_list[1:]:
                        if (ff.stokes != self.stokes):
                            raise(Exception('The model components in FF_sum do not have the same stokes type!'))

    def add(self,ff) :
        """
        Adds a FisherForecast object to the sum.

        Args:
          ff (FisherForecast): An existing FisherForecast object to add.
        """
        self.ff_list.append(ff)
        if (len(self.ff_list)==1) :
            self.size += ff.size
        else :
            self.size += ff.size+2

        if (len(ff.prior_sigma_list)==0) :
            if (len(self.ff_list)==1) :
                self.prior_sigma_list.extend(ff.size*[None])
            else :
                self.prior_sigma_list.extend((ff.size+2)*[None])
        else :
            if (len(self.ff_list)==1) :
                self.prior_sigma_list.extend(ff.prior_sigma_list)
            else :
                self.prior_sigma_list.extend(ff.prior_sigma_list+[None,None])
                
            
            
    def visibilities(self,obs,p,verbosity=0) :
        """
        Generates visibilities associated with the summed model.

        Args:
          obs (ehtim.Obsdata): ehtim data object
          p (numpy.ndarray): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities computed at observations.
        """
        
        k = 0
        # uas2rad = np.pi/180./3600e6

        if (self.stokes == 'I'):
            V = 0.0j*obs.data['u']
            for i,ff in enumerate(self.ff_list) :
                q = p[k:(k+ff.size)]

                if (i==0) :
                    shift_factor = 1.0
                    k += ff.size
                else :
                    dx = p[k+ff.size]
                    dy = p[k+ff.size+1]
                    shift_factor = np.exp( 2.0j*np.pi*(obs.data['u']*dx+obs.data['v']*dy)*uas2rad )
                    k += ff.size + 2
                    
                V = V + ff.visibilities(obs,q,verbosity=verbosity) * shift_factor
            return V

        else:
            RR = 0.0j*obs.data['u']
            LL = 0.0j*obs.data['u']
            RL = 0.0j*obs.data['u']
            LR = 0.0j*obs.data['u']
            for i,ff in enumerate(self.ff_list) :
                q = p[k:(k+ff.size)]

                if (i==0) :
                    shift_factor = 1.0
                    k += ff.size
                else :
                    dx = p[k+ff.size]
                    dy = p[k+ff.size+1]
                    shift_factor = np.exp( 2.0j*np.pi*(obs.data['u']*dx+obs.data['v']*dy)*uas2rad )
                    k += ff.size + 2
                    
                RR_prev, LL_prev, RL_prev, LR_prev = ff.visibilities(obs,q,verbosity=verbosity)
                RR += RR_prev * shift_factor
                LL += LL_prev * shift_factor
                RL += RL_prev * shift_factor
                LR += LR_prev * shift_factor

            return RR, LL, RL, LR

    def visibility_gradients(self,obs,p,verbosity=0) :
        """
        Generates visibility gradients associated with the summed model.

        Args:
          obs (ehtim.Obsdata): ehtim data object
          p (numpy.ndarray): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities computed at observations.
        """

        u = obs.data['u']
        v = obs.data['v']
        # uas2rad = np.pi/180./3600e6

        if (self.stokes == 'I'):
            gradV = []
            k = 0
            for i,ff in enumerate(self.ff_list) :
                q = p[k:(k+ff.size)]

                if (i==0) :
                    shift_factor = 1.0
                    k += ff.size
                else :
                    dx = p[k+ff.size]
                    dy = p[k+ff.size+1]
                    V = ff.visibilities(obs,q,verbosity=verbosity)
                    shift_factor = np.exp( 2.0j*np.pi*(u*dx+v*dy)*uas2rad )
                    k += ff.size + 2
                
                for gV in ff.visibility_gradients(obs,q,verbosity=verbosity).T :
                    gradV.append( gV*shift_factor )

                if (i>0) :
                    gradV.append(-2.0j*np.pi*u*shift_factor*V*uas2rad)
                    gradV.append(-2.0j*np.pi*v*shift_factor*V*uas2rad)

            gradV = np.array(gradV)

            return gradV.T

        else:
            gradRR = []
            gradLL = []
            gradRL = []
            gradLR = []
            k = 0
            for i,ff in enumerate(self.ff_list) :
                q = p[k:(k+ff.size)]

                if (i==0) :
                    shift_factor = 1.0
                    k += ff.size
                else :
                    dx = p[k+ff.size]
                    dy = p[k+ff.size+1]
                    RR,LL,RL,LR = ff.visibilities(obs,q,verbosity=verbosity)
                    shift_factor = np.exp( 2.0j*np.pi*(u*dx+v*dy)*uas2rad )
                    k += ff.size + 2
                
                gradRR_prev, gradLL_prev, gradRL_prev, gradLR_prev = ff.visibility_gradients(obs,q,verbosity=verbosity)
                
                # gradRR = gradRR_prev.T*shift_factor
                # gradLL = gradLL_prev.T*shift_factor
                # gradRL = gradRL_prev.T*shift_factor
                # gradLR = gradLR_prev.T*shift_factor

                for gRR in gradRR_prev.T:
                    gradRR.append(gRR*shift_factor)
                for gLL in gradLL_prev.T:
                    gradLL.append(gLL*shift_factor)
                for gRL in gradRL_prev.T:
                    gradRL.append(gRL*shift_factor)
                for gLR in gradLR_prev.T:
                    gradLR.append(gLR*shift_factor)

                if (i>0) :
                    # gRR_shift = np.array([-2.0j*np.pi*u*shift_factor*RR*uas2rad,-2.0j*np.pi*v*shift_factor*RR*uas2rad])
                    # gLL_shift = np.array([-2.0j*np.pi*u*shift_factor*LL*uas2rad,-2.0j*np.pi*v*shift_factor*LL*uas2rad])
                    # gRL_shift = np.array([-2.0j*np.pi*u*shift_factor*RL*uas2rad,-2.0j*np.pi*v*shift_factor*RL*uas2rad])
                    # gLR_shift = np.array([-2.0j*np.pi*u*shift_factor*LR*uas2rad,-2.0j*np.pi*v*shift_factor*LR*uas2rad])

                    # gradRR = np.concatenate((gradRR,gRR_shift))
                    # gradLL = np.concatenate((gradLL,gLL_shift))
                    # gradRL = np.concatenate((gradRL,gRL_shift))
                    # gradLR = np.concatenate((gradLR,gLR_shift))

                    gradRR.append(-2.0j*np.pi*u*shift_factor*RR*uas2rad)
                    gradRR.append(-2.0j*np.pi*v*shift_factor*RR*uas2rad)
                    gradLL.append(-2.0j*np.pi*u*shift_factor*LL*uas2rad)
                    gradLL.append(-2.0j*np.pi*v*shift_factor*LL*uas2rad)
                    gradRL.append(-2.0j*np.pi*u*shift_factor*RL*uas2rad)
                    gradRL.append(-2.0j*np.pi*v*shift_factor*RL*uas2rad)
                    gradLR.append(-2.0j*np.pi*u*shift_factor*LR*uas2rad)
                    gradLR.append(-2.0j*np.pi*v*shift_factor*LR*uas2rad)

            gradRR = np.array(gradRR)
            gradLL = np.array(gradLL)
            gradRL = np.array(gradRL)
            gradLR = np.array(gradLR)

            return gradRR.T, gradLL.T, gradRL.T, gradLR.T
                
    def parameter_labels(self,verbosity=0) :
        """
        Parameter labels to be used in plotting.

        Args:
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (list): List of strings with parameter labels.
        """        
        pll = []
        for i,ff in enumerate(self.ff_list) :
            for lbl in ff.parameter_labels() :
                pll.append(lbl)
            if (i>0) :
                pll.append(r'$\delta\Delta x~(\mu{\rm as})$')
                pll.append(r'$\delta\Delta y~(\mu{\rm as})$')
        
        return pll
