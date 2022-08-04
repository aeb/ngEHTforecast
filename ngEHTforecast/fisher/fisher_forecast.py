import numpy as np
import matplotlib.pyplot as plt
import copy
import hashlib

# import scipy.linalg as linalg
import numpy.linalg as linalg

def _print_matrix(m) :
    for i in range(m.shape[0]) :
        line = ""
        for j in range(m.shape[1]) :
            line = line + ' %10.3g'%(m[i][j])
        print(line)

def _print_vector(v) :
    line = ""
    for i in range(v.shape[0]) :
        line = line + ' %10.3g'%(v[i])
    print(line)

def _invert_matrix(a) :
    tmp = linalg.inv(a)
    return 0.5*(tmp+tmp.T)
    # return linalg.inv(a)
    # return linalg.pinvh(a)
    # n = a.shape[0]
    # I = np.identity(n)
    # return linalg.solve(a, I, sym_pos = True, overwrite_b = True)

def _vMv(v,M) :
    # return np.matmul(v.T,np.matmul(M,v))
    tmp = np.matmul(v.T,np.matmul(M,v))
    return 0.5*(tmp+tmp.T)
    
    
# Some constants
uas2rad = np.pi/180./3600e6            
rad2uas = 1.0/(uas2rad)
sig2fwhm = np.sqrt(8.0*np.log(2.0))
fwhm2sig = 1.0/sig2fwhm    
    
class FisherForecast :
    """
    Class that collects and contains information for making Fisher-matrix type
    forecasts for observations of various types.  Forms a base class for fully
    analytical versions that may be application specific.

    Attributes:
      size (int): Number of parameters expected by this model.
      stokes (str): If this is a Stokes I model ('I') or a polarized model ('full').
      prior_sigma_list (list): List of standard deviations associated with the parameter priors.
      covar (numpy.ndarray): Internal space for the computation of the covariance matrix.
      argument_hash (str): MD5 hash object indicating last state of covariance computation. Used to determine if the covariance needs to be recomputed.
    """

    def __init__(self) :
        self.size = 0
        self.argument_hash = None
        self.inv_argument_hash = None
        self.prior_sigma_list = []
        self.stokes = 'I'
        self.covar = None
        self.invcovar= None

        self.default_color_list = ['r','b','g','orange','purple']
        self.default_cmap_list = ['Reds_r','Blues_r','Greens_r','Oranges_r','Purples_r']

        
    def visibilities(self,obs,p,verbosity=0) :
        """
        User-defined function in child classes that generates visibilities 
        associated with a given model image object.

        Args:
          obs (ehtim.obsdata.Obsdata): An ehtim Obsdata object with a particular set of observation details (u,v positions, times, etc.)
          p (numpy.ndarray): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities computed at observations.
        """

        raise(RuntimeError("visibilities function not implemented in base class!"))
        return 0*obs.data['u']

            
    def visibility_gradients(self,obs,p,verbosity=0,**kwargs) :
        """
        User-defined function in child classes that generates visibility gradients
        associated with a given model image object.

        Args:
          obs (ehtim.obsdata.Obsdata): An ehtim Obsdata object with a particular set of observation details (u,v positions, times, etc.)
          p (numpy.ndarray): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities gradients computed at observations.
        """

        raise(RuntimeError("visibility_gradients function not implemented in base class!"))
        return 0*obs.data['u']


    def parameter_labels(self,verbosity=0) :
        """
        User-defined function in child classes that returns a list of
        of parameter labels to be used in plotting.

        Args:
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (list): List of strings with parameter labels.
        """

        raise(RuntimeError("parameter_labels function not implemented in base class!"))
        return []

    
    def add_gaussian_prior(self,pindex,sigma,verbosity=0) :
        """
        Add a Gaussian prior on the parameter with the specified index.

        Args:
          pindex (int): Index of the parameter on which to place the prior.
          sigma (float): Standard deviation of the Gaussian prior to apply.
          verbosity (int): Verbosity level. Default: 0.
        """

        if (pindex>self.size) :
            raise(RuntimeError("Parameter %i does not exist, expected in [0,%i]."%(pindex,self.size-1)))
        if (len(self.prior_sigma_list)==0) :
            self.prior_sigma_list = self.size*[None]
        self.prior_sigma_list[pindex] = sigma

        self.argument_hash = None
        
        
    def add_gaussian_prior_list(self,sigma_list,verbosity=0) :
        """
        Add a Gaussian priors on all of the model parameters.

        Args:
          sigma_list (list): List of standard deviations of the Gaussian priors to apply.
          verbosity (int): Verbosity level. Default: 0.
        """

        self.prior_sigma_list = copy.copy(sigma_list)
        if (len(self.prior_sigma_list)!=self.size) :
            raise(RuntimeError("Priors must be specified for all parameters if set by list.  If sigma is None, no prior will be applied."))
        self.argument_hash = None
        

    def generate_image(self,p,limits=None,shape=None,verbosity=0) :
        """
        Generate and return an image for the parent visibility model evaluated
        at a given set of parameter values.  The user is responsible for setting
        the plot limits and shape intelligently.  Note that this uses FFTs and 
        assumes that the resolution and field of view are sufficient to adequately
        resolve model image features.

        Args:
          p (list): List of parameter values.
          limits (float,list): Limits on the field of view in which to construct the image in uas.  Either a single float, in which case the limits are symmetric and set to [-limits,limits,-limits,limits], or a list of floats. Default: 100.
          shape (int,list): Dimensions of the image.  If an int, the dimensions are equal.  If a two-element list of ints, sets the dimensions to the two values. Default: 256.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray,numpy.ndarray,numpy.ndarray): x positions of the pixel centers in uas, y positions of the pixel centers in uas, intensities at pixel centers in Jy/uas^2.
        """

        # Fix image limits
        if (limits is None) :
            limits = [-100,100,-100,100]
        elif (isinstance(limits,list)) :
            limits = limits
        else :
            limits = [-limits, limits, -limits, limits]

        # Set shape
        if (shape is None) :
            shape = [256, 256]
        elif (isinstance(shape,list)) :
            shape = shape
        else :
            shape = [shape, shape]
            
        if (verbosity>0) :
            print("limits:",limits)
            print("shape:",shape)

        umax = 0.5*shape[0]/(uas2rad*(limits[1]-limits[0]))
        vmax = 0.5*shape[1]/(uas2rad*(limits[3]-limits[2]))

        if (verbosity>0) :
            print("Max (u,v) = ",(umax,vmax))


        # FIX
        class obsempty :
            data = {}

        
        u,v = np.mgrid[-umax:umax:(shape[0])*1j, -vmax:vmax:(shape[1])*1j]
        obsgrid = obsempty()
        obsgrid.data['u'] = u
        obsgrid.data['v'] = v

        V = self.visibilities(obsgrid,p,verbosity=verbosity)
        
        I = np.abs(np.fft.fftshift(np.fft.fft2(V)))
        x1d = np.fft.fftshift(np.fft.fftfreq(I.shape[0],np.abs(u[1,1]-u[0,0])))
        y1d = np.fft.fftshift(np.fft.fftfreq(I.shape[1],np.abs(v[1,1]-v[0,0])))

        x,y = np.meshgrid(x1d,y1d,indexing='ij')
        x = x/uas2rad
        y = y/uas2rad

        obsgrid.data['u'] = 0.0
        obsgrid.data['v'] = 0.0
        I = I * np.abs(self.visibilities(obsgrid,p,verbosity=verbosity))/np.sum(I) / ((x[1,1]-x[0,0])*(y[1,1]-y[0,0]))

        if (verbosity>0) :
            print("V00:",self.visibilities(obsgrid,p))
            print("Sum I:", np.sum(I) * ((x[1,1]-x[0,0])*(y[1,1]-y[0,0])) )
        
        return x,y,I

    
    def display_image(self,p,limits=None,shape=None,verbosity=0,**kwargs) :
        """
        Generate and plot an image for the parent visibility model evaluated
        at a given set of parameter values.  The user is responsible for setting
        the plot limits and shape intelligently.  Note that this uses FFTs and 
        assumes that the resolution and field of view are sufficient to adequately
        resolve model image features.

        Args:
          p (list): List of parameter values.
          limits (float,list): Limits on the field of view in which to construct the image in uas.  Either a single float, in which case the limits are symmetric and set to [-limits,limits,-limits,limits], or a list of floats. Default: 100.
          shape (int,list): Dimensions of the image.  If an int, the dimensions are equal.  If a two-element list of ints, sets the dimensions to the two values. Default: 256.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (matplotlib.pyplot.figure,matplotlib.pyplot.axes,matplotlib.pyplot.colorbar): Handles to the figure, axes, and colorbar.
        """

        plt.figure(figsize=(6.5,5))
        axs=plt.axes([0.15,0.15,0.8*5/6.5,0.8])

        x,y,I = self.generate_image(p,limits=limits,shape=shape,verbosity=verbosity)

        Imax = np.max(I)
        if (Imax>1e-1) :
            fu = r'$I~({\rm Jy}/\mu{\rm as}^2)$'
        elif (Imax>1e-4) :
            fu = r'$I~({\rm mJy}/\mu{\rm as}^2)$'
            I = I*1e3
        elif (Imax>1e-7) :
            fu = r'$I~({\rm \mu Jy}/\mu{\rm as}^2)$'
            I = I*1e6
        elif (Imax>1e-10) :
            fu = r'$I~({\rm nJy}/\mu{\rm as}^2)$'
            I = I*1e9
        elif (Imax>1e-13) :
            fu = r'$I~({\rm pJy}/\mu{\rm as}^2)$'
            I = I*1e12

        plt.pcolormesh(x,y,I,cmap='afmhot',vmin=0,shading='auto')

        plt.xlabel(r'$\Delta{\rm RA}~(\mu{\rm as})$')
        plt.ylabel(r'$\Delta{\rm Dec}~(\mu{\rm as})$')

        plt.gca().invert_xaxis()
        
        cbax = plt.axes([0.8*5/6.5+0.05+0.15,0.15,0.05,0.8])
        plt.colorbar(cax=cbax)
        cbax.set_ylabel(fu,rotation=-90,ha='center',va='bottom')

        plt.sca(axs)
        
        return plt.gcf(),axs,cbax
        
        

    def check_gradients(self,obs,p,h=None,verbosity=0) :
        """
        Numerically evaluates the gradients using the visibilities function and
        compares them to the gradients returned by the visibility_gradients
        function.  Numerical differentiation is 2nd-order, centered finite
        differences.  Results are output to standard out.  For best results,
        the user should pass some option for the step size h to use for numerical 
        differentiation.

        Args:
          obs (ehtim.obsdata.Obsdata): An ehtim Obsdata object with a particular set of observation details (u,v positions, times, etc.) at which to check gradients..
          p (list): List of parameter values.
          h (float,list): List of small steps which define the finite differences.  If None, uses 1e-4*p.  If a float, uses h*p.  If a list, sets each step size separately.
          verbosity (int): Verbosity level. Default: 0.
        """

        if (h is None) :
            h = np.array(len(p)*[1e-4])
        elif (not isinstance(h,list)) :
            h = np.array(len(p)*[h])

        if self.stokes == 'I':
            gradV_an = self.visibility_gradients(obs,p,verbosity=verbosity)
            gradV_fd = []
            q = np.copy(p)
            for i in range(self.size) :
                q[i] = p[i]+h[i]
                Vp = self.visibilities(obs,q,verbosity=verbosity)
                q[i] = p[i]-h[i]
                Vm = self.visibilities(obs,q,verbosity=verbosity)
                q[i] = p[i]

                gradV_fd.append((Vp-Vm)/(2.0*h[i]))
            gradV_fd = np.array(gradV_fd).T
            
        else:
            gradV_an_RR, gradV_an_LL, gradV_an_RL, gradV_an_LR = self.visibility_gradients(obs,p,verbosity=verbosity)
            gradV_an = gradV_an_RR + gradV_an_LL + gradV_an_RL + gradV_an_LR
            gradV_fd = []
            q = np.copy(p)
            for i in range(self.size) :
                q[i] = p[i]+h[i]
                Vp_RR, Vp_LL, Vp_RL, Vp_LR = self.visibilities(obs,q,verbosity=verbosity)
                q[i] = p[i]-h[i]
                Vm_RR, Vm_LL, Vm_RL, Vm_LR = self.visibilities(obs,q,verbosity=verbosity)
                q[i] = p[i]

                gradV_fd.append(((Vp_RR-Vm_RR)/(2.0*h[i])) + ((Vp_LL-Vm_LL)/(2.0*h[i])) + ((Vp_RL-Vm_RL)/(2.0*h[i])) + ((Vp_LR-Vm_LR)/(2.0*h[i])))
            gradV_fd = np.array(gradV_fd).T

        lbls = self.parameter_labels()
        err = (gradV_fd-gradV_an)
        print("Gradient Check Report:")
        print("  Sample of errors by parameter")
        print("  %30s %15s %15s %15s %15s %15s"%("Param.","u","v","anal. grad","fd. grad","err"))
        for i in range(self.size) :
            for j in range(min(len(obs.data['u']),10)) :
                print("  %30s %15.8g %15.8g %15.8g %15.8g %15.8g"%(lbls[i],obs.data['u'][j],obs.data['v'][j],gradV_an[j][i],gradV_fd[j][i],err[j][i]))

        mfe_arg = np.argmax(err)
        mfe_i = mfe_arg%self.size
        mfe_j = mfe_arg//self.size
        max_err = np.max(err)
        print("  Global Maximum error:",max_err)
        print("  %30s %15.8g %15.8g %15.8g %15.8g %15.8g"%(lbls[mfe_i],obs.data['u'][mfe_j],obs.data['v'][mfe_j],gradV_an[mfe_j][mfe_i],gradV_fd[mfe_j][mfe_i],err[mfe_j][mfe_i]))
        

            
    def fisher_covar(self,obs,p,verbosity=0,**kwargs) :
        """
        Returns the Fisher matrix as defined in the accompanying documentation.
        Intelligently avoids recomputation if the observation and parameters are
        unchanged.

        Args:
          obs (ehtim.obsdata.Obsdata): An Obsdata object containing the observation particulars.
          p (list): List of model parameters at which to compute the Fisher matrix.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): The Fisher matrix.
        """
        # Get the fisher covariance 
        new_argument_hash = hashlib.md5(bytes(str(obs)+str(p),'utf-8')).hexdigest()
        if ( new_argument_hash == self.argument_hash ) :
            return self.covar
        else :
            self.argument_hash = new_argument_hash
            if self.stokes == 'I':
                obs = obs.switch_polrep('stokes')
                gradV = self.visibility_gradients(obs,p,verbosity=verbosity,**kwargs)
                self.covar = np.zeros((self.size,self.size))
                for i in range(self.size) :
                    for j in range(self.size) :
                        self.covar[i][j] = np.sum( np.conj(gradV[:,i])*gradV[:,j]/obs.data['sigma']**2 + gradV[:,i]*np.conj(gradV[:,j])/obs.data['sigma']**2)
            
            else:
                obs = obs.switch_polrep('circ')
                grad_RR, grad_LL, grad_RL, grad_LR = self.visibility_gradients(obs,p,verbosity=verbosity,**kwargs)
                self.covar = np.zeros((self.size,self.size))
                for i in range(self.size) :
                    for j in range(self.size) :
                        self.covar[i][j] = np.sum( np.conj(grad_RR[:,i])*grad_RR[:,j]/obs.data['rrsigma']**2 + grad_RR[:,i]*np.conj(grad_RR[:,j])/obs.data['rrsigma']**2)
                        self.covar[i][j] += np.sum( np.conj(grad_LL[:,i])*grad_LL[:,j]/obs.data['llsigma']**2 + grad_LL[:,i]*np.conj(grad_LL[:,j])/obs.data['llsigma']**2)
                        self.covar[i][j] += np.sum( np.conj(grad_RL[:,i])*grad_RL[:,j]/obs.data['rlsigma']**2 + grad_RL[:,i]*np.conj(grad_RL[:,j])/obs.data['rlsigma']**2)
                        self.covar[i][j] += np.sum( np.conj(grad_LR[:,i])*grad_LR[:,j]/obs.data['lrsigma']**2 + grad_LR[:,i]*np.conj(grad_LR[:,j])/obs.data['lrsigma']**2)

            if (verbosity>0) :
                print("FisherCovar covar before priors:")
                _print_matrix(self.covar)

            if (len(self.prior_sigma_list)>0) :
                for i in range(self.size) :
                    if (not self.prior_sigma_list[i] is None) :
                        self.covar[i][i] += 2.0/(self.prior_sigma_list[i]**2) # Why factor of 2?
        
            if (verbosity>0) :
                print("FisherCovar covar after priors:")
                _print_matrix(self.covar)

            if (verbosity>1) :
                print("FisherCovar priors:",self.prior_sigma_list)
                print("Dimensions:",self.covar.shape)
                
        return self.covar

    def inverse_fisher_covar(self,obs,p,verbosity=0,**kwargs) :
        """
        Returns the inverse of the Fisher matrix as defined in the accompanying 
        documentation. Intelligently avoids recomputation if the observation and 
        parameters are unchanged.

        Args:
          obs (ehtim.obsdata.Obsdata): An Obsdata object containing the observation particulars.
          p (list): List of model parameters at which to compute the Fisher matrix.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): The inverse of the Fisher matrix.
        """

        new_inv_argument_hash = hashlib.md5(bytes(str(obs)+str(p),'utf-8')).hexdigest()
        if ( new_inv_argument_hash == self.inv_argument_hash ) :
            return self.invcovar
        else :
            self.inv_argument_hash = new_inv_argument_hash
            self.invcovar = _invert_matrix(self.fisher_covar(obs,p,verbosity=verbosity,**kwargs))

        return self.invcovar
    
    
    def uniparameter_uncertainties(self,obs,p,ilist=None,verbosity=0,**kwargs) :
        """
        Computes the uncertainties on a subset of model parameters, fixing all
        others. Note that this is not a covariance; each uncertainty is for a 
        single parameter, fixing all others.  This is probably not what is wanted
        for forecasting, see marginalized_uncertainties.

        Args:
          obs (ehtim.obsdata.Obsdata): An Obsdata object containing the observation particulars.
          p (list): List of parameter values at which to compute uncertainties.
          ilist (int,list) : Index or list of indicies for which to compute the uncertainties. If None will return a list of all uncertainties. Default: None.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (float/list): Parameter uncertainty or list of marginalized uncertainties of desired parameters.
        """
        C = self.fisher_covar(obs,p,verbosity=verbosity,**kwargs)

        if (ilist is None) :
            ilist = np.arange(self.size)

        if (isinstance(ilist,int)) :
            N = C[ilist][ilist]
            Sig_uni = np.sqrt(2.0/N)
        else :
            Sig_uni = np.zeros(self.size)
            for i in ilist :
                N = C[i][i]
                Sig_uni[i] = np.sqrt(2.0/N)
                
        return Sig_uni

    
    def marginalized_uncertainties(self,obs,p,ilist=None,verbosity=0,**kwargs) :
        """
        Computes the uncertainties on a subset of model parameters, marginalized
        over all others. Note that this is not the marginalized covariance; each
        uncertainty is for a single parameter, marginalizing out all others.

        Args:
          obs (ehtim.obsdata.Obsdata): An Obsdata object containing the observation particulars.
          p (list): List of parameter values at which to compute uncertainties.
          ilist (int,list) : Index or list of indicies for which to compute the marginalized uncertainties. If None will return a list of all marginalized uncertainties. Default: None.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (float/list): Marginalized parameter uncertainty or list of marginalized uncertainties of desired parameters.
        """
        C = self.fisher_covar(obs,p,verbosity=verbosity,**kwargs)
        # M = np.zeros((self.size-1,self.size-1))
        # v = np.zeros(self.size-1)
        
        if (ilist is None) :
            ilist = np.arange(self.size)

        if (isinstance(ilist,int)) :
            i = ilist

            Cinv = self.inverse_fisher_covar(obs,p,verbosity=verbosity,**kwargs)
            mN = _invert_matrix(Cinv[[ilist],:][:,[ilist]])
            Sig_marg = np.sqrt(2.0/mN[0,0])
            
            # ilist = np.arange(self.size)
            # ini = ilist[ilist!=i]
            # M = C[ini,:][:,ini]
            # v = C[i,ini]
            # N = C[i][i]
            # v,M = self._condition_vM(v,M)
            # Minv = _invert_matrix(M)
            # mN = (N - _vMv(v,Minv))
            # Sig_marg = np.sqrt(2.0/mN)

        else :
            Cinv = self.inverse_fisher_covar(obs,p,verbosity=verbosity,**kwargs)
            Sig_marg = np.sqrt(2.0*Cinv[ilist,ilist])
            
            # Sig_marg = np.zeros(len(ilist))
            # iall = np.arange(self.size)
            # for k,i in enumerate(ilist) :
            #     ini = iall[iall!=i]
            #     M = C[ini,:][:,ini]
            #     v = C[ini,i]
            #     N = C[i][i]
            #     v,M = self._condition_vM(v,M)
            #     Minv = _invert_matrix(M)
            #     mN = N - _vMv(v,Minv)
            #     Sig_marg[k] = np.sqrt(2.0/mN)
                
        return Sig_marg


    def marginalized_covariance(self,obs,p,ilist=None,verbosity=0,**kwargs) :
        """
        Computes the covariance for a subset of model parameters, marginalized
        over all parameters outside of the subset.

        Args:
          obs (ehtim.obsdata.Obsdata): An Obsdata object containing the observation particulars.
          p (list): List of parameter values at which to compute uncertainties.
          ilist (int,list) : Index or list of indicies for which to compute the marginalized uncertainties. If None will return a list of all marginalized uncertainties. Default: None.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): Marginalized covariance for desired parameter subset.
        """
        C = self.fisher_covar(obs,p,verbosity=verbosity,**kwargs)

        if (ilist is None) :
            return C
        elif (isinstance(ilist,int)) :
            return self.marginalized_uncertainties(obs,p,ilist=ilist,verbosity=0,**kwargs)
        else :
            Cinv = self.inverse_fisher_covar(obs,p,verbosity=verbosity,**kwargs)
            return ( _invert_matrix(Cinv[ilist,:][:,ilist]) )

            
            # iall = np.arange(self.size)
            # isni = (iall!=ilist[0])
            # for i in ilist[1:] :
            #     isni = isni*(iall!=i)
            # ini = iall[isni]
            # n = C[ilist,:][:,ilist]
            # r = C[ini,:][:,ilist]
            # m = C[ini,:][:,ini]
            # r,m = self._condition_vM(r,m)

            # return n - _vMv(r,_invert_matrix(m))
            
    
    def uncertainties(self,obs,p,ilist=None,verbosity=0,**kwargs) :
        """
        Computes the uncertainties on a subset of model parameters, both fixing
        and marginalizing over all others. 

        Args:
          obs (ehtim.obsdata.Obsdata): An Obsdata object containing the observation particulars.
          p (list): List of parameter values at which to compute uncertainties.
          ilist (int,list) : Index or list of indicies for which to compute the marginalized uncertainties. If None will return a list of all marginalized uncertainties. Default: None.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (float/list,float/list): Parameter uncertainty or list of marginalized uncertainties of desired parameters, marginalized parameter uncertainty or list of marginalized uncertainties of desired parameters.
        """
        C = self.fisher_covar(obs,p,verbosity=verbosity,**kwargs)
        Cinv = self.inverse_fisher_covar(obs,p,verbosity=verbosity,**kwargs)

        if (verbosity>1) :
            print("Fisher Matrix:")
            _print_matrix(C)


        Sig_uni = self.uniparameter_uncertainties(obs,p,ilist,verbosity=verbosity,**kwargs)
        Sig_marg = self.marginalized_uncertainties(obs,p,ilist,verbosity=verbosity,**kwarg)
            
        # if (ilist is None) :
        #     ilist = np.arange(self.size)

        # if (isinstance(ilist,int)) :
        #     i = ilist
        #     ilist = np.arange(self.size)
        #     ini = ilist[ilist!=i]
        #     M = C[ini,:][:,ini]
        #     v = C[i,ini]
        #     N = C[i][i]
        #     v,M = self._condition_vM(v,M)
        #     Minv = _invert_matrix(M)
        #     mN = N - _vMv(v,Minv)
        #     Sig_uni[i] = np.sqrt(2.0/N)
        #     Sig_marg[i] = np.sqrt(2.0/mN)
        # else :            
        #     Sig_uni = np.zeros(self.size)
        #     Sig_marg = np.zeros(self.size)
        #     M = np.zeros((self.size-1,self.size-1))
        #     v = np.zeros(self.size-1)
        #     ilist = np.arange(self.size)
        #     for i in ilist :
        #         ini = ilist[ilist!=i]
        #         M = C[ini,:][:,ini]
        #         v = C[i,ini]
        #         N = C[i][i]
        #         v,M = self._condition_vM(v,M)
        #         Minv = _invert_matrix(M)
        #         mN = N - _vMv(v,Minv)
        #         Sig_uni[i] = np.sqrt(2.0/N)
        #         Sig_marg[i] = np.sqrt(2.0/mN)

        #         if (verbosity>1) :
        #             print("Submatrix (%i):"%(i))
        #             _print_matrix(M)
        #             print("Submatrix inverse (%i):"%(i))                
        #             _print_matrix(Minv)
        #             print("Subvectors v1 (%i):"%(i))
        #             _print_vector(v)
        #             print("N,mN (%i):"%(i),N,mN)
        
        return Sig_uni, Sig_marg

    
    def joint_biparameter_chisq(self,obs,p,i1,i2,kind='marginalized',verbosity=0,**kwargs) :
        """
        Computes the ensemble-averaged 2nd-order contribution to the chi-square
        for two parameters after fixing or marginalizing over all others.

        Args:
          obs (ehtim.obsdata.Obsdata): An Obsdata object containing the observation particulars.
          p (list): List of parameter values at which to compute uncertainties.
          i1 (int): Index of first parameter.
          i2 (int): Index of second parameter.
          kind (str): Choice of what to do with other parameters. Choices are 'marginalized', 'fixed'.  Default: 'marginalized'.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray,numpy.ndarray,numpy.ndarray): :math:`p_{i1}`, :math:`p_{i2}`, and :math:`\\chi^2` on a grid of points that covers the inner :math:`4.5\\Sigma` region.
        """
        # C = self.fisher_covar(obs,p,verbosity=verbosity,**kwargs)        
        # M = np.zeros((self.size-2,self.size-2))
        # v1 = np.zeros(self.size-2)
        # v2 = np.zeros(self.size-2)
        # ilist = np.arange(self.size)
        
        # ini12 = ilist[(ilist!=i1)*(ilist!=i2)]
        # for j2,j in enumerate(ini12) :
        #     for k2,k in enumerate(ini12) :
        #         M[j2,k2] = C[j][k]
        #     v1[j2] = C[i1][j]
        #     v2[j2] = C[i2][j]
        # N1 = C[i1][i1]
        # N2 = C[i2][i2]
        # C12 = C[i1][i2]


        ilist = [i1,i2]
            
        if (kind.lower()=='fixed') :
            # pass
            C = self.inverse_fisher_covar(obs,p,verbosity=verbosity,**kwargs)
            Ci1i2 = C[ilist,:][:,ilist]
        
        elif (kind.lower()=='marginalized') :
            Cinv = self.inverse_fisher_covar(obs,p,verbosity=verbosity,**kwargs)
            Ci1i2 = _invert_matrix(Cinv[ilist,:][:,ilist])
            
            # if (verbosity>1) :
            #     print("Fisher Matrix:")
            #     _print_matrix(C)
            #     print("Submatrix (%i,%i):"%(i1,i2))
            #     _print_matrix(M)
            #     print("Subvectors v1:")
            #     _print_vector(v1)
            #     print("Subvectors v2:")
            #     _print_vector(v2)

            # vv = np.vstack([v1,v2]).T
            # vv,M = self._condition_vM(vv,M)
            # v1 = vv[:,0]
            # v2 = vv[:,1]
            
            # Minv = _invert_matrix(M)
            # N1 = N1 - _vMv(v1,Minv)
            # N2 = N2 - _vMv(v2,Minv)
            # C12 = C12 - 0.5*(np.matmul(v1.T,np.matmul(Minv,v2)) + np.matmul(v2.T,np.matmul(Minv,v1)))

        else :
            raise(RuntimeError("Received unexpected value for kind, %s. Allowed values are 'fixed' and 'marginalized'."%(kind)))


        # Set some names
        N1 = Ci1i2[0,0]
        N2 = Ci1i2[1,1]
        C12 = Ci1i2[0,1]
        
                    
        # Find range with eigenvectors/values
        l1 = 0.5*( (N1+N2) + np.sqrt( (N1-N2)**2 + 4.0*C12**2 ) )
        l2 = 0.5*( (N1+N2) - np.sqrt( (N1-N2)**2 + 4.0*C12**2 ) )

        e1 = np.array( [ C12, l1-N1 ] )
        e2 = np.array( [ e1[1], -e1[0] ] )
        e1 = e1/ np.sqrt( e1[0]**2+e1[1]**2 )
        e2 = e2/ np.sqrt( e2[0]**2+e2[1]**2 )

        if (l1<=0 or l2<=0) :
            print("Something's wrong! Variances are nonpositive!")
            print(i1,i2)
            print(l1,l2)
            print(N1,N2,C12)
            # print(mN1,mN2,mC12)
            print(C)

            l1 = max(1e-10,l1)
            l2 = max(1e-10,l2)

        if (l1>l2 ):
            Sig_maj = np.sqrt(1.0/l2)
            Sig_min = np.sqrt(1.0/l1)
            e_maj = np.copy(e2)
            e_min = np.copy(e1)
        else :
            Sig_maj = np.sqrt(1.0/l1)
            Sig_min = np.sqrt(1.0/l2)
            e_maj = np.copy(e1)
            e_min = np.copy(e2)

        dp1 = 4.5*( Sig_maj*np.abs(e_maj[0]) + Sig_min*np.abs(e_min[0]) )
        dp2 = 4.5*( Sig_maj*np.abs(e_maj[1]) + Sig_min*np.abs(e_min[1]) )

        Npx = int(max(128,min(16*Sig_maj/Sig_min,1024)))
        p1,p2 = np.meshgrid(np.linspace(-dp1,dp1,Npx),np.linspace(-dp2,dp2,Npx))
        csq = 0.5*(N1*p1**2 + 2*C12*p1*p2 + N2*p2**2)
            
        return p1,p2,csq


    def plot_1d_forecast(self,obs,p,i1,kind='marginalized',labels=None,fig=None,axes=None,colors=None,alphas=0.25,verbosity=0,**kwargs) :
        """
        Plot the marginalized posterior for a single parameter for a given observation.
        
        Args:
          obs (list,Obsdata): An Obsdata object or list of Obsdata objects containing the observation particulars.
          p (list): List of parameter values at which to compute uncertainties.
          i1 (int): Index of parameter to be plotted.
          kind (str): Choice of what to do with other parameters. Choices are 'marginalized', 'fixed'. Default: 'marginalized'.
          labels (list,str): A list of labels for each Obsdata object. When fewer labels than observations are provided, the first set of observations are labeled. Default: None.
          fig (matplotlib.figure.Figure): Figure on which to place plot. If None, a new figure will be created. Default: None.
          axes (matplotlib.axes.Axes): Axes on which to place plot. If None, a new axes will be created. Default: None.
          colors (list,str): A color or list of colors for the plots. When fewer colors than observations are provided, they will be cycled through. If None, a default list will be used. Default: None.
          alphas (list,float): An alpha value or list of alpha values for the filled portion of the plots. Default: 0.25.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (matplotlib.figure.Figure, matplotlib.axes.Axes): Handles to the figure and axes objects in the plot.
        """

        if (isinstance(obs,list)) :
            obslist = obs
        else :
            obslist = [obs]

        if (labels is None) :
            labels = len(obslist)*[None]
        elif (not isinstance(labels,list)) :
            labels = [labels]
        if (len(labels)<len(obslist)) :
            raise(RuntimeError("Label list must have the same size as the observation list."))

        if (axes is None) :
            if (fig is None) :
                plt.figure(figsize=(5,4))
            plt.axes([0.15,0.15,0.8,0.8])
        else :
            plt.sca(axes)

        if (colors is None) :
            colors = self.default_color_list
        elif (isinstance(colors,list)) :
            pass
        else :
            colors = [colors]

        if (isinstance(alphas,list)) :
            pass
        else :
            alphas = [alphas]
            

        for k,obs in enumerate(obslist) :
            if (kind=='marginalized') :
                Sigma = self.marginalized_uncertainties(obs,p,ilist=i1)
            elif (kind=='fixed') :
                Sigma = self.uniparameter_uncertainties(obs,p,ilist=i1)
            else :
                raise(RuntimeError("Received unexpected value for kind, %s. Allowed values are 'fixed' and 'marginalized'."%(kind)))
            x = np.linspace(-5*Sigma,5*Sigma,256)
            y = np.exp(-x**2/(2.0*Sigma**2)) / np.sqrt(2.0*np.pi*Sigma**2)
            plt.fill_between(x,y,y2=0,alpha=self._choose_from_list(alphas,k),color=self._choose_from_list(colors,k))
            plt.plot(x,y,'-',color=self._choose_from_list(colors,k),label=labels[k])

        plt.xlabel(self.parameter_labels()[i1])
        plt.yticks([])
        plt.ylim(bottom=0)

        if (not (np.array(labels)==None).all()) :
            plt.legend()

        return plt.gcf(),plt.gca()


    def plot_2d_forecast(self,obs,p,i1,i2,kind='marginalized',labels=None,fig=None,axes=None,colors=None,cmaps=None,alphas=0.75,verbosity=0,**kwargs) :
        """
        Plot the joint marginalized posterior for a pair of parameters for a given observation.
        
        Args:
          obs (list,Obsdata): An Obsdata object or list of Obsdata objects containing the observation particulars.
          p (list): List of parameter values at which to compute uncertainties.
          i1 (int): Index of first parameter to be plotted.
          i2 (int): Index of second parameter to be plotted.
          kind (str): Choice of what to do with other parameters. Choices are 'marginalized', 'fixed'. Default: 'marginalized'.
          labels (list,str): A list of labels for each Obsdata object. When fewer labels than observations are provided, the first set of observations are labeled. Default: None.
          fig (matplotlib.figure.Figure): Figure on which to place plot. If None, a new figure will be created. Default: None.
          axes (matplotlib.axes.Axes): Axes on which to place plot. If None, a new axes will be created. Default: None.
          colors (list,str): A color or list of colors for the plots. When fewer colors than observations are provided, they will be cycled through. If None, a default list will be used. Default: None.
          cmaps (list,matplotlib.colors.Colormap): A colormap or list of colormaps for the plots. When fewer colormaps than observations are provided, they will be cycled through. If None, a default list will be used. Default: None.
          alphas (list,float): An alpha value or list of alpha values for the filled portion of the plots. Default: 0.75.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (matplotlib.figure.Figure, matplotlib.axes.Axes): Handles to the figure and axes objects in the plot.
        """

        if (isinstance(obs,list)) :
            obslist = obs
        else :
            obslist = [obs]

        if (labels is None) :
            labels = len(obslist)*[None]
        elif (not isinstance(labels,list)) :
            labels = [labels]
        if (len(labels)<len(obslist)) :
            raise(RuntimeError("Label list must have the same size as the observation list."))

        if (axes is None) :
            if (fig is None) :
                plt.figure(figsize=(5,5))
            plt.axes([0.2,0.2,0.75,0.75])
        else :
            plt.sca(axes)

        if (colors is None) :
            colors = self.default_color_list
        elif (isinstance(colors,list)) :
            pass
        else :
            colors = [colors]

        if (cmaps is None) :
            cmaps = self.default_cmap_list
        elif (isinstance(cmaps,list)) :
            pass
        else :
            cmaps = [cmaps]
            
        if (isinstance(alphas,list)) :
            pass
        else :
            alphas = [alphas]

        for k,obs in enumerate(obslist) :
            d,w,csq = self.joint_biparameter_chisq(obs,p,i1,i2,verbosity=verbosity,kind=kind,**kwargs)
            plt.contourf(d,w,np.sqrt(csq),cmap=self._choose_from_list(cmaps,k),alpha=self._choose_from_list(alphas,k),levels=[0,1,2,3])
            plt.contour(d,w,np.sqrt(csq),colors=self._choose_from_list(colors,k),alpha=1,levels=[0,1,2,3])
            plt.plot([],[],'-',color=self._choose_from_list(colors,k),label=labels[k])

        plt.xlabel(self.parameter_labels()[i1])
        plt.ylabel(self.parameter_labels()[i2])
        plt.grid(True,alpha=0.25)

        if (not (np.array(labels)==None).all()) :
            plt.legend()

        return plt.gcf(),plt.gca()


    def plot_triangle_forecast(self,obs,p,ilist=None,kind='marginalized',labels=None,colors=None,cmaps=None,alphas=0.75,fig=None,axes=None,figsize=None,axis_location=None,verbosity=0,**kwargs) :
        """
        Generate a triangle plot of joint posteriors for a selected list of parameter indexes.
        
        Args:
          obs (list,Obsdata): An Obsdata object or list of Obsdata objects containing the observation particulars.
          p (list): List of parameter values at which to compute uncertainties.
          ilist (list): List of indicies of parameters to include in triangle plot. If None, includes all parameters. Default: None.
          kind (str): Choice of what to do with other parameters. Choices are 'marginalized', 'fixed'. Default: 'marginalized'.
          labels (list,str): A list of labels for each Obsdata object. When fewer labels than observations are provided, the first set of observations are labeled. Default: None.
          colors (list,str): A color or list of colors for the plots. When fewer colors than observations are provided, they will be cycled through. If None, a default list will be used. Default: None.
          cmaps (list,matplotlib.colors.Colormap): A colormap or list of colormaps for the plots. When fewer colormaps than observations are provided, they will be cycled through. If None, a default list will be used. Default: None.
          alphas (list,float): An alpha value or list of alpha values for the filled portion of the plots. Default: 0.75.
          fig (matplotlib.figure.Figure): Figure on which to place plot. If None, a new figure will be created. Default: None.
          axes (matplotlib.axes.Axes): Axes on which to place plot. If None, a new axes will be created. Default: None.
          figsize (list): Figure size in inches.  If None, attempts a guess. Default: None.
          axis_location (list): Location parameters for triangle plot region. If None, attempts a guess. Default: None.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (matplotlib.figure.Figure, dict): Handles to the figure and dictionary of axes objects in the plot.
        """

        if (isinstance(obs,list)) :
            obslist = obs
        else :
            obslist = [obs]

        if (ilist is None) :
            ilist = np.arange(self.size)
            
        if (labels is None) :
            labels = len(obslist)*[None]
        elif (not isinstance(labels,list)) :
            labels = [labels]
        if (len(labels)<len(obslist)) :
            raise(RuntimeError("Label list must have the same size as the observation list."))

        if (colors is None) :
            colors = self.default_color_list
        elif (isinstance(colors,list)) :
            pass
        else :
            colors = [colors]

        if (cmaps is None) :
            cmaps = self.default_cmap_list
        elif (isinstance(cmaps,list)) :
            pass
        else :
            cmaps = [cmaps]
            
        if (isinstance(alphas,list)) :
            pass
        else :
            alphas = [alphas]
            
        if (fig is None) :
            plt.figure(figsize=(len(ilist)*2.5,len(ilist)*2.5))
        else :
            plt.scf(fig)

        if (figsize is None) :
            figsize = plt.gcf().get_size_inches()
        else :
            plg.gcf().set_size_inches(figsize)

        if (axis_location is None) :
            lmarg = 0.625 # Margin in inches
            rmarg = 0.625 # Margin in inches
            tmarg = 0.625 # Margin in inches
            bmarg = 0.625 # Margin in inches
            ax0 = lmarg/figsize[0]
            ay0 = bmarg/figsize[1]
            axw = (figsize[0]-lmarg-rmarg)/figsize[0]
            ayw = (figsize[1]-tmarg-bmarg)/figsize[1]
            axis_location = [ax0, ay0, axw, ayw]

        # Number of rows/columns
        nrow = len(ilist)

        # Make axes dictionary if it doesn't exist already
        if (axes is None) :
            # Get window size details
            gutter_size = 0.0625 # Gutter in inches
            x_gutter = gutter_size/figsize[0]
            y_gutter = gutter_size/figsize[1]
            x_window_size = (axis_location[2] - (nrow-1)*x_gutter)/float(nrow)
            y_window_size = (axis_location[3] - (nrow-1)*y_gutter)/float(nrow)
            
            axes = {}
            for j in range(nrow) :
                for i in range(j+1) :
                    # Find axis location with various gutters, etc.
                    x_window_start = axis_location[0] + i*(x_gutter+x_window_size)
                    y_window_start = axis_location[1] + axis_location[3] - y_window_size - j*(y_gutter+y_window_size)
                    axes[i,j] = plt.axes([x_window_start, y_window_start, x_window_size, y_window_size])

        # Run over panels and plot
        xlim_dict = {}
        for k,obs in enumerate(obslist) :
            if (kind=='marginalized') :
                Sigma = self.marginalized_uncertainties(obs,p)
            elif (kind=='fixed') :
                Sigma = self.uniparameter_uncertainties(obs,p)
            else :
                raise(RuntimeError("Received unexpected value for kind, %s. Allowed values are 'fixed' and 'marginalized'."%(kind)))

            # print("Sigma before:",Sigma)
            for jj in range(len(Sigma)) :
                if (np.isnan(Sigma[jj]) or np.isinf(Sigma[jj])) :
                    Sigma[jj] = 999.0
            # print("Sigma after: ",Sigma)

            for j in range(nrow) :
                jj = ilist[j]
                xtmp = np.linspace(-3.5*Sigma[jj],3.5*Sigma[jj],256)
                ytmp = np.exp(-xtmp**2/(2.0*Sigma[jj]**2))/np.sqrt(2.0*np.pi*Sigma[jj]**2)
                axes[j,j].fill_between(xtmp,ytmp,y2=0,color=self._choose_from_list(colors,k),alpha=self._choose_from_list(alphas,k)/3.0)
                axes[j,j].plot(xtmp,ytmp,color=self._choose_from_list(colors,k))

                if (k==0) :                        
                    xlim_dict[j] = (-3.5*Sigma[jj],3.5*Sigma[jj])
                else :
                    if (3.5*Sigma[jj]>xlim_dict[j][1]) :
                        xlim_dict[j] = (-3.5*Sigma[jj],3.5*Sigma[jj])                    

            for j in range(nrow) :
                for i in range(j) :

                    # ii = ilist[ni-i]
                    ii = ilist[i]
                    jj = ilist[j]

                    plt.sca(axes[i,j])

                    p1,p2,csq = self.joint_biparameter_chisq(obs,p,ii,jj,kind=kind,verbosity=verbosity)
                    plt.contourf(p1,p2,np.sqrt(csq),cmap=self._choose_from_list(cmaps,k),alpha=self._choose_from_list(alphas,k),levels=[0,1,2,3])
                    plt.contour(p1,p2,np.sqrt(csq),colors=self._choose_from_list(colors,k),alpha=1,levels=[0,1,2,3])

                    plt.xlim(xlim_dict[i])
                    plt.ylim(xlim_dict[j])

                    plt.grid(True,alpha=0.25)

                    # plt.text(0.05,0.05,'%i,%i / %i,%i'%(i,j,ii,jj),transform=plt.gca().transAxes)


        for j in range(nrow) :
            axes[j,j].set_xlim(xlim_dict[j])
            axes[j,j].set_ylim(bottom=0)
            axes[j,j].set_yticks([])


        for j in range(nrow-1) :
            for i in range(j+1) :
                axes[i,j].set_xticklabels([])

        for j in range(nrow) :
            for i in range(1,j+1) :
                axes[i,j].set_yticklabels([])

        for j in range(1,nrow) :
            axes[0,j].set_ylabel(self.parameter_labels()[ilist[j]])

        for i in range(nrow) :
            axes[i,nrow-1].set_xlabel(self.parameter_labels()[ilist[i]])


        # Make axis for labels
        if (not (np.array(labels)==None).all()) :
            plt.axes([0.95,0.95,0.01,0.01])
            for k in range(len(obslist)) :
                plt.plot([],[],color=self._choose_from_list(colors,k),label=labels[k])
            plt.gca().spines.right.set_visible(False)
            plt.gca().spines.left.set_visible(False)
            plt.gca().spines.top.set_visible(False)
            plt.gca().spines.bottom.set_visible(False)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.legend(loc='upper right',bbox_to_anchor=(0,0))


        return plt.gcf(),axes

    
    def _condition_vM(self,v,M) :
        """
        A helper function that conditions the vector and matrix associated 
        with marginalization.  This conditioning does not change v.M^{-1}.v,
        but does ensure that all quantities in the matrix multiplication
        and matrix inversion are of similar size.  It cannot address numerical
        instability caused by strongly correlated or nearly degenerate M.
        
        The components of the marginalization process assume a structure for
        the covariance that looks like:

                 ( N v.T )
                 ( v  M  )

        So, if N is an nxn array and M is an mxm array, v should be an mxn array.

        The conditioning procedure is equivalent to normalizing the parameters
        that are being marginalized by the square roots of their variances.

        Args:
          v (nd.array): Vector or matrix associated with the correlations between the parameters to be marginalized over and those to be retained.
          M (nd.array): Covariance of the marginalized parameters.

        Returns:
          (nd.array,nd.array): Conditioned v and M. 
        """
        # Conditions v and M to improve inversion and double-dot product performance
        il = np.arange(M.shape[0])
        dd = 1.0/np.sqrt(M[il,il])
        for i in il :
            M[i,:] = M[i,:]*dd
            M[:,i] = M[:,i]*dd

        if (len(v.shape)==1) :
            v = dd*v
        else :
            for i in np.arange(v.shape[1]) :
                v[:,i] = v[:,i]*dd

        return v,M

    def _choose_from_list(self,v,i) :
        """
        A helper function to cyclicly select from a list.

        Args:
          v (list): A list.
          i (int): An integer index.

        Returns:
          (v element type): Element of v at index i%len(v).
        """
        return v[i%len(v)]





# if __name__ == "__main__" :

#     # Example
    
#     # Make Fisher forecast object
#     ff1 = FF_splined_raster(5,60)
#     ff2 = FF_smoothed_delta_ring()
#     ff = FF_sum([ff1,ff2])

#     # Read in some data
#     obs_ngeht = eh.obsdata.load_uvfits('../data/M87_230GHz_40uas.uvfits')
#     obs_ngeht.add_scans()
#     obs_ngeht = obs_ngeht.avg_coherent(0,scan_avg=True)
#     obs_ngeht = obs_ngeht.add_fractional_noise(0.01)
#     #
#     obs = obs_ngeht.flag_sites(['ALMA','APEX','JCMT','SMA','SMT','LMT','SPT','PV','PDB','KP'])


#     obslist = [obs,obs_ngeht]
#     labels = ['ngEHT w/o EHT','Full ngEHT']
    
    
#     # Choose a default image
#     p = np.zeros(ff.size)
#     rad2uas = 3600e6*180/np.pi
#     for j in range(ff1.npx) :
#         for i in range(ff1.npx) :
#             p[i+ff1.npx*j] =  -((ff1.xcp[i,j]-5.0/rad2uas)**2+ff1.ycp[i,j]**2)*rad2uas**2/(2.0*25.0**2) + np.log( 1.0/( 2.0*np.pi * (25.0/rad2uas)**2 ) )
#     p[-5] = 0.2
#     p[-4] = 40
#     p[-3] = 2
#     p[-2] = 0
#     p[-1] = 0
#     p = np.array(p)

#     # 1D Diameter
#     plot_1d_forecast(ff,p,ff.size-4,obslist,labels=labels)
#     plt.savefig('ring_forecast_1d.png',dpi=300)

#     # 2D diameter vs width
#     plot_2d_forecast(ff,p,ff.size-4,ff.size-3,obslist,labels=labels)
#     plt.savefig('ring_forecast_2d.png',dpi=300)

#     # Triangle
#     plist = np.arange(len(p)-5,len(p))
#     plot_triangle_forecast(ff,p,plist,obslist,labels=labels)
#     plt.savefig('ring_forecast_tri.png',dpi=300)
    
#     # plt.show()

