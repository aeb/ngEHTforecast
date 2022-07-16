import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
import scipy.interpolate as si
import ehtim as eh
import copy
import hashlib
import themispy as ty
import ehtim.parloop as ploop

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
    return linalg.inv(a)
    # return linalg.pinvh(a)
    # n = a.shape[0]
    # I = np.identity(n)
    # return linalg.solve(a, I, sym_pos = True, overwrite_b = True)

    
class FisherForecast :
    """
    Class that collects and contains information for making Fisher-matrix type
    forecasts for observations of various types.  Forms a base class for fully
    analytical versions that may be application specific.

    Attributes:
      size (int): Number of parameters expected by this model.
      stokes (str): If this is a Stokes I model ('I') or a polarized model ('full').
      prior_sigma_list (list): List of standard deviations associated with the parameter priors.
      covar (np.ndarray): Internal space for the computation of the covariance matrix.
      argument_hash (str): MD5 hash object indicating last state of covariance computation. Used to determine if the covariance needs to be recomputed.
    """

    def __init__(self) :
        self.size = 0
        self.argument_hash = None
        self.prior_sigma_list = []
        self.stokes = 'I'
        self.covar = None

        self.default_color_list = ['r','b','g','orange','purple']
        self.default_cmap_list = ['Reds_r','Blues_r','Greens_r','Oranges_r','Purples_r']

        
    def visibilities(self,obs,p,verbosity=0) :
        """
        User-defined function in child classes that generates visibilities 
        associated with a given model image object.

        Args:
          obs (Obsdata): An ehtim Obsdata object with a particular set of observation details (u,v positions, times, etc.)
          p (np.array): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          V (np.array): list of complex visibilities computed at observations.
        """

        raise(RuntimeError("visibilities function not implemented in base class!"))
        return 0*obs.data['u']

            
    def visibility_gradients(self,obs,p,verbosity=0,**kwargs) :
        """
        User-defined function in child classes that generates visibility gradients
        associated with a given model image object.

        Args:
          obs (Obsdata): An ehtim Obsdata object with a particular set of observation details (u,v positions, times, etc.)
          p (np.array): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          gradV (np.ndarray): list of complex visibilities gradients computed at observations.
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
          plabels (list): list of strings with parameter labels.
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
          x (np.ndarray): x positions of the pixel centers in uas.
          y (np.ndarray): y positions of the pixel centers in uas.
          I (np.ndarray): Intensities at pixel centers in Jy/uas^2.
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

        uas2rad = np.pi/180./3600e6            
        umax = shape[0]/(uas2rad*(limits[1]-limits[0]))
        vmax = shape[1]/(uas2rad*(limits[3]-limits[2]))

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

        x,y = np.meshgrid(x1d,y1d)
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
          fig (matplotlib.pyplot.figure): Figure handle
          axs (matplotlib.pyplot.axes): Axes handle
          cb (matplotlib.pyplot.colorbar): Colorbar handle
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

        plt.pcolormesh(x,y,I,cmap='afmhot',vmin=0)

        plt.xlabel(r'$\Delta{\rm RA}~(\mu{\rm as})$')
        plt.ylabel(r'$\Delta{\rm Dec}~(\mu{\rm as})$')

        cbax = plt.axes([0.8*5/6.5+0.05+0.15,0.15,0.05,0.8])
        plt.colorbar(cax=cbax)
        cbax.set_ylabel(fu,rotation=-90,ha='center',va='bottom')

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
          obs (Obsdata): An ehtim Obsdata object with a particular set of observation details (u,v positions, times, etc.) at which to check gradients..
          p (list): List of parameter values.
          h (float,list): List of small steps which define the finite differences.  If None, uses 1e-4*p.  If a float, uses h*p.  If a list, sets each step size separately.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
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
        Returns the Fisher matrix M as defined in the accompanying documentation.
        Intelligently avoids recomputation if the observation and parameters are
        unchanged.

        Args:
          obs (Obsdata): An Obsdata object containing the observation particulars.
          p (list): List of model parameters at which to compute the Fisher matrix.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          M (np.ndarray): The Fisher matrix M.
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

    
    def uniparameter_uncertainties(self,obs,p,ilist=None,verbosity=0,**kwargs) :
        """
        Computes the uncertainties on a subset of model parameters, fixing all
        others. Note that this is not a covariance; each uncertainty is for a 
        single parameter, fixing all others.  This is probably not what is wanted
        for forecasting, see marginalized_uncertainties.

        Args:
          obs (Obsdata): An Obsdata object containing the observation particulars.
          p (list): List of parameter values at which to compute uncertainties.
          ilist (int,list) : Index or list of indicies for which to compute the uncertainties. If None will return a list of all uncertainties. Default: None.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          Sigma (float,list): Parameter uncertainty or list of marginalized uncertainties of desired parameters.
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
          obs (Obsdata): An Obsdata object containing the observation particulars.
          p (list): List of parameter values at which to compute uncertainties.
          ilist (int,list) : Index or list of indicies for which to compute the marginalized uncertainties. If None will return a list of all marginalized uncertainties. Default: None.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          Sigma (float,list): Marginalized parameter uncertainty or list of marginalized uncertainties of desired parameters.
        """
        C = self.fisher_covar(obs,p,verbosity=verbosity,**kwargs)
        # M = np.zeros((self.size-1,self.size-1))
        # v = np.zeros(self.size-1)
        
        if (ilist is None) :
            ilist = np.arange(self.size)

        if (isinstance(ilist,int)) :
            i = ilist
            ilist = np.arange(self.size)
            ini = ilist[ilist!=i]
            # for j2,j in enumerate(ini) :
            #     for k2,k in enumerate(ini) :
            #         M[j2,k2] = C[j][k]
            #     v[j2] = C[i][j]
            M = C[ini,:][:,ini]
            v = C[i,ini]
            N = C[i][i]
            # Minv = _invert_matrix(M)
            # mN = N - np.matmul(v,np.matmul(Minv,v))
            # print("Before mN:",mN)
            v,M = self._condition_vM(v,M)
            Minv = _invert_matrix(M)
            mN = (N - np.matmul(v,np.matmul(Minv,v)))
            # print("After mN:",mN)
            Sig_marg = np.sqrt(2.0/mN)

        else :
            Sig_marg = np.zeros(len(ilist))
            iall = np.arange(self.size)
            for k,i in enumerate(ilist) :
                ini = iall[iall!=i]
                # for j2,j in enumerate(ini) :
                #     for k2,k in enumerate(ini) :
                #         M[j2,k2] = C[j][k]
                #     v[j2] = C[i][j]
                M = C[ini,:][:,ini]
                v = C[ini,i]
                N = C[i][i]
                # Minv = _invert_matrix(M)
                # mN = N - np.matmul(v,np.matmul(Minv,v))
                # print("Before mN:",mN)
                v,M = self._condition_vM(v,M)
                Minv = _invert_matrix(M)
                mN = N - np.matmul(v,np.matmul(Minv,v))
                # print("After mN:",mN)
                Sig_marg[k] = np.sqrt(2.0/mN)
                
        return Sig_marg


    def marginalized_covariance(self,obs,p,ilist=None,verbosity=0,**kwargs) :
        """
        Computes the covariance for a subset of model parameters, marginalized
        over all parameters outside of the subset.

        Args:
          obs (Obsdata): An Obsdata object containing the observation particulars.
          p (list): List of parameter values at which to compute uncertainties.
          ilist (int,list) : Index or list of indicies for which to compute the marginalized uncertainties. If None will return a list of all marginalized uncertainties. Default: None.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          covar (np.ndarray): Marginalized covariance for desired parameter subset.
        """
        C = self.fisher_covar(obs,p,verbosity=verbosity,**kwargs)

        if (ilist is None) :
            return C
        elif (isinstance(ilist,int)) :
            return self.marginalized_uncertainties(obs,p,ilist=ilist,verbosity=0,**kwargs)
        else :
            iall = np.arange(self.size)
            isni = (iall!=ilist[0])
            for i in ilist[1:] :
                isni = isni*(iall!=i)
            ini = iall[isni]
            n = C[ilist,:][:,ilist]
            r = C[ini,:][:,ilist]
            m = C[ini,:][:,ini]
            # print("Before mN:",n - np.matmul(r.T,np.matmul(_invert_matrix(m),r)))
            r,m = self._condition_vM(r,m)
            # print("After mN:",n - np.matmul(r.T,np.matmul(_invert_matrix(m),r)))
            # print("Shapes:",n.shape,r.shape,m.shape)
            return n - np.matmul(r.T,np.matmul(_invert_matrix(m),r))
            
    
    def uncertainties(self,obs,p,ilist=None,verbosity=0,**kwargs) :
        """
        Computes the uncertainties on a subset of model parameters, both fixing
        and marginalizing over all others. 

        Args:
          obs (Obsdata): An Obsdata object containing the observation particulars.
          p (list): List of parameter values at which to compute uncertainties.
          ilist (int,list) : Index or list of indicies for which to compute the marginalized uncertainties. If None will return a list of all marginalized uncertainties. Default: None.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          SigmaF (float,list): Parameter uncertainty or list of marginalized uncertainties of desired parameters.
          SigmaM (float,list): Marginalized parameter uncertainty or list of marginalized uncertainties of desired parameters.
        """
        C = self.fisher_covar(obs,p,verbosity=verbosity,**kwargs)
        if (verbosity>1) :
            print("Fisher Matrix:")
            _print_matrix(C)
        
        if (ilist is None) :
            ilist = np.arange(self.size)

        if (isinstance(ilist,int)) :
            i = ilist
            ilist = np.arange(self.size)
            ini = ilist[ilist!=i]
            # for j2,j in enumerate(ini) :
            #     for k2,k in enumerate(ini) :
            #         M[j2,k2] = C[j][k]
            #     v[j2] = C[i][j]
            M = C[ini,:][:,ini]
            v = C[i,ini]
            N = C[i][i]
            v,M = self._condition_vM(v,M)
            Minv = _invert_matrix(M)
            mN = N - np.matmul(v,np.matmul(Minv,v))
            Sig_uni[i] = np.sqrt(2.0/N)
            Sig_marg[i] = np.sqrt(2.0/mN)
        else :            
            Sig_uni = np.zeros(self.size)
            Sig_marg = np.zeros(self.size)
            M = np.zeros((self.size-1,self.size-1))
            v = np.zeros(self.size-1)
            ilist = np.arange(self.size)
            for i in ilist :
                ini = ilist[ilist!=i]
                # for j2,j in enumerate(ini) :
                #     for k2,k in enumerate(ini) :
                #         M[j2,k2] = C[j][k]
                #     v[j2] = C[i][j]
                M = C[ini,:][:,ini]
                v = C[i,ini]
                N = C[i][i]
                v,M = self._condition_vM(v,M)
                Minv = _invert_matrix(M)
                mN = N - np.matmul(v,np.matmul(Minv,v))
                Sig_uni[i] = np.sqrt(2.0/N)
                Sig_marg[i] = np.sqrt(2.0/mN)

                if (verbosity>1) :
                    print("Submatrix (%i):"%(i))
                    _print_matrix(M)
                    print("Submatrix inverse (%i):"%(i))                
                    _print_matrix(Minv)
                    print("Subvectors v1 (%i):"%(i))
                    _print_vector(v)
                    print("N,mN (%i):"%(i),N,mN)
            
        return Sig_uni,Sig_marg

    
    def joint_biparameter_chisq(self,obs,p,i1,i2,kind='marginalized',verbosity=0,**kwargs) :
        """
        Computes the ensemble-averaged 2nd-order contribution to the chi-square
        for two parameters after fixing or marginalizing over all others.

        Args:
          obs (Obsdata): An Obsdata object containing the observation particulars.
          p (list): List of parameter values at which to compute uncertainties.
          i1 (int): Index of first parameter.
          i2 (int): Index of second parameter.
          kind (str): Choice of what to do with other parameters. Choices are 'marginalized', 'fixed'.  Default: 'marginalized'.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          SigmaF (float,list): Parameter uncertainty or list of marginalized uncertainties of desired parameters.
          SigmaM (float,list): Marginalized parameter uncertainty or list of marginalized uncertainties of desired parameters.
        """
        C = self.fisher_covar(obs,p,verbosity=verbosity,**kwargs)
        M = np.zeros((self.size-2,self.size-2))
        v1 = np.zeros(self.size-2)
        v2 = np.zeros(self.size-2)
        ilist = np.arange(self.size)
        
        ini12 = ilist[(ilist!=i1)*(ilist!=i2)]
        for j2,j in enumerate(ini12) :
            for k2,k in enumerate(ini12) :
                M[j2,k2] = C[j][k]
            v1[j2] = C[i1][j]
            v2[j2] = C[i2][j]
        N1 = C[i1][i1]
        N2 = C[i2][i2]
        C12 = C[i1][i2]

        if (kind.lower()=='fixed') :
            pass
        
        elif (kind.lower()=='marginalized') :
            if (verbosity>1) :
                print("Fisher Matrix:")
                _print_matrix(C)
                print("Submatrix (%i,%i):"%(i1,i2))
                _print_matrix(M)
                print("Subvectors v1:")
                _print_vector(v1)
                print("Subvectors v2:")
                _print_vector(v2)

            # print("i1,i2:",i1,i2)
            # print(v1.shape,v2.shape)
            # print(v1)
            # print(v2)
            vv = np.vstack([v1,v2]).T
            # print(vv.shape)
            # print(vv)
            # print(vv[:,0],v1)
            # print(vv[:,1],v2)
            # print("  =================")
            vv,M = self._condition_vM(vv,M)
            # print("  =================")
            v1 = vv[:,0]
            v2 = vv[:,1]
            # print(v1.shape,v2.shape)
            
            Minv = _invert_matrix(M)
            N1 = N1 - np.matmul(v1.T,np.matmul(Minv,v1))
            N2 = N2 - np.matmul(v2.T,np.matmul(Minv,v2))
            C12 = C12 - 0.5*(np.matmul(v1.T,np.matmul(Minv,v2)) + np.matmul(v2.T,np.matmul(Minv,v1)))

        else :
            raise(RuntimeError("Received unexpected value for kind, %s. Allowed values are 'fixed' and 'marginalized'."))
        
                    
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
                raise(RuntimeError("Received unexpected value for kind, %s. Allowed values are 'fixed' and 'marginalized'."))
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
                raise(RuntimeError("Received unexpected value for kind, %s. Allowed values are 'fixed' and 'marginalized'."))

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
        return v[i%len(v)]

    
class FF_complex_gains_single_epoch(FisherForecast) :
    """
    FisherForecast with complex gain reconstruction assuming a single epoch.
    This is a helper class, and probably not the FisherForecast object that
    should be used to study the impact of gains.  See FF_complex_gains.

    Args:
      ff (FisherForecast): A FisherForecast object to which we wish to add gains.

    Attributes:
      ff (FisherForecast): The FisherForecast object before gain reconstruction.
      plbls (list): List of parameter labels (str) including gains parameter names.
      gain_amplitude_priors (list): list of standard deviations of the log-normal priors on the gain amplitudes. Default: 10.
      gain_phase_priors (list): list of standard deviations on the normal priors on the gain phases. Default: 100.
      gain_ratio_amplitude_priors (list): For polarized gains, list of the log-normal priors on the gain amplitude ratios.  Default: 1e-10.
      gain_ratio_phase_priors (list): For polarized gains, list of the normal priors on the gain phase differences.  Default: 1e-10.
    """

    def __init__(self,ff) :
        super().__init__()
        self.ff = ff
        self.stokes = self.ff.stokes
        self.scans = False
        self.plbls = self.ff.parameter_labels()
        self.prior_sigma_list = self.ff.prior_sigma_list
        self.gain_amplitude_priors = {}
        self.gain_phase_priors = {}
        self.gain_ratio_amplitude_priors = {}
        self.gain_ratio_phase_priors = {}
        
    def visibilities(self,obs,p,verbosity=0,**kwargs) :
        """
        Complex visibilities associated with the underlying FisherForecast object
        evaluated at the data points in the given Obsdata object for the model with
        the given parameter values.

        Args:
          obs (Obsdata): An ehtim Obsdata object with a particular set of observation details (u,v positions, times, etc.)
          p (np.array): list of parameters for the model at which visiblities are desired.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          V (np.array): list of complex visibilities computed at observations.
        """
        
        return self.ff.visibilities(obs,p,verbosity=verbosity,**kwargs)

    def visibility_gradients(self,obs,p,verbosity=0,**kwargs) :
        """
        Gradients of the complex visibilities associated with the underlying 
        FisherForecast object with respect to the model pararmeters evaluated at 
        the data points in the given Obsdata object for the model with the given 
        parameter values, including gradients with respect to the gain amplitudes,
        phases, and if this is a polarized model, R/L gain amplitude ratios and 
        phase differences.  

        Args:
          obs (Obsdata): An ehtim Obsdata object with a particular set of observation details (u,v positions, times, etc.)
          p (np.array): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          gradV (np.ndarray): list of complex visibility gradients computed at observations.
        """
        
        # Generate the gain epochs and update the size
        station_list = np.unique(np.concatenate((obs.data['t1'],obs.data['t2'])))
        nt = len(station_list)
        self.size = self.ff.size
        self.plbls = self.ff.parameter_labels()
        self.prior_sigma_list = copy.copy(self.ff.prior_sigma_list)

        if (self.prior_sigma_list == []) :
            self.prior_sigma_list = [None]*self.size

        if (self.stokes == 'I'):

            V_pg = self.ff.visibilities(obs,p,verbosity=verbosity,**kwargs)
            gradV_pg = self.ff.visibility_gradients(obs,p,verbosity=verbosity,**kwargs)

            # Start with model parameters
            gradV = list(gradV_pg.T)

            # Now add gains
            for station in station_list :
                dVda = 0.0j*V_pg
                dVdp = 0.0j*V_pg

                # G
                inget1 = (obs.data['t1']==station)
                if (np.any(inget1)) :
                    dVda[inget1] = V_pg[inget1]
                    dVdp[inget1] = 1.0j * V_pg[inget1]

                # G*
                inget2 = (obs.data['t2']==station)
                if (np.any(inget2)) :
                    dVda[inget2] = V_pg[inget2]
                    dVdp[inget2] = -1.0j * V_pg[inget2]

                # Check if any cases
                if (np.any(inget1) or np.any(inget2)) :

                    gradV.append(dVda)
                    gradV.append(dVdp)
                    self.size +=2
                    self.plbls.append(r'$\ln(|G|_{%s})$'%(station))
                    self.plbls.append(r'${\rm arg}(G_{%s})$'%(station))

                    if ( len(list(self.gain_amplitude_priors.keys()))>0 or len(list(self.gain_phase_priors.keys()))>0) :
                        
                        # Set amplitude priors
                        if ( station in self.gain_amplitude_priors.keys() ) :
                            self.prior_sigma_list.append(self.gain_amplitude_priors[station])
                        elif ( 'All' in self.gain_amplitude_priors.keys() ) :
                            self.prior_sigma_list.append(self.gain_amplitude_priors['All'])
                        else :
                            self.prior_sigma_list.append(10.0) # Big amplitude

                        # Set phase priors
                        if ( station in self.gain_phase_priors.keys() ) :
                            self.prior_sigma_list.append(self.gain_phase_priors[station])
                        elif ( 'All' in self.gain_phase_priors.keys() ) :
                            self.prior_sigma_list.append(self.gain_phase_priors['All'])
                        else :
                            self.prior_sigma_list.append(100.0) # Big phase

                    else :
                        self.prior_sigma_list.append(10.0) # Big amplitude
                        self.prior_sigma_list.append(100.0) # Big phase

            gradV = np.array(gradV).T

            if (verbosity>0) :
                for j in range(min(10,gradV.shape[0])) :
                    line = "SEG gradV: %10.3g %10.3g"%(obs.data['u'][j],obs.data['v'][j])
                    for k in range(gradV.shape[1]) :
                        line = line + " %8.3g + %8.3gi"%(gradV[j,k].real,gradV[j,k].imag)
                    print(line)

            return gradV

        else:

            RR_pg, LL_pg, RL_pg, LR_pg = self.ff.visibilities(obs,p,verbosity=verbosity,**kwargs)
            gradRR_pg, gradLL_pg, gradRL_pg, gradLR_pg = self.ff.visibility_gradients(obs,p,verbosity=verbosity,**kwargs)

            # start with the current list of model parameters
            gradRR = list(gradRR_pg.T)
            gradLL = list(gradLL_pg.T)
            gradRL = list(gradRL_pg.T)
            gradLR = list(gradLR_pg.T)

            # add gains
            for station in station_list :
                
                dRRda = 0.0j*RR_pg
                dRRdp = 0.0j*RR_pg
                dRRdr = 0.0j*RR_pg
                dRRdt = 0.0j*RR_pg

                dLLda = 0.0j*LL_pg
                dLLdp = 0.0j*LL_pg
                dLLdr = 0.0j*LL_pg
                dLLdt = 0.0j*LL_pg

                dRLda = 0.0j*RL_pg
                dRLdp = 0.0j*RL_pg
                dRLdr = 0.0j*RL_pg
                dRLdt = 0.0j*RL_pg

                dLRda = 0.0j*LR_pg
                dLRdp = 0.0j*LR_pg
                dLRdr = 0.0j*LR_pg
                dLRdt = 0.0j*LR_pg

                # G
                inget1 = (obs.data['t1']==station)
                if (np.any(inget1)) :

                    dRRda[inget1] = RR_pg[inget1]
                    dRRdp[inget1] = 1.0j * RR_pg[inget1]
                    dRRdr[inget1] = 0.5 * RR_pg[inget1]
                    dRRdt[inget1] = 0.5j * RR_pg[inget1]

                    dLLda[inget1] = LL_pg[inget1]
                    dLLdp[inget1] = 1.0j * LL_pg[inget1]
                    dLLdr[inget1] = -0.5 * LL_pg[inget1]
                    dLLdt[inget1] = -0.5j * LL_pg[inget1]

                    dRLda[inget1] = RL_pg[inget1]
                    dRLdp[inget1] = 1.0j * RL_pg[inget1]
                    dRLdr[inget1] = 0.5 * RL_pg[inget1]
                    dRLdt[inget1] = 0.5j * RL_pg[inget1]

                    dLRda[inget1] = LR_pg[inget1]
                    dLRdp[inget1] = 1.0j * LR_pg[inget1]
                    dLRdr[inget1] = -0.5 * LR_pg[inget1]
                    dLRdt[inget1] = -0.5j * LR_pg[inget1]

                # G*
                inget2 = (obs.data['t2']==station)
                if (np.any(inget2)) :

                    dRRda[inget2] = RR_pg[inget2]
                    dRRdp[inget2] = -1.0j * RR_pg[inget2]
                    dRRdr[inget2] = 0.5 * RR_pg[inget2]
                    dRRdt[inget2] = -0.5j * RR_pg[inget2]

                    dLLda[inget2] = LL_pg[inget2]
                    dLLdp[inget2] = -1.0j * LL_pg[inget2]
                    dLLdr[inget2] = -0.5 * LL_pg[inget2]
                    dLLdt[inget2] = 0.5j * LL_pg[inget2]

                    dRLda[inget2] = RL_pg[inget2]
                    dRLdp[inget2] = -1.0j * RL_pg[inget2]
                    dRLdr[inget2] = -0.5 * RL_pg[inget2]
                    dRLdt[inget2] = 0.5j * RL_pg[inget2]

                    dLRda[inget2] = LR_pg[inget2]
                    dLRdp[inget2] = -1.0j * LR_pg[inget2]
                    dLRdr[inget2] = 0.5 * LR_pg[inget2]
                    dLRdt[inget2] = -0.5j * LR_pg[inget2]

                # check if any cases
                if (np.any(inget1) or np.any(inget2)) :

                    gradRR.append(dRRda)
                    gradRR.append(dRRdp)
                    gradRR.append(dRRdr)
                    gradRR.append(dRRdt)

                    gradLL.append(dLLda)
                    gradLL.append(dLLdp)
                    gradLL.append(dLLdr)
                    gradLL.append(dLLdt)

                    gradRL.append(dRLda)
                    gradRL.append(dRLdp)
                    gradRL.append(dRLdr)
                    gradRL.append(dRLdt)

                    gradLR.append(dLRda)
                    gradLR.append(dLRdp)
                    gradLR.append(dLRdr)
                    gradLR.append(dLRdt)

                    self.size += 4

                    # gain geometric mean labels
                    self.plbls.append(r'$\ln(|G|_{%s})$'%(station))
                    self.plbls.append(r'${\rm arg}(G_{%s})$'%(station))

                    # gain ratio labels
                    self.plbls.append(r'$\ln(|R|_{%s})$'%(station))
                    self.plbls.append(r'${\rm arg}(R_{%s})$'%(station))

                    # complex gain priors
                    if ( len(list(self.gain_amplitude_priors.keys()))>0 or len(list(self.gain_phase_priors.keys()))>0) :
                        
                        # Set amplitude priors
                        if ( station in self.gain_amplitude_priors.keys() ) :
                            self.prior_sigma_list.append(self.gain_amplitude_priors[station])
                        elif ( 'All' in self.gain_amplitude_priors.keys() ) :
                            self.prior_sigma_list.append(self.gain_amplitude_priors['All'])
                        else :
                            self.prior_sigma_list.append(10.0) # Big amplitude

                        # Set phase priors
                        if ( station in self.gain_phase_priors.keys() ) :
                            self.prior_sigma_list.append(self.gain_phase_priors[station])
                        elif ( 'All' in self.gain_phase_priors.keys() ) :
                            self.prior_sigma_list.append(self.gain_phase_priors['All'])
                        else :
                            self.prior_sigma_list.append(100.0) # Big phase

                    else :
                        self.prior_sigma_list.append(10.0) # Big amplitude
                        self.prior_sigma_list.append(100.0) # Big phase

                    # complex gain ratio priors
                    if ( len(list(self.gain_ratio_amplitude_priors.keys()))>0 or len(list(self.gain_ratio_phase_priors.keys()))>0) :

                        # Set gain ratio amplitude priors
                        if ( station in self.gain_ratio_amplitude_priors.keys() ) :
                            self.prior_sigma_list.append(self.gain_ratio_amplitude_priors[station])
                        elif ( 'All' in self.gain_ratio_amplitude_priors.keys() ) :
                            self.prior_sigma_list.append(self.gain_ratio_amplitude_priors['All'])
                        else :
                            self.prior_sigma_list.append(1e-10) # small amplitude

                        # Set gain ratio phase priors
                        if ( station in self.gain_ratio_phase_priors.keys() ) :
                            self.prior_sigma_list.append(self.gain_ratio_phase_priors[station])
                        elif ( 'All' in self.gain_ratio_phase_priors.keys() ) :
                            self.prior_sigma_list.append(self.gain_ratio_phase_priors['All'])
                        else :
                            self.prior_sigma_list.append(1e-10) # small phase

                    else :
                        self.prior_sigma_list.append(1e-10) # small amplitude
                        self.prior_sigma_list.append(1e-10) # small phase

            gradRR = np.array(gradRR).T
            gradLL = np.array(gradLL).T
            gradRL = np.array(gradRL).T
            gradLR = np.array(gradLR).T

            if (verbosity>0) :
                for j in range(min(10,gradRR.shape[0])) :
                    line = "SEG gradRR: %10.3g %10.3g"%(obs.data['u'][j],obs.data['v'][j])
                    for k in range(gradRR.shape[1]) :
                        line = line + " %8.3g + %8.3gi"%(gradRR[j,k].real,gradRR[j,k].imag)
                    print(line)
                for j in range(min(10,gradLL.shape[0])) :
                    line = "SEG gradLL: %10.3g %10.3g"%(obs.data['u'][j],obs.data['v'][j])
                    for k in range(gradLL.shape[1]) :
                        line = line + " %8.3g + %8.3gi"%(gradLL[j,k].real,gradLL[j,k].imag)
                    print(line)
                for j in range(min(10,gradRL.shape[0])) :
                    line = "SEG gradRL: %10.3g %10.3g"%(obs.data['u'][j],obs.data['v'][j])
                    for k in range(gradRL.shape[1]) :
                        line = line + " %8.3g + %8.3gi"%(gradRL[j,k].real,gradRL[j,k].imag)
                    print(line)
                for j in range(min(10,gradLR.shape[0])) :
                    line = "SEG gradLR: %10.3g %10.3g"%(obs.data['u'][j],obs.data['v'][j])
                    for k in range(gradLR.shape[1]) :
                        line = line + " %8.3g + %8.3gi"%(gradLR[j,k].real,gradLR[j,k].imag)
                    print(line)


            # print('Prior sigma list:',self.prior_sigma_list)

            return gradRR, gradLL, gradRL, gradLR

    def parameter_labels(self,verbosity=0) :
        """
        Parameter labels to be used in plotting, including those associated
        with the gains.

        Args:
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          plabels (list): list of strings with parameter labels.
        """        
        return self.plbls

    def set_gain_amplitude_prior(self,sigma,station=None) :
        """
        Sets the log-normal priors on the gain amplitudes, either for a
        specified station or for all stations.

        Args:
          sigma (float): Standard deviation of the log-amplitude.
          station (str,list): Station code of the station(s) for which the prior is to be set. If None, will set the prior on all stations. Default: None.

        Returns:
        """
        sigma = min(sigma,10.0)
        if (station is None) :
            self.gain_amplitude_priors = {'All':sigma}
        elif (isinstance(station,list)) :
            for s in station :
                self.gain_amplitude_priors[s] = sigma
        else :
            self.gain_amplitude_priors[station] = sigma
            
        self.argument_hash = None

    def set_gain_phase_prior(self,sigma,station=None) :
        """
        Sets the normal priors on the gain phases, either for a specified 
        station or for all stations. Usually, this should be the default
        (uniformative).  

        Args:
          sigma (float): Standard deviation of the phase.
          station (str,list): Station code of the station(s) for which the prior is to be set. If None, will set the prior on all stations. Default: None.

        Returns:
        """
        sigma = min(sigma,100.0)
        if (station is None) :
            self.gain_phase_priors = {'All':sigma}
        elif (isinstance(station,list)) :
            for s in station :
                self.gain_phase_priors[s] = sigma
        else :
            self.gain_phase_priors[station] = sigma
            
        self.argument_hash = None

    def set_gain_ratio_amplitude_prior(self,sigma,station=None) :
        """
        Sets the log-normal priors on the R/L gain amplitude ratios, either 
        for a specified station or for all stations. Only relevant for
        polarized models. Default: 1e-10 for all stations.

        Args:
          sigma (float): Standard deviation of the log-amplitude ratio.
          station (str,list): Station code of the station(s) for which the prior is to be set. If None, will set the prior on all stations. Default: None.

        Returns:
        """
        sigma = min(sigma,100.0)
        if (station is None) :
            self.gain_ratio_amplitude_priors = {'All':sigma}
        else :
            self.gain_ratio_amplitude_priors[station] = sigma
            
        self.argument_hash = None

    def set_gain_ratio_phase_prior(self,sigma,station=None) :
        """
        Sets the normal priors on the R/L gain phase differences, either 
        for a specified station or for all stations. Only relevant for
        polarized models. Default: 1e-10 for all stations.

        Args:
          sigma (float): Standard deviation of the phase.
          station (str,list): Station code of the station(s) for which the prior is to be set. If None, will set the prior on all stations. Default: None.

        Returns:
        """
        sigma = min(sigma,100.0)
        if (station is None) :
            self.gain_ratio_phase_priors = {'All':sigma}
        else :
            self.gain_ratio_phase_priors[station] = sigma
            
        self.argument_hash = None
        

class FF_complex_gains(FisherForecast) :
    """
    FisherForecast with complex gain reconstruction with multiple epochs.
    This is usually the FisherForecast object that should be used to investigate
    the impact of uncertain station gains.

    Args:
      ff (FisherForecast): A FisherForecast object to which we wish to add gains.

    Attributes:
      ff (FisherForecast): The FisherForecast object before gain reconstruction.
      gain_epochs (np.ndarray): 2D array containing the start and end times for each gain solution epoch.
      gain_amplitude_priors (list): list of standard deviations of the log-normal priors on the gain amplitudes. Default: 10.
      gain_phase_priors (list): list of standard deviations on the normal priors on the gain phases. Default: 100.
      gain_ratio_amplitude_priors (list): For polarized gains, list of the log-normal priors on the gain amplitude ratios.  Default: 1e-10.
      gain_ratio_phase_priors (list): For polarized gains, list of the normal priors on the gain phase differences.  Default: 1e-10.
    """

    def __init__(self,ff) :
        super().__init__()
        self.ff = ff
        self.stokes = self.ff.stokes
        self.scans = False
        self.gain_epochs = None
        self.prior_sigma_list = self.ff.prior_sigma_list
        self.gain_amplitude_priors = {}
        self.gain_phase_priors = {}
        self.gain_ratio_amplitude_priors = {}
        self.gain_phase_priors = {}
        self.ff_cgse = FF_complex_gains_single_epoch(ff)
        self.size = self.ff.size
        self.covar = np.zeros((self.ff.size,self.ff.size))
        
    def set_gain_epochs(self,scans=False,gain_epochs=None) :
        """
        Sets the gain solution intervals (gain epochs) to be used. If neither
        scans nor gain_epochs selected, will solve for gains on each unique
        timestamp in the data.

        Args:
          scans (bool): If True, solves for independent gains by scan. Overrides explicit specification of gain solution intervals. Default: False.
          gain_epochs (nd.array): 2D array containing the start and end times for each gain solution epoch. Default: None.

        Returns:
        """
        
        self.scans = scans
        self.gain_epochs = gain_epochs
        
    def generate_gain_epochs(self,obs) :
        if (self.scans==True) :
            obs.add_scans()
            self.gain_epochs = obs.scans
        elif (self.gain_epochs is None) :
            tu = np.unique(obs.data['time'])
            dt = tu[1:]-tu[:-1]
            dt = np.array([dt[0]]+list(dt)+[dt[-1]])
            self.gain_epochs = np.zeros((len(tu),2))
            self.gain_epochs[:,0] = tu - 0.5*dt[:-1]
            self.gain_epochs[:,1] = tu + 0.5*dt[1:]
    
    def visibilities(self,obs,p,verbosity=0,**kwargs) :
        """
        Complex visibilities associated with the underlying FisherForecast object
        evaluated at the data points in the given Obsdata object for the model with
        the given parameter values.

        Args:
          obs (Obsdata): An ehtim Obsdata object with a particular set of observation details (u,v positions, times, etc.)
          p (np.array): list of parameters for the model at which visiblities are desired.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          V (np.array): list of complex visibilities computed at observations.
        """
        
        return self.ff.visibilities(obs,p,verbosity=verbosity,**kwargs)

    def visibility_gradients(self,obs,p,verbosity=0,**kwargs) :
        """
        Gradients of the complex visibilities associated with the underlying 
        FisherForecast object with respect to the model pararmeters evaluated at 
        the data points in the given Obsdata object for the model with the given 
        parameter values.  

        Args:
          obs (Obsdata): An ehtim Obsdata object with a particular set of observation details (u,v positions, times, etc.)
          p (np.array): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          gradV (np.ndarray): list of complex visibility gradients computed at observations.
        """
        
        return self.ff.visibility_gradients(obs,p,verbosity=verbosity,**kwargs)

    def fisher_covar(self,obs,p,verbosity=0,**kwargs) :
        """
        Returns the Fisher matrix M as defined in the accompanying documentation,
        marginalized over the complex station gains. Intelligently avoids 
        recomputation if the observation and parameters are unchanged.

        Args:
          obs (Obsdata): An Obsdata object containing the observation particulars.
          p (list): List of model parameters at which to compute the Fisher matrix.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          M (np.ndarray): The Fisher matrix M.
        """
        
        # Get the fisher covariance 
        new_argument_hash = hashlib.md5(bytes(str(obs)+str(p),'utf-8')).hexdigest()
        if ( new_argument_hash == self.argument_hash ) :
            return self.covar
        else :
            self.argument_hash = new_argument_hash
            self.covar = np.zeros((self.ff.size,self.ff.size))
            if self.stokes == 'I':
                obs = obs.switch_polrep('stokes')
                self.generate_gain_epochs(obs)
                self.ff_cgse.gain_amplitude_priors = copy.copy(self.gain_amplitude_priors)
                self.ff_cgse.gain_phase_priors = copy.copy(self.gain_phase_priors)
                self.ff_cgse.argument_hash = None

                if (verbosity>0) :
                    print("gain amplitude dicts -- global:",self.gain_amplitude_priors)
                    print("gain amplitude dicts -- local: ",self.ff_cgse.gain_amplitude_priors)
                    
                
                for ige,ge in enumerate(self.gain_epochs) :
                    with ploop.HiddenPrints():
                        obs_ge = obs.flag_UT_range(UT_start_hour=ge[0],UT_stop_hour=ge[1],output='flagged')

                    gradV_wgs = self.ff_cgse.visibility_gradients(obs_ge,p,verbosity=verbosity,**kwargs)
                    covar_wgs = np.zeros((self.ff_cgse.size,self.ff_cgse.size))
                    
                    for i in range(self.ff_cgse.size) :
                        for j in range(self.ff_cgse.size) :
                            covar_wgs[i][j] = np.sum( np.conj(gradV_wgs[:,i])*gradV_wgs[:,j]/obs_ge.data['sigma']**2 + gradV_wgs[:,i]*np.conj(gradV_wgs[:,j])/obs_ge.data['sigma']**2)

                    for i in np.arange(self.ff.size,self.ff_cgse.size) :
                        if (not self.ff_cgse.prior_sigma_list[i] is None) :
                            covar_wgs[i][i] += 2.0/self.ff_cgse.prior_sigma_list[i]**2
                            
                    n = covar_wgs[:self.ff.size,:self.ff.size]
                    r = covar_wgs[self.ff.size:,:self.ff.size]
                    m = covar_wgs[self.ff.size:,self.ff.size:]
                    r,m = self._condition_vM(r,m)
                    mn = n - np.matmul(r.T,np.matmul(_invert_matrix(m),r))
                    
                    if (verbosity>1) :
                        print("gain epoch %g of %g ----------------------"%(ige,self.gain_epochs.shape[0]))
                        print("obs stations:",np.unique(list(obs_ge.data['t1'])+list(obs_ge.data['t2'])))
                        print("obs times:",np.unique(obs_ge.data['time']))
                        print("obs data:\n",obs_ge.data)
                        print("gradV gains:\n",gradV_wgs[self.size:])
                        print("covar:")
                        _print_matrix(covar_wgs)
                        print("n:")
                        _print_matrix(n)
                        print("r:")
                        _print_matrix(r)
                        print("m:")
                        _print_matrix(m)
                        print("minv:")
                        _print_matrix(_invert_matrix(m))
                        print("marginalized n:")
                        _print_matrix(mn)
                        print("marginalized/symmetrized n:")
                        _print_matrix(0.5*(mn+mn.T))

                    #quit()

                    
                    self.covar = self.covar + 0.5*(mn+mn.T)

            else:
                obs = obs.switch_polrep('circ')
                self.generate_gain_epochs(obs)
                self.ff_cgse.gain_amplitude_priors = copy.copy(self.gain_amplitude_priors)
                self.ff_cgse.gain_phase_priors = copy.copy(self.gain_phase_priors)
                self.ff_cgse.gain_ratio_amplitude_priors = copy.copy(self.gain_ratio_amplitude_priors)
                self.ff_cgse.gain_ratio_phase_priors = copy.copy(self.gain_ratio_phase_priors)
                self.ff_cgse.argument_hash = None

                if (verbosity>0) :
                    print("gain amplitude dicts -- global:",self.gain_amplitude_priors)
                    print("gain amplitude dicts -- local: ",self.ff_cgse.gain_amplitude_priors)
                    print("gain phase dicts -- global:",self.gain_phase_priors)
                    print("gain phase dicts -- local: ",self.ff_cgse.gain_phase_priors)
                    print("gain ratio amplitude dicts -- global:",self.gain_ratio_amplitude_priors)
                    print("gain ratio amplitude dicts -- local: ",self.ff_cgse.gain_ratio_amplitude_priors)
                    print("gain ratio phase dicts -- global:",self.gain_ratio_phase_priors)
                    print("gain ratio phase dicts -- local: ",self.ff_cgse.gain_ratio_phase_priors)

                for ige,ge in enumerate(self.gain_epochs) :
                    with ploop.HiddenPrints():
                        obs_ge = obs.flag_UT_range(UT_start_hour=ge[0],UT_stop_hour=ge[1],output='flagged')

                    gradRR_wgs, gradLL_wgs, gradRL_wgs, gradLR_wgs = self.ff_cgse.visibility_gradients(obs_ge,p,verbosity=verbosity,**kwargs)
                    covar_wgs = np.zeros((self.ff_cgse.size,self.ff_cgse.size))

                    for i in range(self.ff_cgse.size) :
                        for j in range(self.ff_cgse.size) :
                            covar_wgs[i][j] = np.sum( np.conj(gradRR_wgs[:,i])*gradRR_wgs[:,j]/obs_ge.data['rrsigma']**2 + gradRR_wgs[:,i]*np.conj(gradRR_wgs[:,j])/obs_ge.data['rrsigma']**2)
                            covar_wgs[i][j] += np.sum( np.conj(gradLL_wgs[:,i])*gradLL_wgs[:,j]/obs_ge.data['llsigma']**2 + gradLL_wgs[:,i]*np.conj(gradLL_wgs[:,j])/obs_ge.data['llsigma']**2)
                            covar_wgs[i][j] += np.sum( np.conj(gradRL_wgs[:,i])*gradRL_wgs[:,j]/obs_ge.data['rlsigma']**2 + gradRL_wgs[:,i]*np.conj(gradRL_wgs[:,j])/obs_ge.data['rlsigma']**2)
                            covar_wgs[i][j] += np.sum( np.conj(gradLR_wgs[:,i])*gradLR_wgs[:,j]/obs_ge.data['lrsigma']**2 + gradLR_wgs[:,i]*np.conj(gradLR_wgs[:,j])/obs_ge.data['lrsigma']**2)

                    for i in np.arange(self.ff.size,self.ff_cgse.size) :
                        if (not self.ff_cgse.prior_sigma_list[i] is None) :
                            covar_wgs[i][i] += 2.0/self.ff_cgse.prior_sigma_list[i]**2

                    n = covar_wgs[:self.ff.size,:self.ff.size]
                    r = covar_wgs[self.ff.size:,:self.ff.size]
                    m = covar_wgs[self.ff.size:,self.ff.size:]
                    r,m = self._condition_vM(r,m)
                    mn = n - np.matmul(r.T,np.matmul(_invert_matrix(m),r))

                    self.covar = self.covar + 0.5*(mn+mn.T)

            if (len(self.prior_sigma_list)>0) :
                for i in range(self.size) :
                    if (not self.prior_sigma_list[i] is None) :
                        self.covar[i][i] += 2.0/self.prior_sigma_list[i]**2 # Why factor of 2?
                    
        return self.covar
    
    def parameter_labels(self,verbosity=0) :
        """
        Parameter labels to be used in plotting.

        Args:
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          plabels (list): list of strings with parameter labels.
        """        
        return self.ff.parameter_labels()

    def set_gain_amplitude_prior(self,sigma,station=None,verbosity=0) :
        """
        Sets the log-normal priors on the gain amplitudes, either for a
        specified station or for all stations. Identical priors are 
        assummed across the entire observation.

        Args:
          sigma (float): Standard deviation of the log-amplitude.
          station (str,list): Station code of the station(s) for which the prior is to be set. If None, will set the prior on all stations. Default: None.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
        """
        sigma = min(sigma,10.0)
        if (station is None) :
            self.gain_amplitude_priors = {'All':sigma}
        else :
            self.gain_amplitude_priors[station] = sigma
            
        if (verbosity>0) :
            print("Gain amplitude dict:",self.gain_amplitude_priors)
            
        self.argument_hash = None

    def set_gain_phase_prior(self,sigma,station=None,verbosity=0) :
        """
        Sets the normal priors on the gain phases, either for a specified 
        station or for all stations. Usually, this should be the default
        (uniformative). Identical priors are assummed across the entire 
        observation.

        Args:
          sigma (float): Standard deviation of the phase.
          station (str,list): Station code of the station(s) for which the prior is to be set. If None, will set the prior on all stations. Default: None.

        Returns:
        """
        sigma = min(sigma,100.0)
        if (station is None) :
            self.gain_phase_priors = {'All':sigma}
        else :
            self.gain_phase_priors[station] = sigma
            
        if (verbosity>0) :
            print("Gain phase dict:",self.gain_phase_priors)

        self.argument_hash = None

    def set_gain_ratio_amplitude_prior(self,sigma,station=None,verbosity=0) :
        """
        Sets the log-normal priors on the R/L gain amplitude ratios, either 
        for a specified station or for all stations. Only relevant for
        polarized models. Default: 1e-10 for all stations. Identical priors 
        are assummed across the entire observation.

        Args:
          sigma (float): Standard deviation of the log-amplitude ratio.
          station (str,list): Station code of the station(s) for which the prior is to be set. If None, will set the prior on all stations. Default: None.

        Returns:
        """
        sigma = min(sigma,100.0)
        if (station is None) :
            self.gain_ratio_amplitude_priors = {'All':sigma}
        else :
            self.gain_ratio_amplitude_priors[station] = sigma
            
        if (verbosity>0) :
            print("Gain ratio amplitude dict:",self.gain_ratio_amplitude_priors)
            
        self.argument_hash = None

    def set_gain_ratio_phase_prior(self,sigma,station=None,verbosity=0) :
        """
        Sets the normal priors on the R/L gain phase differences, either 
        for a specified station or for all stations. Only relevant for
        polarized models. Default: 1e-10 for all stations. Identical priors 
        are assummed across the entire observation.

        Args:
          sigma (float): Standard deviation of the phase.
          station (str,list): Station code of the station(s) for which the prior is to be set. If None, will set the prior on all stations. Default: None.

        Returns:
        """
        sigma = min(sigma,100.0)
        if (station is None) :
            self.gain_ratio_phase_priors = {'All':sigma}
        else :
            self.gain_ratio_phase_priors[station] = sigma
            
        if (verbosity>0) :
            print("Gain ratio phase dict:",self.gain_ratio_phase_priors)

        self.argument_hash = None
        
    
class FF_model_image(FisherForecast) :
    """
    FisherForecast from ThemisyPy model_image objects. Uses FFTs and centered 
    finite-difference to compute the visibilities and gradients. May not always
    produce sensible behaviors. Has not been extensively tested for some time.

    Args:
      img (model_image): A ThemisPy model_image object.
      stokes (str): Indicates if this is limited to Stokes I ('I') or include full polarization ('full'). Default: 'I'.

    Attributes:
      img (model_image): The ThemisPy model_image object for which to make forecasts.
    """

    def __init__(self,img,stokes='I') :
        super().__init__()
        self.img = img
        self.size = self.img.size
        self.stokes = stokes

    def visibilities(self,obs,p,limits=None,shape=None,padding=4,verbosity=0) :
        """
        Generates visibilities associated with a given model image object.  Note 
        that this uses FFTs to create the visibilities and then interpolates.

        Args:
          obs (ehtim.Obsdata): ehtim data object
          p (np.array): list of parameters for the model image used to create object.
          limits (list): May be single number, list with four elements specifying in :math:`\\mu as` the symmetric limits or explicit limits in [xmin,xmax,ymin,ymax] order.  Default: 100.
          shape (list): May be single number or list with two elements indicating the number of pixels in the two directions.  Default: 256.
          padding (int): Factor by which to pad image with zeros prior to FFT.  Default: 4.
          verbosity (int): Verbosity level. 0 prints nothing. 1 prints various elements of the plotting process. Passed to :func:`model_image.generate_intensity_map`. Default: 0.

        Returns:
          V (np.array): list of complex visibilities computed at observations.
        """

        if self.stokes != 'I':
            raise(Exception('Polarized visibilities for FF_model_image are not yet implemented!'))

        # Generate the image at current location
        self.img.generate(p)

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
            
        # Generate a set of pixels
        x,y = np.mgrid[limits[0]:limits[1]:shape[0]*1j,limits[2]:limits[3]:shape[1]*1j]
        # Generate a set of intensities (Jy/uas^2)
        I = self.img.intensity_map(x,y,p,verbosity=verbosity)
        V00_img = np.sum(I*np.abs((x[1,1]-x[0,0])*(y[1,1]-y[0,0])))

        uas2rad = np.pi/180./3600e6
        x = x * uas2rad
        y = y * uas2rad
        
        # Generate the complex visibilities, gridded
        V2d = np.fft.fftshift(np.conj(np.fft.fft2(I,s=padding*np.array(shape))))
        u1d = np.fft.fftshift(np.fft.fftfreq(padding*I.shape[0],np.abs(x[1,1]-x[0,0])))
        v1d = np.fft.fftshift(np.fft.fftfreq(padding*I.shape[1],np.abs(y[1,1]-y[0,0])))
        
        # Center image
        i2d,j2d = np.meshgrid(np.arange(V2d.shape[0]),np.arange(V2d.shape[1]))
        V2d = V2d * np.exp(1.0j*np.pi*(i2d+j2d))
        
        
        # Generate the interpolator and interpolate to the u,v list
        Vrinterp = si.RectBivariateSpline(u1d,v1d,np.real(V2d))
        Viinterp = si.RectBivariateSpline(u1d,v1d,np.imag(V2d))

        V00_fft = Vrinterp(0.0,0.0)
        V2d = V2d * V00_img/V00_fft
        
        V = 0.0j * u
        for i in range(len(u)) :
            V[i] = (Vrinterp(obs.data['u'][i],obs.data['v'][i]) + 1.0j*Viinterp(obs.data['u'][i],obs.data['v'][i]))

        return V
            
    def visibility_gradients(self,obs,p,h=1e-2,limits=None,shape=None,padding=4,verbosity=0) :
        """
        Generates visibilities associated with a given model image object.  Note 
        that this uses FFTs to create the visibilities and then interpolates.

        Args:
          obs (ehtim.Obsdata): ehtim data object
          p (np.array): list of parameters for the model image used to create object.
          h (float): fractional step for finite-difference gradient computation.  Default: 0.01.
          limits (list): May be single number, list with four elements specifying in :math:`\\mu as` the symmetric limits or explicit limits in [xmin,xmax,ymin,ymax] order.  Default: 100.
          shape (list): May be single number or list with two elements indicating the number of pixels in the two directions.  Default: 256.
          padding (int): Factor by which to pad image with zeros prior to FFT.  Default: 4.
          verbosity (int): Verbosity level. 0 prints nothing. 1 prints various elements of the plotting process. Passed to :func:`model_image.generate_intensity_map`. Default: 0.

        Returns:
          gradV (np.ndarray): list of complex visibilities gradients computed at observation.
        """

        if self.stokes != 'I':
            raise(Exception('Polarized gradients for FF_model_image are not yet implemented!'))

        pp = np.copy(p)
        gradV = np.zeros((len(u),self.img.size))

        for i in range(self.img.size) :
            if (np.abs(p[i])>0) :
                hq = h*np.abs(p[i])
            else :
                hq = h**2
            pp[i] = p[i] + hq
            Vmp = np.copy(self.visibilities(obs,pp,limits=limits,shape=shape,padding=padding,verbosity=verbosity))
            pp[i] = p[i] - hq
            Vmm = np.copy(self.visibilities(obs,pp,limits=limits,shape=shape,padding=padding,verbosity=verbosity))
            pp[i] = p[i]
            gradV[:,i] = (Vmp-Vmm)/(2.0*hq)

        return gradV

    def parameter_labels(self,verbosity=0) :
        """
        Parameter labels to be used in plotting.

        Args:
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          plabels (list): list of strings with parameter labels.
        """        
        return self.img.parameter_name_list()
    

class FF_smoothed_delta_ring(FisherForecast) :
    """
    FisherForecast object for a delta-ring convolved with a circular Gaussian.
    Parameter vector is:
      p[0] ... Total flux in Jy.
      p[1] ... Diameter in uas.
      p[2] ... Twice the standard deviation of the Gaussian smoothing kernel in uas.

    Args:
      stokes (str): Indicates if this is limited to Stokes I ('I') or include full polarization ('full'). Default: 'I'.
    """

    def __init__(self,stokes='I') :
        super().__init__()
        self.size = 3
        self.stokes = stokes
        
    def visibilities(self,obs,p,verbosity=0) :
        """
        Generates visibilities associated with Gaussian-convolved delta-ring.

        Args:
          obs (ehtim.Obsdata): ehtim data object
          p (np.array): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          V (np.array): list of complex visibilities computed at observations.
        """
        
        # Takes:
        #  p[0] ... flux in Jy
        #  p[1] ... diameter in uas
        #  p[2] ... width in uas

        if self.stokes != 'I':
            raise(Exception('Polarized visibilities for FF_smoothed_delta_ring are not yet implemented!'))

        u = obs.data['u']
        v = obs.data['v']
        
        d = p[1]
        w = p[2] 
        uas2rad = np.pi/180./3600e6
        piuv = np.pi*np.sqrt(u**2+v**2)*uas2rad
        x = piuv*d
        y = piuv*w
        J0 = ss.jv(0,x)
        J1 = ss.jv(1,x)
        ey2 = np.exp(-0.5*y**2)
        V = p[0] * J0 * ey2
        return V
        
    def visibility_gradients(self,obs,p,verbosity=0) :
        """
        Generates visibility gradients associated with Gaussian-convolved delta-ring.

        Args:
          obs (ehtim.Obsdata): ehtim data object
          p (np.array): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          gradV (np.array): list of complex visibilities computed at observations.
        """

        # Takes:
        #  p[0] ... flux in Jy
        #  p[1] ... diameter in uas
        #  p[2] ... width in uas

        if self.stokes != 'I':
            raise(Exception('Polarized gradients for FF_smoothed_delta_ring are not yet implemented!'))

        u = obs.data['u']
        v = obs.data['v']
                
        d = p[1]
        w = p[2]
        uas2rad = np.pi/180./3600e6
        piuv = np.pi*np.sqrt(u**2+v**2)*uas2rad
        x = piuv*d
        y = piuv*w
        J0 = ss.jv(0,x)
        J1 = ss.jv(1,x)
        ey2 = np.exp(-0.5*y**2)

        gradV = np.array([ J0*ey2,  # dV/dp[0]
                           -p[0]*piuv*J1*ey2, # dV/dd
                           -p[0]*piuv**2*w*J0*ey2 ]) # dV/dw

        return gradV.T
        

    def parameter_labels(self,verbosity=0) :
        """
        Parameter labels to be used in plotting.

        Args:
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          plabels (list): list of strings with parameter labels.
        """        
        return [r'$\delta F~({\rm Jy})$',r'$\delta d~(\mu{\rm as})$',r'$\delta w~(\mu{\rm as})$']
    

class FF_symmetric_gaussian(FisherForecast) :
    """
    FisherForecast object for a circular Gaussian.
    Parameter vector is:
      p[0] ... Total flux in Jy.
      p[1] ... FWHM in uas.

    Args:
      stokes (str): Indicates if this is limited to Stokes I ('I') or include full polarization ('full'). Default: 'I'.
    """

    def __init__(self,stokes='I') :
        super().__init__()
        self.size = 2
        self.stokes = stokes
        
    def visibilities(self,obs,p,verbosity=0) :
        """
        Generates visibilities associated with a circular Gaussian.

        Args:
          obs (ehtim.Obsdata): ehtim data object
          p (np.array): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          V (np.array): list of complex visibilities computed at observations.
        """

        # Takes:
        #  p[0] ... flux in Jy
        #  p[1] ... diameter in uas

        if self.stokes != 'I':
            raise(Exception('Polarized visibilities for FF_symmetric_gaussian are not yet implemented!'))

        u = obs.data['u']
        v = obs.data['v']

        d = p[1]
        uas2rad = np.pi/180./3600e6
        piuv = 2.0*np.pi*np.sqrt(u**2+v**2)*uas2rad / np.sqrt(8.0*np.log(2.0))
        y = piuv * d
        ey2 = np.exp(-0.5*y**2)
        return p[0]*ey2
        
    def visibility_gradients(self,obs,p,verbosity=0) :
        """
        Generates visibility gradients associated with a circular Gaussian.

        Args:
          obs (ehtim.Obsdata): ehtim data object
          p (np.array): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          gradV (np.array): list of complex visibilities computed at observations.
        """

        # Takes:
        #  p[0] ... flux in Jy
        #  p[1] ... diameter in uas

        if self.stokes != 'I':
            raise(Exception('Polarized gradients for FF_symmetric_gaussian are not yet implemented!'))

        u = obs.data['u']
        v = obs.data['v']

        d = p[1]
        uas2rad = np.pi/180./3600e6
        piuv = 2.0*np.pi*np.sqrt(u**2+v**2)*uas2rad / np.sqrt(8.0*np.log(2.0))
        y = piuv * d
        ey2 = np.exp(-0.5*y**2)

        gradV = np.array([ ey2, # dV/dp[0]
                           -p[0]*piuv**2*d*ey2 ]) # dV/dd

        return gradV.T
        

    def parameter_labels(self,verbosity=0) :
        """
        Parameter labels to be used in plotting.

        Args:
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          plabels (list): list of strings with parameter labels.
        """        
        return [r'$\delta F~({\rm Jy})$',r'$\delta FWHM~(\mu{\rm as})$']


class FF_asymmetric_gaussian(FisherForecast) :
    """
    FisherForecast object for a noncircular Gaussian.  
    Parameter vector is:
      p[0] ... Total flux in Jy.
      p[1] ... Symmetrized mean of the FHWM in the major and minor axes in uas.
      p[2] ... Asymmetry parameter, A, expected to be in the range [0,1).
      p[3] ... Position angle in radians.
    The major axis FWHM is p[1]/sqrt(1-A); the minor axis FWHM is p[1]/sqrt(1+A).

    Args:
      stokes (str): Indicates if this is limited to Stokes I ('I') or include full polarization ('full'). Default: 'I'.
    """

    def __init__(self,stokes='I') :
        super().__init__()
        self.size = 4
        self.stokes = stokes
        
    def visibilities(self,obs,p,verbosity=0) :
        """
        Generates visibilities associated with a noncircular Gaussian.

        Args:
          obs (ehtim.Obsdata): ehtim data object
          p (np.array): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          V (np.array): list of complex visibilities computed at observations.
        """

        # Takes:
        #  p[0] ... flux in Jy
        #  p[1] ... mean diameter in uas
        #  p[2] ... asymmetry parameter 
        #  p[3] ... position angle in radians

        if self.stokes != 'I':
            raise(Exception('Polarized visibilities for FF_asymmetric_gaussian are not yet implemented!'))

        u = obs.data['u']
        v = obs.data['v']

        uas2rad = np.pi/180./3600e6
        
        F = p[0]
        sig = p[1]*uas2rad / np.sqrt(8.0*np.log(2.0))
        A = p[2]

        cpa = np.cos(p[3])
        spa = np.sin(p[3])
        ur = cpa*u + spa*v
        vr = -spa*u + cpa*v

        sigmaj = sig / np.sqrt(1-A)
        sigmin = sig / np.sqrt(1+A)

        pimaj = 2.0*np.pi*ur
        pimin = 2.0*np.pi*vr

        ymaj = pimaj*sigmaj
        ymin = pimin*sigmin

        ey2 = np.exp(-0.5*(ymaj**2+ymin**2))

        return F*ey2

    
    def visibility_gradients(self,obs,p,verbosity=0) :
        """
        Generates visibility gradients associated with a noncircular Gaussian.

        Args:
          obs (ehtim.Obsdata): ehtim data object
          p (np.array): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          V (np.array): list of complex visibilities computed at observations.
        """
        
        # Takes:
        #  p[0] ... flux in Jy
        #  p[1] ... mean diameter in uas
        #  p[2] ... asymmetry parameter 
        #  p[3] ... position angle in radians

        if self.stokes != 'I':
            raise(Exception('Polarized gradients for FF_asymmetric_gaussian are not yet implemented!'))

        u = obs.data['u']
        v = obs.data['v']

        uas2rad_fwhm2sig = np.pi/180./3600e6 / np.sqrt(8.0*np.log(2.0))
        
        F = p[0]
        sig = p[1]
        A = p[2]

        cpa = np.cos(p[3])
        spa = np.sin(p[3])
        ur = cpa*u + spa*v
        vr = -spa*u + cpa*v

        sigmaj = sig / np.sqrt(1-A)
        sigmin = sig / np.sqrt(1+A)

        pimaj = 2.0*np.pi*ur * uas2rad_fwhm2sig 
        pimin = 2.0*np.pi*vr * uas2rad_fwhm2sig 

        ymaj = pimaj*sigmaj
        ymin = pimin*sigmin

        ey2 = np.exp(-0.5*(ymaj**2+ymin**2))

        gradV = np.array([ ey2, # dV/dF
                           -F*ey2*( pimaj**2*sig/(1-A) + pimin**2*sig/(1+A) ), # dV/dd
                           -F*ey2*0.5*( (pimaj*sig/(1-A))**2 - (pimin*sig/(1+A))**2 ), # dV/dA
                           -F*ey2* pimaj*pimin * ( sigmaj**2 - sigmin**2 ) ])
        
        
        return gradV.T

    def parameter_labels(self,verbosity=0) :
        """
        Parameter labels to be used in plotting.

        Args:
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          plabels (list): list of strings with parameter labels.
        """        
        return [r'$\delta F~({\rm Jy})$',r'$\delta FWHM~(\mu{\rm as})$',r'$\delta A$',r'$\delta {\rm PA}~({\rm rad})$']

    

class FF_splined_raster(FisherForecast) :
    """
    FisherForecast object for a splined raster (i.e., themage).
    Parameter vector is the log of the intensity at the control points:
      p[0] ....... ln(I[0,0])
      p[1] ....... ln(I[1,0])
       ...
      p[N-1] ..... ln(I[N-1,0])
      p[N] ....... ln(I[0,1])
       ...
      p[N*N-1] ... ln(I[N-1,N-1])
    where each I[i,j] is measured in Jy/sr.

    Args:
      N (int): Raster dimension (only supports square raster).
      fov (float): Raster field of view in uas (only supports fixed rasters).
      stokes (str): Indicates if this is limited to Stokes I ('I') or include full polarization ('full'). Default: 'I'.

    Attributes:
      xcp (np.ndarray): x-positions of raster control points.
      ycp (np.ndarray): y-positions of raster control points.
      apx (float): Raster pixel size.
    """

    def __init__(self,N,fov,stokes='I') :
        super().__init__()
        self.npx = N
        self.size = self.npx**2
        fov = fov * np.pi/(3600e6*180) * (N-1)/N
        
        self.xcp,self.ycp = np.meshgrid(np.linspace(-0.5*fov,0.5*fov,self.npx),np.linspace(-0.5*fov,0.5*fov,self.npx))

        self.apx = (self.xcp[1,1]-self.xcp[0,0])*(self.ycp[1,1]-self.ycp[0,0])

        self.stokes = stokes

        if self.stokes != 'I':
            self.size *= 4

    def visibilities(self,obs,p,verbosity=0) :
        """
        Generates visibilities associated with a splined raster model.

        Args:
          obs (ehtim.Obsdata): ehtim data object
          p (np.array): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          V (np.array): list of complex visibilities computed at observations.
        """
        
        # Takes:
        #  p[0] ... p[0,0]
        #  p[1] ... p[1,0]
        #  ...
        #  p[NxN-1] = p[N-1,N-1]

        if self.stokes != 'I':
            raise(Exception('Polarized visibilities for FF_splined_raster are not yet implemented!'))

        u = obs.data['u']
        v = obs.data['v']
        
        # Add themage
        if self.stokes == 'I':
            V = 0.0j*u
            for i in range(self.npx) :
                for j in range(self.npx) :
                    V = V + np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[i+self.npx*j]) * ty.vis.W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*ty.vis.W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx
            return V
        
        else:
            I = 0.0j*u
            Q = 0.0j*u
            U = 0.0j*u
            V = 0.0j*u

            countI = 0
            for i in range(self.npx) :
                for j in range(self.npx) :
                    I = I + np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[i+self.npx*j]) * ty.vis.W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*ty.vis.W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx
                    countI += 1
            for i in range(self.npx) :
                for j in range(self.npx) :
                    Q = Q + np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[countI + i+self.npx*j]) * ty.vis.W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*ty.vis.W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx
                    U = U + np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[2*countI + i+self.npx*j]) * ty.vis.W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*ty.vis.W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx
                    V = V + np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[3*countI + i+self.npx*j]) * ty.vis.W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*ty.vis.W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx

            RR = I + V
            LL = I - V
            RL = Q + (1.0j)*U
            LR = Q - (1.0j)*U

            return RR, LL, RL, LR
        
    def visibility_gradients(self,obs,p,verbosity=0) :
        """
        Generates visibility gradients associated with a splined raster model.

        Args:
          obs (ehtim.Obsdata): ehtim data object
          p (np.array): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          gradV (np.array): list of complex visibilities computed at observations.
        """

        # Takes:
        #  p[0] ... p[0,0]
        #  p[1] ... p[1,0]
        #  ...
        #  p[NxN-1] = p[N-1,N-1]

        if self.stokes != 'I':
            raise(Exception('Polarized gradients for FF_splined_raster are not yet implemented!'))
        
        u = obs.data['u']
        v = obs.data['v']
        
        # Add themage
        if self.stokes == 'I':
            gradV = list()
            for j in range(self.npx) :
                for i in range(self.npx) :
                    gradV.append( np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[i+self.npx*j]) * ty.vis.W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*ty.vis.W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx )
            gradV = np.array(gradV)
            return gradV.T
        else:
            gradI = list()
            gradQ = list()
            gradU = list()
            gradV = list()
            
            countI = 0
            for j in range(self.npx) :
                for i in range(self.npx) :
                    gradI.append( np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[i+self.npx*j]) * ty.vis.W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*ty.vis.W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx )
                    countI += 1
            for j in range(self.npx) :
                for i in range(self.npx) :
                    gradQ.append( np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[countI + i+self.npx*j]) * ty.vis.W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*ty.vis.W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx )
                    gradU.append( np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[2*countI + i+self.npx*j]) * ty.vis.W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*ty.vis.W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx )
                    gradV.append( np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[3*countI + i+self.npx*j]) * ty.vis.W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*ty.vis.W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx )
            gradI_small = np.array(gradI).T
            gradQ_small = np.array(gradQ).T
            gradU_small = np.array(gradU).T
            gradV_small = np.array(gradV).T

            gradI = np.block([1.0*gradI_small,0.0*gradQ_small,0.0*gradU_small,0.0*gradV_small])
            gradQ = np.block([0.0*gradI_small,1.0*gradQ_small,0.0*gradU_small,0.0*gradV_small])
            gradU = np.block([0.0*gradI_small,0.0*gradQ_small,1.0*gradU_small,0.0*gradV_small])
            gradV = np.block([0.0*gradI_small,0.0*gradQ_small,0.0*gradU_small,1.0*gradV_small])

            grad_RR = gradI + gradV
            grad_LL = gradI - gradV
            grad_RL = gradQ + (1.0j)*gradU
            grad_LR = gradQ - (1.0j)*gradU
            
            return grad_RR, grad_LL, grad_RL, grad_LR


    def parameter_labels(self,verbosity=0) :
        """
        Parameter labels to be used in plotting.

        Args:
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          plabels (list): list of strings with parameter labels.
        """        
        pll = []
        for j in range(self.npx) :
            for i in range(self.npx) :
                pll.append( r'$\delta I_{%i,%i}$'%(i,j) )

        if self.stokes != 'I':
            for j in range(self.npx) :
                for i in range(self.npx) :
                    pll.append( r'$\delta Q_{%i,%i}$'%(i,j) )
            for j in range(self.npx) :
                for i in range(self.npx) :
                    pll.append( r'$\delta U_{%i,%i}$'%(i,j) )
            for j in range(self.npx) :
                for i in range(self.npx) :
                    pll.append( r'$\delta V_{%i,%i}$'%(i,j) )

        return pll

    def generate_parameter_list(self,glob,p=None) :
        """
        Utility function to quickly and easily generate a set of raster values
        associated with various potential initialization schemes. These include:
        an existing FisherForecast object, a FisherForecast class, or a FITS file
        name.

        TBD.

        Args:
          glob (str): Can be an existing FisherForecast child object, the name of a FisherForecast child class, or a string with a FITS file name.
          p (list): If glob is a FisherForecast child object or the name of a FisherForecast child class, a parameter list must be passed from which the image will be generated.
          
        Returns:
          p (list): Approximate parameter list for associated splined raster object.
        """

        if ( issubclass(type(glob),FisherForecast) ) :
            # This is a FF object, set the parameters, make the images and go
            pass
        elif ( issubclass(glob,FisherForecast) ) :
            # This is a FF class, create a FF obj set the parameters, make the images and go
            pass
        elif ( isinstance(glob,str) ) :
            # This is a string, check for .fits and go
            pass
        else :
            raise(RuntimeError("Unrecognized glob from which to generate acceptable parameter list."))                  
        
        pass

    

class FF_smoothed_delta_ring_themage(FisherForecast) :
    """
    FisherForecast object for a splined raster (i.e., themage) plus a
    Gaussian-convolved delta-ring.

    Parameter vector is the log of the intensity at the control points:
      p[0] ....... ln(I[0,0])
      p[1] ....... ln(I[1,0])
       ...
      p[N-1] ..... ln(I[N-1,0])
      p[N] ....... ln(I[0,1])
       ...
      p[N*N-1] ... ln(I[N-1,N-1])
      p[N*N+0] ... Total flux in Jy.
      p[N*N+1] ... Diameter in uas.
      p[N*N+2] ... Twice the standard deviation of the Gaussian smoothing kernel in uas.

    where each I[i,j] is measured in Jy/sr.

    Args:
      N (int): Raster dimension (only supports square raster).
      fov (float): Raster field of view in uas (only supports fixed rasters).
      stokes (str): Indicates if this is limited to Stokes I ('I') or include full polarization ('full'). Default: 'I'.

    Attributes:
      xcp (np.ndarray): x-positions of raster control points.
      ycp (np.ndarray): y-positions of raster control points.
      apx (float): Raster pixel size.
    """

    def __init__(self,N,fov,stokes='I') :
        super().__init__()
        self.npx = N
        self.size = self.npx**2 + 3
        fov = fov * np.pi/(3600e6*180) * (N-1)/N
        
        self.xcp,self.ycp = np.meshgrid(np.linspace(-0.5*fov,0.5*fov,self.npx),np.linspace(-0.5*fov,0.5*fov,self.npx))

        self.apx = (self.xcp[1,1]-self.xcp[0,0])*(self.ycp[1,1]-self.ycp[0,0])

        self.stokes = stokes

    def visibilities(self,obs,p,verbosity=0) :
        """
        Generates visibilities associated with splined raster + Gaussian-convolved delta-ring.

        Args:
          obs (ehtim.Obsdata): ehtim data object
          p (np.array): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          V (np.array): list of complex visibilities computed at observations.
        """

        # Takes:
        #  p[0] ... p[0,0]
        #  p[1] ... p[1,0]
        #  ...
        #  p[NxN-1] = p[N-1,N-1]
        #  p[NxN+0] ... ring flux in Jy
        #  p[NxN+1] ... ring diameter in uas
        #  p[NxN+2] ... ring width in uas

        if self.stokes != 'I':
            raise(Exception('Polarized visibilities for FF_smoothed_delta_ring_themage are not yet implemented!'))

        u = obs.data['u']
        v = obs.data['v']

        V = 0.0j*u

        # print("p:",p)
        
        # Add themage
        for i in range(self.npx) :
            for j in range(self.npx) :
                V = V + np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[i+self.npx*j]) * ty.vis.W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*ty.vis.W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx

        # print(u[:5],v[:5],V[:5])

                
        # Add ring
        d = p[-2]
        w = p[-1]
        uas2rad = np.pi/180./3600e6
        piuv = np.pi*np.sqrt(u**2+v**2)*uas2rad
        x = piuv*d
        y = piuv*w
        J0 = ss.jv(0,x)
        J1 = ss.jv(1,x)
        ey2 = np.exp(-0.5*y**2)
        V = V + p[-3] * J0 * ey2
        
        return V
        
    def visibility_gradients(self,obs,p,verbosity=0) :
        """
        Generates visibility gradients associated with splined-raster + Gaussian-convolved delta-ring.

        Args:
          obs (ehtim.Obsdata): ehtim data object
          p (np.array): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          gradV (np.array): list of complex visibilities computed at observations.
        """
        
        # Takes:
        #  p[0] ... p[0,0]
        #  p[1] ... p[1,0]
        #  ...
        #  p[NxN-1] = p[N-1,N-1]
        #  p[NxN+0] ... ring flux in Jy
        #  p[NxN+1] ... ring diameter in uas
        #  p[NxN+2] ... ring width in uas

        if self.stokes != 'I':
            raise(Exception('Polarized gradients for FF_smoothed_delta_ring_themage are not yet implemented!'))

        u = obs.data['u']
        v = obs.data['v']

        gradV = []

        # Add themage
        for j in range(self.npx) :
            for i in range(self.npx) :
                gradV.append( np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[i+self.npx*j]) * ty.vis.W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*ty.vis.W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx )
        
        # Add ring
        d = p[-2]
        w = p[-1]
        uas2rad = np.pi/180./3600e6
        piuv = np.pi*np.sqrt(u**2+v**2)*uas2rad
        x = piuv*d
        y = piuv*w
        J0 = ss.jv(0,x)
        J1 = ss.jv(1,x)
        ey2 = np.exp(-0.5*y**2)

        gradV.append( J0*ey2 ) # dV/dF
        gradV.append( -p[-3]*piuv*J1*ey2 ) # dV/dd
        gradV.append( -p[-3]*piuv**2*w*J0*ey2 ) # dV/dw

        gradV = np.array(gradV)
        
        # print("FOO:",gradV.shape)
        
        return gradV.T
        

    def parameter_labels(self,verbosity=0) :
        """
        Parameter labels to be used in plotting.

        Args:
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          plabels (list): list of strings with parameter labels.
        """        
        pll = []
        for j in range(self.npx) :
            for i in range(self.npx) :
                pll.append( r'$\delta I_{%i,%i}$'%(i,j) )

        pll.append(r'$\delta F~({\rm Jy})$')
        pll.append(r'$\delta d~(\mu{\rm as})$')
        pll.append(r'$\delta w~(\mu{\rm as})$')

        return pll

    
class FF_thick_mring(FisherForecast) :
    """
    FisherForecast object for an m-ring model (based on ehtim).
    Parameter vector is:
      p[0] ... DOM, PLEASE FILL THIS IN.

    Args:
      m (int): Stokes I azimuthal Fourier series order.
      mp (int): Linear polarization azimuthal Fourier series order. Only used if stokes=='full'. Default: None.
      mc (int): Circular polarization azimuthal Fourier series order. Only used if stokes=='full'. Default: None.
      stokes (str): Indicates if this is limited to Stokes I ('I') or include full polarization ('full'). Default: 'I'.

    Attributes:
      m (int): Stokes I azimuthal Fourier series order.
      mp (int): Linear polarization azimuthal Fourier series order. Only used if stokes=='full'.
      mc (int): Circular polarization azimuthal Fourier series order. Only used if stokes=='full'.

    """

    def __init__(self,m,mp=None,mc=None,stokes='I') :
        super().__init__()
        self.m = m
        self.mp = mp
        self.mc = mc
        self.stokes = stokes
        self.size = 3 + 2*m
        if self.stokes != 'I':
            if (self.mp > 0):
                self.size += 2 + 4*self.mp
            if (self.mc > 0):
                self.size += 2 + 4*self.mc

    def param_wrapper(self,p) :
        """
        Converts parameters from a flattened list to an ehtim-style dictionary.
        
        Args:
          p (float): Parameter list as used by FisherForecast.
        
        Returns:
          params (dict): Dictionary containing parameter values as used by ehtim.
        """
        
        # convert unwrapped parameter list to ehtim-readable version

        params = {}
        params['F0'] = p[0]
        params['d'] = p[1] * eh.RADPERUAS
        params['alpha'] = p[2] * eh.RADPERUAS
        params['x0'] = 0.0
        params['y0'] = 0.0
        beta_list = np.zeros(self.m, dtype="complex")
        ind_start = 3
        for i in range(self.m):
            beta_list[i] = p[ind_start+(2*i)]*np.exp((1j)*p[ind_start+1+(2*i)])
        params['beta_list'] = beta_list

        if self.stokes != 'I':
            if self.mp > 0:
                beta_list_pol = np.zeros(1 + 2*self.mp, dtype="complex")
                ind_start += 2*len(beta_list)
                for i in range(1+2*self.mp):
                    beta_list_pol[i] = p[ind_start+(2*i)]*np.exp((1j)*p[ind_start+1+(2*i)])
                params['beta_list_pol'] = beta_list_pol
            else:
                params['beta_list_pol'] = np.zeros(0, dtype="complex")

            if self.mc > 0:
                beta_list_cpol = np.zeros(1 + 2*self.mc, dtype="complex")
                ind_start += 2*len(beta_list_pol)
                for i in range(1+2*self.mc):
                    beta_list_cpol[i] = p[ind_start+(2*i)]*np.exp((1j)*p[ind_start+1+(2*i)])
                params['beta_list_cpol'] = beta_list_cpol
            else:
                params['beta_list_cpol'] = np.zeros(0, dtype="complex")

        return params

    def visibilities(self,obs,p,verbosity=0) :
        """
        Generates visibilities associated with m-ring model.

        Args:
          obs (ehtim.Obsdata): ehtim data object
          p (np.array): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          V (np.array): list of complex visibilities computed at observations.
        """

        # Takes:
        # p[0] ... total flux of the ring (Jy), which is also beta_0.
        # p[1] ... ring diameter (radians)
        # p[2] ... ring thickness (FWHM of Gaussian convolution) (radians)
        # p[...] ... beta list; list of complex Fourier coefficients, [beta_1, beta_2, ..., beta_m]
        #          Negative indices are determined by the condition beta_{-m} = beta_m*.
        #          Indices are all scaled by F0 = beta_0, so they are dimensionless.
        # p[...] ... beta list for linear polarization (if present)
        #          list of complex Fourier coefficients, [beta_{-mp}, beta_{-mp+1}, ..., beta_{mp-1}, beta_{mp}]
        # p[...] ... beta list for circular polarization (if present)
        #          list of complex Fourier coefficients, [beta_{-mc}, beta_{-mc+1}, ..., beta_{mc-1}, beta_{mc}]

        # set up parameter dictionary for ehtim
        params = self.param_wrapper(p)

        # read (u,v)-coordinates
        u = obs.data['u']
        v = obs.data['v']

        # compute model visibilities
        if self.stokes == 'I':
            vis = eh.model.sample_1model_uv(u,v,'thick_mring',params,pol='I')
            return vis
        else:
            vis_RR = eh.model.sample_1model_uv(u,v,'thick_mring',params,pol='RR')
            vis_LL = eh.model.sample_1model_uv(u,v,'thick_mring',params,pol='LL')
            vis_RL = eh.model.sample_1model_uv(u,v,'thick_mring',params,pol='RL')
            vis_LR = eh.model.sample_1model_uv(u,v,'thick_mring',params,pol='LR')
            return vis_RR, vis_LL, vis_RL, vis_LR
        
    def visibility_gradients(self,obs,p,verbosity=0) :
        """
        Generates visibility gradients associated with m-ring model.

        Args:
          obs (ehtim.Obsdata): ehtim data object
          p (np.array): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          gradV (np.array): list of complex visibilities computed at observations.
        """
        
        # Takes:
        # p[0] ... total flux of the ring (Jy), which is also beta_0.
        # p[1] ... ring diameter (radians)
        # p[2] ... ring thickness (FWHM of Gaussian convolution) (radians)
        # p[...] ... beta list; list of complex Fourier coefficients, [beta_1, beta_2, ..., beta_m]
        #          Negative indices are determined by the condition beta_{-m} = beta_m*.
        #          Indices are all scaled by F0 = beta_0, so they are dimensionless.
        # p[...] ... beta list for linear polarization (if present)
        #          list of complex Fourier coefficients, [beta_{-mp}, beta_{-mp+1}, ..., beta_{mp-1}, beta_{mp}]
        # p[...] ... beta list for circular polarization (if present)
        #          list of complex Fourier coefficients, [beta_{-mc}, beta_{-mc+1}, ..., beta_{mc-1}, beta_{mc}]

        # set up parameter dictionary for ehtim
        params = self.param_wrapper(p)

        # read (u,v)-coordinates
        u = obs.data['u']
        v = obs.data['v']

        # compute model gradients
        if self.stokes == 'I':
            grad = eh.model.sample_1model_grad_uv(u,v,'thick_mring',params,pol='I',fit_pol=False,fit_cpol=False)
            
            # unit conversion
            grad[1,:] *= eh.RADPERUAS
            grad[2,:] *= eh.RADPERUAS

            # remove (x0,y0) parameters
            grad = np.concatenate((grad[:3, :], grad[5:, :]))
            
            return grad.T

        else:
            grad_RR = eh.model.sample_1model_grad_uv(u,v,'thick_mring',params,pol='RR',fit_pol=True,fit_cpol=True)
            grad_LL = eh.model.sample_1model_grad_uv(u,v,'thick_mring',params,pol='LL',fit_pol=True,fit_cpol=True)
            grad_RL = eh.model.sample_1model_grad_uv(u,v,'thick_mring',params,pol='RL',fit_pol=True,fit_cpol=True)
            grad_LR = eh.model.sample_1model_grad_uv(u,v,'thick_mring',params,pol='LR',fit_pol=True,fit_cpol=True)
            
            # unit conversion
            grad_RR[1,:] *= eh.RADPERUAS
            grad_RR[2,:] *= eh.RADPERUAS

            grad_LL[1,:] *= eh.RADPERUAS
            grad_LL[2,:] *= eh.RADPERUAS

            grad_RL[1,:] *= eh.RADPERUAS
            grad_RL[2,:] *= eh.RADPERUAS

            grad_LR[1,:] *= eh.RADPERUAS
            grad_LR[2,:] *= eh.RADPERUAS

            # remove (x0,y0) parameters
            grad_RR = np.concatenate((grad_RR[:3, :], grad_RR[5:, :]))
            grad_LL = np.concatenate((grad_LL[:3, :], grad_LL[5:, :]))
            grad_RL = np.concatenate((grad_RL[:3, :], grad_RL[5:, :]))
            grad_LR = np.concatenate((grad_LR[:3, :], grad_LR[5:, :]))

            return grad_RR.T, grad_LL.T, grad_RL.T, grad_LR.T

    def parameter_labels(self,verbosity=0) :
        """
        Parameter labels to be used in plotting.

        Args:
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          plabels (list): list of strings with parameter labels.
        """        
        labels = list()
        labels.append(r'$\delta F~({\rm Jy})$')
        labels.append(r'$\delta d~(\mu{\rm as})$')
        labels.append(r'$\delta \alpha~(\mu{\rm as})$')
        # labels.append(r'$\delta x_0~(\mu{\rm as})$')
        # labels.append(r'$\delta y_0~(\mu{\rm as})$')
        
        for i in range(self.m):
            labels.append(r'$\delta |\beta_{m=' + str(i+1) + r'}|$')
            labels.append(r'$\delta {\rm arg}\beta_{m=' + str(i+1) + r'}$')

        if self.stokes != 'I':
            
            if self.mp > 0:
                for i in range(self.mp):
                    labels.append(r'$\delta |\beta_{mp=-' + str(self.mp - i) + r'}|$')
                    labels.append(r'$\delta {\rm arg}\beta_{mp=-' + str(self.mp - i) + r'}$')
                labels.append(r'$\delta |\beta_{mp=0}|$')
                labels.append(r'$\delta {\rm arg}\beta_{mp=0}$')
                for i in range(self.mp):
                    labels.append(r'$\delta |\beta_{mp=' + str(i + 1) + r'}|$')
                    labels.append(r'$\delta {\rm arg}\beta_{mp=' + str(i + 1) + r'}$')
            if self.mc > 0:
                for i in range(self.mc):
                    labels.append(r'$\delta |\beta_{mc=-' + str(self.mc - i) + r'}|$')
                    labels.append(r'$\delta {\rm arg}\beta_{mc=-' + str(self.mc - i) + r'}$')
                labels.append(r'$\delta |\beta_{mc=0}|$')
                labels.append(r'$\delta {\rm arg}\beta_{mc=0}$')
                for i in range(self.mc):
                    labels.append(r'$\delta |\beta_{mc=' + str(i + 1) + r'}|$')
                    labels.append(r'$\delta {\rm arg}\beta_{mc=' + str(i + 1) + r'}$')

        return labels
    
class FF_sum(FisherForecast) :
    """
    FisherForecast object constructed from the sum of multiple 
    FisherForecast objects. For example, a binary might be generated from
    the sum of two Gaussians.  

    The parameter vector is constructed from the concatenation of the parameter
    vectors from each individual object, with all objects after the first
    gaining a pair of offset parameters.  That is:
      p[0] ............ Obj1 p[0]
      p[1] ............ Obj1 p[1]
        ...
      p[n1] ........... Obj2 p[0]
      p[n1+1] ......... Obj2 p[0]
        ...
      p[n1+n2] ........ Obj2 dx
      p[n1+n2+1] ...... Obj2 dy
      p[n1+n2+2] ...... Obj3 p[0]
        ...
      p[n1+n2+n3+2] ... Obj3 dx
      p[n1+n2+n3+3] ... Obj3 dy
        ...
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
          p (np.array): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          V (np.array): list of complex visibilities computed at observations.
        """
        
        k = 0
        uas2rad = np.pi/180./3600e6

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
                    shift_factor = np.exp( -2.0j*np.pi*(obs.data['u']*dx+obs.data['v']*dy)*uas2rad )
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
                    shift_factor = np.exp( -2.0j*np.pi*(obs.data['u']*dx+obs.data['v']*dy)*uas2rad )
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
          p (np.array): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          gradV (np.array): list of complex visibilities computed at observations.
        """

        u = obs.data['u']
        v = obs.data['v']
        uas2rad = np.pi/180./3600e6

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
                    shift_factor = np.exp( -2.0j*np.pi*(u*dx+v*dy)*uas2rad )
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
                    shift_factor = np.exp( -2.0j*np.pi*(u*dx+v*dy)*uas2rad )
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
          plabels (list): list of strings with parameter labels.
        """        
        pll = []
        for i,ff in enumerate(self.ff_list) :
            for lbl in ff.parameter_labels() :
                pll.append(lbl)
            if (i>0) :
                pll.append(r'$\delta\Delta x~(\mu{\rm as})$')
                pll.append(r'$\delta\Delta y~(\mu{\rm as})$')
        
        return pll


            

    




if __name__ == "__main__" :

    # Example
    
    # Make Fisher forecast object
    ff1 = FF_splined_raster(5,60)
    ff2 = FF_smoothed_delta_ring()
    ff = FF_sum([ff1,ff2])

    # Read in some data
    obs_ngeht = eh.obsdata.load_uvfits('../data/M87_230GHz_40uas.uvfits')
    obs_ngeht.add_scans()
    obs_ngeht = obs_ngeht.avg_coherent(0,scan_avg=True)
    obs_ngeht = obs_ngeht.add_fractional_noise(0.01)
    #
    obs = obs_ngeht.flag_sites(['ALMA','APEX','JCMT','SMA','SMT','LMT','SPT','PV','PDB','KP'])


    obslist = [obs,obs_ngeht]
    labels = ['ngEHT w/o EHT','Full ngEHT']
    
    
    # Choose a default image
    p = np.zeros(ff.size)
    rad2uas = 3600e6*180/np.pi
    for j in range(ff1.npx) :
        for i in range(ff1.npx) :
            p[i+ff1.npx*j] =  -((ff1.xcp[i,j]-5.0/rad2uas)**2+ff1.ycp[i,j]**2)*rad2uas**2/(2.0*25.0**2) + np.log( 1.0/( 2.0*np.pi * (25.0/rad2uas)**2 ) )
    p[-5] = 0.2
    p[-4] = 40
    p[-3] = 2
    p[-2] = 0
    p[-1] = 0
    p = np.array(p)

    # 1D Diameter
    plot_1d_forecast(ff,p,ff.size-4,obslist,labels=labels)
    plt.savefig('ring_forecast_1d.png',dpi=300)

    # 2D diameter vs width
    plot_2d_forecast(ff,p,ff.size-4,ff.size-3,obslist,labels=labels)
    plt.savefig('ring_forecast_2d.png',dpi=300)

    # Triangle
    plist = np.arange(len(p)-5,len(p))
    plot_triangle_forecast(ff,p,plist,obslist,labels=labels)
    plt.savefig('ring_forecast_tri.png',dpi=300)
    
    # plt.show()

