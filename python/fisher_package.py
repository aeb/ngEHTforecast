import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
import scipy.interpolate as si
import ehtim as eh
import copy
import hashlib
import themispy as ty
import ehtim.parloop as ploop

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


class FisherForecast :
    """
    Class that collects and contains information for making Fisher-matrix type
    forecasts for observations of various types.  Forms a base class for fully
    analytical versions that may be application specific.
    """

    def __init__(self) :
        self.size = 0
        self.argument_hash = None
        self.prior_sigma_list = []
        self.stokes = 'I'
        self.covar = None
        
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
        User-defined function in child classes that generates visibilit gradients
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
        Generate and return an image for the given visibility model evaluated
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
        
    def plot_image(self,p,limits=None,shape=None,verbosity=0,**kwargs) :

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

    
    def uniparameter_uncertainties(self,obs,p,verbosity=0,**kwargs) :
        C = self.fisher_covar(obs,p,verbosity=verbosity,**kwargs)
        Sig_uni = np.zeros(self.size)
        ilist = np.arange(self.size)
        for i in ilist :
            N = C[i][i]
            Sig_uni[i] = np.sqrt(2.0/N)
        return Sig_uni

    
    def marginalized_uncertainties(self,obs,p,verbosity=0,**kwargs) :
        C = self.fisher_covar(obs,p,verbosity=verbosity,**kwargs)
        Sig_marg = np.zeros(self.size)
        M = np.zeros((self.size-1,self.size-1))
        v = np.zeros(self.size-1)
        ilist = np.arange(self.size)
        for i in ilist :
            ini = ilist[ilist!=i]
            for j2,j in enumerate(ini) :
                for k2,k in enumerate(ini) :
                    M[j2,k2] = C[j][k]
                v[j2] = C[i][j]
            N = C[i][i]
            Minv = np.linalg.inv(M)
            mN = N - np.matmul(v,np.matmul(Minv,v))
            Sig_marg[i] = np.sqrt(2.0/mN)
        return Sig_marg

    
    def uncertainties(self,obs,p,verbosity=0,**kwargs) :
        C = self.fisher_covar(obs,p,verbosity=verbosity,**kwargs)
        if (verbosity>1) :
            print("Fisher Matrix:")
            _print_matrix(C)
        Sig_uni = np.zeros(self.size)
        Sig_marg = np.zeros(self.size)
        M = np.zeros((self.size-1,self.size-1))
        v = np.zeros(self.size-1)
        ilist = np.arange(self.size)
        for i in ilist :
            ini = ilist[ilist!=i]
            for j2,j in enumerate(ini) :
                for k2,k in enumerate(ini) :
                    M[j2,k2] = C[j][k]
                v[j2] = C[i][j]
            N = C[i][i]
            Minv = np.linalg.inv(M)
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

    
    def joint_biparameter_chisq(self,obs,p,i1,i2,verbosity=0,**kwargs) :
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

        if (verbosity>1) :
            print("Fisher Matrix:")
            _print_matrix(C)
            print("Submatrix (%i,%i):"%(i1,i2))
            _print_matrix(M)
            print("Subvectors v1:")
            _print_vector(v1)
            print("Subvectors v2:")
            _print_vector(v2)

        
        Minv = np.linalg.inv(M)
        mN1 = N1 - np.matmul(v1.T,np.matmul(Minv,v1))
        mN2 = N2 - np.matmul(v2.T,np.matmul(Minv,v2))
        mC12 = C12 - 0.5*(np.matmul(v1.T,np.matmul(Minv,v2)) + np.matmul(v2.T,np.matmul(Minv,v1)))

        # Find range with eigenvectors/values
        l1 = 0.5*( (mN1+mN2) + np.sqrt( (mN1-mN2)**2 + 4.0*mC12**2 ) )
        l2 = 0.5*( (mN1+mN2) - np.sqrt( (mN1-mN2)**2 + 4.0*mC12**2 ) )

        e1 = np.array( [ mC12, l1-mN1 ] )
        e2 = np.array( [ e1[1], -e1[0] ] )
        e1 = e1/ np.sqrt( e1[0]**2+e1[1]**2 )
        e2 = e2/ np.sqrt( e2[0]**2+e2[1]**2 )
        
        if (l1<=0 or l2<=0) :
            print("Something's wrong! Variances are nonpositive!")
            print(i1,i2)
            print(l1,l2)
            print(N1,N2,C12)
            print(mN1,mN2,mC12)
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
        mcsq = 0.5*(mN1*p1**2 + 2*mC12*p1*p2 + mN2*p2**2)

        return p1,p2,csq,mcsq


class FF_complex_gains_single_epoch(FisherForecast) :
    """
    
    FisherForecast with complex gain reconstruction assuming a single epoch.

    Args:
      ff (FisherForecast): A FisherForecast object to which we wish to add gains.

    Attributes:
      ff (FisherForecast): The FisherForecast object before gain reconstruction.
    """

    def __init__(self,ff) :
        super().__init__()
        self.ff = ff
        self.scans = False
        self.gain_epochs = None
        self.plbls = self.ff.parameter_labels()
        self.prior_sigma_list = self.ff.prior_sigma_list
        self.gain_amplitude_priors = {}
        self.gain_phase_priors = {}
        
    def visibilities(self,obs,p,verbosity=0,**kwargs) :
        return self.ff.visibilities(obs,p,verbosity=verbosity,**kwargs)

    def visibility_gradients(self,obs,p,verbosity=0,**kwargs) :
        V_pg = self.ff.visibilities(obs,p,verbosity=verbosity,**kwargs)
        gradV_pg = self.ff.visibility_gradients(obs,p,verbosity=verbosity,**kwargs)
        
        # Start with model parameters
        gradV = list(gradV_pg.T)

        # Generate the gain epochs and update the size
        station_list = obs.tarr['site']
        nt = len(station_list)
        self.size = self.ff.size
        self.plbls = self.ff.parameter_labels()
        self.prior_sigma_list = copy.copy(self.ff.prior_sigma_list)
        
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

                if (verbosity>1) :
                    print("SEG %s amp   priors:"%(station),self.gain_amplitude_priors)
                    print("SEG %s phase priors:"%(station),self.gain_phase_priors)
                    print("SEG %s priors:"%(station),self.prior_sigma_list)

        gradV = np.array(gradV).T

        if (verbosity>0) :
            for j in range(min(10,gradV.shape[0])) :
                line = "SEG gradV: %10.3g %10.3g"%(obs.data['u'][j],obs.data['v'][j])
                for k in range(gradV.shape[1]) :
                    line = line + " %8.3g + %8.3gi"%(gradV[j,k].real,gradV[j,k].imag)
                print(line)
                
        return gradV

    def parameter_labels(self) :
        return self.plbls

    def set_gain_amplitude_prior(self,sigma,station=None) :
        sigma = min(sigma,10.0)
        if (station is None) :
            self.gain_amplitude_priors = {'All':sigma}
        else :
            self.gain_amplitude_priors[station] = sigma
            
        self.argument_hash = None

    def set_gain_phase_prior(self,sigma,station=None) :
        sigma = min(sigma,100.0)
        if (station is None) :
            self.gain_phase_priors = {'All':sigma}
        else :
            self.gain_phase_priors[station] = sigma
            
        self.argument_hash = None
        

class FF_complex_gains(FisherForecast) :
    """
    
    FisherForecast with complex gain reconstruction.

    Args:
      ff (FisherForecast): A FisherForecast object to which we wish to add gains.

    Attributes:
      ff (FisherForecast): The FisherForecast object before gain reconstruction.
    """

    def __init__(self,ff) :
        super().__init__()
        self.ff = ff
        self.scans = False
        self.gain_epochs = None
        self.plbls = self.ff.parameter_labels()
        self.prior_sigma_list = self.ff.prior_sigma_list
        self.gain_amplitude_priors = {}
        self.ff_cgse = FF_complex_gains_single_epoch(ff)
        self.size = self.ff.size
        self.covar = np.zeros((self.ff.size,self.ff.size))
        
    def set_gain_epochs(self,scans=False,gain_epochs=None) :
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
        return self.ff.visibilities(obs,p,verbosity=verbosity,**kwargs)

    def visibility_gradients(self,obs,p,verbosity=0,**kwargs) :
        return self.ff.visibilities(obs,p,verbosity=verbosity,**kwargs)

    def fisher_covar(self,obs,p,verbosity=0,**kwargs) :
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
                
                for ige,ge in enumerate(self.gain_epochs) :
                    with ploop.HiddenPrints():
                        obs_ge = obs.flag_UT_range(UT_start_hour=ge[0],UT_stop_hour=ge[1],output='flagged')

                    gradV_wgs = self.ff_cgse.visibility_gradients(obs_ge,p,**kwargs)
                    covar_wgs = np.zeros((self.ff_cgse.size,self.ff_cgse.size))
                    
                    for i in range(self.ff_cgse.size) :
                        for j in range(self.ff_cgse.size) :
                            covar_wgs[i][j] = np.sum( np.conj(gradV_wgs[:,i])*gradV_wgs[:,j]/obs_ge.data['sigma']**2 + gradV_wgs[:,i]*np.conj(gradV_wgs[:,j])/obs_ge.data['sigma']**2)

                    for i in np.arange(self.ff.size,self.ff_cgse.size) :
                        if (not self.ff_cgse.prior_sigma_list[i] is None) :
                            covar_wgs[i][i] += 1.0/self.ff_cgse.prior_sigma_list[i]**2
                            
                    n = covar_wgs[:self.ff.size,:self.ff.size]
                    r = covar_wgs[self.ff.size:,:self.ff.size]
                    m = covar_wgs[self.ff.size:,self.ff.size:]

                    # print("gain epoch %g of %g ----------------------"%(ige,self.gain_epochs.shape[0]))
                    # print("obs stations:",np.unique(list(obs_ge.data['t1'])+list(obs_ge.data['t2'])))
                    # print("obs times:",np.unique(obs_ge.data['time']))
                    # print("obs data:\n",obs_ge.data)
                    # print("gradV gains:\n",gradV_wgs[self.size:])
                    # print("covar:\n",covar_wgs)
                    # print("n:\n",n)
                    # print("r:\n",r)
                    # print("m:\n",m)
                    # print("minv:\n",np.linalg.inv(m))

                    #quit()
                    
                    self.covar = self.covar + n - np.matmul(r.T,np.matmul(np.linalg.inv(m),r))

            else:
                raise(RuntimeError("Polarized models with gains remains to be implemented."))
                # obs = obs.switch_polrep('circ')
                # grad_RR, grad_LL, grad_RL, grad_LR = self.visibility_gradients(obs,p,**kwargs)
                # covar = np.zeros((self.size,self.size))
                # for i in range(self.size) :
                #     for j in range(self.size) :
                #         covar[i][j] = 0.5*np.sum( np.conj(grad_RR[:,i])*grad_RR[:,j]/obs.data['rrsigma']**2 + grad_RR[:,i]*np.conj(grad_RR[:,j])/obs.data['rrsigma']**2)
                #         covar[i][j] += 0.5*np.sum( np.conj(grad_LL[:,i])*grad_LL[:,j]/obs.data['llsigma']**2 + grad_LL[:,i]*np.conj(grad_LL[:,j])/obs.data['llsigma']**2)
                #         covar[i][j] += 0.5*np.sum( np.conj(grad_RL[:,i])*grad_RL[:,j]/obs.data['rlsigma']**2 + grad_RL[:,i]*np.conj(grad_RL[:,j])/obs.data['rlsigma']**2)
                #         covar[i][j] += 0.5*np.sum( np.conj(grad_LR[:,i])*grad_LR[:,j]/obs.data['lrsigma']**2 + grad_LR[:,i]*np.conj(grad_LR[:,j])/obs.data['lrsigma']**2)

            if (len(self.prior_sigma_list)>0) :
                for i in range(self.size) :
                    if (not self.prior_sigma_list[i] is None) :
                        self.covar[i][i] += 2.0/self.prior_sigma_list[i]**2 # Why factor of 2?
                    
        return self.covar
    
    def parameter_labels(self) :
        return self.ff.parameter_labels()

    def set_gain_amplitude_prior(self,sigma,station=None) :
        sigma = min(sigma,10.0)
        if (station is None) :
            self.gain_amplitude_priors = {'All':sigma}
        else :
            self.gain_amplitude_priors[station] = sigma
            
        self.argument_hash = None

    def set_gain_phase_prior(self,sigma,station=None) :
        sigma = min(sigma,100.0)
        if (station is None) :
            self.gain_phase_priors = {'All':sigma}
        else :
            self.gain_phase_priors[station] = sigma
            
        self.argument_hash = None
            
            
    

    
class FF_model_image(FisherForecast) :
    """
    
    FisherForecast from ThemisyPy model_image objects.  Uses FFTs and centered 
    finite-difference to compute the visibilities and gradients.  May not always
    produce sensible behaviors.

    Args:
      img (model_image): A ThemisPy model_image object.

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

    def parameter_labels(self) :
        return self.img.parameter_name_list()
    

class FF_smoothed_delta_ring(FisherForecast) :

    def __init__(self,stokes='I') :
        super().__init__()
        self.size = 3
        self.stokes = stokes
        
    def visibilities(self,obs,p,verbosity=0) :
        # Takes:
        #  p[0] ... flux in Jy
        #  p[1] ... diameter in uas
        #  p[2] ... width in uas

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
        # Takes:
        #  p[0] ... flux in Jy
        #  p[1] ... diameter in uas
        #  p[2] ... width in uas

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
        

    def parameter_labels(self) :
        return [r'$\delta F~({\rm Jy})$',r'$\delta d~(\mu{\rm as})$',r'$\delta w~(\mu{\rm as})$']
    

class FF_symmetric_gaussian(FisherForecast) :

    def __init__(self,stokes='I') :
        super().__init__()
        self.size = 2
        self.stokes = stokes
        
    def visibilities(self,obs,p,verbosity=0) :
        # Takes:
        #  p[0] ... flux in Jy
        #  p[1] ... diameter in uas

        u = obs.data['u']
        v = obs.data['v']

        d = p[1]
        uas2rad = np.pi/180./3600e6
        piuv = 2.0*np.pi*np.sqrt(u**2+v**2)*uas2rad / np.sqrt(8.0*np.log(2.0))
        y = piuv * d
        ey2 = np.exp(-0.5*y**2)
        return p[0]*ey2
        
    def visibility_gradients(self,obs,p,verbosity=0) :
        # Takes:
        #  p[0] ... flux in Jy
        #  p[1] ... diameter in uas

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
        

    def parameter_labels(self) :
        return [r'$\delta F~({\rm Jy})$',r'$\delta FWHM~(\mu{\rm as})$']


class FF_asymmetric_gaussian(FisherForecast) :

    def __init__(self,stokes='I') :
        super().__init__()
        self.size = 4
        self.stokes = stokes
        
    def visibilities(self,obs,p,verbosity=0) :
        # Takes:
        #  p[0] ... flux in Jy
        #  p[1] ... mean diameter in uas
        #  p[2] ... asymmetry parameter 
        #  p[3] ... position angle in radians

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
        # Takes:
        #  p[0] ... flux in Jy
        #  p[1] ... mean diameter in uas
        #  p[2] ... asymmetry parameter 
        #  p[3] ... position angle in radians

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

    def parameter_labels(self) :
        return [r'$\delta F~({\rm Jy})$',r'$\delta FWHM~(\mu{\rm as})$',r'$\delta A$',r'$\delta {\rm PA}~({\rm rad})$']

    

class FF_splined_raster(FisherForecast) :

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
        # Takes:
        #  p[0] ... p[0,0]
        #  p[1] ... p[1,0]
        #  ...
        #  p[NxN-1] = p[N-1,N-1]

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
        # Takes:
        #  p[0] ... p[0,0]
        #  p[1] ... p[1,0]
        #  ...
        #  p[NxN-1] = p[N-1,N-1]
        
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

    def parameter_labels(self) :

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


class FF_smoothed_delta_ring_themage(FisherForecast) :

    def __init__(self,N,fov,stokes='I') :
        super().__init__()
        self.npx = N
        self.size = self.npx**2 + 3
        fov = fov * np.pi/(3600e6*180) * (N-1)/N
        
        self.xcp,self.ycp = np.meshgrid(np.linspace(-0.5*fov,0.5*fov,self.npx),np.linspace(-0.5*fov,0.5*fov,self.npx))

        self.apx = (self.xcp[1,1]-self.xcp[0,0])*(self.ycp[1,1]-self.ycp[0,0])

        self.stokes = stokes

    def visibilities(self,obs,p,verbosity=0) :
        # Takes:
        #  p[0] ... p[0,0]
        #  p[1] ... p[1,0]
        #  ...
        #  p[NxN-1] = p[N-1,N-1]
        #  p[NxN+0] ... ring flux in Jy
        #  p[NxN+1] ... ring diameter in uas
        #  p[NxN+2] ... ring width in uas

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
        # Takes:
        #  p[0] ... p[0,0]
        #  p[1] ... p[1,0]
        #  ...
        #  p[NxN-1] = p[N-1,N-1]
        #  p[NxN+0] ... ring flux in Jy
        #  p[NxN+1] ... ring diameter in uas
        #  p[NxN+2] ... ring width in uas

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
        

    def parameter_labels(self) :

        pll = []
        for j in range(self.npx) :
            for i in range(self.npx) :
                pll.append( r'$\delta I_{%i,%i}$'%(i,j) )

        pll.append(r'$\delta F~({\rm Jy})$')
        pll.append(r'$\delta d~(\mu{\rm as})$')
        pll.append(r'$\delta w~(\mu{\rm as})$')

        return pll

class FF_thick_mring(FisherForecast) :

    def __init__(self,m,mp,mc,stokes='I') :
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

    def param_wrapper(self,p):
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

    def visibilities(self,obs,p,verbosity=0):
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
        
    def visibility_gradients(self,obs,p,verbosity=0):
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

    def parameter_labels(self):
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
                
    def parameter_labels(self) :

        pll = []
        for i,ff in enumerate(self.ff_list) :
            for lbl in ff.parameter_labels() :
                pll.append(lbl)
            if (i>0) :
                pll.append(r'$\delta\Delta x~(\mu{\rm as})$')
                pll.append(r'$\delta\Delta y~(\mu{\rm as})$')
        
        return pll


            

    


_ff_color_list = ['r','b','g','orange','purple']
_ff_cmap_list = ['Reds_r','Blues_r','Greens_r','Oranges_r','Purples_r']
_ff_color_size = 5
    
def plot_1d_forecast(ff,p,i1,obslist,labels=None) :

    if (labels is None) :
        labels = len(obslist)*[None]
    
    plt.figure(figsize=(5,4))
    plt.axes([0.15,0.15,0.8,0.8])
    
    n = 256
    
    sigdict = {}
    for k,obs in enumerate(obslist) :
        _,Sigm = ff.uncertainties(obs,p)

        x = np.linspace(-5*Sigm[i1],5*Sigm[i1],n)
        y = np.exp(-x**2/(2.0*Sigm[i1]**2)) / np.sqrt(2.0*np.pi*Sigm[i1]**2)

        plt.fill_between(x,y,y2=0,alpha=0.25,color=_ff_color_list[k%_ff_color_size])
        plt.plot(x,y,'-',color=_ff_color_list[k%_ff_color_size],label=labels[k])

    plt.xlabel(ff.parameter_labels()[i1])
    plt.yticks([])
    plt.ylim(bottom=0)
    plt.legend()

    return plt.gcf(),plt.gca()


def plot_2d_forecast(ff,p,i1,i2,obslist,labels=None) :
    
    if (labels is None) :
        labels = len(obslist)*[None]

    plt.figure(figsize=(5,5))
    plt.axes([0.2,0.2,0.75,0.75])

    for k,obs in enumerate(obslist) :
        d,w,csq,mcsq = ff.joint_biparameter_chisq(obs,p,i1,i2)
        plt.contourf(d,w,np.sqrt(mcsq),cmap=_ff_cmap_list[k%_ff_color_size],alpha=0.75,levels=[0,1,2,3])
        plt.contour(d,w,np.sqrt(mcsq),colors=_ff_color_list[k%_ff_color_size],alpha=1,levels=[0,1,2,3])
        plt.plot([],[],'-',color=_ff_color_list[k%_ff_color_size],label=labels[k])
    
    plt.xlabel(ff.parameter_labels()[i1])
    plt.ylabel(ff.parameter_labels()[i2])
    plt.grid(True,alpha=0.25)
    plt.legend()

    return plt.gcf(),plt.gca()


def plot_triangle_forecast(ff,p,ilist,obslist,labels=None) :

    wdx = 2
    wdy = 2
    wgx = 0.25
    wgy = 0.25
    wmx = 1
    wmy = 0.75

    ni = len(ilist)

    fx = wmx + ni*(wdx+wgx)
    fy = wmy + ni*(wdy+wgy)

    
    plt.figure(figsize=(fx,fy))
    axs = {}
    for i in range(ni) :
        for j in range(ni-i) :
            ax0 = wmx + i*(wdx+wgx)
            ay0 = wmy + j*(wdy+wgy)
            axs[i,j] = plt.axes([ax0/fx,ay0/fy,wdx/fx,wdy/fy])

    xlim_dict = {}
    for k,obs in enumerate(obslist) :
        _,Sigm = ff.uncertainties(obs,p)

        for j in range(ni) :
            jj = ilist[j]
            xtmp = np.linspace(-3.5*Sigm[jj],3.5*Sigm[jj],256)
            ytmp = np.exp(-xtmp**2/(2.0*Sigm[jj]**2))/np.sqrt(2.0*np.pi*Sigm[jj]**2)
            axs[j,ni-j-1].fill_between(xtmp,ytmp,y2=0,color=_ff_color_list[k%_ff_color_size],alpha=0.25)
            axs[j,ni-j-1].plot(xtmp,ytmp,color=_ff_color_list[k%_ff_color_size])

            if (k==0) :
                xlim_dict[j] = (-3.5*Sigm[jj],3.5*Sigm[jj])
            else :
                if (3.5*Sigm[jj]>xlim_dict[j][1]) :
                    xlim_dict[j] = (-3.5*Sigm[jj],3.5*Sigm[jj])                    
                
        for i in range(ni) :
            for j in range(ni-i-1) :
                ii = ilist[i]
                jj = ilist[ni-j-1]

                plt.sca(axs[i,j])
        
                p1,p2,csq,mcsq = ff.joint_biparameter_chisq(obs,p,ii,jj)
                plt.contourf(p1,p2,np.sqrt(mcsq),cmap=_ff_cmap_list[k%_ff_color_size],alpha=0.75,levels=[0,1,2,3])
                plt.contour(p1,p2,np.sqrt(mcsq),colors=_ff_color_list[k%_ff_color_size],alpha=1,levels=[0,1,2,3])

                plt.xlim(xlim_dict[i])
                plt.ylim(xlim_dict[ni-j-1])

                plt.grid(True,alpha=0.25)


    for j in range(ni) :
        axs[j,ni-j-1].set_xlim(xlim_dict[j])
        axs[j,ni-j-1].set_ylim(bottom=0)
        axs[j,ni-j-1].set_yticks([])

    for i in range(ni) :
        for j in range(1,ni-i) :
            axs[i,j].set_xticklabels([])
            
    for i in range(1,ni) :
        for j in range(ni-i-1) :
            axs[i,j].set_yticklabels([])

    for j in range(ni-1) :
        axs[0,j].set_ylabel(ff.parameter_labels()[ilist[ni-j-1]])
        
    for i in range(ni) :
        axs[i,0].set_xlabel(ff.parameter_labels()[ilist[i]])


    # Make axis for labels
    if (not labels is None) :
        plt.axes([0.95,0.95,0.01,0.01])
        for k in range(len(obslist)) :
            plt.plot([],[],_ff_color_list[k%_ff_color_size],label=labels[k])
        plt.gca().spines.right.set_visible(False)
        plt.gca().spines.left.set_visible(False)
        plt.gca().spines.top.set_visible(False)
        plt.gca().spines.bottom.set_visible(False)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.legend(loc='upper right',bbox_to_anchor=(0,0))

        
    return plt.gcf(),axs


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

