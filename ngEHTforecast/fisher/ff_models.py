import numpy as np
import scipy.special as ss
import scipy.interpolate as si
import ehtim as eh

import ngEHTforecast.fisher.fisher_forecast as ff

# Some constants
uas2rad = np.pi/180./3600e6            
rad2uas = 1.0/(uas2rad)
sig2fwhm = np.sqrt(8.0*np.log(2.0))
fwhm2sig = 1.0/sig2fwhm

class FF_model_image(ff.FisherForecast) :
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
          p (numpy.ndarray): list of parameters for the model image used to create object.
          limits (list): May be single number, list with four elements specifying in :math:`\\mu as` the symmetric limits or explicit limits in [xmin,xmax,ymin,ymax] order.  Default: 100.
          shape (list): May be single number or list with two elements indicating the number of pixels in the two directions.  Default: 256.
          padding (int): Factor by which to pad image with zeros prior to FFT.  Default: 4.
          verbosity (int): Verbosity level. 0 prints nothing. 1 prints various elements of the plotting process. Passed to :func:`model_image.generate_intensity_map`. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities computed at observations.
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
          p (numpy.ndarray): list of parameters for the model image used to create object.
          h (float): fractional step for finite-difference gradient computation.  Default: 0.01.
          limits (list): May be single number, list with four elements specifying in :math:`\\mu as` the symmetric limits or explicit limits in [xmin,xmax,ymin,ymax] order.  Default: 100.
          shape (list): May be single number or list with two elements indicating the number of pixels in the two directions.  Default: 256.
          padding (int): Factor by which to pad image with zeros prior to FFT.  Default: 4.
          verbosity (int): Verbosity level. 0 prints nothing. 1 prints various elements of the plotting process. Passed to :func:`model_image.generate_intensity_map`. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities gradients computed at observation.
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
          (list): List of strings with parameter labels.
        """        
        return self.img.parameter_name_list()
    

class FF_smoothed_delta_ring(ff.FisherForecast) :
    """
    FisherForecast object for a delta-ring convolved with a circular Gaussian.
    Parameter vector is:

    * p[0] ... Total flux in Jy.
    * p[1] ... Diameter in uas.
    * p[2] ... Twice the standard deviation of the Gaussian smoothing kernel in uas.

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
          p (numpy.ndarray): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities computed at observations.
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
          p (numpy.ndarray): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities computed at observations.
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
          (list): List of strings with parameter labels.
        """        
        return [r'$\delta F~({\rm Jy})$',r'$\delta d~(\mu{\rm as})$',r'$\delta w~(\mu{\rm as})$']
    

class FF_symmetric_gaussian(ff.FisherForecast) :
    """
    FisherForecast object for a circular Gaussian.
    Parameter vector is:

    * p[0] ... Total flux in Jy.
    * p[1] ... FWHM in uas.

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
          p (numpy.ndarray): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities computed at observations.
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
          p (numpy.ndarray): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities computed at observations.
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
          (list): List of strings with parameter labels.
        """        
        return [r'$\delta F~({\rm Jy})$',r'$\delta FWHM~(\mu{\rm as})$']


class FF_asymmetric_gaussian(ff.FisherForecast) :
    """
    FisherForecast object for a noncircular Gaussian.  
    Parameter vector is:

    * p[0] ... Total flux in Jy.
    * p[1] ... Symmetrized mean of the FHWM in the major and minor axes in uas: :math:`{\\rm FWHM}^{-2} = {\\rm FMHM}_{\\rm min}^{-2}+{\\rm FWHM}_{\\rm max}^{-2}`
    * p[2] ... Asymmetry parameter, :math:`A`, expected to be in the range [0,1).
    * p[3] ... Position angle in radians of the major axis E of N.

    In terms of these, :math:`{\\rm FWHM}_{\\rm maj}={\\rm FWHM}/\\sqrt{1-A}` and :math:`{\\rm FWHM}_{\\rm min}={\\rm FWHM}/\\sqrt{1-A}`.

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
          p (numpy.ndarray): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities computed at observations.
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

        cpa = np.sin(p[3])
        spa = np.cos(p[3])
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
          p (numpy.ndarray): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities computed at observations.
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

        # uas2rad_fwhm2sig = np.pi/180./3600e6 / np.sqrt(8.0*np.log(2.0))
        
        F = p[0]
        sig = p[1]
        A = p[2]

        cpa = np.sin(p[3])
        spa = np.cos(p[3])
        ur = cpa*u + spa*v
        vr = -spa*u + cpa*v

        sigmaj = sig / np.sqrt(1-A)
        sigmin = sig / np.sqrt(1+A)

        pimaj = 2.0*np.pi*ur * uas2rad*fwhm2sig 
        pimin = 2.0*np.pi*vr * uas2rad*fwhm2sig 

        ymaj = pimaj*sigmaj
        ymin = pimin*sigmin

        ey2 = np.exp(-0.5*(ymaj**2+ymin**2))

        gradV = np.array([ ey2, # dV/dF
                           -F*ey2*( pimaj**2*sig/(1-A) + pimin**2*sig/(1+A) ), # dV/dd
                           -F*ey2*0.5*( (pimaj*sig/(1-A))**2 - (pimin*sig/(1+A))**2 ), # dV/dA
                           F*ey2* pimaj*pimin * ( sigmaj**2 - sigmin**2 ) ])
        
        
        return gradV.T

    def parameter_labels(self,verbosity=0) :
        """
        Parameter labels to be used in plotting.

        Args:
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (list): List of strings with parameter labels.
        """        
        return [r'$\delta F~({\rm Jy})$',r'$\delta FWHM~(\mu{\rm as})$',r'$\delta A$',r'$\delta {\rm PA}~({\rm rad})$']

    

def W_cubic_spline_1d(k,a=-0.5) :
    """
    Fourier domain convolution function for the approximate 1D cubic spline.
    Useful for the splined raster image classes.

    Args:
      k (numpy.ndarray): Wavenumber.
      a (float): Cubic spline control parameter. Default: -0.5.

    Returns:
      (numpy.ndarray): 1D array of values of the cubic spline weight function evaluated at each value of k.
    """

    abk = np.abs(k)
    ok4 = 1.0/(abk**4+1e-10)
    ok3 = np.sign(k)/(abk**3+1e-10)

    return np.where( np.abs(k)<1e-2,
                     1 - ((2*a+1)/15.0)*k**2 + ((16*a+1)/560.0)*k**4,
                     ( 12.0*ok4*( a*(1.0-np.cos(2*k)) + 2.0*(1.0-np.cos(k)) )
                       - 4.0*ok3*np.sin(k)*( 2.0*a*np.cos(k) + (4*a+3) ) ) )

    
class FF_splined_raster(ff.FisherForecast) :
    """
    FisherForecast object for a splined raster (i.e., themage).
    Parameter vector is the log of the intensity at the control points:

    * p[0] ....... :math:`\\ln(I[0,0])`
    * p[1] ....... :math:`\\ln(I[1,0])`
    * ...
    * p[N-1] ..... :math:`\\ln(I[N-1,0])`
    * p[N] ....... :math:`\\ln(I[0,1])`
    * ...
    * p[N*N-1] ... :math:`\\ln(I[N-1,N-1])`

    where each :math:`I[i,j]` is measured in Jy/sr.

    Args:
      N (int): Raster dimension (only supports square raster).
      fov (float): Raster field of view in uas (only supports fixed rasters).
      stokes (str): Indicates if this is limited to Stokes I ('I') or include full polarization ('full'). Default: 'I'.

    Attributes:
      xcp (numpy.ndarray): x-positions of raster control points.
      ycp (numpy.ndarray): y-positions of raster control points.
      apx (float): Raster pixel size.
    """

    def __init__(self,N,fov,stokes='I') :
        super().__init__()
        self.npx = N
        self.size = self.npx**2
        # fov = fov * np.pi/(3600e6*180) * (N-1)/N
        fov = fov * uas2rad * (N-1)/N
        
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
          p (numpy.ndarray): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities computed at observations.
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
                    V = V + np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[i+self.npx*j]) * W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx
            return V
        
        else:
            I = 0.0j*u
            Q = 0.0j*u
            U = 0.0j*u
            V = 0.0j*u

            countI = 0
            for i in range(self.npx) :
                for j in range(self.npx) :
                    I = I + np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[i+self.npx*j]) * W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx
                    countI += 1
            for i in range(self.npx) :
                for j in range(self.npx) :
                    Q = Q + np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[countI + i+self.npx*j]) * W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx
                    U = U + np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[2*countI + i+self.npx*j]) * W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx
                    V = V + np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[3*countI + i+self.npx*j]) * W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx

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
          p (numpy.ndarray): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities computed at observations.
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
                    gradV.append( np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[i+self.npx*j]) * W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx )
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
                    gradI.append( np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[i+self.npx*j]) * W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx )
                    countI += 1
            for j in range(self.npx) :
                for i in range(self.npx) :
                    gradQ.append( np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[countI + i+self.npx*j]) * W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx )
                    gradU.append( np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[2*countI + i+self.npx*j]) * W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx )
                    gradV.append( np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[3*countI + i+self.npx*j]) * W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx )
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
          (list): List of strings with parameter labels.
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


    

class FF_smoothed_delta_ring_themage(ff.FisherForecast) :
    """
    FisherForecast object for a splined raster (i.e., themage) plus a
    Gaussian-convolved delta-ring.

    Parameter vector is the log of the intensity at the control points:

    * p[0] ....... :math:`\\ln(I[0,0])`
    * p[1] ....... :math:`\\ln(I[1,0])`
    * ...
    * p[N-1] ..... :math:`\\ln(I[N-1,0])`
    * p[N] ....... :math:`\\ln(I[0,1])`
    * ...
    * p[N*N-1] ... :math:`\\ln(I[N-1,N-1])`
    * p[N*N+0] ... Total flux in Jy.
    * p[N*N+1] ... Diameter in uas.
    * p[N*N+2] ... Twice the standard deviation of the Gaussian smoothing kernel in uas.

    where each :math:`I[i,j]` is measured in Jy/sr.

    Args:
      N (int): Raster dimension (only supports square raster).
      fov (float): Raster field of view in uas (only supports fixed rasters).
      stokes (str): Indicates if this is limited to Stokes I ('I') or include full polarization ('full'). Default: 'I'.

    Attributes:
      xcp (numpy.ndarray): x-positions of raster control points.
      ycp (numpy.ndarray): y-positions of raster control points.
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
          p (numpy.ndarray): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities computed at observations.
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
                V = V + np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[i+self.npx*j]) * W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx

        # print(u[:5],v[:5],V[:5])

                
        # Add ring
        d = p[-2]
        w = p[-1]
        # uas2rad = np.pi/180./3600e6
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
          p (numpy.ndarray): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities computed at observations.
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
                gradV.append( np.exp(-2.0j*np.pi*(u*self.xcp[i,j]+v*self.ycp[i,j]) + p[i+self.npx*j]) * W_cubic_spline_1d(2.0*np.pi*self.xcp[i,j]*u)*W_cubic_spline_1d(2.0*np.pi*self.ycp[i,j]*v) * self.apx )
        
        # Add ring
        d = p[-2]
        w = p[-1]
        # uas2rad = np.pi/180./3600e6
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
          (list): List of strings with parameter labels.
        """        
        pll = []
        for j in range(self.npx) :
            for i in range(self.npx) :
                pll.append( r'$\delta I_{%i,%i}$'%(i,j) )

        pll.append(r'$\delta F~({\rm Jy})$')
        pll.append(r'$\delta d~(\mu{\rm as})$')
        pll.append(r'$\delta w~(\mu{\rm as})$')

        return pll

    
class FF_thick_mring(ff.FisherForecast) :
    """
    FisherForecast object for an m-ring model (based on ehtim).
    Parameter vector is:

    * p[0] ... DOM, PLEASE FILL THIS IN.

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
          (dict): Dictionary containing parameter values as used by ehtim.
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
          p (numpy.ndarray): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities computed at observations.
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
          p (numpy.ndarray): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities computed at observations.
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
          (list): List of strings with parameter labels.
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
    
