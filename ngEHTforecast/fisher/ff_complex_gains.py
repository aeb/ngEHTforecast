import numpy as np
import copy
import hashlib
import ehtim.parloop as ploop

import ngEHTforecast.fisher.fisher_forecast as ff

class FF_complex_gains_single_epoch(ff.FisherForecast) :
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
      gain_phase_priors (list): list of standard deviations on the normal priors on the gain phases. Default: 30.
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
          obs (ehtim.obsdata.Obsdata): An ehtim Obsdata object with a particular set of observation details (u,v positions, times, etc.)
          p (numpy.ndarray): list of parameters for the model at which visiblities are desired.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities computed at observations.
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
          obs (ehtim.obsdata.Obsdata): An ehtim Obsdata object with a particular set of observation details (u,v positions, times, etc.)
          p (numpy.ndarray): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibility gradients computed at observations.
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
                            self.prior_sigma_list.append(30.0) # Big phase

                    else :
                        self.prior_sigma_list.append(10.0) # Big amplitude
                        self.prior_sigma_list.append(30.0) # Big phase

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
                            self.prior_sigma_list.append(30.0) # Big phase

                    else :
                        self.prior_sigma_list.append(10.0) # Big amplitude
                        self.prior_sigma_list.append(30.0) # Big phase

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
          (list): List of strings with parameter labels.
        """        
        return self.plbls

    def set_gain_amplitude_prior(self,sigma,station=None) :
        """
        Sets the log-normal priors on the gain amplitudes, either for a
        specified station or for all stations.

        Args:
          sigma (float): Standard deviation of the log-amplitude.
          station (str,list): Station code of the station(s) for which the prior is to be set. If None, will set the prior on all stations. Default: None.
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
        """
        sigma = min(sigma,30.0)
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
        """
        sigma = min(sigma,30.0)
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
        """
        sigma = min(sigma,30.0)
        if (station is None) :
            self.gain_ratio_phase_priors = {'All':sigma}
        else :
            self.gain_ratio_phase_priors[station] = sigma
            
        self.argument_hash = None
        

class FF_complex_gains(ff.FisherForecast) :
    """
    FisherForecast with complex gain reconstruction with multiple epochs.
    This is usually the FisherForecast object that should be used to investigate
    the impact of uncertain station gains.

    Args:
      ff (FisherForecast): A FisherForecast object to which we wish to add gains.
      marg_method (str): Method used for intermediate marginalization. Options are 'covar' and 'vMv'. Default: 'covar'.

    Attributes:
      ff (FisherForecast): The FisherForecast object before gain reconstruction.
      gain_epochs (numpy.ndarray): 2D array containing the start and end times for each gain solution epoch.
      gain_amplitude_priors (list): list of standard deviations of the log-normal priors on the gain amplitudes. Default: 10.
      gain_phase_priors (list): list of standard deviations on the normal priors on the gain phases. Default: 30.
      gain_ratio_amplitude_priors (list): For polarized gains, list of the log-normal priors on the gain amplitude ratios.  Default: 1e-10.
      gain_ratio_phase_priors (list): For polarized gains, list of the normal priors on the gain phase differences.  Default: 1e-10.
      marg_method (str): Method used for intermediate marginalization. Options are 'covar' and 'vMv'. Based on preliminary tests, 'covar' is faster and more accurate, though this may not be the case for models with very many parameters.
    """

    def __init__(self,ff,marg_method='covar') :
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
        self.marg_method = marg_method
        
    def set_gain_epochs(self,scans=False,gain_epochs=None) :
        """
        Sets the gain solution intervals (gain epochs) to be used. If neither
        scans nor gain_epochs selected, will solve for gains on each unique
        timestamp in the data.

        Args:
          scans (bool): If True, solves for independent gains by scan. Overrides explicit specification of gain solution intervals. Default: False.
          gain_epochs (nd.array): 2D array containing the start and end times for each gain solution epoch. Default: None.
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
          obs (ehtim.obsdata.Obsdata): An ehtim Obsdata object with a particular set of observation details (u,v positions, times, etc.)
          p (numpy.ndarray): list of parameters for the model at which visiblities are desired.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibilities computed at observations.
        """
        
        return self.ff.visibilities(obs,p,verbosity=verbosity,**kwargs)

    def visibility_gradients(self,obs,p,verbosity=0,**kwargs) :
        """
        Gradients of the complex visibilities associated with the underlying 
        FisherForecast object with respect to the model pararmeters evaluated at 
        the data points in the given Obsdata object for the model with the given 
        parameter values.  

        Args:
          obs (ehtim.obsdata.Obsdata): An ehtim Obsdata object with a particular set of observation details (u,v positions, times, etc.)
          p (numpy.ndarray): list of parameters for the model image used to create object.
          verbosity (int): Verbosity level. Default: 0.

        Returns:
          (numpy.ndarray): List of complex visibility gradients computed at observations.
        """
        
        return self.ff.visibility_gradients(obs,p,verbosity=verbosity,**kwargs)

    def fisher_covar(self,obs,p,verbosity=0,**kwargs) :
        """
        Returns the Fisher matrix as defined in the accompanying documentation,
        marginalized over the complex station gains. Intelligently avoids 
        recomputation if the observation and parameters are unchanged.

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
                            
                    if (self.marg_method == 'covar'):
                        mn = ff._invert_matrix(ff._invert_matrix(covar_wgs)[:self.ff.size,:self.ff.size])
                    elif (self.marg_method == 'vMv'):
                        n = covar_wgs[:self.ff.size,:self.ff.size]
                        r = covar_wgs[self.ff.size:,:self.ff.size]
                        m = covar_wgs[self.ff.size:,self.ff.size:]
                        r,m = self._condition_vM(r,m)
                        mn = n - ff._vMv(r,ff._invert_matrix(m))
                    else :
                        raise(RuntimeError("Received unexpected intermediate margnilazation method, %s. Allowed values are 'covar' and 'vMv'."%(self.marg_method)))

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
                        _print_matrix(ff._invert_matrix(m))
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
                    # mn = n - np.matmul(r.T,np.matmul(ff._invert_matrix(m),r))
                    mn = n - ff._vMv(r,ff._invert_matrix(m))

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
          (list): List of strings with parameter labels.
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
        """
        sigma = min(sigma,30.0)
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
        """
        sigma = min(sigma,10.0)
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
        """
        sigma = min(sigma,30.0)
        if (station is None) :
            self.gain_ratio_phase_priors = {'All':sigma}
        else :
            self.gain_ratio_phase_priors[station] = sigma
            
        if (verbosity>0) :
            print("Gain ratio phase dict:",self.gain_ratio_phase_priors)

        self.argument_hash = None
