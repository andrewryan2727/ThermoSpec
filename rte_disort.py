import numpy as np
from pydisort import Disort, DisortOptions, scattering_moments
import torch
torch.set_default_dtype(torch.float64)
from config import SimulationConfig
from grid import LayerGrid
from scipy.interpolate import interpn


class DisortRTESolver:
    def __init__(self, config: SimulationConfig, grid: LayerGrid, n_cols = 1, output_radiance=False, uniform_props=False,planck=True, observer_mu=1.0, observer_phi=0.0, solver_mode=None, spectral_component=None):
        self.cfg = config
        self.grid = grid
        self.output_radiance = output_radiance
        self.uniform_props = uniform_props
        self.planck = planck
        self.n_cols = n_cols
        self.observer_mu = observer_mu
        self.observer_phi = observer_phi
        
        # New mode parameters
        self.solver_mode = solver_mode
        self.spectral_component = spectral_component
        
        # Determine effective mode for this instance
        self._determine_effective_mode()
        
        #For multi-wavelength cases, load optical and solar constants from files
        if(self.is_multi_wave):
            self._load_constants()
            if self.effective_mode == 'hybrid_thermal':
                self._filter_thermal_wavelengths()
                
        #Construct optical properties tensors
        if(self.is_multi_wave):
            #More complicated set up for multiple wavelengths
            self.prop = self._setup_optical_properties_advanced() 
        else:
            #Simple set up for globally uniform properties. 
            self.prop = self._setup_optical_properties() 
        self.op = self._setup_disort_options() #Initialize disort options
        self.ds = Disort(self.op) #initialize disort
        if(self.is_multi_wave):
            nwave = len(self.wavenumbers)
        else:
            nwave = 1
        #Initialize boundary condition arrays.
        self.nwave = nwave 
        self.umu0 = torch.zeros(n_cols)
        self.phi0 = torch.zeros(n_cols) #Leaving at zero for now. No changes later. 
        self.fbeam = torch.zeros((nwave,n_cols))
        self.albedo = torch.zeros((nwave,n_cols))
        self.fisot = torch.zeros((nwave,n_cols))
        self.temis = torch.ones((nwave,n_cols)) #Top emissivity (space). Stays at 1. 
        self.btemp = torch.zeros(n_cols)
        self.ttemp  = torch.full((n_cols,), self.cfg.disort_space_temp) #Space temperature. Stays unchanged, usually set to 0 for space.  
        self.bc = {
            "umu0": self.umu0, #Cosine of solar incidence angle
            "phi0": self.phi0, #Azimuth angle of solar incidence. 
            "albedo": self.albedo, #Albedo of bottom boundary
            "btemp": self.btemp, #Brightness temperature of bottom boundary. 
            "ttemp": self.ttemp, #Top boundary temperature (space)
            "temis": self.temis, #Emissivity of top boundary (space)
            "fisot": self.fisot, #Intensity of top-boundary isotropic illumination (W/m^2)
            "fbeam": self.fbeam #Intensity of incidence parallel beam Solar flux (W/m^2)
        }

    def _determine_effective_mode(self):
        """Determine the effective operating mode for this DISORT instance."""
        # If solver_mode is explicitly provided, use it
        if self.solver_mode is not None:
            if self.solver_mode == 'hybrid':
                if self.spectral_component == 'thermal_only':
                    self.effective_mode = 'hybrid_thermal'
                    self.is_multi_wave = True
                    self.should_load_solar_spectrum = False  # Thermal-only doesn't need solar spectrum
                elif self.spectral_component == 'visible_only':
                    self.effective_mode = 'hybrid_visible'
                    self.is_multi_wave = False
                    self.should_load_solar_spectrum = False  # Visible-only uses cfg.J
                else:
                    raise ValueError("Hybrid mode requires spectral_component to be 'thermal_only' or 'visible_only'")
            elif self.solver_mode == 'multi_wave':
                self.effective_mode = 'multi_wave'
                self.is_multi_wave = True
                self.should_load_solar_spectrum = True  # Multi-wave needs solar spectrum
            elif self.solver_mode == 'two_wave':
                self.effective_mode = 'two_wave'
                self.is_multi_wave = False
                self.should_load_solar_spectrum = False  # Two-wave uses cfg.J
            else:
                raise ValueError(f"Unknown solver_mode: {self.solver_mode}")
        else:
            # Legacy mode: determine from existing logic
            raise ValueError(f"Solver mode must be specified!")
            # if self.cfg.multi_wave:
            #     self.effective_mode = 'multi_wave'
            #     self.is_multi_wave = True
            #     self.should_load_solar_spectrum = True  # Legacy multi-wave needs solar spectrum
            # else:
            #     self.effective_mode = 'two_wave'
            #     self.is_multi_wave = False
            #     self.should_load_solar_spectrum = False  # Legacy two-wave uses cfg.J
    
    def _filter_thermal_wavelengths(self):
        """Filter loaded constants to thermal wavelengths only (for hybrid_thermal mode)."""
        if not hasattr(self, 'wavenumbers'):
            raise ValueError("Cannot filter wavelengths: constants not loaded yet")
            
        # Convert cutoff wavelength to wavenumber
        cutoff_wn = 10000.0 / self.cfg.hybrid_wavelength_cutoff  # Convert μm to cm-1
        thermal_mask = self.wavenumbers <= cutoff_wn
        
        if not any(thermal_mask):
            raise ValueError(f"No thermal wavelengths found below cutoff {self.cfg.hybrid_wavelength_cutoff} μm")
            
        # Filter all wavelength-dependent arrays
        self.wavenumbers = self.wavenumbers[thermal_mask]
        self.g_array = self.g_array[thermal_mask]
        self.Cext_array = self.Cext_array[thermal_mask]
        self.Csca_array = self.Csca_array[thermal_mask]
        self.ssalb_array = self.ssalb_array[thermal_mask]
        
        # Filter solar spectrum if it exists
        if hasattr(self, 'solar'):
            self.solar = self.solar[thermal_mask]
            
        # Filter emissivity spectrum if it exists
        if hasattr(self, 'emiss_base'):
            self.emiss_base = self.emiss_base[thermal_mask]
            
        #print(f"Hybrid thermal mode: filtered to {len(self.wavenumbers)} thermal wavelengths (< {self.cfg.hybrid_wavelength_cutoff} μm)")

    def _load_constants(self):
        #self.wavenumbers, self.ssalb_array, self.Cext_array, self.Csca_array, self.Cabs_array, self.alpha1_array = load_mie_folder(self.cfg.folder,self.config.nmom)
        if(self.output_radiance):
            mie_file = self.cfg.mie_file_out
            solar_file = self.cfg.solar_spectrum_file_out
            emissivity_file = self.cfg.substrate_spectrum_out
        else:
            mie_file = self.cfg.mie_file
            solar_file = self.cfg.solar_spectrum_file
            emissivity_file = self.cfg.substrate_spectrum
        mie_params = np.loadtxt(mie_file)
        sortidx = np.argsort(10000./mie_params[:,0])
        self.wavenumbers = 10000./mie_params[sortidx,0]
        self.g_array = mie_params[sortidx,1]
        self.Cext_array = mie_params[sortidx,2]
        self.Csca_array = mie_params[sortidx,3]
        self.ssalb_array = mie_params[sortidx,4]
        # Load solar spectrum only if we need spectral solar data
        # In hybrid mode, thermal-only solvers don't need solar spectrum (no visible wavelengths)
        # In hybrid mode, visible-only solvers shouldn't be multi-wave (use cfg.J instead)
        if (self.cfg.diurnal or self.cfg.sun) and self.should_load_solar_spectrum:
            #Load multi-wave solar flux file. 
            solar_array = np.loadtxt(solar_file)
            if(len(solar_array[:,0]) != len(self.wavenumbers) or np.max(solar_array[:,0]-self.wavenumbers)>0.1):
                print("Warning: Solar spectrum file wavenumbers do not match scattering files!")
            self.solar = solar_array[:,1]/self.cfg.R**2.
            self.solar_sum = np.sum(self.solar)
        if(self.cfg.use_spec):
            #Load emissivity spectrum for substrate. 
            emiss_spec = np.loadtxt(emissivity_file)
            if(len(emiss_spec[:,0]) != len(self.wavenumbers) or np.max(emiss_spec[:,0]-self.wavenumbers)>0.1):
                print("Warning: Emissivity spectrum file wavenumbers do not match scattering files!")
            self.emiss_base = emiss_spec[:,1]
        #Determine TIR wavenumber range for uniform properties case
        if(self.uniform_props):
            #Set wavenumber range for averaging TIR optical properties for making a smooth planck function. 
            #Used only for producing final emissivity spectra. I.e., the spectra from other runs are divided by this smooth planck function. 
            wn_min = 100.0
            wn_max = 2000.0
            self.wn_min = np.argmin(np.abs(self.wavenumbers - wn_min))
            self.wn_max = np.argmin(np.abs(self.wavenumbers - wn_max))
        

    def _setup_optical_properties(self):
        #Set up optical properties for simple two-wave disort case (vis and thermal)
        n_layers = len(self.grid.x_RTE)
        prop = torch.zeros((1, self.n_cols, n_layers, 2 + self.cfg.nmom), dtype=torch.float64)

        tau_boundaries = self.grid.x_boundaries
        if not self.planck: tau_boundaries *= self.cfg.eta #Convert tau vaues using visible extinction coefficient ratio. 
        dtau = tau_boundaries[1:] - tau_boundaries[:-1] #tau at boundaries of each layer
        tau_layer = torch.tensor(dtau)
        if self.planck:
            ssalb_val = self.cfg.ssalb_therm
            g_val = self.cfg.g_therm
        else:
            ssalb_val = self.cfg.ssalb_vis
            g_val = self.cfg.g_vis
        ssa_layer = torch.full((n_layers,), ssalb_val)
        moms = scattering_moments(self.cfg.nmom, "henyey-greenstein", g_val)
        moments = torch.stack(
            [torch.full((n_layers,), moms[i]) for i in range(self.cfg.nmom)],
            dim=-1
        )
        prop[0, :, :, 0] = tau_layer
        prop[0, :, :, 1] = ssa_layer
        prop[0, :, :, 2:] = moments
        return prop


    def _setup_optical_properties_advanced(self):
        """Set up optical properties for DISORT
           Handles wavelength-dependent and/or depth-dependent scenarios
           Must specify input file paths in config file."""
        n_layers = len(self.grid.x_RTE) 
        n_waves = len(self.wavenumbers) if self.is_multi_wave else 1
        n_mom = self.cfg.nmom 
        n_cols = self.n_cols #number of columns. Always 1 unless we add variations in the future for different properties but at the same wavelength. 

        #initialize properties tensor. Last dimension is tau + ss_albedo + scattering moments
        prop = torch.zeros((n_waves, n_cols, n_layers, 2 + n_mom), dtype=torch.float64)
        Vp = (4./3.)*np.pi*self.cfg.radius**3. #particle volume, m^3
        if self.cfg.depth_dependent_properties:
            #Get particle fill fraction from depth-dependent density
            fill_frac_layers = self.grid.rho_depth / self.cfg.rho_particle
            # Calculate number density at each layer: n_p(z) = φ(z) / V_p
            n_p = fill_frac_layers / Vp  # particles per m³ at each layer
        else:
            n_p= self.cfg.fill_frac / Vp #Use user-prescribed fill fraction
        Et_mean = np.mean(n_p * self.Cext_array*1e-12)
        if self.cfg.scale_Et:
            #Calculate a scale factor to apply to the calculated Et values so that the mean is equivalent to the 
            Et_scale = self.cfg.Et/Et_mean
            print("Multi_wave mean Et rescaled to user input value!")
        else:
            Et_scale = 1.0
        print(f"Multi_wave mean Et = {Et_scale*Et_mean}")

        for i_wave in range(n_waves):
            #Calculate dtau at this wavelength using the extinction cross section Cext

            Et = Et_scale*n_p * self.Cext_array[i_wave]*1e-12 #extinction coefficient at this wavelength, converting Cext from µm^2 to m^2 in the process. 
            if(self.uniform_props):
                Et = Et_scale*n_p * np.mean(self.Cext_array[self.wn_min:self.wn_max])*1e-12
            #Et = Et*10.0 + 0.0 #manual override to fixed value for testing 
            tau_boundaries = Et*self.grid.x_boundaries/self.cfg.Et #tau at boundaries of each layer, convert from global Et to wavelength-specific Et.  
            dtau = tau_boundaries[1:] - tau_boundaries[:-1] #tau thickness of each layer
            tau_layer = torch.tensor(dtau,dtype=torch.float64)
            ssalb_val = self.ssalb_array[i_wave].copy()
            if(self.cfg.force_vis_disort and self.wavenumbers[i_wave] >3330.0):
                #Force all visible wavelengths to use the DISORT default value for visible scattering albedo
                ssalb_val = self.cfg.ssalb_vis
            if(self.uniform_props):
                ssalb_val = np.mean(self.ssalb_array[self.wn_min:self.wn_max])
            # if self.wavenumbers[i_wave] >3330.0: 
            #     ssalb_val = ssalb_val*0.0 + 0.5 #manual override to fixed value for testing VISIBLE. 
            # else:
            #     ssalb_val = ssalb_val*0.0 + 0.8 #manual override for fixed value testing THERMAL
            ssa_layer = torch.full([n_layers], ssalb_val,dtype=torch.float64)
            g_val = self.g_array[i_wave]
            if(self.cfg.force_vis_disort and self.wavenumbers[i_wave] >3330.0):
                #Force all visible wavelengths to use the DISORT default value for visible scattering asymmetry factor
                g_val = self.cfg.g_vis
            #g_val = g_val*0.0 #manual override to fixed value for testing. 
            if(self.uniform_props):
                g_val = np.mean(self.g_array[self.wn_min:self.wn_max])
            moms = scattering_moments(self.cfg.nmom, "henyey-greenstein", g_val)
            moments = torch.stack(
                [torch.full((n_layers,), moms[i]) for i in range(n_mom)],
                dim=-1
            )
            #Populate properties tensor
            prop[i_wave, :, :, 0] = tau_layer
            prop[i_wave, :, :, 1] = ssa_layer
            prop[i_wave, :, :, 2:] = moments
        return prop

    def _setup_disort_options(self):
        """Configure DISORT options"""
        op = DisortOptions()
        if(self.output_radiance):
            #calculate radiances on output (slower) by omitting the onlyfl option. 
            if self.planck:
                op.flags("lamber,quiet,usrang,planck")
            else:
                op.flags("lamber,quiet,usrang") 
            op.ds().nmom = self.cfg.nmom_out
            op.ds().nstr = self.cfg.nstr_out
            op.ds().nphase = self.cfg.nmom_out
            # Handle both scalar and array inputs for observer angles
            if hasattr(self.observer_mu, '__len__'):
                op.user_mu(np.array(self.observer_mu)) #Array of zenith angles for measuring intensity
            else:
                op.user_mu(np.array([self.observer_mu])) #Single zenith angle for measuring intensity
            
            if hasattr(self.observer_phi, '__len__'):
                op.user_phi(np.array(self.observer_phi)) #Array of azimuth angles for measuring intensity
            else:
                op.user_phi(np.array([self.observer_phi])) #Single azimuth angle for measuring intensity 
        else:
            if self.planck:
                op.flags("lamber,quiet,onlyfl,intensity_correction,planck")
            else:
                op.flags("lamber,quiet,onlyfl,intensity_correction") #Calculate flux only during model run (faster). 
            op.ds().nmom = self.cfg.nmom
            op.ds().nstr = self.cfg.nstr
            op.ds().nphase = self.cfg.nmom
        if self.is_multi_wave:
            self.lower_wns, self.upper_wns = self._compute_wn_bounds()
            self.wn_bin_widths = np.array(self.upper_wns) - np.array(self.lower_wns)
            self.wn_bins = np.append(self.lower_wns,self.upper_wns[-1])
            op.wave_lower(self.lower_wns)
            op.wave_upper(self.upper_wns)
        else:
            #Broadband. Define range for planck function to a very wide range. 
            op.wave_lower([20]) #500 µm
            op.wave_upper([10000]) #1 µm
        n_waves = len(self.wavenumbers) if self.is_multi_wave else 1
        op.nwave(n_waves)
        op.ncol(self.n_cols)
        op.ds().nlyr = len(self.grid.x_RTE)

        return op

    def _compute_wn_bounds(self):
        if(self.output_radiance):
            file = self.cfg.wn_bounds_out
        else:
            file = self.cfg.wn_bounds
        wn_bounds = np.loadtxt(file)
        wn_bounds = np.sort(wn_bounds)
        if(len(wn_bounds) != len(self.wavenumbers)+1):
            raise ValueError(f"Expected {len(self.wavenumbers)+1} wavenumber bounds, got {len(wn_bounds)} from {file}.")
        lower_bounds = wn_bounds[:-1].tolist()  # lower edge for each bin
        upper_bounds = wn_bounds[1:].tolist()   # upper edge for each bin
        return lower_bounds, upper_bounds
    



    def disort_run(self, T, mu, F, Q=None, phi=None):
        #Solve RTE using DISORT for both visible and thermal bands.
        #mu, F, Q, and phi should all either be supplied as scalars or as arrays with length=n_cols

        #Account for some crater shadow cases where the sun is up but the facet is not illuminated. 
        F *= mu > 0.001
        if self.n_cols > 1:
            if not hasattr(F, 'shape'):
                #If F is a scalar, convert to array with n_cols elements
                F = np.full(self.n_cols, F )
            if not hasattr(mu, 'shape'):
                #If mu is a scalar, convert to array with n_cols elements
                mu = np.full(self.n_cols, mu )

        #Initialization state and dimensions of boundary condition arrays. 
        # umu0 = torch.zeros(n_cols)
        # phi0 = torch.zeros(n_cols)
        # fbeam = torch.zeros((nwave,n_cols))
        # albedo = torch.zeros((nwave,n_cols))
        # fisot = torch.zeros((nwave,n_cols))
        # temis = torch.ones((nwave,n_cols)) #Top emissivity (space). Stays at 1. 
        # btemp = torch.zeros(n_cols)
        # ttemp  = torch.zeros(n_cols)

        if self.is_multi_wave:
            #fbeam = torch.tensor(self.solar*F).unsqueeze(1) #Solar flux integrated within each wavenumber band.
            if np.any(F>0): 
                if hasattr(self, 'solar'):
                    # Use loaded solar spectrum (multi-wave mode)
                    if self.n_cols==1:
                        self.fbeam[:] = torch.from_numpy(self.solar*F)[:,None]
                    else:
                        self.fbeam[:] = torch.from_numpy(np.tile(self.solar[:,None],(1,self.n_cols))*np.tile(F,(self.nwave,1)))
                else:
                    # No solar spectrum loaded (e.g., hybrid thermal mode) - use broadband approximation
                    if self.n_cols==1:
                        self.fbeam[:] = F * self.cfg.J / self.nwave  # Distribute broadband flux evenly
                    else:
                        self.fbeam[:] = torch.from_numpy(np.tile((F * self.cfg.J / self.nwave)[None,:],(self.nwave,1)))
            #temis = torch.full([len(self.wavenumbers)],1.0).unsqueeze(1)
            #if Q==None: 
                #fisot = torch.tensor(np.zeros_like(self.wavenumbers)).unsqueeze(1)
            else:
                self.fbeam*=0.0
            if np.any(Q != None):
                #fisot = torch.tensor(Q).unsqueeze(1)
                #For now passing global value. TO DO: update to wavelength-dependent later.
                #fisot = torch.full([len(self.wavenumbers)],Q).unsqueeze(1)
                #Not correct for multi-column data.
                if self.n_cols==1: 
                    #only works if Q array is size [n_waves]
                    self.fisot[:] = torch.from_numpy(Q)[:,None]
                else:
                    self.fisot[:] = torch.from_numpy(Q)
            if(self.cfg.use_spec and not self.uniform_props):
                #albedo = torch.tensor(1.0 - self.emiss_base).unsqueeze(1) #Albedo spectrum for the bottom boundary.
                self.albedo[:] = torch.from_numpy(1.0 - self.emiss_base)[:,None] #emiss_base must have same dimensions as n_wave. 
            else:
                #albedo = torch.full([len(self.wavenumbers)],self.cfg.R_base).unsqueeze(1) 
                self.albedo.fill_(self.cfg.R_base)
        else:
            #Set up basic two-wave boundary conditions. 
            if self.planck:
                #fbeam = torch.tensor([0.0]) #We're in the thermal two-wave case. No sunlight here. 
                self.fbeam.fill_(0.0)
            else:
                if(self.n_cols>1):
                    self.fbeam[:]= torch.from_numpy(F*self.cfg.J)[None,:] #Visible two-wave case. Broadband flux from solar const. F must have dimension n_col. 
                else:
                    self.fbeam.fill_(F*self.cfg.J)
            #temis = torch.tensor([1.0])
            if np.any(Q==None):
                self.fisot.fill_(0.0) 
            else:
                if(self.n_cols>1):
                    self.fisot[:] = torch.from_numpy(Q)[None,:] #Q must have dimensions n_col
                else:
                    self.fisot.fill_(Q)
            self.albedo.fill_(self.cfg.R_base)

        if(self.cfg.single_layer):
            if(self.n_cols>1):
                self.btemp[:] = torch.from_numpy(T[-1,:])
                if(np.any(T[-1,:]<0)):
                    print('Negative temperatures passed to disort. Aborting run.')
                    return
            else:
                self.btemp.fill_(T[-1])
        else:
            T_interface = calculate_interface_T(T, self.grid.nlay_dust, self.grid.alpha, self.grid.beta)
            if(self.n_cols>1):
                self.btemp[:] = torch.from_numpy(T_interface)
            else:
                self.btemp.fill_(T_interface)
        
        # Set solar incidence angles
        if(self.n_cols>1):
            if hasattr(mu,'shape'):
                self.umu0[:] = torch.from_numpy(mu)
            else:
                self.umu0.fill_(mu)
            # Set solar azimuth angles if provided
            if phi is not None:
                self.phi0[:] = torch.from_numpy(phi)
            else:
                self.phi0.fill_(0.0)  # Default to 0 if not provided
        else:
            self.umu0[:] = mu
            if phi is not None:
                self.phi0.fill_(phi)
            else:
                self.phi0.fill_(0.0)  # Default to 0 if not provided

        # self.bc = {
        #     "umu0": self.umu0, #Cosine of solar incidence angle
        #     "phi0": self.phi0, #Azimuth angle of solar incidence. 
        #     "albedo": self.albedo, #Albedo of bottom boundary
        #     "btemp": self.btemp, #Brightness temperature of bottom boundary. 
        #     "ttemp": self.ttemp, #Top boundary temperature (space)
        #     "temis": self.temis, #Emissivity of top boundary (space)
        #     "fisot": self.fisot, #Intensity of top-boundary isotropic illumination (W/m^2)
        #     "fbeam": self.fbeam #Intensity of incidence parallel beam Solar flux (W/m^2)
        # }

        #Need to interpolate temperature field to be at boundaries of computational layers, rather than at the center.
        #Assuming that the temperature grid is sufficiently dense to allow for a fast linear interpolation.
        if self.n_cols==1:  
            T_interp = np.interp(self.grid.x_boundaries,self.grid.x_RTE,T[1:self.grid.nlay_dust+1])
            T_tensor = torch.tensor(T_interp).unsqueeze(0)
        else:
            #T_interp = interpn((self.grid.x_RTE,),T[1:self.grid.nlay_dust+1,:],self.grid.x_boundaries,bounds_error=False,fill_value=None)
            #T_tensor = torch.tensor(T_interp).swapaxes(0,1)
            T_interp = np.vstack([
                np.interp(self.grid.x_boundaries, self.grid.x_RTE, T[1:self.grid.nlay_dust+1, col])
                for col in range(self.n_cols)
            ])
            T_tensor = torch.tensor(T_interp)

        #Run disort. Result returns up and down total fluxes. 
        result = self.ds.forward(self.prop,'', T_tensor, **self.bc)
        if self.is_multi_wave:
            fl_up = result[:,:,0,0].numpy() #total upwards diffuse flux from top layer, for rough surface scattering modeling. 
        else:
            fl_up = result[0,:,0,0].numpy()
        #result[nwave,ncol,ndepth,up or down]

        #DISORT outputs from the gather_flx() method are: 
        # 0.	RFLDIR(lu) - Direct-beam flux (without delta-M scaling)
        # 1.	RFLDN(lu) - Diffuse down-flux (total minus direct-beam) (without delta-M scaling)
        # 2.	FLUP(lu) - Diffuse up-flux – same as first output of pydisort output
        # 3.	DFDT(lu) - Flux divergence  d(net flux)/d(optical depth),where 'net flux' includes the direct beam. An exact result;  not from differencing fluxes)
        # 4.	UAVG(lu) - Mean intensity (including the direct beam)  (Not corrected for delta-M-scaling effects)
        # 5.	UU(iu,lu,j) - Intensity ( if ONLYFL = FALSE;  zero otherwise )
        # 6.	ALBMED(iu) - Albedo of the medium as a function of incident beam angle cosine UMU(IU)  (IBCND = 1 case only)
        # 7.	TRNMED(iu) - Transmissivity of the medium as a function of incident beam angle cosine UMU(IU)  (IBCND = 1 case only)

        # Up stream minus down stream rough calculation of flux divergence. Not as accurate as what disort provides. 
        F_net = result[:,:,:,0] - result[:,:,:,1] #[nwave, ncol, ndepth,up/down]
        # Calculate heating rate as flux divergence
        Q_rad_simple = np.diff(F_net.numpy()) / self.grid.dtau
        if self.is_multi_wave:
            #Sum over the different wavelengths. 
            Q_rad_simple = np.sum(Q_rad_simple,axis=0)
        Q_rad_simple = np.squeeze(Q_rad_simple)
        if(self.output_radiance):
            #Just return the radiance as viewed by the observer, currently fixed at zenith. 
            rad = self.ds.gather_rad() #rad[nwave,ncol,ndepth,n_obs_direction mu, n_obs_direction phi]
            return(rad[:,:,0,:,:],fl_up)
        else:
            # #Get flux divergence. This method didn't work well for very slow rotators (e.g., the Moon). Let to fluxes very deep that shouldn't exist. 
            flx = self.ds.gather_flx() #flx[nwave,ncol,ndepth,8 properties] See table above for list of properties. 
            # if self.is_multi_wave:
            #     #Summation of values from different wavelength bins
            #     flux_divergence = flx[:,:,:,3]
            #     Q_rad = torch.sum(flux_divergence,dim=0)
            # else:
            #     Q_rad = flx[0,:,:,3] #returns flux divergence Qrad[ncol,ndepth_boundaries]
            if(self.n_cols==1):
                #Q_rad_interp = np.interp(self.grid.x_RTE,self.grid.x_boundaries,Q_rad[0,:])
                source = np.zeros(self.grid.x_num)
                source[1:self.grid.nlay_dust+1] = Q_rad_simple #used to be Q_rad_interp here. 
                source_term = source * self.grid.K * self.cfg.q
                if(not self.cfg.single_layer):
                    #Two-layer mode. Need to add terms for the boundary. 
                    #emission = np.pi*np.sum(planck_wn_integrated(self.wn_bins,T_interface)*self.emiss_base)
                    #Radiative flux source term at the rock/dust boundary. They are not multipled by Kq because they are not volumetric. 
                    #Terms on right are down direct-beam flux + down diffuse flux - up diffuse flux. 
                    source_term[self.grid.nlay_dust+1] = torch.sum((flx[:,0,-1,0] + flx[:,0,-1,1] - flx[:,0,-1,2]),dim=0)
            else:
                #Multiple columns. Return source_term in format [n_colns, n_depth]
                #Q_rad_interp = interpn((self.grid.x_boundaries,),Q_rad.numpy().swapaxes(0,1),self.grid.x_RTE,bounds_error=False,fill_value=None)
                # Q_rad_interp = np.vstack([
                #     np.interp(self.grid.x_RTE, self.grid.x_boundaries, Q_rad[col,:])
                #     for col in range(self.n_cols)
                # ])
                source = np.zeros((self.n_cols,self.grid.x_num))
                source[:,1:self.grid.nlay_dust+1] = Q_rad_simple #used to be Q_rad_interp here. 
                source_term = source * self.grid.K * self.cfg.q
                if(not self.cfg.single_layer):
                    #emission = np.pi*np.sum(planck_wn_integrated(self.wn_bins,T_interface)*self.emiss_base)
                    #Radiative flux source term at the rock/dust boundary. They are not multipled by Kq because they are not volumetric. 
                    #Terms on right are down direct-beam flux + down diffuse flux - up diffuse flux. 
                    source_term[:,self.grid.nlay_dust+1] = torch.sum((flx[:,:,-1,0] + flx[:,:,-1,1] - flx[:,:,-1,2]),dim=0)
            return source_term, fl_up



def calculate_interface_T(T,i,alpha,beta):
    return((alpha*T[i] + beta*T[i+1])/(alpha + beta))

