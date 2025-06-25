import numpy as np
from pydisort import Disort, DisortOptions, scattering_moments
import torch

from config import SimulationConfig
from grid import LayerGrid

torch.set_default_dtype(torch.float64)

class DisortRTESolver:
    def __init__(self, config: SimulationConfig, grid: LayerGrid, output_radiance=False, uniform_props=False):
        self.cfg = config
        self.grid = grid
        self.output_radiance = output_radiance
        self.uniform_props = uniform_props
        #For multi-wavelength cases, load optical and solar constants from files
        if(self.cfg.multi_wave):
            self._load_constants()
        #Construct optical properties tensors
        if(self.cfg.multi_wave or self.cfg.depth_dependent):
            #More complicated set up for multiple wavelengths and/or depth-dependent properties
            self.prop = self._setup_optical_properties_advanced() 
        else:
            #Simple set up for globally uniform properties. 
            self.prop = self._setup_optical_properties() 
        self.op = self._setup_disort_options() #Initialize disort options
        self.ds = Disort(self.op) #initialize disort

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
        solar_array = np.loadtxt(solar_file)
        if(len(solar_array[:,0]) != len(self.wavenumbers) or np.max(solar_array[:,0]-self.wavenumbers)>0.1):
            print("Warning: Solar spectrum file wavenumbers do not match scattering files!")
        self.solar = solar_array[:,1]/self.cfg.R**2.
        emiss_spec = np.loadtxt(emissivity_file)
        if(len(emiss_spec[:,0]) != len(self.wavenumbers) or np.max(emiss_spec[:,0]-self.wavenumbers)>0.1):
            print("Warning: Emissivity spectrum file wavenumbers do not match scattering files!")
        self.emiss_base = emiss_spec[:,1]
        #Determine TIR wavenumber range for uniform properties case
        if(self.uniform_props):
            wn_min = 100.0
            wn_max = 2000.0
            self.wn_min = np.argmin(np.abs(self.wavenumbers - wn_min))
            self.wn_max = np.argmin(np.abs(self.wavenumbers - wn_max))
        

    def _setup_optical_properties(self):
        """Set up optical properties for DISORT"""
        n_layers = len(self.grid.dtau)
        tau_layer = torch.tensor(self.grid.dtau)
        ssa_layer = torch.full((n_layers,), self.cfg.ssalb_vis)
        moms = scattering_moments(self.cfg.nmom, "henyey-greenstein", self.cfg.g)
        moments = torch.stack(
            [torch.full((n_layers,), moms[i]) for i in range(self.cfg.nmom)],
            dim=-1
        )
        optical_props = torch.cat([
            tau_layer.unsqueeze(-1),
            ssa_layer.unsqueeze(-1),
            moments
        ], dim=-1)
        return optical_props.unsqueeze(0).unsqueeze(0)
    
    def _setup_optical_properties_advanced(self):
        """Set up optical properties for DISORT
           Handles wavelength-dependent and/or depth-dependent scenarios
           Must specify input file paths in config file."""
        n_layers = len(self.grid.dtau)
        n_waves = len(self.wavenumbers) if self.cfg.multi_wave else 1
        n_mom = self.cfg.nmom 
        n_cols = 1 #number of columns. Always 1 unless we add variations in the future for different properties but at the same wavelength. 

        #initialize properties tensor. Last dimension is tau + ss_albedo + scattering moments
        prop = torch.zeros((n_waves, n_cols, n_layers, 2 + n_mom), dtype=torch.float64)
        for i_wave in range(n_waves):
            # Select dtau for this wavelength
            #if self.cfg.multi_wave:
            #    tau_layer = torch.tensor(self.grid.dtau[i_wave])
            #else:
            #    tau_layer = torch.tensor(self.grid.dtau)

            if self.cfg.depth_dependent:
                #Depth-dependent properties
                raise NotImplementedError("Depth-dependent properties not implemented yet.")
                #Need to rewrite and finish this section! Incomplete! 
                # if self.cfg.multi_wave:
                #     ssa_layer = self.ssalb_array[i_wave, :]  # Shape: (n_layers,)
                #     g_layer = self.cfg.g_profile[i_wave, :]      # Shape: (n_layers,)
                #     moments = torch.stack([
                #         torch.tensor(self.cfg.moments_profile[i_wave, :, i_mom])
                #         for i_mom in range(n_mom)
                #     ], dim=-1)  # (n_layers, n_mom)
                # else:
                #     ssa_layer = self.ssalb_array[:]          # Shape: (n_layers,)
                #     g_layer = self.cfg.g_profile[:]              # Shape: (n_layers,)
                #     moments = torch.stack([
                #         torch.tensor(self.cfg.moments_profile[:, i_mom])
                #         for i_mom in range(n_mom)
                #     ], dim=-1)  # (n_layers, n_mom)
            else:
                # Uniform properties with depth
                if self.cfg.multi_wave:
                    #Calculate dtau at this wavelength using the extinction cross section Cext
                    node_depth_meters = (self.grid.x_RTE/self.cfg.Et)
                    Vp = (4./3.)*np.pi*self.cfg.radius**3. #particle volume, m^3
                    n_p= self.cfg.fill_frac / Vp #number of particles /m3
                    Et = n_p * self.Cext_array[i_wave]*1e-12 #extinction coefficient at this wavelength, converting Cext from µm^2 to m^2 in the process. 
                    if(self.uniform_props):
                        Et = n_p * np.mean(self.Cext_array[self.wn_min:self.wn_max])*1e-12
                    #Et = Et*0.0 + 9000
                    tau = node_depth_meters*Et
                    dtau = np.insert(np.diff(tau),0,tau[0]*2)
                    tau_layer = torch.tensor(dtau,dtype=torch.float64)
                    ssalb_val = self.ssalb_array[i_wave].copy()
                    if(self.uniform_props):
                        ssalb_val = np.mean(self.ssalb_array[self.wn_min:self.wn_max])
                    #ssalb_val = ssalb_val*0.0 + 0.1
                    ssa_layer = torch.full([n_layers], ssalb_val,dtype=torch.float64)
                    #g_layer = np.full(n_layers, self.cfg.g[i_wave])
                    # Moments same for all layers
                    #moms = self.alpha1_array[i_wave,:]
                    g_val = self.g_array[i_wave]
                    if(self.uniform_props):
                        g_val = np.mean(self.g_array[self.wn_min:self.wn_max])
                    moms = scattering_moments(self.cfg.nmom, "henyey-greenstein", g_val)
                    moments = torch.stack(
                        [torch.full((n_layers,), moms[i]) for i in range(n_mom)],
                        dim=-1
                    )
                else:
                    ssa_layer = np.full(n_layers, self.cfg.ssalb_vis)
                    g_layer = np.full(n_layers, self.cfg.g)
                    moms = scattering_moments(self.cfg.nmom, "henyey-greenstein", g_layer[0])
                    moments = torch.stack(
                        [torch.full((n_layers,), moms[i],dtype=torch.float64) for i in range(n_mom)],
                        dim=-1
                    )
            #Populate properties tensor
            prop[i_wave, 0, :, 0] = tau_layer
            prop[i_wave, 0, :, 1] = ssa_layer
            prop[i_wave, 0, :, 2:] = moments

        return prop



    def _setup_disort_options(self):
        """Configure DISORT options"""
        op = DisortOptions()
        if(self.output_radiance):
            op.flags("lamber,quiet,planck,usrang") #calculate radiances on output (slower)
            op.ds().nmom = self.cfg.nmom_out
            op.ds().nstr = self.cfg.nstr_out
            op.ds().nphase = self.cfg.nmom_out
            op.user_mu(np.array([1.0])) #Specify zenith angle for measuring intensity
            op.user_phi(np.array([0.0])) #Specify azimuth angle for measuring intensity. 
        else:
            op.flags("lamber,quiet,onlyfl,planck") #Calculate flux only during model run (faster). 
            op.ds().nmom = self.cfg.nmom
            op.ds().nstr = self.cfg.nstr
            op.ds().nphase = self.cfg.nmom
        if(self.cfg.multi_wave):
            lower, upper = self._compute_wn_bounds()
            op.wave_lower(lower)
            op.wave_upper(upper)
        else:
            #Broadband. Define range for planck function to a very wide range. 
            op.wave_lower([20]) #500 µm
            op.wave_upper([10000]) #1 µm
        n_waves = len(self.wavenumbers) if self.cfg.multi_wave else 1
        op.nwave(n_waves)
        op.ds().nlyr = len(self.grid.dtau)

        return op

    def _compute_wn_bounds(self):
        if(self.output_radiance):
            file = self.cfg.wn_bounds_out
        else:
            file = self.cfg.wn_bounds
        wn_boudns = np.loadtxt(file)
        if(len(wn_boudns) != len(self.wavenumbers)+1):
            raise ValueError(f"Expected {len(self.wavenumbers)+1} wavenumber bounds, got {len(wn_boudns)} from {file}.")
        lower_bounds = wn_boudns[:-1].tolist()  # lower edge for each bin
        upper_bounds = wn_boudns[1:].tolist()   # upper edge for each bin
        return lower_bounds, upper_bounds

    def disort_run(self,  T, mu, F ):
        #Solve RTE using DISORT for both visible and thermal bands.

        if(self.cfg.multi_wave):
            fbeam = torch.tensor(self.solar*F).unsqueeze(1) #Solar flux integrated within each wavenumber band. 
            temis = torch.full([len(self.wavenumbers)],1.0).unsqueeze(1)
            if(self.cfg.use_spec and not self.uniform_props):
                albedo = torch.tensor(1.0 - self.emiss_base).unsqueeze(1) #Albedo spectrum for the bottom boundary.
            else:
                albedo = torch.full([len(self.wavenumbers)],self.cfg.R_base).unsqueeze(1) 
        else:
            fbeam = torch.tensor([self.cfg.J * F]) #Broadband. Use solar constant. 
            temis = torch.tensor([1.0])
            albedo = torch.tensor([self.cfg.R_base])

        bc = {
            "umu0": torch.tensor([mu]), #Cosine of solar incidence angle
            "phi0": torch.tensor([0.0]), #Azimuth angle of solar incidence. 
            "albedo": albedo, #Albedo of bottom boundary
            "btemp": torch.tensor([T[-1]]), #Brightness temperature of bottom boundary. 
            "ttemp": torch.tensor([0.0]), #Top boundary temperature (space)
            "temis": temis, #Emissivity of top boundary (space)
            "fbeam": fbeam #Solar flux
        }
        
        #Need to interpolate temperature field to be at boundaries of computational layers, rather than at the center.
        #Assuming that the temperature grid is sufficiently dense to allow for a fast linear interpolation.  
        T_interp = np.interp(self.grid.x_boundaries,self.grid.x_RTE,T[1:self.grid.nlay_dust+1])
        T_tensor = torch.tensor(T_interp).unsqueeze(0)

        #Run disort. 
        result = self.ds.forward(self.prop,'', T_tensor, **bc)

        # Up stream minus down stream. 
        #F_net = result[0,0,:,0] - result[0,0,:,1] 
        # Calculate heating rate as flux divergence
        #Q_rad = np.diff(F_net.numpy()) / self.grid.dtau
        if(self.output_radiance):
            rad = self.ds.gather_rad()
            return(rad[:,0,0,0,0])
        else:
            #flux divergence from disort. According to documention, this is an exact result not from differencing of fluxes. 
            if(self.cfg.multi_wave):
                #Summation of the different wavelengths
                Q_rad = torch.sum(self.ds.gather_flx()[:,0,:,3],dim=0)
            else:
                Q_rad = self.ds.gather_flx()[0,0,:,3]
            Q_rad_interp = np.interp(self.grid.x_RTE,self.grid.x_boundaries,Q_rad)
            source = np.zeros(self.grid.x_num)
            source[1:self.grid.nlay_dust+1] = Q_rad_interp
            source_term = source * self.grid.K * self.cfg.q        
            return source_term

