import numpy as np
from pydisort import Disort, DisortOptions, scattering_moments
import torch

from config import SimulationConfig
from grid import LayerGrid

torch.set_default_dtype(torch.float64)

class DisortRTESolver:
    def __init__(self, config: SimulationConfig, grid: LayerGrid):
        self.cfg = config
        self.grid = grid
        #Construct optical properties tensors
        if(self.cfg.multi_wave or self.cfg.depth_dependent):
            #More complicated set up for multiple wavelengths and/or depth-dependent properties
            self.prop = self._setup_optical_properties_advanced() 
        else:
            #Simple set up for globally uniform properties. 
            self.prop = self._setup_optical_properties() 
        self.op = self._setup_disort_options() #Initialize disort options
        self.ds = Disort(self.op) #initialize disort


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
        n_waves = len(self.grid.wavenumbers) if self.cfg.multi_wave else 1
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
                #Need to rewrite and finish this section! Incomplete! 
                if self.cfg.multi_wave:
                    ssa_layer = self.grid.ssalb_array[i_wave, :]  # Shape: (n_layers,)
                    g_layer = self.cfg.g_profile[i_wave, :]      # Shape: (n_layers,)
                    moments = torch.stack([
                        torch.tensor(self.cfg.moments_profile[i_wave, :, i_mom])
                        for i_mom in range(n_mom)
                    ], dim=-1)  # (n_layers, n_mom)
                else:
                    ssa_layer = self.grid.ssalb_array[:]          # Shape: (n_layers,)
                    g_layer = self.cfg.g_profile[:]              # Shape: (n_layers,)
                    moments = torch.stack([
                        torch.tensor(self.cfg.moments_profile[:, i_mom])
                        for i_mom in range(n_mom)
                    ], dim=-1)  # (n_layers, n_mom)
            else:
                # Uniform properties with depth
                if self.cfg.multi_wave:
                    #Calculate dtau at this wavelength using the extinction cross section Cext
                    node_depth_meters = (self.grid.x_RTE/self.cfg.Et)
                    Vp = (4./3.)*np.pi*self.cfg.radius**3.
                    n_p= self.cfg.fill_frac / Vp
                    Et = n_p * self.grid.Cext_array[i_wave]*1e-12 #extinction coefficient at this wavelength, converting Cext from µm^2 to m^2 in the process. 
                    tau = node_depth_meters*Et
                    dtau = np.insert(np.diff(tau),0,tau[0]*2)
                    tau_layer = torch.tensor(dtau,dtype=torch.float64)
                    ssalb_val = self.grid.ssalb_array[i_wave].copy()
                    ssa_layer = torch.full([n_layers], ssalb_val,dtype=torch.float64)
                    #g_layer = np.full(n_layers, self.cfg.g[i_wave])
                    # Moments same for all layers
                    moms = self.grid.alpha1_array[i_wave,:]
                    moms = scattering_moments(self.cfg.nmom, "henyey-greenstein", self.cfg.g)
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
        op.flags("lamber,quiet,onlyfl,planck")
        if(self.cfg.multi_wave):
            lower, upper = self._compute_wn_bounds()
            op.wave_lower(lower)
            op.wave_upper(upper)
        else:
            #Broadband. Define range for planck function to a very wide range. 
            op.wave_lower([20]) #500 µm
            op.wave_upper([10000]) #1 µm
        n_waves = len(self.grid.wavenumbers) if self.cfg.multi_wave else 1
        op.nwave(n_waves)
        op.ds().nlyr = len(self.grid.dtau)
        op.ds().nmom = self.cfg.nmom
        op.ds().nstr = self.cfg.nstr
        op.ds().nphase = self.cfg.nmom
        return op

    def _compute_wn_bounds(self):
        wn_centers = np.array(self.grid.wavenumbers)
        n_bins = len(wn_centers)

        # Compute edges first (N+1 edges)
        edges = np.zeros(n_bins + 1)

        # Interior edges
        edges[1:-1] = 0.5 * (wn_centers[:-1] + wn_centers[1:])
        #Boundaries
        edges[0] = wn_centers[0]
        edges[-1] = wn_centers[-1]

        lower_bounds = edges[:-1].tolist()  # lower edge for each bin
        upper_bounds = edges[1:].tolist()   # upper edge for each bin

        return lower_bounds, upper_bounds

    def disort_run(self,  T, mu, F ,R_base=0.0):
        #Solve RTE using DISORT for both visible and thermal bands.

        if(self.cfg.multi_wave):
            fbeam = torch.tensor(self.grid.solar*F).unsqueeze(1) #Solar flux integrated within each wavenumber band. 
            temis = torch.full([len(self.grid.wavenumbers)],1.0).unsqueeze(1)
            albedo = torch.full([len(self.grid.wavenumbers)],self.cfg.R_base).unsqueeze(1) #eventually change this to allow inport of rock reflectance spectrum?
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

