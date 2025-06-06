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
        self.prop = self._setup_optical_properties() #Construct disort optical properties tensor
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

    def _setup_disort_options(self):
        """Configure DISORT options"""
        op = DisortOptions()
        op.flags("lamber,quiet,planck,onlyfl")
        op.wave_lower([self.cfg.wavenums[0]])
        op.wave_upper([self.cfg.wavenums[1]])
        op.nwave(1)
        op.ds().nlyr = len(self.grid.dtau)
        op.ds().nmom = self.cfg.nmom
        op.ds().nstr = self.cfg.nstr
        op.ds().nphase = self.cfg.nmom
        return op


    def disort_run(self,  T, mu, F ,R_base=0.0):
        #Solve RTE using DISORT for both visible and thermal bands.
        
        bc = {
            "umu0": torch.tensor([mu]), #Cosine of solar incidence angle
            "phi0": torch.tensor([0.0]), #Azimuth angle of solar incidence. 
            "albedo": torch.tensor([R_base]), #Albedo of bottom boundary
            "btemp": torch.tensor([T[-1]]), #Brightness temperature of bottom boundary. 
            "ttemp": torch.tensor([0.0]), #Top boundary temperature (space)
            "temis": torch.tensor([1.0]), #Emissivity of top boundary (space)
            "fisot": torch.tensor([0.0]), #Diffusve flux term at top. Could be self-heating. 
            "fbeam": torch.tensor([self.cfg.J * F]) #Solar flux
        }
        
        #Need to interpolate temperature field to be at boundaries of computational layers, rather than at the center.
        #Assuming that the temperature grid is sufficiently dense to allow for a fast linear interpolation.  
        T_interp = np.interp(self.grid.x_boundaries,self.grid.x_RTE,T[1:self.grid.nlay_dust+1])
        T_tensor = torch.tensor(T_interp).unsqueeze(0)

        #Run disort. 
        result = self.ds.forward(self.prop, '', T_tensor, **bc)

        # Up stream minus down stream. 
        F_net = result[0,0,:,0] - result[0,0,:,1] 
        
        # Calculate heating rate as flux divergence
        #Q_rad = np.diff(F_net.numpy()) / self.grid.dtau

        #flux divergence from disort. According to documention, this is an exact result not from differencing of fluxes. 
        Q_rad = self.ds.gather_flx()[0,0,:,3]
        Q_rad_interp = np.interp(self.grid.x_RTE,self.grid.x_boundaries,Q_rad)
        source = np.zeros(self.grid.x_num)
        source[1:self.grid.nlay_dust+1] = Q_rad_interp
        source_term = source * self.grid.K * self.cfg.q        
        return source_term

