import numpy as np
from scipy.integrate import solve_bvp

from config import SimulationConfig
from grid import LayerGrid
from stencils import build_bvp_stencil, build_jacobian_vis_banded, build_jacobian_therm_banded
from bvp_solvers import solve_bvp_vis, solve_bvp_therm
import scipy.integrate

# -----------------------------------------------------------------------------
# File: rte_hapke.py
# Description: Planetary regolith thermal model, including Hapke's 1996 radiative transfer 2-stream approximation. .
# Author: Andrew J. Ryan, 2025
#
# This code is free to use, modify, and distribute for any purpose.
# Please contact the author (ajryan4@arizona.edu) to discuss applications or if you
# use this code in your research or projects.
# -----------------------------------------------------------------------------

class RadiativeTransfer:
    """
    Encapsulates radiative-transfer solves for visible and thermal source terms.
    """
    def __init__(self, config: SimulationConfig, grid: LayerGrid):
        self.cfg = config
        self.grid = grid

        if config.RTE_solver == 'hapke':
            # Original 2-stream initialization
            if config.single_layer:
                self.A_im1, self.A_i, self.A_ip1, self.h0, self.hN = build_bvp_stencil(grid.x_boundaries, grid.nlay_dust)
            else:
                self.A_im1, self.A_i, self.A_ip1, self.h0, self.hN = build_bvp_stencil(grid.x_boundaries, grid.nlay_dust)
            A = (self.A_im1, self.A_i, self.A_ip1)
            self.J_vis_ab   = build_jacobian_vis_banded((self.A_im1, self.A_i, self.A_ip1, self.h0, self.hN), config.gamma_vis)
            self.J_therm_ab = build_jacobian_therm_banded((self.A_im1, self.A_i, self.A_ip1, self.h0, self.hN), config.gamma_therm, config.single_layer)
            self.A_bvp = A

        # Previous solutions
        n = grid.nlay_dust
        self.dphi_vis_prev  = np.zeros(n)
        self.dphi_therm_prev= np.zeros(n)

    def solve_visible(self,x,phi_vis_prev, T, mu, F, Q_vis):
        #Hapke visible 2-stream RTE solver
        phi_vis_new = solve_bvp_vis(
            x, self._vis_fun, phi_vis_prev,
            self.J_vis_ab, self.A_bvp, self.h0, self.hN, Q_vis, T,
            tol=self.cfg.bvp_tol, max_iter=self.cfg.bvp_max_iter
        )
        return phi_vis_new

    def solve_thermal(self,x,phi_therm_prev, T, mu, F, Q_therm):
        #Hapke thermal 2-stream RTE solver
        if self.cfg.single_layer:
            # Use Dirichlet boundary condition for single-layer. Heat flux is balanced at bottom boundary. 
            D = self.cfg.sigma/np.pi * T[-1]**4
        else:
            #Upstream value for lower boundary solver is defined by thermal emission from the substrate. 
            #D = self.cfg.sigma/np.pi * T[self.grid.nlay_dust]**4
            T_interface = calculate_interface_T(T, self.grid.nlay_dust, self.grid.alpha, self.grid.beta)
            D = self.cfg.sigma/np.pi * T_interface**4
        phi_therm_new = solve_bvp_therm(
            x, self._therm_fun, phi_therm_prev,
            self.J_therm_ab, self.A_bvp,
            self.h0, self.hN,D,Q_therm,T,self.cfg.single_layer,
            tol=self.cfg.bvp_tol, max_iter=self.cfg.bvp_max_iter
        )
        return phi_therm_new

    def compute_source(self,T_layers,x_RTE,x,phi_vis_prev,phi_therm_prev,mu, F, Q_therm=0.0, Q_vis=0.0):
        # Using Hapke RTE model. 
        self.mu = mu
        self.F = F
        #Interpolate temperature from layer centers to boundaries. 
        T = np.interp(x,x_RTE,T_layers[1:self.grid.nlay_dust+1])
        if F > 0 or Q_vis > 1.0e-3:
            phi_vis_new = self.solve_visible(x,phi_vis_prev,T, mu, F, Q_vis)
            F2 = F*(mu>0.001)
            source_vis = self.cfg.eta * self.cfg.gamma_vis**2 * self.cfg.q * (
                self.cfg.J * F2 * np.exp(-x / mu) + 4*np.pi*phi_vis_new)
        else:
            source_vis = np.zeros(len(x))
            phi_vis_new = np.zeros(len(x))
            
        phi_therm_new = self.solve_thermal(x,phi_therm_prev,T, mu, F, Q_therm)
        if self.cfg.single_layer:
            d2 = self._therm_fun(x, phi_therm_new, T)
        else:
            d2 = self._therm_fun(x, phi_therm_new, T[:self.grid.nlay_dust+1])
        source_therm = np.pi * self.cfg.q * d2

        if not self.cfg.single_layer:
            #Calculate boundary fluxes. 
            T_interface = calculate_interface_T(T, self.grid.nlay_dust, self.grid.alpha, self.grid.beta)
            therm_up = self.cfg.sigma / np.pi * T_interface**4
            #therm_up = self.cfg.sigma/np.pi * T[self.grid.nlay_dust]**4
            #downstream visible at bottom boundary is equal to phi_vis*2
            vis_down = 2.0 * phi_vis_new[-1]
            #downstream thermal at bottom boundary is therm_up + therm_down = phi_therm*2
            therm_down = 2.0 * phi_therm_new[-1] - therm_up
            # Vis diffuse down seems to consitently underestimate disort predictions by about 11–15%. Cause unclear. 
            # The equivalency of the other terms (direct beam, diffuse therm_down, and therm_up) were also verified against DISORT. 
            boundary_flux = ( therm_down*np.pi - therm_up*np.pi)
            if F > 0:
                boundary_flux += self.F*self.cfg.J * np.exp(-self.grid.x_boundaries[-1]/self.mu)
            if F > 0 or Q_vis > 1.0e-3:
                boundary_flux += vis_down*np.pi

        source_combined = source_vis + source_therm
        source_centers = np.interp(x_RTE,x,source_combined)
        source = np.zeros(self.grid.x_num)
        if self.cfg.single_layer:
            source[1:-1] = source_centers
            source *= self.grid.K
        else:
            source[1:self.grid.nlay_dust+1] = source_centers
            source *= self.grid.K
            source[self.grid.nlay_dust+1] = boundary_flux #store the boundary flux term here. 
        return source, phi_vis_new,phi_therm_new


    # Internal ODE functions
    def _vis_fun(self, x, phi, T):
        #Hapke visible RTE equation
        if(self.mu>0.001):
            source = self.cfg.J * self.cfg.ssalb_vis * np.exp(-x/self.mu) / (4*np.pi)
        else:
            source = 0.0
        d2 = 4.0*(self.cfg.gamma_vis**2 * phi - source)
        return d2

    def _therm_fun(self, x, phi, T):
        #Hapke thermal RTE equation
        d2 = 4.0*(self.cfg.gamma_therm**2)*(phi - (1/np.pi)*self.cfg.sigma*T**4)
        return d2

def planck_wn_integrated(wn_edges, T):
    """
    Integrate the Planck function over each wavenumber bin (edges in cm^-1).
    Returns band-integrated radiance (W/m^2/sr per band).
    """
    h = 6.62607015e-34  # Planck constant (J s)
    c = 2.99792458e8    # Speed of light (m/s)
    k = 1.380649e-23    # Boltzmann constant (J/K)
    def planck_wn(wn, T):
        wn_m = wn * 100.0  # Convert from cm^-1 to m^-1
        return (2 * h * c**2 * wn_m**3) / (np.exp(h * c * wn_m / (k * T)) - 1)
    B_bands = np.zeros(len(wn_edges)-1)
    for i in range(len(B_bands)):
        # Integrate over each bin
        B_bands[i], _ = scipy.integrate.quad(planck_wn, wn_edges[i], wn_edges[i+1], args=(T,), limit=100)
    # Removed division by bin width to match DISORT's band-integrated output
    return B_bands

def calculate_interface_T(T,i,alpha,beta):
    return((alpha*T[i] + beta*T[i+1])/(alpha + beta))