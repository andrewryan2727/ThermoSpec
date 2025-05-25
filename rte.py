import numpy as np
from scipy.integrate import solve_bvp
from scipy.linalg import solve_banded

from config import SimulationConfig
from grid import LayerGrid
from stencils import build_bvp_stencil, build_jacobian_vis_banded, build_jacobian_therm_banded
from bvp_solvers import solve_bvp_vis, solve_bvp_therm

# -----------------------------------------------------------------------------
# File: rte.py
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
        self.custom_bvp = config.custom_bvp

        # Precompute stencil and Jacobians
        if config.single_layer:
            self.A_im1, self.A_i, self.A_ip1, self.h0, self.hN = build_bvp_stencil(grid.x, grid.nlay_dust)
        else:
            self.A_im1, self.A_i, self.A_ip1, self.h0, self.hN = build_bvp_stencil(grid.x, grid.nlay_dust)
        A = (self.A_im1, self.A_i, self.A_ip1)
        self.J_vis_ab   = build_jacobian_vis_banded((self.A_im1, self.A_i, self.A_ip1, self.h0, self.hN), config.gamma_vis)
        self.J_therm_ab = build_jacobian_therm_banded((self.A_im1, self.A_i, self.A_ip1, self.h0, self.hN), config.gamma_therm, config.single_layer)
        self.A_bvp = A

        # Previous solutions
        n = grid.nlay_dust
        self.phi_vis_prev   = np.zeros(n)
        self.dphi_vis_prev  = np.zeros(n)
        self.phi_therm_prev = np.zeros(n)
        self.dphi_therm_prev= np.zeros(n)

    def solve_visible(self, T, mu, F):
        x = self.grid.x_RTE
        if self.custom_bvp:
            u0 = self.phi_vis_prev.copy()
            self.phi_vis_prev = solve_bvp_vis(
                x, self._vis_fun, u0,
                self.J_vis_ab, self.A_bvp, self.h0, self.hN, T,
                tol=self.cfg.bvp_tol, max_iter=self.cfg.bvp_max_iter
            )
        else:
            sol = solve_bvp(self._vis_fun, self._vis_bc, x,
                            np.vstack((self.phi_vis_prev, self.dphi_vis_prev)), tol=self.cfg.bvp_tol)
            y = sol.sol(x)
            self.phi_vis_prev, self.dphi_vis_prev = y[0], y[1]
        return self.phi_vis_prev

    def solve_thermal(self, T, mu, F):
        x = self.grid.x_RTE
        if self.custom_bvp:
            u0 = self.phi_therm_prev.copy()
            if self.cfg.single_layer:
                # Use Dirichlet boundary condition for single-layer. Heat flux is balanced at bottom boundary. 
                D = -self.cfg.sigma/np.pi * T[-1]**4
            else:
                # Use Dirichlet boundary condition for two-layer
                if(F>0):
                    D = -self.cfg.sigma/np.pi * T[self.grid.nlay_dust]**4 + self.phi_vis_prev[-1] + self.cfg.J*np.exp(-self.cfg.eta*x[-1]/mu)/(4*np.pi)
                else:
                    D = -self.cfg.sigma/np.pi * T[self.grid.nlay_dust]**4
            self.phi_therm_prev = solve_bvp_therm(
                x, self._therm_fun, u0,
                self.J_therm_ab, self.A_bvp,
                self.h0, self.hN, D,T,
                tol=self.cfg.bvp_tol, max_iter=self.cfg.bvp_max_iter
            )
        else:
            ode = lambda xx, C: self._therm_fun(xx, C, T)
            bc  = lambda Ca, Cb: self._therm_bc(Ca, Cb, T)
            sol = solve_bvp(ode, bc, x,
                            np.vstack((self.phi_therm_prev, self.dphi_therm_prev)), tol=self.cfg.bvp_tol)
            y = sol.sol(x)
            self.phi_therm_prev, self.dphi_therm_prev = y[0], y[1]
        return self.phi_therm_prev

    def compute_source(self, T, mu, F):
        self.mu = mu
        self.F  = F
        #visible RTE source
        if(F>0):
          #The sun is up. Do the calculations. 
          phi_vis = self.solve_visible(T, mu, F)
          source_vis = self.cfg.eta * self.cfg.gamma_vis**2 * self.cfg.q * (
            self.cfg.J * F * np.exp(-self.grid.x_RTE / mu) + 4*np.pi*phi_vis)
        else:
          #The sun is not up. Skip. 
          source_vis = np.zeros(len(self.grid.x_RTE))
          self.phi_vis_prev = np.zeros(len(self.grid.x_RTE))
        #Thermal RTE source
        phi_therm = self.solve_thermal(T, mu, F)
        # Obtain second derivative - use appropriate temperature range based on mode
        if self.cfg.single_layer:
            _, d2 = self._therm_fun(self.grid.x_RTE, (phi_therm, 0.0), T[1:-1])
        else:
            _, d2 = self._therm_fun(self.grid.x_RTE, (phi_therm, 0.0), T[1:self.grid.nlay_dust+1])
        source_therm = np.pi * self.cfg.q * d2
        source = np.zeros(self.grid.x_num)
        if self.cfg.single_layer:
            source[1:-1] = source_vis + source_therm
        else:
            source[1:self.grid.nlay_dust+1] = source_vis + source_therm
        source = source*self.grid.K
        return source

    # Internal ODE functions
    def _vis_fun(self, x, C, T):
        phi, dphidx = C
        source = self.cfg.J * self.cfg.ssalb_vis * np.exp(-x/self.mu) / (4*np.pi)
        d2 = 4.0*(self.cfg.gamma_vis**2 * phi - source)
        return [dphidx, d2]

    def _vis_bc(self, Ca, Cb):
        phi_a, dphidx_a = Ca
        phi_b, dphidx_b = Cb
        return np.array([phi_a - 0.5*dphidx_a, phi_b + 0.5*dphidx_b])

    def _therm_fun(self, x, C, T):
        phi, dphidx = C
        #temp = np.interp(x, self.grid.x_orig, T)
        #temp = T[1:self.grid.nlay_dust+1]
        d2 = 4.0*(self.cfg.gamma_therm**2)*(phi - (1/np.pi)*self.cfg.sigma*T**4)
        return [dphidx, d2]

    def _therm_bc(self, Ca, Cb, T):
        phi_a, dphidx_a = Ca
        phi_b, dphidx_b = Cb
        top = phi_a - 0.5*dphidx_a
        bottom = phi_b - (self.cfg.sigma/np.pi)*T[self.grid.nlay_dust]**4 + self.phi_vis_prev[-1]
        return np.array([top, bottom])
