import numpy as np
from config import SimulationConfig
from stencils import (
    fd1d_heat_implicit_diagonal_nonuniform_kieffer,
    fd1d_heat_implicit_matrix_nonuniform_kieffer
)

# -----------------------------------------------------------------------------
# File: grid.py
# Description: Planetary regolith thermal model, including Hapke's 1996 radiative transfer 2-stream approximation. .
# Author: Andrew J. Ryan, 2025
#
# This code is free to use, modify, and distribute for any purpose.
# Please contact the author (ajryan4@arizona.edu) to discuss applications or if you
# use this code in your research or projects.
# -----------------------------------------------------------------------------

class LayerGrid:
    """
    Builds spatial discretization layers and finite-difference matrix for heat diffusion.
    """
    def __init__(self, config: SimulationConfig):
        self.config = config
        self._build_layers()
        self._build_fd_matrix()

    def _build_layers(self):
        cfg = self.config
        dust_lthick, rock_lthick = self._calculate_lthick()
        if cfg.single_layer:
            # Single layer case - use dust properties for everything
            if(cfg.geometric_spacing):
                #Spacing increases by geometric factor spacing_factor
                s = dust_lthick  #first layer spacing, tau units. 
                L = cfg.dust_thickness * cfg.Et #Total dust column thickness, converted to tau units
                x_nodes = [-dust_lthick/2.] #virtual node. 


                # keep adding nodes until the next would go past L
                while x_nodes[-1] + s < L:
                    x_nodes.append(x_nodes[-1] + s)
                    s *= cfg.spacing_factor  # increase spacing by factor

                last = L + (L - x_nodes[-1])
                x_nodes.append(last)
   
                x = np.array(x_nodes)
                x_num = len(x)
                self.nlay_dust = x_num-2  # For RTE, exclude virtual top/bottom nodes

            else:
                # Uniform layer thickness. 
                dust_tau = cfg.dust_thickness * cfg.Et
                nlay_dust_init = int(round(dust_tau / dust_lthick))
                # revise actual layer thickness to match integer layer count
                dust_lthick = dust_tau / nlay_dust_init
                self.nlay_dust = nlay_dust_init 
                
                # Total nodes (including virtual top/bottom)
                x_num = self.nlay_dust + 2
                x = np.zeros(x_num)
                
                # Virtual top node and first real node
                x[0] = -dust_lthick / 2.0
                x[1] = dust_lthick / 2.0
                
                # Remaining nodes
                for i in range(2, self.nlay_dust):
                    x[i] = x[i-1] + dust_lthick

                #Ensure last real node is in the correct position, accouting for changes due to rounding in conversion from m to tau. 
                x[-2] = cfg.dust_thickness * cfg.Et - dust_lthick / 2.0
                x[-1] = x[-2] + dust_lthick

        else:
            # Two-layer case - dust and rock layers
            # Calculate dust layer thicknesses in terms of optical depth tau
            dust_tau = cfg.dust_thickness * cfg.Et
            nlay_dust_init = int(round(dust_tau / dust_lthick))
            # revise actual layer thickness to match integer layer count
            dust_lthick = dust_tau / nlay_dust_init
            self.nlay_dust = nlay_dust_init + 1

            # Ensure minimum dust layer count
            if(self.nlay_dust < 10):
                dust_lthick = dust_tau / 10.0
                self.nlay_dust = 11

            # Rock layer count and thickness in tau units
            nlay_rock = int(round(cfg.rock_thickness / rock_lthick))
            rock_tau = cfg.rock_thickness * cfg.Et
            rock_lthick_tau = rock_tau / nlay_rock
            self.nlay_rock = nlay_rock

            # Total nodes (including virtual top/bottom)
            x_num = self.nlay_dust + self.nlay_rock + 3
            x = np.zeros(x_num)

            # Virtual top node and first real node in dust
            x[0] = -dust_lthick / 2.0
            x[1] = dust_lthick / 2.0
            for i in range(2, self.nlay_dust):
                x[i] = x[i-1] + dust_lthick
            
            # Node at dust/rock interface
            x[self.nlay_dust] = cfg.dust_thickness * cfg.Et

            # First two rock nodes
            x[self.nlay_dust+1] = 2.0*x[self.nlay_dust] - x[self.nlay_dust-1]
            x[self.nlay_dust+2] = x[self.nlay_dust+1] + 0.5*dust_lthick + 0.5*rock_lthick_tau
            # Remaining rock layers
            for i in range(self.nlay_dust+3, self.nlay_dust + self.nlay_rock + 3):
                x[i] = x[i-1] + rock_lthick_tau

            # Adjust bottom nodes exactly
            x[-2] = (cfg.dust_thickness + cfg.rock_thickness) * cfg.Et
            if x[-1] < x[-2]:
                x[-1] = x[-2] + rock_lthick_tau

        # Store coordinates and counts
        self.x = x
        self.x_RTE = x[1:-1].copy() if cfg.single_layer else x[1:self.nlay_dust+1].copy()
        self.x_orig = x.copy()
        self.x_num = x_num

    def _calculate_lthick(self):
        """
        Calculate layer thicknesses based on skin depth. 
        For single layer, returns dust layer thickness.
        For two layers, returns dust and rock layer thicknesses.
        """
        cfg = self.config
        if(cfg.auto_thickness):
            if cfg.single_layer:
                #Single layer cases. Simpler. Use dust properties for everything.
                dust_lthick = (cfg.dust_skin_depth * cfg.flay) #in tau units
                rock_lthick = cfg.rock_lthick #not used. 
            else:
                #Two layer case. 
                dust_lthick = (cfg.dust_skin_depth * cfg.flay) #in tau units
                nlay = cfg.dust_thickness * cfg.Et / dust_lthick #approx number of layers in dust column
                if(nlay < 10):
                    #If we have less than 10 layers, increase the layer thickness to ensure sufficient resolution.
                    dust_lthick = (cfg.dust_thickness * cfg.Et) / 10.0
                rock_lthick = cfg.rock_skin_depth * cfg.flay * 0.25 / cfg.Et #in meters

            if(cfg.use_RTE):
                # If using RTE, make sure we have sufficient resolution in the first optical depth. 
                #Currently setting the max first layer thickness as no more than 1/50 of optical depth. 
                dust_lthick = max(dust_lthick, 0.02)
        else:
            # Use user-defined layer thicknesses
            dust_lthick = cfg.dust_lthick
            rock_lthick = cfg.rock_lthick   
        return dust_lthick, rock_lthick
        

    def _build_fd_matrix(self):
        cfg = self.config
        x = self.x
        x_num = len(x)


        # Layer thickness array
        lthick = np.zeros(x_num)
        lthick[1:-1] = (
            (x[1:-1] + 0.5*(x[2:] - x[1:-1]))
            - (x[1:-1] - 0.5*(x[1:-1] - x[0:-2]))
        )
        lthick[0] = x[1] - x[0]
        lthick[-1] = x[-1] - x[-2]

        # Diffusivity for dust and rock
        K_dust = cfg.k_dust * cfg.Et**2 / (cfg.rho_dust * cfg.cp_dust)
        K_rock = cfg.k_rock * cfg.Et**2 / (cfg.rho_rock * cfg.cp_rock)

        # Initialize property arrays
        K = np.zeros(x_num)
        cond = np.zeros(x_num)
        dens = np.zeros(x_num)
        heat = np.zeros(x_num)

        if(cfg.single_layer):
            # Single layer case - all nodes are dust. All layers use dust properties.
            K[:] = K_dust
            cond[:] = cfg.k_dust * cfg.Et**2
            dens[:] = cfg.rho_dust
            heat[:] = cfg.cp_dust
        else:
            # Dust nodes
            K[:self.nlay_dust] = K_dust
            cond[:self.nlay_dust] = cfg.k_dust * cfg.Et**2
            dens[:self.nlay_dust] = cfg.rho_dust
            heat[:self.nlay_dust] = cfg.cp_dust

            # Interface harmonic mean
            K[self.nlay_dust] = 2/(1/K_dust + 1/K_rock)
            cond[self.nlay_dust] = 2/(1/(cfg.k_dust*cfg.Et**2) + 1/(cfg.k_rock*cfg.Et**2))
            dens[self.nlay_dust] = 2/(1/cfg.rho_dust + 1/cfg.rho_rock)
            heat[self.nlay_dust] = 2/(1/cfg.cp_dust + 1/cfg.cp_rock)

            # Rock nodes
            K[self.nlay_dust+1:] = K_rock
            cond[self.nlay_dust+1:] = cfg.k_rock * cfg.Et**2
            dens[self.nlay_dust+1:] = cfg.rho_rock
            heat[self.nlay_dust+1:] = cfg.cp_rock

        self.K = K

        #Calculate time increment here. 
        if cfg.auto_dt:
            # Calculate dt based on stability criterion
            dt_stability = np.min(10*lthick / K)
            # Adjust dt to be nearly divisible by period
            steps_per_day = np.ceil(cfg.P / dt_stability)
            #Never fewer than 200 steps per day
            if(steps_per_day < 200):
                steps_per_day = 200
            dt = cfg.P / steps_per_day
            self.steps_per_day = int(steps_per_day)
            self.dt = dt
        else:
            # Compute time increment
            t_num = cfg.tsteps_day * cfg.ndays
            dt = cfg.P * cfg.ndays / (t_num - 1)
            self.steps_per_day = cfg.tsteps_day
            self.dt = dt
        print(f"Time step: {dt:.6f} s, Steps per day: {self.steps_per_day}")

        # Finite-difference stencil (non-uniform)
        A1 = (
            2*dt*cond[1:-1]
            / (dens[1:-1]*heat[1:-1] * lthick[1:-1]**2
               * (1 + (lthick[0:-2]*cond[1:-1])/(lthick[1:-1]*cond[0:-2])))
        )
        A3 = (
            (1 + (lthick[0:-2]*cond[1:-1])/(lthick[1:-1]*cond[0:-2]))
            / (1 + (lthick[2:]*cond[1:-1])/(lthick[1:-1]*cond[2:]))
        )
        A2 = -1.0 * (1.0 + A3)

        # Build banded matrix
        self.diag = fd1d_heat_implicit_diagonal_nonuniform_kieffer(x_num, A1, A2, A3)
