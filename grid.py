import numpy as np
from config import SimulationConfig
from stencils import (
    fd1d_heat_implicit_diagonal_nonuniform_kieffer,
    fd1d_heat_implicit_matrix_nonuniform_kieffer
)
from parse_input_files import load_mie_folder

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

                l_thick = [dust_lthick]  # List to store layer thicknesses

                # keep adding nodes until the next would go past L
                while x_nodes[-1] + s < L:
                    x_nodes.append(x_nodes[-1] + s)
                    l_thick.append(s)
                    s *= cfg.spacing_factor  # increase spacing by factor

                last = L + (L - x_nodes[-1])
                l_thick.append(2*(L - x_nodes[-1]))  # last layer thickness
                x_nodes.append(last)
   
                x = np.array(x_nodes)
                x_num = len(x)
                self.nlay_dust = x_num-2  # For RTE, exclude virtual top/bottom nodes
                self.l_thick = np.array(l_thick)  

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

                self.l_thick = np.zeros(x_num)
                self.l_thick[:] = dust_lthick  # Set layer thicknesses

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
            self.nlay_dust = nlay_dust_init

            # Ensure minimum number of nodes in dust column
            if(self.nlay_dust < cfg.min_nlay_dust):
                dust_lthick = dust_tau / cfg.min_nlay_dust
                self.nlay_dust = cfg.min_nlay_dust

            # Rock layer count and thickness in tau units
            rock_tau = cfg.rock_thickness * cfg.Et
            rock_lthick_tau = rock_lthick*cfg.Et


            # Virtual top node and first real node in dust
            x = [-dust_lthick / 2.0] #virtual node on top. 
            l_thick = [dust_lthick]
            x.append(dust_lthick / 2.0) #First real dust node. 
            l_thick.append(dust_lthick)  
            for i in range(2, self.nlay_dust+1):
                x.append(x[-1] + dust_lthick)
                l_thick.append(dust_lthick)


            s = rock_lthick_tau  #first rock layer thickness, tau units. 
            L = dust_tau + rock_tau #Total dust column thickness in tau units. 
            x_nodes = [-dust_lthick/2.] #virtual node. 

            x.append(x[-1] + dust_lthick/2. + s/2.)  #First rock node
            l_thick.append(s)  
            s *= cfg.spacing_factor

            #Rock node spacing increased geometrically. 
            # keep adding nodes until the next would go past L
            while x[-1] + s < L:
                x.append(x[-1] + s)
                l_thick.append(s)
                s *= cfg.spacing_factor  # increase spacing by factor
            
            last = L + (L - x[-1])
            x.append(last) #Add final virtual node. 
            l_thick.append(2*(L - x[-1]))
            x = np.array(x)
            x_num = len(x)
            self.l_thick = np.array(l_thick)  # Store layer thicknesses
            


        # Store coordinates and counts
        self.x = x #depth in tau units
        self.x_RTE = x[1:-1].copy() if cfg.single_layer else x[1:self.nlay_dust+1].copy()
        self.x_orig = x.copy()
        self.x_num = x_num
        self.dtau = np.insert(np.diff(self.x_RTE),0,self.x_RTE[0]*2) #Array of thickness values of computational layers, for disort. Assumed uniform thickness.  
        #Calculate coordinates at the edge of each computational layer, for disort. 
        x_boundaries = np.zeros(len(self.dtau)+1)
        for i in np.arange(len(self.dtau)):
            if(i!=0):
                x_boundaries[i] = x_boundaries[i-1] + self.dtau[i-1]
        x_boundaries[-1] = x_boundaries[-2] + self.dtau[-1]
        self.x_boundaries = x_boundaries

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
                if(nlay < cfg.min_nlay_dust):
                    #If we have less than the minimum number of required layers, increase the layer thickness to ensure sufficient resolution.
                    dust_lthick = (cfg.dust_thickness * cfg.Et) / cfg.min_nlay_dust
                #Rock layer thickness is based on skin depth
                #This often leads to excessively thick rock layers in a two-layer scenario, so we use a factor to reduce it.
                rock_lthick = cfg.rock_skin_depth * cfg.flay * cfg.rock_lthick_fac / cfg.Et #in meters

            if(cfg.use_RTE):
                # If using RTE, make sure we have sufficient resolution at the optical extinction scale. 
                # If auto-calculated layers are too coarse, force them to be finer. 
                dust_lthick = min(dust_lthick, cfg.dust_rte_max_lthick)
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
        # lthick = np.zeros(x_num)
        # lthick[1:-1] = (
        #     (x[1:-1] + 0.5*(x[2:] - x[1:-1]))
        #     - (x[1:-1] - 0.5*(x[1:-1] - x[0:-2]))
        # )
        # lthick[0] = x[1] - x[0]
        # lthick[-1] = x[-1] - x[-2]
        lthick = self.l_thick

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
            K[:self.nlay_dust+1] = K_dust
            cond[:self.nlay_dust+1] = cfg.k_dust * cfg.Et**2
            dens[:self.nlay_dust+1] = cfg.rho_dust
            heat[:self.nlay_dust+1] = cfg.cp_dust

            # Rock nodes
            K[self.nlay_dust+1:] = K_rock
            cond[self.nlay_dust+1:] = cfg.k_rock * cfg.Et**2
            dens[self.nlay_dust+1:] = cfg.rho_rock
            heat[self.nlay_dust+1:] = cfg.cp_rock

        self.K = K
        self.cond = cond
        self.dens = dens
        self.heat = heat

        self.alpha = 2.0*cond[self.nlay_dust] / self.l_thick[self.nlay_dust]  # Thermal diffusivity at dust-rock interface
        self.beta = 2.0*cond[self.nlay_dust+1] / self.l_thick[self.nlay_dust+1]  # Thermal diffusivity at rock-dust interface

        #Calculate time increment here. 
        if cfg.auto_dt:
            # Calculate dt based on stability criterion
            dt_stability = cfg.dtfac*np.min(lthick / K / cfg.Et)
            # Adjust dt to be nearly divisible by period
            steps_per_day = np.ceil(cfg.P / dt_stability)
            #Never fewer than 200 steps per day
            if(steps_per_day < cfg.minsteps):
                steps_per_day = cfg.minsteps
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
        