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
                #First two layers have the same thickness (virtual and real nodes)
                x_nodes = [-dust_lthick/2., dust_lthick/2.] #First two nodes, virtual and real. 
                l_thick = [dust_lthick, dust_lthick]  # List to store layer thicknesses
                s *= cfg.spacing_factor  # increase spacing by factor

                # keep adding nodes until the next would go past L
                while x_nodes[-1] + s < L:
                    #Add layer thickness to our list. 
                    l_thick.append(s)
                    #Put the next node in the center of the next layer. 
                    x_nodes.append(x_nodes[-1] + 0.5*l_thick[-2] + 0.5*l_thick[-1])
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

            #Calculate dust nodes. All layers have same thickness. 
            # Virtual top node and first real node in dust
            x = [-dust_lthick/2.0, dust_lthick/2.0] #virtual node and first node. 
            l_thick = [dust_lthick, dust_lthick] #Virtual node and first node have same thickness
            for i in range(2, self.nlay_dust+1):
                x.append(x[-1] + dust_lthick)
                l_thick.append(dust_lthick)


            s = rock_lthick_tau  #first rock layer thickness, tau units. 
            L = dust_tau + rock_tau #Total dust column thickness in tau units. 

            x.append(x[-1] + dust_lthick/2. + s/2.)  #First rock node
            l_thick.append(s)  
            s *= cfg.spacing_factor

            #Rock node spacing increased geometrically. 
            # keep adding nodes until the next would go past L
            while x[-1] + s < L:
                l_thick.append(s)
                x.append(x[-1] + 0.5*l_thick[-1] + 0.5*l_thick[-1])
                s *= cfg.spacing_factor  # increase spacing by factor
            
            last = L + (L - x[-1])
            x.append(last) #Add final virtual node. 
            l_thick.append(2*(L - x[-1]))
            x = np.array(x)
            x_num = len(x)
            self.l_thick = np.array(l_thick)  # Store layer thicknesses
            
        self.x = x.copy() #depth in tau units
        # Check for depth-dependent properties
        self._calculate_depth_dependent_properties()
        if self.rho_depth is not None:
            #We need to update the grid tau values to account for changing Et with changing density (with depth)
            #
            #First, calculate Et for each layer, which scales linearly with density
            #We assume that cfg.Et is corresponds to the surface density. 
            Et_depth = cfg.Et*(self.rho_depth/cfg.rho_surface) #Et value for each layer
            Et_depth[0] = Et_depth[1] #Force virtual nodes Et values to be equal to first node, for simplicity. 
            Et_depth[-1] = Et_depth[-2]
            #Convert lthick back to meters using the global Et value that was used in its creation. 
            l_thick_m = self.l_thick.copy()/cfg.Et
            self.l_thick = (l_thick_m) * Et_depth #Layer thickness in new depth-dependent Et units. 
            self.x = np.cumsum(self.l_thick) -0.5*self.l_thick - self.l_thick[0] #Layer node centers
            self.x_m = np.cumsum(l_thick_m) -0.5*l_thick_m - l_thick_m[0] #Layer node centers in meters. 
            #Calculate a new Et array value for the layer center node points, required later for converting back and forth between tau and meters. 
            cfg.Et = self.x/self.x_m

        # Store coordinates and counts
        self.x_RTE = self.x[1:-1].copy() if cfg.single_layer else self.x[1:self.nlay_dust+1].copy()
        self.x_orig = self.x.copy()
        self.x_num = x_num
        #self.dtau = np.insert(np.diff(self.x_RTE),0,self.x_RTE[0]*2) #Array of thickness values of computational layers, for disort. Assumed uniform thickness.  
        self.dtau = self.l_thick[1:self.nlay_dust+1]
        #Calculate coordinates at the edge of each RTE computational layers, for disort and hapke rte models. 
        self.x_boundaries = np.insert(np.cumsum(self.l_thick[1:self.nlay_dust+1]),0,0) #in tau units. 

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
    
    def _calculate_depth_dependent_properties(self):
        """
        Calculate depth-dependent density and thermal conductivity.
        Uses exponential profile for density and linear relationship for conductivity.
        Only applies to single-layer models.
        
        Returns:
            tuple: (density_array, conductivity_array) for all grid points
        """
        cfg = self.config
        
        if not cfg.depth_dependent_properties:
            self.rho_depth = None
            self.k_depth = None
            return
        
        # Convert depth coordinates from tau units to meters
        depths_m = self.x / cfg.Et
        
        # Calculate depth-dependent density using exponential profile
        # ρ(z) = ρ_deep - (ρ_deep - ρ_surface) * exp(-z/H)
        density_variation = (cfg.rho_deep - cfg.rho_surface) * np.exp(-depths_m / cfg.density_scale_height)
        rho_depth = cfg.rho_deep - density_variation
        
        # Calculate depth-dependent conductivity
        # k_dust(z) = k_deep - (k_deep - k_surface) * ((ρ_deep - ρ(z))/(ρ_deep - ρ_surface))
        # This ensures k varies linearly with the density variation
        density_factor = (cfg.rho_deep - rho_depth) / (cfg.rho_deep - cfg.rho_surface)
        k_depth = cfg.k_deep - (cfg.k_deep - cfg.k_surface) * density_factor
        self.rho_depth = rho_depth
        self.k_depth = k_depth
    
    def check_and_update_temperature_dependent_properties(self, T_current):
        """
        Check if temperature-dependent properties need updating and do so if needed.
        Only calculates new cp and/or k values if temperature change exceeds threshold.
        
        Args:
            T_current: Current temperature field array
            
        Returns:
            bool: True if matrix was updated, False otherwise
        """
        cfg = self.config
        
        if not cfg.temperature_dependent_properties or (not cfg.temp_dependent_cp and not cfg.temp_dependent_k):
            return False
        
        # Check if update is needed based on temperature changes since last update
        if hasattr(self, 'T_last_cp_update'):
            temp_changes = np.abs(T_current - self.T_last_cp_update)
            max_temp_change = np.max(temp_changes)
            
            if max_temp_change > cfg.temp_change_threshold:
                # Calculate new temperature-dependent properties
                cp_temp_dependent = None
                k_temp_dependent = None
                
                # Temperature-dependent heat capacity
                if cfg.temp_dependent_cp:
                    c0, c1, c2, c3, c4 = cfg.cp_coeffs
                    T = T_current
                    cp_temp_dependent = c0 + c1*T + c2*T**2 + c3*T**3 + c4*T**4

                    self.heat = cp_temp_dependent
                
                # Temperature-dependent thermal conductivity
                if cfg.temp_dependent_k:
                    # Start with base conductivity (depth-dependent if enabled, otherwise uniform)
                    if hasattr(self, 'k_depth') and self.k_depth is not None:
                        # Use depth-dependent base conductivity that was calculated at the start of the run
                        k_base = self.k_depth.copy()
                    else:
                        # Use uniform base conductivity
                        k_base = np.full(len(T_current), cfg.k_dust)
                    
                    # Add temperature-dependent radiative term: k_total = k_base * (1 + B*T³)
                    T = T_current
                    k_temp_dependent = k_base * (1.0 + cfg.k_temp_coeff * T**3)
                    
                    # Convert to thermal diffusion units
                    k_temp_dependent = k_temp_dependent * cfg.Et**2

                    # Update stored conductivity arrays
                    self.cond = k_temp_dependent.copy()
                    self.K = self.cond / (self.dens * self.heat) 

                    #Update q arrays if we're runing an RTE model (otherwise, q is not used)
                    if cfg.use_RTE:
                        cfg.q = 1 / (self.cond * cfg.Et)                
                        cfg.q_bound = np.interp(self.x_boundaries,self.x,cfg.q) 
                
                # Update the finite difference matrix with new properties
                self._update_fd_matrix(heat=cp_temp_dependent, cond=k_temp_dependent)
                
                # Store current temperatures as the last update time
                self.T_last_cp_update = T_current.copy()
                
                # print(f"Matrix updated: max temperature change = {max_temp_change:.2f} K > {cfg.temp_change_threshold:.2f} K")
                # if cfg.temp_dependent_cp:
                #     print(f"  Updated heat capacity: {cp_temp_dependent.min():.1f} - {cp_temp_dependent.max():.1f} J/kg/K")
                # if cfg.temp_dependent_k:
                #     k_display = k_temp_dependent / cfg.Et**2  # Convert back to W/m/K for display
                #     print(f"  Updated thermal conductivity: {k_display.min():.2e} - {k_display.max():.2e} W/m/K")
                return True
            else:
                # No update needed
                return False
        else:
            # First call - should not happen if initialization was done correctly
            print("Warning: T_last_cp_update not initialized")
            return False
    
    def _update_fd_matrix(self, dens=None, heat=None, cond=None):
        """
        Update the finite difference matrix with new material properties.
        Used for temperature-dependent property updates.
        
        Args:
            dens: Updated density array (optional, uses self.dens if None)
            heat: Updated heat capacity array (optional, uses self.heat if None)  
            cond: Updated conductivity array (optional, uses self.cond if None)
        """
        cfg = self.config
        
        # Use provided arrays or fall back to stored arrays
        if dens is None:
            dens = self.dens
        if heat is None:
            heat = self.heat
        if cond is None:
            cond = self.cond
            
        lthick = self.l_thick
        dt = self.dt
        x_num = len(self.x)
        
        # Recalculate finite-difference stencil coefficients
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

        # Rebuild banded matrix
        self.diag = fd1d_heat_implicit_diagonal_nonuniform_kieffer(x_num, A1, A2, A3)
        
        # Update stored heat capacity if it was modified
        if heat is not self.heat:
            self.heat = heat.copy()

        if cond is not self.cond:
            self.cond = cond.copy()
        
        #print("Finite difference matrix updated for temperature-dependent properties")
    

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
            
            
            if self.rho_depth is not None and self.k_depth is not None:
                # Use depth-dependent properties
                dens[:] = self.rho_depth
                cond[:] = self.k_depth * cfg.Et**2  # Convert to thermal diffusion units
                K[:] = self.k_depth * cfg.Et**2 / (self.rho_depth * cfg.cp_dust)  # Diffusivity
                heat[:] = cfg.cp_dust  # Start with uniform heat capacity (may be updated later)
                #Need to update q, which was conputed in the config init. 
                cfg.q = 1.0 / (self.k_depth * cfg.Et) #Depth-dependent q array of size x_num. 
                cfg.q_bound = np.interp(self.x_boundaries,self.x,cfg.q) #Need a version of q array at layer boundaries for hapke. 
                print(f"Using depth-dependent properties:")
                print(f"  Surface density: {self.rho_depth[0]:.1f} kg/m³")
                print(f"  Deep density: {self.rho_depth[-1]:.1f} kg/m³")
                print(f"  Surface conductivity: {self.k_depth[0]:.2e} W/m/K")
                print(f"  Deep conductivity: {self.k_depth[-1]:.2e} W/m/K")
            else:
                # Use uniform properties
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
            method1 = cfg.dtfac*np.min(lthick / K/ cfg.Et)
            method2 = np.min(lthick**2. / K)
            dt_stability = np.max((method1,method2))
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
        
        # Initialize temperature-dependent property tracking
        if cfg.temperature_dependent_properties:
            self.temp_dependent_enabled = True
            # Initialize with starting temperature for first calculation
            T_initial = np.full(x_num, cfg.T_bottom)
            
            heat_temp_dependent = None
            k_temp_dependent = None
            
            if cfg.temp_dependent_cp:
                # Calculate initial temperature-dependent heat capacity
                c0, c1, c2, c3, c4 = cfg.cp_coeffs
                T = T_initial
                heat_temp_dependent = c0 + c1*T + c2*T**2 + c3*T**3 + c4*T**4
                print(f"Initial temperature-dependent heat capacity calculated (T={cfg.T_bottom:.1f}K)")
                print(f"  cp range: {heat_temp_dependent.min():.1f} - {heat_temp_dependent.max():.1f} J/kg/K")
                self.heat = heat_temp_dependent.copy()
                
            if cfg.temp_dependent_k:
                # Calculate initial temperature-dependent thermal conductivity
                # Start with base conductivity (depth-dependent if enabled, otherwise uniform)
                if hasattr(self, 'k_depth') and self.k_depth is not None:
                    k_base = self.k_depth.copy()
                else:
                    k_base = np.full(x_num, cfg.k_dust)
                
                # Add temperature-dependent radiative term: k_total = k_base * (1 + B*T³)
                T = T_initial
                k_temp_dependent = k_base * (1.0 + cfg.k_temp_coeff * T**3)
                
                # Convert to thermal diffusion units for finite difference
                k_temp_dependent_fd = k_temp_dependent * cfg.Et**2
                
                # Update stored conductivity arrays
                self.cond = k_temp_dependent_fd.copy()
                self.K = k_temp_dependent_fd / (dens * heat)

                #Update q arrays, shouldn't need to do this for RTE cases where q is actually used. 
                if self.cfg.use_RTE:
                    cfg.q = 1/ (self.cond * cfg.Et)                
                    cfg.q_bound = np.interp(self.x_boundaries,self.x,cfg.q)
                
                print(f"Initial temperature-dependent thermal conductivity calculated (T={cfg.T_bottom:.1f}K)")
                print(f"  k range: {k_temp_dependent.min():.2e} - {k_temp_dependent.max():.2e} W/m/K")
                
                # Pass the diffusion units version to matrix update
                k_temp_dependent = k_temp_dependent_fd
            
            # Update matrix with new properties if any were calculated
            if heat_temp_dependent is not None or k_temp_dependent is not None:
                self._update_fd_matrix(heat=heat_temp_dependent, cond=k_temp_dependent)
                
            # Store the temperature field from when properties were last updated
            self.T_last_cp_update = T_initial.copy()
            print("Temperature-dependent property tracking initialized")
        else:
            self.temp_dependent_enabled = False
        