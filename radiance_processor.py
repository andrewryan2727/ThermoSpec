"""
Unified radiance computation module for thermal simulation post-processing.

This module consolidates all radiance calculation functionality from the thermal model,
providing a clean interface for post-processing thermal simulation results.

Radiance calculations always use DISORT regardless of which solver was used for 
thermal evolution, since DISORT provides superior spectral radiance capabilities.

Author: Andrew J. Ryan, 2025
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import pickle
from copy import deepcopy
import warnings

from config import SimulationConfig
from grid import LayerGrid
from rte_disort import DisortRTESolver


class CraterRadianceProcessor:
    """
    Calculate summed crater radiance as seen by observers at different geometries.
    Creates DISORT instances dynamically for each facet with correct viewing angles.
    
    This class is moved from observer_radiance.py and integrated into the unified
    radiance processing system.
    """
    
    def __init__(self, config: SimulationConfig, grid: LayerGrid, observer_vectors: List[List[float]]):
        """
        Store configuration and observer vectors for dynamic DISORT creation.
        
        Args:
            config: Simulation configuration
            grid: Spatial grid
            observer_vectors: list of [x,y,z] observer direction vectors
        """
        self.config = config
        self.grid = grid
        self.observer_vectors = [np.array(vec) / np.linalg.norm(vec) for vec in observer_vectors]
        
        # Store observers without pre-creating DISORT instances
        self.observers = []
        for obs_vec in self.observer_vectors:
            self.observers.append({'vector': obs_vec})
    
    def compute_facet_visibility(self, crater_shadowtester, observer_vec: np.ndarray) -> np.ndarray:
        """
        Use existing ShadowTester to determine facet visibility from observer.
        
        Args:
            crater_shadowtester: ShadowTester instance from crater module
            observer_vec: normalized observer direction vector [x,y,z]
            
        Returns:
            visibility: fractional visibility (0-1) for each facet
        """
        # Use the existing illuminated_facets method with observer vector instead of sun vector
        return crater_shadowtester.illuminated_facets(observer_vec)
    
    def compute_facet_observer_angles(self, crater_mesh, observer_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate local viewing angles for each facet relative to observer.
        Uses precomputed local coordinate system from crater_mesh for consistency.
        
        Args:
            crater_mesh: CraterMesh object with precomputed tangent vectors
            observer_vec: normalized observer direction vector
            
        Returns:
            facet_mu: cosine of local viewing angle for each facet
            facet_phi: local azimuth angle for each facet (in facet reference frame)
        """
        # Dot product for mu
        facet_mu = np.dot(crater_mesh.normals, observer_vec)
        facet_mu[facet_mu < 0] = 0.0
        
        # Use precomputed tangent vectors for phi calculation
        facet_phi = np.zeros(len(crater_mesh.normals))
        
        for i in range(len(crater_mesh.normals)):
            if facet_mu[i] <= 0:
                continue
                
            # Project observer into facet plane
            obs_in_plane = observer_vec - facet_mu[i] * crater_mesh.normals[i]
            obs_in_plane_norm = np.linalg.norm(obs_in_plane)
            
            if obs_in_plane_norm > 1e-10:
                obs_in_plane = obs_in_plane / obs_in_plane_norm
                cos_phi = np.dot(obs_in_plane, crater_mesh.tangent1[i])
                sin_phi = np.dot(obs_in_plane, crater_mesh.tangent2[i])
                facet_phi[i] = np.arctan2(sin_phi, cos_phi)
                if facet_phi[i] < 0:
                    facet_phi[i] += 2 * np.pi
            else:
                # Observer is along the normal direction, phi is arbitrary
                facet_phi[i] = 0.0
        
        return facet_mu, facet_phi
    
    def create_disort_for_facet(self, mu: float, phi: float) -> Tuple:
        """
        Create DISORT instances for a specific viewing geometry.
        
        Args:
            mu: cosine of viewing angle
            phi: azimuth angle (radians)
            
        Returns:
            tuple: (disort_thermal, disort_vis) instances
        """
        disort_thermal = DisortRTESolver(
            self.config, self.grid, n_cols=1,
            output_radiance=True, planck=True,
            observer_mu=mu, observer_phi=phi, 
            solver_mode=self.config.output_radiance_mode, 
            spectral_component='thermal_only'
        )
        
        disort_vis = None
        if self.config.output_radiance_mode in ['two_wave', 'hybrid']:
            disort_vis = DisortRTESolver(
                self.config, self.grid, n_cols=1,
                output_radiance=True, planck=False,
                observer_mu=mu, observer_phi=phi,
                solver_mode=self.config.output_radiance_mode, 
                spectral_component='visible_only'
            )
        
        return disort_thermal, disort_vis
    
    def compute_crater_radiance(self, T_crater_facets: np.ndarray, crater_mesh, crater_shadowtester,
                               observer_idx: int, mu_sun: float = 0.0, F_sun: float = 0.0, 
                               sun_vec: Optional[np.ndarray] = None, crater_radtrans=None, 
                               therm_flux_facets: Optional[np.ndarray] = None, 
                               illuminated: Optional[np.ndarray] = None, albedo: Optional[float] = None, 
                               emissivity: Optional[float] = None) -> Union[np.ndarray, float]:
        """
        Calculate total crater radiance as seen by a specific observer.
        Creates DISORT instances dynamically for each facet with correct viewing angles.
        
        Args:
            T_crater_facets: temperature profiles for all facets [depth, n_facets]
            crater_mesh: CraterMesh object
            crater_shadowtester: ShadowTester object 
            observer_idx: index of observer
            mu_sun: solar cosine (for scattered light calculation)
            F_sun: solar illumination flag
            sun_vec: solar direction vector [x, y, z] (for scattered light calculation)
            crater_radtrans: CraterRadiativeTransfer object (for scattered light)
            therm_flux_facets: thermal flux from all facets [n_facets, n_waves]
            illuminated: illuminated fraction for each facet [n_facets]
            albedo: surface albedo (scalar or array)
            emissivity: surface emissivity (scalar or array)
            
        Returns:
            total_radiance: area-weighted summed radiance from all visible facets
                          - For multi_wave: array of shape [n_waves]
                          - For two-wave: scalar value
        """
        observer_vec = self.observers[observer_idx]['vector']
        
        # Get fractional visibility for each facet using existing ShadowTester
        visibility = self.compute_facet_visibility(crater_shadowtester, observer_vec)
        
        # Get local viewing angles for each facet
        facet_mu, facet_phi = self.compute_facet_observer_angles(crater_mesh, observer_vec)
        
        # Initialize radiance arrays based on output_radiance_mode
        if self.config.output_radiance_mode == 'multi_wave':
            # We need to get n_waves by creating a temporary DISORT instance
            temp_disort = DisortRTESolver(self.config, self.grid, n_cols=1, output_radiance=True, planck=True,
                                        solver_mode=self.config.output_radiance_mode)
            n_waves = len(temp_disort.wavenumbers)
            total_radiance = np.zeros(n_waves)
        elif self.config.output_radiance_mode == 'hybrid':
            # Hybrid mode: thermal wavelengths only
            temp_disort = DisortRTESolver(self.config, self.grid, n_cols=1, output_radiance=True, planck=True,
                                        solver_mode=self.config.output_radiance_mode, spectral_component='thermal_only')
            n_waves = len(temp_disort.wavenumbers)
            total_radiance = np.zeros(n_waves)
        else:
            n_waves = 1
            total_radiance = 0.0
        
        # Calculate scattered energy incident upon the surface (Q_scat and Q_selfheat)
        Q_scat_facets = np.zeros((len(crater_mesh.normals), n_waves))
        Q_selfheat_facets = np.zeros((len(crater_mesh.normals), n_waves))
        
        if (crater_radtrans is not None and sun_vec is not None and 
            illuminated is not None and albedo is not None and 
            therm_flux_facets is not None):
            
            # Calculate scattered solar energy and self-heating for all facets
            _, Q_scat_all, Q_selfheat_all, _ = crater_radtrans.compute_fluxes(
                sun_vec, illuminated, therm_flux_facets, albedo, emissivity, 
                F_sun, n_waves, multiple_scatter=True
            )
            
            # Convert to intensity by dividing by pi (as done in main model)
            if n_waves == 1:
                Q_scat_facets[:, 0] = Q_scat_all / np.pi
                Q_selfheat_facets[:, 0] = Q_selfheat_all / np.pi
            else:
                Q_scat_facets = Q_scat_all / np.pi
                Q_selfheat_facets = Q_selfheat_all / np.pi
            
        total_projected_area = 0.0
        
        # Calculate radiance for each visible facet using per-facet DISORT approach
        print(f"Computing crater radiance for {len(crater_mesh.normals)} facets...")
        for i in range(len(crater_mesh.normals)):
            print(i)
            # Skip facets that are not visible or facing away from observer
            if visibility[i] <= 0 or facet_mu[i] <= 0:
                continue
                
            # Temperature profile for this facet
            T_facet = T_crater_facets[:, i]
            
            # Create DISORT instances for this facet's local viewing angles
            try:
                disort_thermal, disort_vis = self.create_disort_for_facet(facet_mu[i], facet_phi[i])
                
                # Get total scattered energy for this facet (Q_scat + Q_selfheat)
                if n_waves == 1:
                    Q_vis = Q_scat_facets[i, 0]
                    Q_therm = Q_selfheat_facets[i, 0]
                else:
                    Q_vis = Q_scat_facets[i, :]
                    Q_therm = Q_selfheat_facets[i, :]
                Q_total = Q_vis + Q_therm
                
                # Calculate radiance from this facet using facet-specific DISORT with scattered energy
                if self.config.output_radiance_mode in ['multi_wave', 'hybrid']:
                    # Multi-wave or hybrid case: return spectral radiance
                    if self.config.output_radiance_mode == 'multi_wave':
                        radiance, _ = disort_thermal.disort_run(T_facet, mu_sun, F_sun, Q=Q_total)
                    else:
                        # Hybrid mode: thermal wavelengths only, no sun. 
                        radiance, _ = disort_thermal.disort_run(T_facet, 0.0, 0.0, Q=Q_therm)
                    if hasattr(radiance, 'numpy'):
                        radiance = radiance.numpy()
                    # radiance should be shape [n_waves] or [n_waves, 1]
                    if radiance.ndim > 1:
                        facet_radiance = radiance[:, 0, 0, 0]  # Take first column if 2D
                    else:
                        facet_radiance = radiance
                else:
                    # Two-wave case: thermal + visible, return single value
                    rad_thermal, _ = disort_thermal.disort_run(T_facet, mu_sun, F_sun, Q=Q_therm)
                    
                    if hasattr(rad_thermal, 'numpy'):
                        rad_thermal = rad_thermal.numpy()
                        
                    # Ensure scalar values
                    rad_thermal = rad_thermal.item() if hasattr(rad_thermal, 'item') else rad_thermal
                    facet_radiance = rad_thermal
                    
            except Exception as e:
                print(f"Warning: DISORT failed for facet {i} (mu={facet_mu[i]:.3f}, phi={facet_phi[i]:.3f}): {e}")
                if self.config.output_radiance_mode in ['multi_wave', 'hybrid']:
                    facet_radiance = np.zeros(n_waves)
                else:
                    facet_radiance = 0.0
            
            # Weight by facet area, projected area (cosine factor), and visibility
            facet_area = crater_mesh.areas[i]
            projected_area = facet_area * facet_mu[i] * visibility[i]
            
            total_radiance += facet_radiance * projected_area
            total_projected_area += projected_area
        
        # Return area-averaged radiance
        if total_projected_area > 0:
            return total_radiance / total_projected_area
        else:
            if self.config.output_radiance_mode in ['multi_wave', 'hybrid']:
                return np.zeros(n_waves)
            else:
                return 0.0
    
    def compute_all_observers(self, T_crater_facets: np.ndarray, crater_mesh, crater_shadowtester,
                            mu_sun: float = 0.0, F_sun: float = 0.0, sun_vec: Optional[np.ndarray] = None, 
                            crater_radtrans=None, therm_flux_facets: Optional[np.ndarray] = None, 
                            illuminated: Optional[np.ndarray] = None, albedo: Optional[float] = None, 
                            emissivity: Optional[float] = None) -> np.ndarray:
        """
        Calculate crater radiance for all observers.
        
        Args:
            T_crater_facets: temperature profiles [depth, n_facets]
            crater_mesh: CraterMesh object
            crater_shadowtester: ShadowTester object
            mu_sun: solar cosine
            F_sun: solar illumination flag
            sun_vec: solar direction vector [x, y, z] (for scattered light calculation)
            crater_radtrans: CraterRadiativeTransfer object (for scattered light)
            therm_flux_facets: thermal flux from all facets [n_facets, n_waves]
            illuminated: illuminated fraction for each facet [n_facets]
            albedo: surface albedo (scalar or array)
            emissivity: surface emissivity (scalar or array)
            
        Returns:
            radiances: array of radiance for each observer
                      - For multi_wave: shape [n_observers, n_waves]  
                      - For two-wave: shape [n_observers]
        """
        if self.config.output_radiance_mode == 'multi_wave':
            # Get number of wavelengths by creating a temporary DISORT instance
            temp_disort = DisortRTESolver(self.config, self.grid, n_cols=1, output_radiance=True, planck=True,
                                        solver_mode=self.config.output_radiance_mode)
            n_waves = len(temp_disort.wavenumbers)
            radiances = np.zeros((len(self.observers), n_waves))
        elif self.config.output_radiance_mode == 'hybrid':
            # Hybrid mode: thermal wavelengths only
            temp_disort = DisortRTESolver(self.config, self.grid, n_cols=1, output_radiance=True, planck=True,
                                        solver_mode=self.config.output_radiance_mode, spectral_component='thermal_only')
            n_waves = len(temp_disort.wavenumbers)
            radiances = np.zeros((len(self.observers), n_waves))
        else:
            radiances = np.zeros(len(self.observers))
        
        for i in range(len(self.observers)):
            radiances[i] = self.compute_crater_radiance(
                T_crater_facets, crater_mesh, crater_shadowtester, i, mu_sun, F_sun,
                sun_vec, crater_radtrans, therm_flux_facets, illuminated, albedo, emissivity
            )
            
        return radiances


class RadianceProcessor:
    """
    Unified radiance processor for thermal simulation post-processing.
    
    This class consolidates functionality from modelmain.py, observer_radiance.py,
    and postprocessing/radiance_calculator.py into a single, clean interface.
    
    All radiance calculations use DISORT regardless of the thermal evolution solver.
    """
    
    def __init__(self, config: SimulationConfig, grid: Optional[LayerGrid] = None):
        """
        Initialize the radiance processor.
        
        Parameters:
        -----------
        config : SimulationConfig
            Simulation configuration
        grid : LayerGrid, optional
            Spatial grid. If None, will be reconstructed from config
        """
        self.config = config
        self.grid = grid if grid is not None else LayerGrid(config)
        
        # Cache for DISORT solvers to avoid recreating them
        self._disort_solvers = {}
        
    def calculate_surface_radiance(self, 
                                 thermal_results: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                 observer_angles: Optional[List[float]] = None,
                                 observer_mu: Optional[Union[float, List[float]]] = None,
                                 spectral_mode: Optional[str] = None,
                                 time_indices: Optional[List[int]] = None,
                                 solar_conditions: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Calculate radiance from flat surface using thermal simulation results.
        
        Parameters:
        -----------
        thermal_results : tuple
            (T_out, T_surf_out, t_out) from thermal simulation
        observer_angles : list of float, optional
            Observer zenith angles in degrees. Converted to mu = cos(angle)
        observer_mu : float or list of float, optional  
            Observer cosines directly. If provided, overrides observer_angles
        spectral_mode : str, optional
            Override config spectral mode ('two_wave', 'multi_wave', 'hybrid')
        time_indices : list of int, optional
            Which time indices to compute (default: all)
        solar_conditions : tuple, optional
            (mu_sun_array, F_sun_array) for solar illumination at each time step
            
        Returns:
        --------
        dict : Radiance calculation results
        """
        T_out, T_surf_out, t_out = thermal_results
        
        # Handle observer angles
        if observer_mu is not None:
            if isinstance(observer_mu, (int, float)):
                observer_mus = [float(observer_mu)]
            else:
                observer_mus = list(observer_mu)
        elif observer_angles is not None:
            observer_mus = [np.cos(np.radians(angle)) for angle in observer_angles]
        else:
            # Default to nadir viewing
            observer_mus = [getattr(self.config, 'observer_mu', 1.0)]
            
        # Handle time indices
        if time_indices is None:
            time_indices = list(range(T_out.shape[1]))
            
        # Handle spectral mode override
        effective_mode = spectral_mode or getattr(self.config, 'output_radiance_mode', 'hybrid')
        
        # Handle solar conditions
        if solar_conditions is not None:
            mu_sun_array, F_sun_array = solar_conditions
        else:
            # Default solar conditions
            F_default = 1.0 if getattr(self.config, 'sun', True) else 0.0
            mu_sun_array = np.ones(len(time_indices))
            F_sun_array = np.full(len(time_indices), F_default)
        
        results = {
            'observer_mus': observer_mus,
            'observer_angles_deg': [np.degrees(np.arccos(mu)) for mu in observer_mus],
            'time_indices': time_indices,
            'times': t_out[time_indices],
            'spectral_mode': effective_mode,
            'config': self.config
        }
        
        # Initialize output arrays based on spectral mode and number of observers
        n_obs = len(observer_mus)
        n_times = len(time_indices)
        
        if effective_mode == 'multi_wave':
            # Get number of wavelengths from solver
            solver = self._get_disort_solver(observer_mus[0], 0.0, effective_mode, 'thermal_only')
            n_waves = len(solver.wavenumbers)
            results['wavenumbers'] = solver.wavenumbers.copy()
            results['radiance'] = np.zeros((n_obs, n_waves, n_times))
            results['flux'] = np.zeros((n_obs, n_waves, n_times))
            results['brightness_temps'] = np.zeros((n_obs, n_times))
            
        elif effective_mode == 'hybrid':
            # Thermal wavelengths + broadband visible
            solver = self._get_disort_solver(observer_mus[0], 0.0, effective_mode, 'thermal_only')
            n_waves = len(solver.wavenumbers)
            results['wavenumbers'] = solver.wavenumbers.copy()
            results['radiance_thermal'] = np.zeros((n_obs, n_waves, n_times))
            results['radiance_visible'] = np.zeros((n_obs, n_times))
            results['flux_thermal'] = np.zeros((n_obs, n_waves, n_times))
            results['flux_visible'] = np.zeros((n_obs, n_times))
            results['brightness_temps'] = np.zeros((n_obs, n_times))
            
        else:  # two_wave
            results['radiance_thermal'] = np.zeros((n_obs, n_times))
            results['radiance_visible'] = np.zeros((n_obs, n_times))
            results['radiance_total'] = np.zeros((n_obs, n_times))
            results['flux_thermal'] = np.zeros((n_obs, n_times))
            results['flux_visible'] = np.zeros((n_obs, n_times))
            results['brightness_temps'] = np.zeros((n_obs, n_times))
        
        # Compute radiances for each observer and time
        print(f"Computing surface radiances for {n_obs} observers and {n_times} time points...")
        
        for obs_idx, mu in enumerate(observer_mus):
            if obs_idx % max(1, n_obs//4) == 0 or n_obs <= 4:
                print(f"  Observer {obs_idx+1}/{n_obs} (mu={mu:.3f}, angle={np.degrees(np.arccos(mu)):.1f}°)")
                
            for time_idx, t_idx in enumerate(time_indices):
                # Get temperature profile at this time
                T_profile = T_out[:, t_idx]
                
                # Get solar conditions for this time
                if t_idx < len(mu_sun_array) and t_idx < len(F_sun_array):
                    mu_sun = mu_sun_array[t_idx]
                    F_sun = F_sun_array[t_idx]
                else:
                    mu_sun = 1.0
                    F_sun = 1.0 if getattr(self.config, 'sun', True) else 0.0
                
                # Calculate radiance based on spectral mode
                if effective_mode == 'multi_wave':
                    rad, flux = self._calculate_multiwave_radiance(T_profile, mu, mu_sun, F_sun, effective_mode)
                    results['radiance'][obs_idx, :, time_idx] = rad
                    results['flux'][obs_idx, :, time_idx] = flux
                    # Calculate brightness temperature from radiance
                    results['brightness_temps'][obs_idx, time_idx] = self._fit_brightness_temperature(
                        rad, results['wavenumbers']
                    )
                    
                elif effective_mode == 'hybrid':
                    # Thermal: multi-wave with no solar
                    rad_th, flux_th = self._calculate_multiwave_radiance(T_profile, mu, 0.0, 0.0, effective_mode, 'thermal_only')
                    # Visible: broadband with solar if present
                    rad_vis, flux_vis = self._calculate_twowave_radiance(T_profile, mu, mu_sun, F_sun, 'visible_only')
                    
                    results['radiance_thermal'][obs_idx, :, time_idx] = rad_th
                    results['radiance_visible'][obs_idx, time_idx] = rad_vis
                    results['flux_thermal'][obs_idx, :, time_idx] = flux_th  
                    results['flux_visible'][obs_idx, time_idx] = flux_vis
                    results['brightness_temps'][obs_idx, time_idx] = self._fit_brightness_temperature(
                        rad_th, results['wavenumbers']
                    )
                    
                else:  # two_wave
                    rad_th, flux_th = self._calculate_twowave_radiance(T_profile, mu, mu_sun, F_sun, 'thermal_only')
                    rad_vis, flux_vis = self._calculate_twowave_radiance(T_profile, mu, mu_sun, F_sun, 'visible_only')
                    
                    results['radiance_thermal'][obs_idx, time_idx] = rad_th
                    results['radiance_visible'][obs_idx, time_idx] = rad_vis
                    results['radiance_total'][obs_idx, time_idx] = rad_th + rad_vis
                    results['flux_thermal'][obs_idx, time_idx] = flux_th
                    results['flux_visible'][obs_idx, time_idx] = flux_vis
                    # Use surface temperature for brightness temp in two-wave mode
                    results['brightness_temps'][obs_idx, time_idx] = T_surf_out[t_idx]
        
        print(f"Surface radiance calculation completed.")
        return results
    
    def calculate_crater_radiance(self,
                                thermal_results: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                crater_data: Dict[str, Any],
                                observer_vectors: List[List[float]],
                                spectral_mode: Optional[str] = None,
                                time_indices: Optional[List[int]] = None,
                                solar_conditions: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Calculate radiance from crater surface using thermal simulation results.
        
        Parameters:
        -----------
        thermal_results : tuple
            (T_crater_facets, T_surf_crater, t_out) from crater thermal simulation
            T_crater_facets shape: [depth, n_facets, n_times]
        crater_data : dict
            Crater geometry and thermal data containing 'crater_mesh', 'crater_shadowtester',
            'crater_radtrans', 'flux_therm_crater', 'illuminated', 'albedo', 'emissivity'
        observer_vectors : list of lists
            Observer direction vectors [[x,y,z], ...]
        spectral_mode : str, optional
            Override config spectral mode
        time_indices : list of int, optional
            Which time indices to compute
        solar_conditions : tuple, optional
            (mu_sun_array, F_sun_array) for solar illumination at each time step
            
        Returns:
        --------
        dict : Crater radiance results
        """
        T_crater_facets, T_surf_crater, t_out = thermal_results
        
        # Handle spectral mode override
        effective_mode = spectral_mode or getattr(self.config, 'output_radiance_mode', 'hybrid')
        
        # Handle time indices
        if time_indices is None:
            if T_crater_facets.ndim == 3:
                time_indices = list(range(T_crater_facets.shape[2]))
            else:
                time_indices = [0]  # Single time point for steady-state
                
        # Handle solar conditions
        if solar_conditions is not None:
            mu_sun_array, F_sun_array = solar_conditions
        else:
            # Default solar conditions
            F_default = 1.0 if getattr(self.config, 'sun', True) else 0.0
            mu_sun_array = np.ones(len(time_indices))
            F_sun_array = np.full(len(time_indices), F_default)
        
        # Create crater radiance processor
        crater_processor = CraterRadianceProcessor(self.config, self.grid, observer_vectors)
        
        # Initialize results structure
        n_obs = len(observer_vectors)
        n_times = len(time_indices)
        
        results = {
            'observer_vectors': observer_vectors,
            'observer_angles_deg': [np.degrees(np.arccos(np.dot([0,0,1], vec/np.linalg.norm(vec)))) for vec in observer_vectors],
            'time_indices': time_indices,
            'times': t_out[time_indices] if hasattr(t_out, '__getitem__') else [t_out],
            'spectral_mode': effective_mode,
            'config': self.config
        }
        
        # Initialize output arrays based on spectral mode
        if effective_mode == 'multi_wave':
            solver = DisortRTESolver(self.config, self.grid, n_cols=1, output_radiance=True, planck=True,
                                   solver_mode=effective_mode)
            n_waves = len(solver.wavenumbers)
            results['wavenumbers'] = solver.wavenumbers.copy()
            results['radiance'] = np.zeros((n_obs, n_waves, n_times))
            
        elif effective_mode == 'hybrid':
            solver = DisortRTESolver(self.config, self.grid, n_cols=1, output_radiance=True, planck=True,
                                   solver_mode=effective_mode, spectral_component='thermal_only')
            n_waves = len(solver.wavenumbers)
            results['wavenumbers'] = solver.wavenumbers.copy()
            results['radiance_thermal'] = np.zeros((n_obs, n_waves, n_times))
            results['radiance_visible'] = np.zeros((n_obs, n_times))
            
        else:  # two_wave
            results['radiance_thermal'] = np.zeros((n_obs, n_times))
            results['radiance_visible'] = np.zeros((n_obs, n_times))
            results['radiance_total'] = np.zeros((n_obs, n_times))
        
        # Extract crater data
        crater_mesh = crater_data['crater_mesh']
        crater_shadowtester = crater_data['crater_shadowtester'] 
        crater_radtrans = crater_data.get('crater_radtrans')
        flux_therm_crater = crater_data.get('flux_therm_crater')
        illuminated = crater_data.get('illuminated')
        albedo = crater_data.get('albedo')
        emissivity = crater_data.get('emissivity')
        
        # Compute radiances for each time step
        print(f"Computing crater radiances for {n_obs} observers and {n_times} time points...")
        
        for time_idx, t_idx in enumerate(time_indices):
            if time_idx % max(1, n_times//4) == 0 or n_times <= 4:
                print(f"  Time step {time_idx+1}/{n_times} (t={results['times'][time_idx]:.1f}s)")
            
            # Extract crater temperature profiles at this time
            if T_crater_facets.ndim == 3:
                T_crater_at_time = T_crater_facets[:, :, t_idx]  # [depth, n_facets]
            else:
                T_crater_at_time = T_crater_facets  # Already [depth, n_facets] for steady-state
            
            # Get solar conditions for this time
            if t_idx < len(mu_sun_array) and t_idx < len(F_sun_array):
                mu_sun = mu_sun_array[t_idx]
                F_sun = F_sun_array[t_idx]
            else:
                mu_sun = 1.0
                F_sun = 1.0 if getattr(self.config, 'sun', True) else 0.0
            
            # Create sun vector (approximate - could be passed in crater_data for accuracy)
            sun_vec = np.array([0, 0, 1]) if F_sun > 0 else None
            
            # Calculate crater radiance for all observers at this time
            observer_radiances = crater_processor.compute_all_observers(
                T_crater_at_time, crater_mesh, crater_shadowtester, mu_sun, F_sun,
                sun_vec, crater_radtrans, flux_therm_crater, illuminated, albedo, emissivity
            )
            
            # Store results
            if effective_mode == 'multi_wave':
                results['radiance'][:, :, time_idx] = observer_radiances
            elif effective_mode == 'hybrid':
                results['radiance_thermal'][:, :, time_idx] = observer_radiances
                # Note: visible component would need separate calculation
            else:  # two_wave
                results['radiance_thermal'][:, time_idx] = observer_radiances
        
        print(f"Crater radiance calculation completed.")
        return results
    
    def _get_disort_solver(self, observer_mu: float, observer_phi: float, 
                          spectral_mode: str, spectral_component: str = None) -> DisortRTESolver:
        """Get or create a DISORT solver with given parameters."""
        
        # Create cache key
        cache_key = f"{observer_mu:.6f}_{observer_phi:.6f}_{spectral_mode}_{spectral_component or 'all'}"
        
        if cache_key not in self._disort_solvers:
            self._disort_solvers[cache_key] = DisortRTESolver(
                self.config, self.grid, n_cols=1, 
                output_radiance=True, planck=True,
                observer_mu=observer_mu, observer_phi=observer_phi,
                solver_mode=spectral_mode, spectral_component=spectral_component
            )
        
        return self._disort_solvers[cache_key]
    
    def _calculate_multiwave_radiance(self, T_profile: np.ndarray, observer_mu: float,
                                    mu_sun: float, F_sun: float, spectral_mode: str,
                                    spectral_component: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate multi-wavelength radiance and flux using DISORT."""
        
        solver = self._get_disort_solver(observer_mu, 0.0, spectral_mode, spectral_component)
        
        radiance, flux = solver.disort_run(T_profile, mu_sun, F_sun)
        
        # Convert torch tensors to numpy if needed
        if hasattr(radiance, 'numpy'):
            radiance = radiance.numpy()
        if hasattr(flux, 'numpy'):
            flux = flux.numpy()
        
        # Extract surface values (first column if 2D)
        if radiance.ndim > 1:
            radiance = np.squeeze(radiance)
        if flux.ndim > 1:
            flux = np.squeeze(flux)
            
        return radiance, flux
            
    def _calculate_twowave_radiance(self, T_profile: np.ndarray, observer_mu: float,
                                  mu_sun: float, F_sun: float, spectral_component: str) -> Tuple[float, float]:
        """Calculate two-wave (broadband) radiance and flux using DISORT."""
        
        solver = self._get_disort_solver(observer_mu, 0.0, 'two_wave', spectral_component)
        
        radiance, flux = solver.disort_run(T_profile, mu_sun, F_sun)
        
        # Convert and extract scalar values
        if hasattr(radiance, 'numpy'):
            radiance = radiance.numpy()
        if hasattr(flux, 'numpy'):
            flux = flux.numpy()
            
        # Ensure scalar output
        radiance = float(radiance.item() if hasattr(radiance, 'item') else radiance)
        flux = float(flux.item() if hasattr(flux, 'item') else flux)
        
        return radiance, flux
    
    def _fit_brightness_temperature(self, radiance_spectrum: np.ndarray, 
                                   wavenumbers: np.ndarray) -> float:
        """
        Fit brightness temperature to radiance spectrum.
        
        Uses the existing blackbody fitting functions from modelmain.py when available.
        """
        try:
            # Try to use the existing max_btemp_blackbody function
            from modelmain import max_btemp_blackbody
            
            # Create wavenumber bounds (approximate)
            dwn = np.diff(wavenumbers)
            dwn = np.append(dwn, dwn[-1])  # Extend last bin width
            wn_bounds = np.zeros(len(wavenumbers) + 1)
            wn_bounds[1:-1] = wavenumbers[:-1] + dwn[:-1]/2
            wn_bounds[0] = wavenumbers[0] - dwn[0]/2
            wn_bounds[-1] = wavenumbers[-1] + dwn[-1]/2
            
            # Create a minimal sim-like object for the function
            class MinimalSim:
                def __init__(self, config, wavenumbers):
                    self.cfg = config
                    self.wavenumbers_out = wavenumbers
            
            sim_like = MinimalSim(self.config, wavenumbers)
            T_fit, _, _, _ = max_btemp_blackbody(sim_like, wn_bounds, radiance_spectrum)
            return float(T_fit)
            
        except (ImportError, AttributeError, Exception):
            # Fallback: simple Stefan-Boltzmann approximation
            try:
                # Approximate total radiance by integrating spectrum
                total_radiance = np.trapz(radiance_spectrum, wavenumbers * 100)  # Convert cm^-1 to m^-1
                sigma = 5.670374419e-8  # Stefan-Boltzmann constant
                T_bright = (total_radiance * np.pi / sigma) ** 0.25
                return float(T_bright)
            except:
                # Ultimate fallback
                return 300.0


# Main interface functions for easy use
def calculate_radiances_from_results(thermal_results_or_sim: Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, Any], object],
                                   config: Optional[SimulationConfig] = None,
                                   observer_angles: Optional[List[float]] = None,
                                   observer_mu: Optional[Union[float, List[float]]] = None,
                                   observer_vectors: Optional[List[List[float]]] = None,
                                   surface_type: str = 'smooth',
                                   spectral_mode: Optional[str] = None,
                                   time_indices: Optional[List[int]] = None,
                                   grid: Optional[LayerGrid] = None,
                                   solar_conditions: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                   crater_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Calculate radiances from thermal simulation results using DISORT.
    
    This is the main convenience function for post-processing thermal simulation results.
    Works regardless of which solver was used for thermal evolution.
    Supports both smooth surface and crater radiance calculations, including dual outputs.
    
    **FLEXIBLE INTERFACE: Supports both simulation objects and manual data**
    
    Parameters:
    -----------
    thermal_results_or_sim : Simulator object, tuple, or dict
        **Option 1 (EASIEST)**: Simulator object - automatically extracts all needed data
        **Option 2**: Tuple (T_out, T_surf_out, t_out) for manual data
        **Option 3**: Dict with 'smooth'/'crater' keys for dual outputs
    config : SimulationConfig, optional
        Simulation configuration (auto-extracted from sim if not provided)
    observer_angles : list of float, optional
        Observer zenith angles in degrees (for smooth surface viewing)
    observer_mu : float or list of float, optional
        Observer cosines (overrides observer_angles if provided)
    observer_vectors : list of lists, optional
        Observer direction vectors [[x,y,z], ...] for crater observations
    surface_type : str, default 'smooth'
        Type of surface calculation: 'smooth', 'crater', or 'both'
    spectral_mode : str, optional
        Override spectral mode ('two_wave', 'multi_wave', 'hybrid')  
    time_indices : list of int, optional
        Which time steps to compute (default: all)
    grid : LayerGrid, optional
        Spatial grid (auto-extracted from sim if not provided)
    solar_conditions : tuple, optional
        (mu_sun_array, F_sun_array) for each time step (auto-extracted from sim if not provided)
    crater_data : dict, optional
        Crater geometry and thermal data (auto-extracted from sim if not provided)
        
    Returns:
    --------
    dict : Radiance calculation results
        For surface_type='smooth': standard smooth surface results
        For surface_type='crater': crater radiance results 
        For surface_type='both': {'smooth': {...}, 'crater': {...}}
        
    Examples:
    ---------
    # **NEW EASY WAY**: Direct from simulation object
    sim = Simulator(config)
    T_out, phi_vis, phi_therm, T_surf, t_out = sim.run()
    
    # Smooth surface radiance
    results = calculate_radiances_from_results(sim, observer_angles=[0, 15, 30, 45, 60])
    
    # Crater radiance (auto-extracts crater data!)
    crater_results = calculate_radiances_from_results(
        sim, surface_type='crater', observer_vectors=[[0,0,1], [0.5,0,1]]
    )
    
    # Dual output comparison  
    dual_results = calculate_radiances_from_results(
        sim, surface_type='both', 
        observer_angles=[0, 30, 60], observer_vectors=[[0,0,1], [0.5,0,1]]
    )
    
    # **LEGACY WAY**: Manual data (still supported for flexibility)
    radiance_results = calculate_radiances_from_results(
        (T_out, T_surf, t_out), config, observer_angles=[0, 15, 30]
    )
    
    # Crater with manual data + crater_data
    crater_results = calculate_radiances_from_results(
        crater_thermal_results, config, surface_type='crater',
        observer_vectors=[[0,0,1]], crater_data=crater_geometry_data
    )
    """
    # Step 1: Detect input type and extract data accordingly
    is_sim_object = hasattr(thermal_results_or_sim, 'run') and hasattr(thermal_results_or_sim, 'cfg')
    
    if is_sim_object:
        # **SIMULATION OBJECT MODE**: Auto-extract everything
        sim = thermal_results_or_sim
        print(f"Auto-detected simulation object, extracting data...")
        
        # Extract config, grid, solar conditions automatically
        if config is None:
            config = sim.cfg
        if grid is None:
            grid = sim.grid
        if solar_conditions is None and hasattr(sim, 'mu_array') and hasattr(sim, 'F_array'):
            solar_conditions = (sim.mu_array, sim.F_array)
            
        # Extract thermal results based on surface type
        if surface_type == 'smooth':
            thermal_results = (sim.T_out, sim.T_surf_out, sim.t_out)
        elif surface_type == 'crater':
            if not hasattr(sim, 'T_crater_out'):
                raise ValueError("Simulation object does not have crater results. Set config.crater=True to enable crater modeling.")
            thermal_results = (sim.T_crater_out, sim.T_surf_crater_out, sim.t_out)
            # Auto-extract crater data
            if crater_data is None:
                crater_data = extract_crater_data_from_sim(sim)
        elif surface_type == 'both':
            if not hasattr(sim, 'T_crater_out'):
                raise ValueError("Simulation object does not have crater results. Set config.crater=True to enable crater modeling.")
            thermal_results = {
                'smooth': (sim.T_out, sim.T_surf_out, sim.t_out),
                'crater': (sim.T_crater_out, sim.T_surf_crater_out, sim.t_out)
            }
            # Auto-extract crater data
            if crater_data is None:
                crater_data = extract_crater_data_from_sim(sim)
        else:
            raise ValueError(f"Unknown surface_type: {surface_type}")
            
    else:
        # **MANUAL DATA MODE**: Use provided data as-is
        thermal_results = thermal_results_or_sim
        if config is None:
            raise ValueError("config parameter is required when providing manual thermal data")
    
    # Step 2: Validate config and grid
    if config is None:
        raise ValueError("config is required")
    
    processor = RadianceProcessor(config, grid)
    
    # Step 3: Process based on surface type
    if surface_type == 'smooth':
        # Standard smooth surface calculation
        if isinstance(thermal_results, dict) and 'smooth' in thermal_results:
            smooth_results = thermal_results['smooth']
        else:
            smooth_results = thermal_results
        return processor.calculate_surface_radiance(
            smooth_results, observer_angles, observer_mu, spectral_mode, time_indices, solar_conditions
        )
    
    elif surface_type == 'crater':
        # Crater-only calculation
        if isinstance(thermal_results, dict) and 'crater' in thermal_results:
            crater_thermal = thermal_results['crater']
        else:
            crater_thermal = thermal_results
            
        if crater_data is None:
            raise ValueError(
                "crater_data is required for crater radiance calculations. "
                "Either pass a simulation object with crater enabled, or provide crater_data manually."
            )
        if observer_vectors is None:
            observer_vectors = [[0, 0, 1]]  # Default nadir viewing
        return processor.calculate_crater_radiance(
            crater_thermal, crater_data, observer_vectors, spectral_mode, time_indices, solar_conditions
        )
    
    elif surface_type == 'both':
        # Dual output calculation
        if not isinstance(thermal_results, dict) or 'smooth' not in thermal_results or 'crater' not in thermal_results:
            raise ValueError("For surface_type='both', need either a simulation object with crater enabled, or thermal_results dict with 'smooth'/'crater' keys")
        
        results = {'surface_type': 'both'}
        
        # Calculate smooth surface radiance
        results['smooth'] = processor.calculate_surface_radiance(
            thermal_results['smooth'], observer_angles, observer_mu, spectral_mode, time_indices, solar_conditions
        )
        
        # Calculate crater radiance
        if crater_data is None:
            raise ValueError("crater_data is required for crater radiance calculations") 
        if observer_vectors is None:
            observer_vectors = [[0, 0, 1]]  # Default nadir viewing
        results['crater'] = processor.calculate_crater_radiance(
            thermal_results['crater'], crater_data, observer_vectors, spectral_mode, time_indices, solar_conditions
        )
        
        return results
    
    else:
        raise ValueError(f"Unknown surface_type: {surface_type}. Must be 'smooth', 'crater', or 'both'")


def calculate_radiances_from_file(file_path: str,
                                observer_settings: Dict[str, Any] = None,
                                config_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Calculate radiances from saved thermal simulation file using DISORT.
    
    Parameters:
    -----------
    file_path : str
        Path to saved thermal results (pickle file)
    observer_settings : dict, optional
        Observer configuration {'angles': [...], 'mu': [...], 'time_indices': [...], 'spectral_mode': '...'}
    config_overrides : dict, optional
        Configuration parameter overrides
        
    Returns:
    --------
    dict : Radiance calculation results
        
    Examples:
    ---------  
    # Basic usage
    results = calculate_radiances_from_file('thermal_results.pkl')
    
    # With custom observer angles
    results = calculate_radiances_from_file(
        'thermal_results.pkl',
        observer_settings={'angles': [0, 15, 30, 45, 60, 75]}
    )
    
    # Override config for different spectral mode
    results = calculate_radiances_from_file(
        'thermal_results.pkl',
        observer_settings={'mu': [1.0, 0.5, 0.2]},
        config_overrides={'output_radiance_mode': 'multi_wave'}
    )
    """
    # Load saved thermal data
    with open(file_path, 'rb') as f:
        saved_data = pickle.load(f)
    
    # Extract thermal results and config
    if 'thermal_results' in saved_data and 'config' in saved_data:
        thermal_results = saved_data['thermal_results']  
        config = saved_data['config']
        grid = saved_data.get('grid')
    elif hasattr(saved_data, 'T_out') and hasattr(saved_data, 'cfg'):
        # Handle simulation object directly
        thermal_results = (saved_data.T_out, saved_data.T_surf_out, saved_data.t_out)
        config = saved_data.cfg
        grid = getattr(saved_data, 'grid', None)
    else:
        raise ValueError("Saved file format not recognized. Expected 'thermal_results' and 'config' keys or simulation object.")
    
    # Apply config overrides
    if config_overrides:
        config = deepcopy(config)
        for key, value in config_overrides.items():
            setattr(config, key, value)
    
    # Extract observer settings
    observer_angles = None
    observer_mu = None
    time_indices = None
    spectral_mode = None
    solar_conditions = None
    
    if observer_settings:
        observer_angles = observer_settings.get('angles')
        observer_mu = observer_settings.get('mu')
        time_indices = observer_settings.get('time_indices')
        spectral_mode = observer_settings.get('spectral_mode')
        solar_conditions = observer_settings.get('solar_conditions')
    
    return calculate_radiances_from_results(
        thermal_results, config, observer_angles, observer_mu, spectral_mode, 
        time_indices, grid, solar_conditions
    )


# Backward compatibility functions
def add_radiance_to_simulation_results(sim, observer_angles: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Add radiance computation to existing simulation object for backward compatibility.
    
    Parameters:
    -----------
    sim : Simulator
        Completed simulation object
    observer_angles : list of float, optional
        Observer angles in degrees
        
    Returns:
    --------
    dict : Radiance results (also stored in sim object)
    """
    warnings.warn("add_radiance_to_simulation_results is deprecated. Use calculate_radiances_from_results directly.", 
                  DeprecationWarning, stacklevel=2)
    
    # Get thermal results from simulation
    thermal_results = (sim.T_out, sim.T_surf_out, sim.t_out)
    
    # Get solar conditions if available
    solar_conditions = None
    if hasattr(sim, 'mu_array') and hasattr(sim, 'F_array'):
        solar_conditions = (sim.mu_array, sim.F_array)
    
    # Calculate radiances
    radiance_results = calculate_radiances_from_results(
        thermal_results, sim.cfg, observer_angles=observer_angles, 
        grid=sim.grid, solar_conditions=solar_conditions
    )
    
    # Store in simulation object for backward compatibility
    sim._radiance_results = radiance_results
    
    # Add attributes to sim object to mimic old behavior
    if 'radiance' in radiance_results:
        sim.radiance_out = radiance_results['radiance'][0, :, :]  # First observer
        sim.flux_out = radiance_results['flux'][0, :, :]
        sim.wavenumbers_out = radiance_results['wavenumbers']
    elif 'radiance_thermal' in radiance_results:
        sim.radiance_out = radiance_results['radiance_thermal'][0, :, :]
        sim.flux_out = radiance_results['flux_thermal'][0, :, :]
        if 'wavenumbers' in radiance_results:
            sim.wavenumbers_out = radiance_results['wavenumbers']
    
    return radiance_results


def recompute_radiance_with_angles(sim, new_angles: List[float]) -> Dict[str, Any]:
    """
    Recompute radiances with different observer angles - perfect for Prem study style analysis.
    
    Parameters:
    -----------
    sim : Simulator  
        Completed simulation object
    new_angles : list of float
        New observer angles in degrees
        
    Returns:
    --------
    dict : New radiance results
    """
    thermal_results = (sim.T_out, sim.T_surf_out, sim.t_out)
    
    # Get solar conditions if available
    solar_conditions = None
    if hasattr(sim, 'mu_array') and hasattr(sim, 'F_array'):
        solar_conditions = (sim.mu_array, sim.F_array)
    
    return calculate_radiances_from_results(
        thermal_results, sim.cfg, observer_angles=new_angles, 
        grid=sim.grid, solar_conditions=solar_conditions
    )


def extract_crater_data_from_sim(sim) -> Optional[Dict[str, Any]]:
    """
    Extract crater geometry and thermal data from a simulation object.
    
    Parameters:
    -----------
    sim : Simulator
        Completed simulation object with crater enabled
        
    Returns:
    --------
    dict or None : Crater data dictionary or None if no crater data
    """
    if not hasattr(sim, 'crater_mesh') or sim.crater_mesh is None:
        return None
    
    crater_data = {
        'crater_mesh': sim.crater_mesh,
        'crater_shadowtester': getattr(sim, 'crater_shadowtester', None),
        'crater_radtrans': getattr(sim, 'crater_radtrans', None),
        'flux_therm_crater': getattr(sim, 'flux_therm_crater', None),
        'illuminated': getattr(sim, 'illuminated', None),
        'albedo': getattr(sim, 'crater_albedo', getattr(sim.cfg, 'albedo', 0.0)),
        'emissivity': getattr(sim, 'crater_emissivity', getattr(sim.cfg, 'em', 1.0))
    }
    
    return crater_data


def calculate_crater_radiance_from_sim(sim, observer_vectors: List[List[float]], 
                                     spectral_mode: Optional[str] = None,
                                     time_indices: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Calculate crater radiance directly from simulation object.
    
    Parameters:
    -----------
    sim : Simulator
        Completed simulation object with crater enabled
    observer_vectors : list of lists
        Observer direction vectors [[x,y,z], ...]
    spectral_mode : str, optional
        Override spectral mode
    time_indices : list of int, optional
        Which time indices to compute
        
    Returns:
    --------
    dict : Crater radiance results
    """
    if not hasattr(sim, 'T_crater_out'):
        raise ValueError("Simulation object does not have crater thermal results (T_crater_out)")
    
    # Extract crater thermal results
    crater_thermal_results = (sim.T_crater_out, sim.T_surf_crater_out, sim.t_out)
    
    # Extract crater data
    crater_data = extract_crater_data_from_sim(sim)
    if crater_data is None:
        raise ValueError("Simulation object does not have crater geometry data")
    
    # Get solar conditions if available
    solar_conditions = None
    if hasattr(sim, 'mu_array') and hasattr(sim, 'F_array'):
        solar_conditions = (sim.mu_array, sim.F_array)
    
    return calculate_radiances_from_results(
        crater_thermal_results, sim.cfg, 
        surface_type='crater',
        observer_vectors=observer_vectors,
        spectral_mode=spectral_mode,
        time_indices=time_indices,
        grid=sim.grid,
        solar_conditions=solar_conditions,
        crater_data=crater_data
    )


def calculate_dual_radiance_from_sim(sim, 
                                   observer_angles: Optional[List[float]] = None,
                                   observer_vectors: Optional[List[List[float]]] = None,
                                   spectral_mode: Optional[str] = None,
                                   time_indices: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Calculate both smooth and crater radiance from simulation object.
    
    Parameters:
    -----------
    sim : Simulator
        Completed simulation object with crater enabled
    observer_angles : list of float, optional
        Observer angles for smooth surface (degrees)
    observer_vectors : list of lists, optional
        Observer vectors for crater surface [[x,y,z], ...]
    spectral_mode : str, optional
        Override spectral mode
    time_indices : list of int, optional
        Which time indices to compute
        
    Returns:
    --------
    dict : Dual radiance results with 'smooth' and 'crater' keys
    """
    if not hasattr(sim, 'T_crater_out'):
        raise ValueError("Simulation object does not have crater thermal results (T_crater_out)")
    
    # Prepare thermal results
    smooth_thermal = (sim.T_out, sim.T_surf_out, sim.t_out)
    crater_thermal = (sim.T_crater_out, sim.T_surf_crater_out, sim.t_out)
    dual_thermal = {'smooth': smooth_thermal, 'crater': crater_thermal}
    
    # Extract crater data
    crater_data = extract_crater_data_from_sim(sim)
    if crater_data is None:
        raise ValueError("Simulation object does not have crater geometry data")
    
    # Default observer vectors if not provided
    if observer_vectors is None:
        observer_vectors = [[0, 0, 1], [0.5, 0, 1], [0, 0.5, 1]]  # Nadir, 30° off-nadir
    
    # Get solar conditions if available
    solar_conditions = None
    if hasattr(sim, 'mu_array') and hasattr(sim, 'F_array'):
        solar_conditions = (sim.mu_array, sim.F_array)
    
    return calculate_radiances_from_results(
        dual_thermal, sim.cfg,
        surface_type='both',
        observer_angles=observer_angles,
        observer_vectors=observer_vectors,
        spectral_mode=spectral_mode,
        time_indices=time_indices,
        grid=sim.grid,
        solar_conditions=solar_conditions,
        crater_data=crater_data
    )