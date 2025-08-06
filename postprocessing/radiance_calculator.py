"""
Standalone radiance calculation module for thermal simulation post-processing.

Extracts radiance calculations from main thermal simulation to enable
independent post-processing of saved thermal data for ML dataset generation.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from config import SimulationConfig
from grid import LayerGrid
from rte_disort import DisortRTESolver
from rte_hapke import RadiativeTransfer
from observer_radiance import ObserverRadianceCalculator
from crater import CraterMesh, ShadowTester, CraterRadiativeTransfer, SelfHeatingList


class StandaloneRadianceCalculator:
    """
    Calculates radiance spectra from saved thermal simulation data.
    
    This class recreates the necessary solver instances and performs
    radiance calculations independently of the main thermal simulation.
    """
    
    def __init__(self, config: SimulationConfig, thermal_data: Dict[str, Any], 
                 grid_data: Dict[str, Any], crater_data: Optional[Dict[str, Any]] = None):
        """
        Initialize radiance calculator with saved simulation data.
        
        Args:
            config: Simulation configuration
            thermal_data: Thermal evolution data (T_out, T_surf_out, etc.)
            grid_data: Spatial grid data  
            crater_data: Crater geometry data (if crater simulation)
        """
        self.cfg = config
        self.thermal_data = thermal_data
        self.grid_data = grid_data
        self.crater_data = crater_data
        
        # Reconstruct grid object
        self.grid = self._reconstruct_grid()
        
        # Initialize RTE solvers for radiance calculations
        if self.cfg.use_RTE:
            self._setup_radiance_solvers()
            
        # Initialize crater components if needed
        if self.cfg.crater and crater_data:
            self._setup_crater_components()
    
    def _reconstruct_grid(self) -> LayerGrid:
        """Reconstruct grid object from saved grid data."""
        # Create a minimal grid object with saved data
        grid = LayerGrid(self.cfg)
        
        # Override with saved grid data
        if 'x_grid' in self.grid_data:
            grid.x_grid = self.grid_data['x_grid']
        if 'x_boundaries' in self.grid_data:
            grid.x_boundaries = self.grid_data['x_boundaries']
        if 'x_RTE' in self.grid_data:
            grid.x_RTE = self.grid_data['x_RTE']
        if 'nlay_dust' in self.grid_data:
            grid.nlay_dust = self.grid_data['nlay_dust']
            
        return grid
    
    def _setup_radiance_solvers(self):
        """Setup RTE solvers for radiance calculations."""
        if self.cfg.RTE_solver == 'disort':
            # Setup output radiance solvers based on configuration
            self.rte_disort_out = DisortRTESolver(
                self.cfg, self.grid, n_cols=1, output_radiance=True, planck=True,
                observer_mu=self.cfg.observer_mu, observer_phi=0.0,
                solver_mode=self.cfg.output_radiance_mode
            )
            
            # Setup crater radiance solvers if needed
            if self.cfg.crater and self.crater_data:
                n_facets = self.crater_data.get('normals', np.array([])).shape[0]
                if n_facets > 0:
                    self.rte_crater_thermal = DisortRTESolver(
                        self.cfg, self.grid, n_cols=n_facets, output_radiance=True, planck=True,
                        solver_mode=self.cfg.output_radiance_mode, spectral_component='thermal_only'
                    )
                    
        elif self.cfg.RTE_solver == 'hapke':
            self.rte_hapke = RadiativeTransfer(self.cfg, self.grid)
    
    def _setup_crater_components(self):
        """Setup crater mesh and related components from saved data."""
        if not self.crater_data:
            return
            
        # Reconstruct crater mesh from saved geometry
        self.crater_mesh = self._reconstruct_crater_mesh()
        
        # Setup shadow tester
        self.crater_shadowtester = ShadowTester(self.crater_mesh)
        
        # Setup observer radiance calculator
        if 'observer_vectors' in self.crater_data:
            observer_vectors = self.crater_data['observer_vectors']
            self.observer_radiance_calc = ObserverRadianceCalculator(
                self.cfg, self.grid, observer_vectors
            )
    
    def _reconstruct_crater_mesh(self):
        """Reconstruct crater mesh object from saved data."""
        class ReconstructedCraterMesh:
            def __init__(self, crater_data):
                self.vertices = crater_data['mesh_vertices']
                self.faces = crater_data['mesh_faces'] 
                self.normals = crater_data['mesh_normals']
                self.areas = crater_data['mesh_areas']
                self.centroids = crater_data['mesh_centroids']
                
                # Precompute local coordinate systems (copy from crater.py)
                self.tangent1, self.tangent2 = self._compute_local_coordinates()
                
            def _compute_local_coordinates(self):
                """Precompute local coordinate system for each facet."""
                n_facets = len(self.normals)
                tangent1 = np.zeros((n_facets, 3))
                tangent2 = np.zeros((n_facets, 3))
                
                for i, normal in enumerate(self.normals):
                    if abs(normal[2]) < 0.9:
                        tangent1[i] = np.cross([0, 0, 1], normal)
                    else:
                        tangent1[i] = np.cross([1, 0, 0], normal)
                    
                    tangent1[i] = tangent1[i] / np.linalg.norm(tangent1[i])
                    tangent2[i] = np.cross(normal, tangent1[i])
                
                return tangent1, tangent2
        
        return ReconstructedCraterMesh(self.crater_data)
    
    def calculate_smooth_surface_radiance(self, time_indices: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Calculate radiance for smooth surface at specified time indices.
        
        Args:
            time_indices: List of time indices to calculate radiance for.
                         If None, calculates for all available times.
                         
        Returns:
            Dictionary containing radiance results
        """
        if not self.cfg.use_RTE:
            raise ValueError("Radiance calculation requires use_RTE=True")
            
        # Determine time indices to process
        n_times = self.thermal_data['T_out'].shape[1]
        if time_indices is None:
            time_indices = list(range(n_times))
            
        results = {
            'radiance_out': [],
            'brightness_temperature': [],
            'time_indices': time_indices
        }
        
        # Get solar geometry if available
        mu_values = self.thermal_data.get('mu_out', np.ones(n_times))
        F_values = np.ones(n_times) if self.cfg.sun else np.zeros(n_times)
        
        for t_idx in time_indices:
            # Extract temperature profile at this time
            T_profile = self.thermal_data['T_out'][:, t_idx]
            
            # Solar parameters
            mu_sun = mu_values[t_idx] if t_idx < len(mu_values) else 0.0
            F_sun = F_values[t_idx] if t_idx < len(F_values) and mu_sun > 0 else 0.0
            
            # Calculate radiance using appropriate solver
            if self.cfg.RTE_solver == 'disort':
                radiance, _ = self.rte_disort_out.disort_run(T_profile, mu_sun, F_sun)
                
                # Convert to numpy and handle tensor format
                if hasattr(radiance, 'numpy'):
                    radiance = radiance.numpy()
                
                # Extract radiance from result tensor [n_waves, 1, 1, 1, 1] -> [n_waves] or scalar
                if radiance.ndim > 1:
                    radiance = radiance.squeeze()
                    
            elif self.cfg.RTE_solver == 'hapke':
                # Use Hapke solver
                radiance = self.rte_hapke.calculate_radiance(T_profile, mu_sun, F_sun)
            
            results['radiance_out'].append(radiance)
            
            # Calculate brightness temperature if spectral data available
            if self.cfg.output_radiance_mode in ['multi_wave', 'hybrid']:
                # For spectral data, calculate brightness temperature
                # This would use the max_brightness_temperature_blackbody function
                try:
                    from modelmain import max_brightness_temperature_blackbody
                    wn_bounds = np.loadtxt(self.cfg.wn_bounds_out)
                    
                    # Create dummy simulator object with minimal required attributes
                    class DummySim:
                        def __init__(self, T_out):
                            self.T_out = T_out
                            
                    dummy_sim = DummySim(self.thermal_data['T_out'])
                    T_bright, _, _, _ = max_brightness_temperature_blackbody(
                        dummy_sim, wn_bounds, radiance, t_idx
                    )
                    results['brightness_temperature'].append(T_bright)
                except:
                    # Fallback to surface temperature if brightness temp calculation fails
                    results['brightness_temperature'].append(self.thermal_data['T_surf_out'][t_idx])
            else:
                # For broadband, use Stefan-Boltzmann relation
                results['brightness_temperature'].append(self.thermal_data['T_surf_out'][t_idx])
        
        # Convert lists to arrays
        results['radiance_out'] = np.array(results['radiance_out'])
        results['brightness_temperature'] = np.array(results['brightness_temperature'])
        
        return results
    
    def calculate_crater_radiance(self, time_indices: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Calculate crater radiance for all observers at specified time indices.
        
        Args:
            time_indices: List of time indices to calculate radiance for.
                         If None, calculates for all available times.
                         
        Returns:
            Dictionary containing crater radiance results
        """
        if not (self.cfg.crater and self.crater_data):
            raise ValueError("Crater radiance calculation requires crater=True and crater data")
            
        # Determine time indices to process
        n_times = self.thermal_data['T_crater_out'].shape[2] 
        if time_indices is None:
            time_indices = list(range(n_times))
            
        n_observers = len(self.crater_data['observer_vectors'])
        
        # Initialize results based on output mode
        if self.cfg.output_radiance_mode in ['multi_wave', 'hybrid']:
            # Get number of wavelengths from solver
            n_waves = len(self.rte_crater_thermal.wavenumbers)
            observer_radiances = np.zeros((n_observers, n_waves, len(time_indices)))
        else:
            observer_radiances = np.zeros((n_observers, len(time_indices)))
            
        # Get solar geometry
        mu_values = self.thermal_data.get('mu_out', np.ones(n_times))
        F_values = np.ones(n_times) if self.cfg.sun else np.zeros(n_times)
        
        for i, t_idx in enumerate(time_indices):
            # Extract crater temperature profiles at this time
            T_crater_facets = self.thermal_data['T_crater_out'][:, :, t_idx]  # [depth, n_facets]
            
            # Solar parameters
            mu_sun = mu_values[t_idx] if t_idx < len(mu_values) else 0.0
            F_sun = F_values[t_idx] if t_idx < len(F_values) and mu_sun > 0 else 0.0
            
            # Calculate thermal flux for crater facets
            therm_flux_facets = self._calculate_crater_thermal_flux(T_crater_facets)
            
            # Calculate radiance for all observers
            radiances = self.observer_radiance_calc.compute_all_observers(
                T_crater_facets, self.crater_mesh, self.crater_shadowtester,
                mu_sun=mu_sun, F_sun=F_sun, therm_flux_facets=therm_flux_facets
            )
            
            observer_radiances[..., i] = radiances
        
        return {
            'observer_radiance_out': observer_radiances,
            'time_indices': time_indices,
            'observer_vectors': self.crater_data['observer_vectors']
        }
    
    def _calculate_crater_thermal_flux(self, T_crater_facets: np.ndarray) -> np.ndarray:
        """
        Calculate thermal flux for crater facets.
        
        Args:
            T_crater_facets: Temperature profiles [depth, n_facets]
            
        Returns:
            Thermal flux array [n_facets, n_waves] or [n_facets]
        """
        n_facets = T_crater_facets.shape[1]
        
        if self.cfg.output_radiance_mode in ['multi_wave', 'hybrid']:
            n_waves = len(self.rte_crater_thermal.wavenumbers)
            therm_flux = np.zeros((n_facets, n_waves))
            
            # Calculate flux for each facet
            for facet_idx in range(n_facets):
                T_facet = T_crater_facets[:, facet_idx]
                _, flux_up = self.rte_crater_thermal.disort_run(
                    T_facet.reshape(-1, 1), 0.0, 0.0
                )
                therm_flux[facet_idx, :] = flux_up[:, 0]
        else:
            # Two-wave case
            therm_flux = np.zeros(n_facets)
            for facet_idx in range(n_facets):
                T_facet = T_crater_facets[:, facet_idx]
                _, flux_up = self.rte_crater_thermal.disort_run(
                    T_facet, 0.0, 0.0
                )
                therm_flux[facet_idx] = flux_up[0]
                
        return therm_flux
    
    def calculate_all_radiances(self, time_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Calculate all available radiance products.
        
        Args:
            time_indices: Time indices to process (None for all)
            
        Returns:
            Complete radiance results dictionary
        """
        results = {}
        
        # Smooth surface radiance
        if self.cfg.use_RTE:
            results['smooth_surface'] = self.calculate_smooth_surface_radiance(time_indices)
        
        # Crater radiance
        if self.cfg.crater and self.crater_data:
            results['crater'] = self.calculate_crater_radiance(time_indices)
            
        return results


def calculate_radiance_from_file(results_file_path: str, 
                               time_indices: Optional[List[int]] = None,
                               output_file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    High-level function to calculate radiance from saved thermal results.
    
    Args:
        results_file_path: Path to saved thermal results HDF5 file
        time_indices: Time indices to process (None for all)
        output_file_path: Path to save radiance results (None to skip saving)
        
    Returns:
        Dictionary containing all radiance results
    """
    from core.data_persistence import ThermalResultsLoader
    
    # Load thermal simulation data
    loader = ThermalResultsLoader(results_file_path)
    all_data = loader.load_all_data()
    
    # Create radiance calculator
    calculator = StandaloneRadianceCalculator(
        config=all_data['config'],
        thermal_data=all_data['thermal_data'],
        grid_data=all_data['grid_data'],
        crater_data=all_data.get('crater_data')
    )
    
    # Calculate radiances
    radiance_results = calculator.calculate_all_radiances(time_indices)
    
    # Save results if requested
    if output_file_path:
        save_radiance_results(radiance_results, output_file_path, all_data['config'])
    
    return radiance_results


def save_radiance_results(radiance_results: Dict[str, Any], 
                         output_path: str, 
                         config: SimulationConfig):
    """
    Save radiance calculation results to HDF5 file.
    
    Args:
        radiance_results: Results from radiance calculations
        output_path: Path to save radiance results
        config: Configuration used for calculations
    """
    import h5py
    from dataclasses import asdict
    import yaml
    
    with h5py.File(output_path, 'w') as f:
        # Save metadata
        meta_group = f.create_group('metadata')
        config_yaml = yaml.dump(asdict(config), default_flow_style=False)
        meta_group.create_dataset('config_yaml', data=config_yaml.encode('utf-8'))
        meta_group.attrs['calculation_type'] = 'radiance_postprocessing'
        
        # Save smooth surface results
        if 'smooth_surface' in radiance_results:
            smooth_group = f.create_group('smooth_surface')
            smooth_data = radiance_results['smooth_surface']
            
            for key, value in smooth_data.items():
                if isinstance(value, np.ndarray):
                    smooth_group.create_dataset(key, data=value, compression='gzip')
                else:
                    smooth_group.attrs[key] = value
        
        # Save crater results  
        if 'crater' in radiance_results:
            crater_group = f.create_group('crater')
            crater_data = radiance_results['crater']
            
            for key, value in crater_data.items():
                if isinstance(value, np.ndarray):
                    crater_group.create_dataset(key, data=value, compression='gzip')
                else:
                    crater_group.attrs[key] = value
    
    print(f"Radiance results saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Standalone radiance calculator ready.")
    print("Use calculate_radiance_from_file() to process saved thermal results.")