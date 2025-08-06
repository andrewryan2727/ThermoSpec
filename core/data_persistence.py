"""
Data persistence layer for thermal simulation results.

Handles saving and loading thermal simulation data to enable
independent post-processing and batch ML dataset generation.
"""

import h5py
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import asdict
import json
import datetime

from config import SimulationConfig


class ThermalResultsWriter:
    """
    Saves thermal simulation results to HDF5 format for efficient storage and retrieval.
    """
    
    def __init__(self, output_path: str, config: SimulationConfig, compression: str = "gzip"):
        """
        Initialize results writer.
        
        Args:
            output_path: Path to output HDF5 file
            config: Configuration used for simulation
            compression: HDF5 compression method ("gzip", "lzf", or None)
        """
        self.output_path = Path(output_path)
        self.config = config
        self.compression = compression
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def save_thermal_results(self, simulation_data: Dict[str, Any]):
        """
        Save complete thermal simulation results to HDF5 file.
        
        Args:
            simulation_data: Dictionary containing all simulation results
        """
        with h5py.File(self.output_path, 'w') as f:
            # Save metadata
            self._save_metadata(f, simulation_data)
            
            # Save thermal data
            self._save_thermal_data(f, simulation_data)
            
            # Save grid information  
            self._save_grid_data(f, simulation_data)
            
            # Save crater data if applicable
            if self.config.crater and 'crater_data' in simulation_data:
                self._save_crater_data(f, simulation_data)
                
            # Save solar/illumination data
            if self.config.sun or self.config.diurnal:
                self._save_solar_data(f, simulation_data)
                
        print(f"Thermal results saved to: {self.output_path}")
    
    def _save_metadata(self, f: h5py.File, data: Dict[str, Any]):
        """Save simulation metadata and configuration."""
        meta_group = f.create_group('metadata')
        
        # Save configuration as YAML string
        config_dict = asdict(self.config)
        config_yaml = yaml.dump(config_dict, default_flow_style=False)
        meta_group.create_dataset('config_yaml', data=config_yaml.encode('utf-8'))
        
        # Save run information
        meta_group.attrs['timestamp'] = datetime.datetime.now().isoformat()
        meta_group.attrs['simulation_type'] = 'thermal_evolution'
        meta_group.attrs['has_crater'] = self.config.crater
        meta_group.attrs['has_rte'] = self.config.use_RTE
        meta_group.attrs['solver_type'] = self.config.RTE_solver if self.config.use_RTE else 'thermal_only'
        
        # Save array shapes for quick reference
        if 'T_out' in data:
            meta_group.attrs['T_out_shape'] = data['T_out'].shape
        if 'T_crater_out' in data:
            meta_group.attrs['T_crater_out_shape'] = data['T_crater_out'].shape
            
    def _save_thermal_data(self, f: h5py.File, data: Dict[str, Any]):
        """Save thermal evolution results."""
        thermal_group = f.create_group('thermal_data')
        
        # Core thermal arrays (always present)
        required_arrays = ['T_out', 'T_surf_out', 't_out']
        for array_name in required_arrays:
            if array_name in data:
                thermal_group.create_dataset(
                    array_name, 
                    data=data[array_name],
                    compression=self.compression
                )
        
        # Optional thermal arrays
        optional_arrays = ['mu_out', 'F_array']
        for array_name in optional_arrays:
            if array_name in data:
                thermal_group.create_dataset(
                    array_name,
                    data=data[array_name], 
                    compression=self.compression
                )
                
        # RTE-specific flux data
        if self.config.use_RTE:
            rte_arrays = ['phi_vis_out', 'phi_therm_out']
            for array_name in rte_arrays:
                if array_name in data:
                    thermal_group.create_dataset(
                        array_name,
                        data=data[array_name],
                        compression=self.compression
                    )
    
    def _save_grid_data(self, f: h5py.File, data: Dict[str, Any]):
        """Save spatial grid information."""
        grid_group = f.create_group('grid_data')
        
        # Grid arrays
        grid_arrays = ['x_grid', 'x_boundaries', 'x_RTE']
        for array_name in grid_arrays:
            if array_name in data:
                grid_group.create_dataset(array_name, data=data[array_name])
        
        # Grid parameters
        grid_params = ['nlay_dust', 'dust_thickness', 'rock_thickness']
        for param_name in grid_params:
            if param_name in data:
                grid_group.attrs[param_name] = data[param_name]
                
    def _save_crater_data(self, f: h5py.File, data: Dict[str, Any]):
        """Save crater-specific data."""
        crater_group = f.create_group('crater_data')
        
        # Crater thermal arrays
        crater_thermal = ['T_crater_out', 'T_surf_crater_out']
        for array_name in crater_thermal:
            if array_name in data:
                crater_group.create_dataset(
                    array_name,
                    data=data[array_name],
                    compression=self.compression
                )
        
        # Crater geometry (from crater_data dict)
        crater_data = data.get('crater_data', {})
        if crater_data:
            # Mesh geometry
            geom_arrays = ['vertices', 'faces', 'normals', 'areas', 'centroids']
            for array_name in geom_arrays:
                if array_name in crater_data:
                    crater_group.create_dataset(f'mesh_{array_name}', data=crater_data[array_name])
            
            # Observer information
            if 'observer_vectors' in crater_data:
                crater_group.create_dataset('observer_vectors', data=crater_data['observer_vectors'])
                
    def _save_solar_data(self, f: h5py.File, data: Dict[str, Any]):
        """Save solar/illumination data."""
        solar_group = f.create_group('solar_data')
        
        solar_arrays = ['mu_out', 'F_sun_out']
        for array_name in solar_arrays:
            if array_name in data:
                solar_group.create_dataset(
                    array_name,
                    data=data[array_name],
                    compression=self.compression
                )


class ThermalResultsLoader:
    """
    Loads thermal simulation results from HDF5 files for post-processing.
    """
    
    def __init__(self, results_path: str):
        """
        Initialize results loader.
        
        Args:
            results_path: Path to saved HDF5 results file
        """
        self.results_path = Path(results_path)
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
            
    def load_config(self) -> SimulationConfig:
        """Load the simulation configuration used for this run."""
        with h5py.File(self.results_path, 'r') as f:
            config_yaml = f['metadata/config_yaml'][()].decode('utf-8')
            config_dict = yaml.safe_load(config_yaml)
            return SimulationConfig(**config_dict)
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load simulation metadata."""
        metadata = {}
        with h5py.File(self.results_path, 'r') as f:
            meta_group = f['metadata']
            # Load attributes
            for key in meta_group.attrs:
                metadata[key] = meta_group.attrs[key]
        return metadata
    
    def load_thermal_data(self) -> Dict[str, np.ndarray]:
        """Load thermal evolution data."""
        thermal_data = {}
        with h5py.File(self.results_path, 'r') as f:
            thermal_group = f['thermal_data']
            for dataset_name in thermal_group:
                thermal_data[dataset_name] = thermal_group[dataset_name][:]
        return thermal_data
    
    def load_grid_data(self) -> Dict[str, Any]:
        """Load spatial grid data."""
        grid_data = {}
        with h5py.File(self.results_path, 'r') as f:
            if 'grid_data' in f:
                grid_group = f['grid_data']
                # Load datasets
                for dataset_name in grid_group:
                    grid_data[dataset_name] = grid_group[dataset_name][:]
                # Load attributes  
                for attr_name in grid_group.attrs:
                    grid_data[attr_name] = grid_group.attrs[attr_name]
        return grid_data
    
    def load_crater_data(self) -> Optional[Dict[str, np.ndarray]]:
        """Load crater-specific data if available."""
        crater_data = {}
        with h5py.File(self.results_path, 'r') as f:
            if 'crater_data' in f:
                crater_group = f['crater_data']
                for dataset_name in crater_group:
                    crater_data[dataset_name] = crater_group[dataset_name][:]
                return crater_data
        return None
    
    def load_solar_data(self) -> Optional[Dict[str, np.ndarray]]:
        """Load solar/illumination data if available."""
        solar_data = {}
        with h5py.File(self.results_path, 'r') as f:
            if 'solar_data' in f:
                solar_group = f['solar_data']
                for dataset_name in solar_group:
                    solar_data[dataset_name] = solar_group[dataset_name][:]
                return solar_data
        return None
    
    def load_all_data(self) -> Dict[str, Any]:
        """Load all available data from the results file."""
        all_data = {
            'config': self.load_config(),
            'metadata': self.load_metadata(),
            'thermal_data': self.load_thermal_data(),
            'grid_data': self.load_grid_data()
        }
        
        # Add optional data if available
        crater_data = self.load_crater_data()
        if crater_data:
            all_data['crater_data'] = crater_data
            
        solar_data = self.load_solar_data()
        if solar_data:
            all_data['solar_data'] = solar_data
            
        return all_data
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the simulation results."""
        metadata = self.load_metadata()
        thermal_data = self.load_thermal_data()
        
        summary = {
            'timestamp': metadata.get('timestamp', 'unknown'),
            'simulation_type': metadata.get('simulation_type', 'unknown'),
            'has_crater': metadata.get('has_crater', False),
            'has_rte': metadata.get('has_rte', False),
            'solver_type': metadata.get('solver_type', 'unknown'),
            'n_time_outputs': thermal_data['t_out'].shape[0] if 't_out' in thermal_data else 0,
            'n_depth_points': thermal_data['T_out'].shape[0] if 'T_out' in thermal_data else 0,
            'time_span_hours': (thermal_data['t_out'][-1] - thermal_data['t_out'][0]) / 3600 if 't_out' in thermal_data else 0,
        }
        
        if 'T_crater_out' in thermal_data:
            summary['n_crater_facets'] = thermal_data['T_crater_out'].shape[1]
            
        return summary


def create_simulation_data_dict(simulator) -> Dict[str, Any]:
    """
    Extract data from a simulator instance for saving.
    
    Args:
        simulator: ThermalSimulator instance after run completion
        
    Returns:
        Dictionary containing all simulation data for saving
    """
    data = {}
    
    # Core thermal data (always present after _make_outputs)
    data['T_out'] = simulator.T_out
    data['T_surf_out'] = simulator.T_surf_out  
    data['t_out'] = simulator.t_out
    
    # Solar/illumination data
    if hasattr(simulator, 'mu_out'):
        data['mu_out'] = simulator.mu_out
    if hasattr(simulator, 'F_array'):
        data['F_array'] = simulator.F_array
        
    # Grid information
    data['x_grid'] = simulator.grid.x_grid
    data['x_boundaries'] = simulator.grid.x_boundaries
    data['x_RTE'] = simulator.grid.x_RTE
    data['nlay_dust'] = simulator.grid.nlay_dust
    data['dust_thickness'] = simulator.cfg.dust_thickness
    data['rock_thickness'] = simulator.cfg.rock_thickness
    
    # RTE flux data
    if hasattr(simulator, 'phi_vis_out'):
        data['phi_vis_out'] = simulator.phi_vis_out
    if hasattr(simulator, 'phi_therm_out'):
        data['phi_therm_out'] = simulator.phi_therm_out
    
    # Crater data
    if simulator.cfg.crater:
        data['T_crater_out'] = simulator.T_crater_out
        data['T_surf_crater_out'] = simulator.T_surf_crater_out
        
        # Crater geometry
        crater_data = {}
        if hasattr(simulator, 'crater_mesh'):
            crater_data['vertices'] = simulator.crater_mesh.vertices
            crater_data['faces'] = simulator.crater_mesh.faces
            crater_data['normals'] = simulator.crater_mesh.normals
            crater_data['areas'] = simulator.crater_mesh.areas
            crater_data['centroids'] = simulator.crater_mesh.centroids
            
        if hasattr(simulator, 'observer_radiance_calc'):
            crater_data['observer_vectors'] = np.array([
                obs['vector'] for obs in simulator.observer_radiance_calc.observers
            ])
            
        data['crater_data'] = crater_data
    
    return data


if __name__ == "__main__":
    # Example usage
    from config import SimulationConfig
    
    # Test data persistence with dummy data
    config = SimulationConfig(ndays=1, diurnal=True, crater=True)
    
    # Create dummy simulation data
    n_time = 24
    n_depth = 20  
    n_facets = 100
    
    dummy_data = {
        'T_out': np.random.uniform(200, 400, (n_depth, n_time)),
        'T_surf_out': np.random.uniform(250, 350, n_time),
        't_out': np.linspace(0, 86400, n_time),
        'mu_out': np.cos(np.linspace(0, 2*np.pi, n_time)),
        'x_grid': np.linspace(0, 0.1, n_depth),
        'x_boundaries': np.linspace(0, 0.1, n_depth+1),
        'x_RTE': np.linspace(0, 0.1, n_depth),
        'nlay_dust': n_depth,
        'dust_thickness': 0.1,
        'rock_thickness': 1.0,
        'T_crater_out': np.random.uniform(200, 400, (n_depth, n_facets, n_time)),
        'T_surf_crater_out': np.random.uniform(250, 350, (n_facets, n_time))
    }
    
    # Test saving
    writer = ThermalResultsWriter('test_output.h5', config)
    writer.save_thermal_results(dummy_data)
    
    # Test loading
    loader = ThermalResultsLoader('test_output.h5')
    summary = loader.get_summary()
    print("Simulation summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
        
    # Clean up
    Path('test_output.h5').unlink()