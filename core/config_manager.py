"""
Configuration management system for thermal model batch processing.

Supports loading base configurations from YAML files and applying 
parameter overrides for batch ML dataset generation.
"""

import yaml
import numpy as np
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any, List, Union
import copy

from config import SimulationConfig


class ConfigManager:
    """
    Manages configuration loading and parameter overrides for batch processing.
    
    Workflow:
    1. Load base configuration from YAML file
    2. Apply parameter overrides (for batch sweeps)
    3. Validate and create SimulationConfig instance
    """
    
    def __init__(self, base_config_path: str):
        """
        Initialize with path to base configuration file.
        
        Args:
            base_config_path: Path to YAML base configuration file
        """
        self.base_config_path = Path(base_config_path)
        self.base_config_dict = self._load_base_config()
        
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration from YAML file."""
        if not self.base_config_path.exists():
            raise FileNotFoundError(f"Base config file not found: {self.base_config_path}")
            
        with open(self.base_config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Flatten nested dictionaries for direct parameter access
        return self._flatten_config_dict(config_dict)
    
    def _flatten_config_dict(self, nested_dict: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """
        Flatten nested configuration dictionary for easier parameter access.
        
        Args:
            nested_dict: Nested configuration dictionary
            parent_key: Parent key for current recursion level
            
        Returns:
            Flattened dictionary with all parameters at top level
        """
        flattened = {}
        
        for key, value in nested_dict.items():
            if isinstance(value, dict):
                # Special handling for observer_vectors format
                if key == 'observer_vectors' and self._is_formatted_observer_vectors(value):
                    flattened[key] = self._convert_observer_vectors_to_list(value)
                else:
                    # Recursively flatten other nested dictionaries
                    flattened.update(self._flatten_config_dict(value, parent_key))
            else:
                # Handle numpy expressions in YAML (e.g., "np.radians(30)")
                if isinstance(value, str) and value.startswith('np.'):
                    try:
                        value = eval(value)
                    except:
                        pass  # Keep as string if evaluation fails
                flattened[key] = value
                
        return flattened
    
    def _is_formatted_observer_vectors(self, obj: Dict[str, Any]) -> bool:
        """Check if object is a formatted observer vectors dictionary."""
        if not isinstance(obj, dict):
            return False
        
        # Check if all values are dictionaries with x, y, z keys
        for key, value in obj.items():
            if not isinstance(value, dict):
                return False
            required_keys = {'x', 'y', 'z'}
            if not required_keys.issubset(value.keys()):
                return False
        
        return True
    
    def _convert_observer_vectors_to_list(self, formatted_vectors: Dict[str, Any]) -> List[List[float]]:
        """Convert formatted observer vectors back to list format."""
        vectors = []
        for vector_name, vector_data in formatted_vectors.items():
            vectors.append([vector_data['x'], vector_data['y'], vector_data['z']])
        return vectors
    
    def create_config(self, parameter_overrides: Dict[str, Any] = None) -> SimulationConfig:
        """
        Create SimulationConfig instance with optional parameter overrides.
        
        Args:
            parameter_overrides: Dictionary of parameters to override
            
        Returns:
            Configured SimulationConfig instance
        """
        # Start with base configuration
        config_dict = copy.deepcopy(self.base_config_dict)
        
        # Apply parameter overrides
        if parameter_overrides:
            config_dict.update(parameter_overrides)
            
        # Handle special cases for numpy arrays/expressions
        config_dict = self._process_special_values(config_dict)
        
        # Create SimulationConfig instance
        # Filter to only include valid SimulationConfig fields
        valid_fields = {f.name for f in SimulationConfig.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        try:
            return SimulationConfig(**filtered_config)
        except TypeError as e:
            raise ValueError(f"Invalid configuration parameters: {e}")
    
    def _process_special_values(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process special configuration values (numpy expressions, etc.)."""
        processed = {}
        
        for key, value in config_dict.items():
            if isinstance(value, str):
                # Handle numpy expressions
                if value.startswith('np.'):
                    try:
                        processed[key] = eval(value)
                    except:
                        processed[key] = value
                else:
                    processed[key] = value
            else:
                processed[key] = value
                
        return processed
    
    def validate_config(self, config: SimulationConfig) -> List[str]:
        """
        Validate configuration and return list of warnings/errors.
        
        Args:
            config: SimulationConfig instance to validate
            
        Returns:
            List of validation messages
        """
        warnings = []
        
        # Check file paths exist
        file_params = [
            'mie_file', 'solar_spectrum_file', 'substrate_spectrum', 'wn_bounds',
            'mie_file_out', 'solar_spectrum_file_out', 'substrate_spectrum_out', 'wn_bounds_out'
        ]
        
        for param in file_params:
            if hasattr(config, param):
                file_path = Path(getattr(config, param))
                if not file_path.exists():
                    warnings.append(f"File not found: {param} = {file_path}")
        
        # Check parameter combinations
        if config.use_RTE and config.RTE_solver not in ['disort', 'hapke']:
            warnings.append(f"Invalid RTE_solver: {config.RTE_solver}")
            
        if config.multi_wave and not config.use_RTE:
            warnings.append("multi_wave=True requires use_RTE=True")
            
        # Check physical parameter ranges
        if config.Et <= 0:
            warnings.append("Extinction coefficient (Et) must be positive")
            
        if config.radius <= 0:
            warnings.append("Particle radius must be positive")
            
        return warnings


class BatchParameterGenerator:
    """
    Generates parameter combinations for batch ML dataset creation.
    """
    
    @staticmethod
    def load_batch_config(batch_config_path: str) -> Dict[str, Any]:
        """Load batch parameter configuration from YAML file."""
        with open(batch_config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def generate_parameter_combinations(batch_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations for batch processing.
        
        Args:
            batch_config: Batch configuration dictionary
            
        Returns:
            List of parameter override dictionaries
        """
        parameter_sweeps = batch_config.get('parameter_sweeps', [])
        fixed_overrides = batch_config.get('fixed_overrides', {})
        
        # Generate parameter values for each sweep
        sweep_values = {}
        for sweep in parameter_sweeps:
            param_name = sweep['parameter']
            sweep_type = sweep['type']
            
            if sweep_type == 'linear':
                values = np.linspace(sweep['min'], sweep['max'], sweep['n_samples'])
            elif sweep_type == 'log_uniform':
                values = np.logspace(np.log10(sweep['min']), np.log10(sweep['max']), sweep['n_samples'])
            elif sweep_type == 'uniform':
                values = np.random.uniform(sweep['min'], sweep['max'], sweep['n_samples'])
            elif sweep_type == 'list':
                values = sweep['values']
            else:
                raise ValueError(f"Unknown sweep type: {sweep_type}")
                
            sweep_values[param_name] = values
        
        # Generate all combinations (Cartesian product)
        if not sweep_values:
            return [fixed_overrides] if fixed_overrides else [{}]
            
        import itertools
        param_names = list(sweep_values.keys())
        param_value_lists = [sweep_values[name] for name in param_names]
        
        combinations = []
        for value_combo in itertools.product(*param_value_lists):
            combo_dict = dict(zip(param_names, value_combo))
            combo_dict.update(fixed_overrides)
            combinations.append(combo_dict)
            
        return combinations


def create_default_base_config(output_path: str, planetary_body: str = "generic"):
    """
    Create a default base configuration YAML file following the organized structure from config.py.
    
    Args:
        output_path: Path where to save the configuration file
        planetary_body: Name of planetary body for defaults
    """
    # Get default values from SimulationConfig (excludes computed fields from __post_init__)
    default_config = SimulationConfig()
    config_dict = asdict(default_config)
    
    # Remove computed fields that are calculated in __post_init__
    computed_fields = ['gamma_vis', 'gamma_therm', 'J', 'q', 'rock_skin_depth', 'dust_skin_depth']
    for field in computed_fields:
        config_dict.pop(field, None)
    
    # Organize into logical groups following config.py structure
    organized_config = {
        'simulation_metadata': {
            'description': f"Default thermal model configuration for {planetary_body}"
        },
        
        'solver_settings': {
            'use_RTE': config_dict['use_RTE'],
            'RTE_solver': config_dict['RTE_solver'], 
            'diurnal': config_dict['diurnal'],
            'sun': config_dict['sun'],
            'T_fixed': config_dict['T_fixed']
        },
        
        'output_settings': {
            'freq_out': config_dict['freq_out'],
            'last_day': config_dict['last_day'],
            'compute_observer_radiance': config_dict['compute_observer_radiance'],
            'observer_mu': f"np.cos(np.radians({np.degrees(np.arccos(config_dict['observer_mu']))}))"
        },
        
        'disort_spectral_modes': {
            'thermal_evolution_mode': config_dict['thermal_evolution_mode'],
            'output_radiance_mode': config_dict['output_radiance_mode']
        },
        
        'optical_properties_rte': {
            'Et': config_dict['Et'],
            'eta': config_dict['eta'],
            'ssalb_vis': config_dict['ssalb_vis'],
            'ssalb_therm': config_dict['ssalb_therm'],
            'g_vis': config_dict['g_vis'],
            'g_therm': config_dict['g_therm'],
            'R_base': config_dict['R_base']
        },
        
        'optical_properties_non_rte': {
            'em': config_dict['em'],
            'albedo': config_dict['albedo']
        },
        
        'planetary_properties': {
            'R': config_dict['R'],
            'S': config_dict['S'],
            'latitude': f"np.radians({np.degrees(config_dict['latitude'])})",
            'dec': f"np.radians({np.degrees(config_dict['dec'])})",
            'P': config_dict['P'],
            'steady_state_mu': f"np.cos(np.radians({np.degrees(np.arccos(config_dict['steady_state_mu']))}))"
        },
        
        'surface_roughness': {
            'crater': config_dict['crater'],
            'illum_freq': config_dict['illum_freq'],
            'compute_crater_radiance': config_dict['compute_crater_radiance'],
            # Better formatting for observer vectors - convert to more readable format
            'observer_vectors': _format_observer_vectors(config_dict['observer_vectors'])
        },
        
        'material_properties_dust': {
            'single_layer': config_dict['single_layer'],
            'dust_thickness': config_dict['dust_thickness'],
            'k_dust': config_dict['k_dust'],
            'rho_dust': config_dict['rho_dust'],
            'cp_dust': config_dict['cp_dust'],
            'k_dust_auto': config_dict['k_dust_auto']
        },
        
        'material_properties_rock': {
            'rock_thickness': config_dict['rock_thickness'],
            'k_rock': config_dict['k_rock'],
            'rho_rock': config_dict['rho_rock'],
            'cp_rock': config_dict['cp_rock']
        },
        
        'boundary_conditions': {
            'T_bottom': config_dict['T_bottom'],
            'bottom_bc': config_dict['bottom_bc']
        },
        
        'grid_settings': {
            'auto_thickness': config_dict['auto_thickness'],
            'flay': config_dict['flay'],
            'geometric_spacing': config_dict['geometric_spacing'],
            'spacing_factor': config_dict['spacing_factor'],
            'min_nlay_dust': config_dict['min_nlay_dust'],
            'rock_lthick_fac': config_dict['rock_lthick_fac'],
            'dust_rte_max_lthick': config_dict['dust_rte_max_lthick']
        },
        
        'time_stepping': {
            'ndays': config_dict['ndays'],
            'auto_dt': config_dict['auto_dt'],
            'tsteps_day': config_dict['tsteps_day'],
            'dtfac': config_dict['dtfac'],
            'minsteps': config_dict['minsteps']
        },
        
        'disort_parameters': {
            'nmom': config_dict['nmom'],
            'nstr': config_dict['nstr'],
            'hybrid_wavelength_cutoff': config_dict['hybrid_wavelength_cutoff'],
            'mie_file': config_dict['mie_file'],
            'solar_spectrum_file': config_dict['solar_spectrum_file'],
            'wn_bounds': config_dict['wn_bounds'],
            'substrate_spectrum': config_dict['substrate_spectrum'],
            'use_spec': config_dict['use_spec'],
            'fill_frac': config_dict['fill_frac'],
            'radius': config_dict['radius']
        },
        
        'disort_output_settings': {
            'mie_file_out': config_dict['mie_file_out'],
            'solar_spectrum_file_out': config_dict['solar_spectrum_file_out'],
            'wn_bounds_out': config_dict['wn_bounds_out'],
            'substrate_spectrum_out': config_dict['substrate_spectrum_out'],
            'otesT1_out': config_dict['otesT1_out'],
            'otesT2_out': config_dict['otesT2_out'],
            'nstr_out': config_dict['nstr_out'],
            'nmom_out': config_dict['nmom_out']
        },
        
        'disort_visible_properties': {
            'force_vis_disort': config_dict['force_vis_disort']
        },
        
        'disort_depth_dependent': {
            'depth_dependent': config_dict['depth_dependent']
        },
        
        'physical_constants': {
            'sigma': config_dict['sigma']
        },
        
        'advanced_solver_settings': {
            'steady_tol': config_dict['steady_tol'],
            'bvp_tol': config_dict['bvp_tol'],
            'bvp_max_iter': config_dict['bvp_max_iter'],
            'T_surf_tol': config_dict['T_surf_tol'],
            'T_surf_max_iter': config_dict['T_surf_max_iter'],
            'disort_space_temp': config_dict['disort_space_temp'],
            'dust_lthick': config_dict['dust_lthick'],
            'rock_lthick': config_dict['rock_lthick']
        },
        
        'diurnal_convergence_settings': {
            'enable_diurnal_convergence': config_dict['enable_diurnal_convergence'],
            'diurnal_convergence_method': config_dict['diurnal_convergence_method'],
            'diurnal_convergence_temp_tol': config_dict['diurnal_convergence_temp_tol'],
            'diurnal_convergence_energy_tol': config_dict['diurnal_convergence_energy_tol'],
            'diurnal_convergence_min_cycles': config_dict['diurnal_convergence_min_cycles'],
            'diurnal_convergence_check_interval': config_dict['diurnal_convergence_check_interval'],
            'diurnal_convergence_window': config_dict['diurnal_convergence_window']
        }
    }
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to YAML file with better formatting
    with open(output_path, 'w') as f:
        f.write(f"# Thermal Model Configuration for {planetary_body}\n")
        f.write(f"# Generated automatically - modify as needed\n\n")
        yaml.dump(organized_config, f, default_flow_style=False, indent=2, sort_keys=False)
        
    print(f"Default configuration saved to: {output_path}")


def _format_observer_vectors(observer_vectors):
    """
    Format observer vectors for better YAML readability.
    
    Instead of the default YAML list format, create a more readable structure.
    """
    formatted_vectors = {}
    for i, vector in enumerate(observer_vectors):
        # Create descriptive names based on vector components
        x, y, z = vector
        if x == 0 and y == 0 and z == 1:
            name = "overhead"
        elif y == 0:
            if x > 0:
                name = f"east_{abs(x):.1f}_deg"
            elif x < 0:
                name = f"west_{abs(x):.1f}_deg"
            else:
                name = f"vector_{i}"
        elif x == 0:
            if y > 0:
                name = f"north_{abs(y):.1f}_deg"
            elif y < 0:
                name = f"south_{abs(y):.1f}_deg"
            else:
                name = f"vector_{i}"
        else:
            name = f"vector_{i}"
            
        formatted_vectors[name] = {
            'x': float(x),
            'y': float(y), 
            'z': float(z),
            'description': f"Observer direction vector [{x}, {y}, {z}]"
        }
    
    return formatted_vectors


if __name__ == "__main__":
    # Example usage and testing
    
    # Create default configuration file
    create_default_base_config("configs/base_configs/default.yaml")
    
    # Example of loading and using configuration
    config_manager = ConfigManager("configs/base_configs/default.yaml")
    
    # Test with parameter overrides
    overrides = {'Et': 50000, 'radius': 10e-6, 'ndays': 2}
    config = config_manager.create_config(overrides)
    
    print(f"Created config with Et={config.Et}, radius={config.radius}")
    
    # Validate configuration
    warnings = config_manager.validate_config(config)
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")