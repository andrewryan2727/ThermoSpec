"""
Extracted thermal simulation core with data persistence integration.

This module provides a clean interface for running thermal simulations
with automatic data saving for post-processing workflows.
"""

import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Import existing thermal simulation code
sys.path.append(str(Path(__file__).parent.parent))
from modelmain import Simulator
from config import SimulationConfig
from core.data_persistence import ThermalResultsWriter, create_simulation_data_dict


class ThermalSimulator:
    """
    Enhanced thermal simulator with automatic data persistence.
    
    Uses composition with the original Simulator to support:
    - Automatic saving of thermal results
    - Optional separation of thermal evolution and radiance calculations
    - Integration with batch processing workflows
    """
    
    def __init__(self, config: SimulationConfig, 
                 save_thermal_data: bool = False,
                 thermal_output_path: Optional[str] = None,
                 calculate_radiance: bool = True):
        """
        Initialize enhanced thermal simulator.
        
        Args:
            config: Simulation configuration
            save_thermal_data: Whether to save thermal evolution data
            thermal_output_path: Path to save thermal data (auto-generated if None)
            calculate_radiance: Whether to calculate radiance outputs
        """
        self.config = config
        self.save_thermal_data = save_thermal_data
        self.thermal_output_path = thermal_output_path
        self.calculate_radiance = calculate_radiance
        
        # Create the core simulator instance
        self.simulator = Simulator(config)
        
        # Track whether thermal evolution has completed
        self._thermal_evolution_completed = False
        self._thermal_data_saved = False
    
    def run(self, save_results: bool = None) -> Dict[str, Any]:
        """
        Run thermal simulation with optional data persistence.
        
        Args:
            save_results: Override instance setting for saving results
            
        Returns:
            Dictionary with run results and file paths
        """
        # Use instance setting if not overridden
        if save_results is None:
            save_results = self.save_thermal_data
        
        # Run thermal evolution using refactored Simulator
        thermal_results = self.simulator.run(calculate_radiance=self.calculate_radiance)
        self._thermal_evolution_completed = True
        
        # Save thermal data if requested
        thermal_file_path = None
        if save_results:
            thermal_file_path = self._save_thermal_data()
        
        return {
            'status': 'completed',
            'thermal_file_path': thermal_file_path,
            'thermal_results': thermal_results,
            'has_crater': self.config.crater,
            'has_rte': self.config.use_RTE,
            'simulation_config': self.config
        }
    
    
    def _save_thermal_data(self) -> str:
        """Save thermal evolution data to HDF5 file."""
        if not self._thermal_evolution_completed:
            raise RuntimeError("Cannot save thermal data before thermal evolution is completed")
        
        # Generate output path if not provided
        if self.thermal_output_path is None:
            output_dir = Path("thermal_outputs")
            output_dir.mkdir(exist_ok=True)
            
            # Generate unique filename based on configuration
            config_hash = hash(str(sorted(self.config.__dict__.items())))
            self.thermal_output_path = output_dir / f"thermal_sim_{abs(config_hash):08x}.h5"
        
        # Create simulation data dictionary using the contained simulator
        simulation_data = create_simulation_data_dict(self.simulator)
        
        # Save using data persistence layer
        writer = ThermalResultsWriter(self.thermal_output_path, self.config)
        writer.save_thermal_results(simulation_data)
        
        self._thermal_data_saved = True
        return str(self.thermal_output_path)
    
    def calculate_radiance_from_saved_data(self, thermal_file_path: Optional[str] = None,
                                         radiance_output_path: Optional[str] = None) -> str:
        """
        Calculate radiance using saved thermal data.
        
        Args:
            thermal_file_path: Path to saved thermal data (uses instance path if None)
            radiance_output_path: Output path for radiance results
            
        Returns:
            Path to saved radiance results
        """
        from postprocessing.radiance_calculator import calculate_radiance_from_file
        
        # Use instance thermal file path if not provided
        if thermal_file_path is None:
            if not self._thermal_data_saved:
                thermal_file_path = self._save_thermal_data()
            else:
                thermal_file_path = self.thermal_output_path
        
        # Generate radiance output path if not provided
        if radiance_output_path is None:
            thermal_path = Path(thermal_file_path)
            radiance_output_path = thermal_path.parent / f"radiance_{thermal_path.stem}.h5"
        
        # Calculate radiance using post-processing module
        calculate_radiance_from_file(thermal_file_path, output_file_path=radiance_output_path)
        
        return str(radiance_output_path)


def run_thermal_simulation_with_config_file(config_file_path: str,
                                           parameter_overrides: Optional[Dict[str, Any]] = None,
                                           save_thermal_data: bool = True,
                                           calculate_radiance: bool = True) -> Dict[str, Any]:
    """
    High-level function to run thermal simulation from configuration file.
    
    Args:
        config_file_path: Path to YAML configuration file
        parameter_overrides: Optional parameter overrides
        save_thermal_data: Whether to save thermal evolution data
        calculate_radiance: Whether to calculate radiance (post-processing)
        
    Returns:
        Dictionary with simulation results and file paths
    """
    from core.config_manager import ConfigManager
    
    # Load configuration
    config_manager = ConfigManager(config_file_path)
    config = config_manager.create_config(parameter_overrides)
    
    # Validate configuration
    warnings = config_manager.validate_config(config)
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Run thermal simulation
    simulator = ThermalSimulator(
        config, 
        save_thermal_data=save_thermal_data,
        calculate_radiance=calculate_radiance
    )
    
    thermal_results = simulator.run()
    
    # Calculate radiance if requested and not done during simulation
    radiance_file_path = None
    if calculate_radiance and not simulator.calculate_radiance:
        radiance_file_path = simulator.calculate_radiance_from_saved_data()
        
    return {
        'thermal_results': thermal_results,
        'radiance_file_path': radiance_file_path,
        'config_warnings': warnings
    }


if __name__ == "__main__":
    # Example usage and testing
    from config import SimulationConfig
    
    # Test basic functionality
    config = SimulationConfig(
        ndays=1,
        diurnal=True,
        sun=True,
        crater=False,  # Start simple
        use_RTE=False,
        freq_out=24
    )
    
    print("Testing enhanced thermal simulator...")
    
    # Test thermal-only simulation (fast)
    simulator = ThermalSimulator(
        config, 
        save_thermal_data=True, 
        calculate_radiance=False
    )
    
    results = simulator.run()
    print(f"Thermal simulation completed: {results['thermal_file_path']}")
    
    # Test post-processing radiance calculation
    if results['thermal_file_path']:
        radiance_path = simulator.calculate_radiance_from_saved_data()
        print(f"Radiance calculation completed: {radiance_path}")
    
    print("Enhanced thermal simulator test completed successfully!")