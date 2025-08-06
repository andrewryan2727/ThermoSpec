"""
Batch runner for generating ML training datasets from thermal simulations.

Handles parameter sweeps, parallel execution, and dataset organization
for machine learning model training.
"""

import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import multiprocessing as mp
from dataclasses import asdict
import yaml
import h5py
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.config_manager import ConfigManager, BatchParameterGenerator
from core.data_persistence import ThermalResultsWriter, create_simulation_data_dict
from postprocessing.radiance_calculator import calculate_radiance_from_file
from modelmain import Simulator


class BatchSimulationRunner:
    """
    Runs batches of thermal simulations for ML dataset generation.
    """
    
    def __init__(self, base_config_path: str, batch_config_path: str, 
                 output_dir: str, n_workers: Optional[int] = None):
        """
        Initialize batch runner.
        
        Args:
            base_config_path: Path to base configuration YAML file
            batch_config_path: Path to batch parameter configuration YAML file  
            output_dir: Directory to save all simulation results
            n_workers: Number of parallel workers (None for auto-detect)
        """
        self.base_config_path = Path(base_config_path)
        self.batch_config_path = Path(batch_config_path)
        self.output_dir = Path(output_dir)
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'thermal_results').mkdir(exist_ok=True)
        (self.output_dir / 'radiance_results').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load configurations
        self.config_manager = ConfigManager(self.base_config_path)
        self.batch_config = BatchParameterGenerator.load_batch_config(self.batch_config_path)
        
        # Generate parameter combinations
        self.parameter_combinations = BatchParameterGenerator.generate_parameter_combinations(
            self.batch_config
        )
        
        self.logger.info(f"Generated {len(self.parameter_combinations)} parameter combinations")
        
    def _setup_logging(self):
        """Setup logging for batch processing."""
        log_file = self.output_dir / 'logs' / 'batch_runner.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def run_single_simulation(self, run_id: int, parameter_overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single thermal simulation with given parameters.
        
        Args:
            run_id: Unique identifier for this simulation run
            parameter_overrides: Parameter values to override from base config
            
        Returns:
            Dictionary with run results and metadata
        """
        start_time = time.time()
        
        try:
            # Create configuration for this run
            config = self.config_manager.create_config(parameter_overrides)
            
            # Validate configuration
            warnings = self.config_manager.validate_config(config)
            if warnings:
                self.logger.warning(f"Run {run_id} configuration warnings: {warnings}")
            
            # Check if radiance should be calculated during simulation or post-processed
            calculate_radiance_during_sim = self.batch_config.get('output_settings', {}).get('calculate_radiance_during_simulation', True)
            save_radiance_data = self.batch_config.get('output_settings', {}).get('save_radiance_data', True)
            
            # Run thermal simulation with optional radiance calculation
            simulator = Simulator(config)
            thermal_results = simulator.run(calculate_radiance=calculate_radiance_during_sim)
            
            # Save thermal results
            thermal_output_path = self.output_dir / 'thermal_results' / f'thermal_run_{run_id:06d}.h5'
            simulation_data = create_simulation_data_dict(simulator)
            
            writer = ThermalResultsWriter(thermal_output_path, config)
            writer.save_thermal_results(simulation_data)
            
            # Post-process radiance if not calculated during simulation but requested
            radiance_output_path = None
            if save_radiance_data and not calculate_radiance_during_sim:
                radiance_output_path = self.output_dir / 'radiance_results' / f'radiance_run_{run_id:06d}.h5'
                try:
                    calculate_radiance_from_file(
                        str(thermal_output_path), 
                        output_file_path=str(radiance_output_path)
                    )
                except Exception as e:
                    self.logger.error(f"Run {run_id} radiance post-processing failed: {e}")
                    radiance_output_path = None
            elif save_radiance_data and calculate_radiance_during_sim:
                # Radiance was calculated during simulation, indicate success
                radiance_output_path = f"calculated_during_simulation_run_{run_id:06d}"
            
            run_time = time.time() - start_time
            
            # Return run metadata
            return {
                'run_id': run_id,
                'status': 'success',
                'run_time': run_time,
                'parameters': parameter_overrides,
                'thermal_output_path': str(thermal_output_path),
                'radiance_output_path': str(radiance_output_path) if radiance_output_path else None,
                'config_hash': hash(str(sorted(parameter_overrides.items()))),
                'warnings': warnings
            }
            
        except Exception as e:
            run_time = time.time() - start_time
            error_msg = f"Run {run_id} failed: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            
            return {
                'run_id': run_id,
                'status': 'failed',
                'error': str(e),
                'run_time': run_time,
                'parameters': parameter_overrides
            }
    
    def run_batch(self, max_runs: Optional[int] = None, 
                  save_progress: bool = True) -> Dict[str, Any]:
        """
        Run complete batch of simulations.
        
        Args:
            max_runs: Maximum number of runs to execute (None for all)
            save_progress: Whether to save progress periodically
            
        Returns:
            Batch execution results and statistics
        """
        # Limit number of runs if requested
        combinations_to_run = self.parameter_combinations[:max_runs] if max_runs else self.parameter_combinations
        n_runs = len(combinations_to_run)
        
        self.logger.info(f"Starting batch execution: {n_runs} runs with {self.n_workers} workers")
        
        # Track results
        batch_results = {
            'batch_metadata': {
                'base_config': str(self.base_config_path),
                'batch_config': str(self.batch_config_path),
                'output_dir': str(self.output_dir),
                'n_runs': n_runs,
                'n_workers': self.n_workers,
                'start_time': time.time()
            },
            'run_results': [],
            'statistics': {
                'completed': 0,
                'failed': 0,
                'total_time': 0.0
            }
        }
        
        # Run simulations in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(run_single_simulation_worker, 
                              str(self.base_config_path), self.batch_config, run_id, params, str(self.output_dir)): run_id
                for run_id, params in enumerate(combinations_to_run)
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                run_id = futures[future]
                
                try:
                    result = future.result()
                    batch_results['run_results'].append(result)
                    
                    if result['status'] == 'success':
                        batch_results['statistics']['completed'] += 1
                        self.logger.info(f"Completed run {run_id} ({result['run_time']:.1f}s)")
                    else:
                        batch_results['statistics']['failed'] += 1
                        self.logger.error(f"Failed run {run_id}: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    batch_results['statistics']['failed'] += 1
                    self.logger.error(f"Future exception for run {run_id}: {e}")
                
                # Save progress periodically
                if save_progress and len(batch_results['run_results']) % 10 == 0:
                    self._save_batch_progress(batch_results)
        
        # Final statistics
        batch_results['batch_metadata']['end_time'] = time.time()
        batch_results['statistics']['total_time'] = (
            batch_results['batch_metadata']['end_time'] - 
            batch_results['batch_metadata']['start_time']
        )
        
        # Save final results
        self._save_batch_results(batch_results)
        
        self.logger.info(f"Batch execution completed: {batch_results['statistics']['completed']} success, "
                        f"{batch_results['statistics']['failed']} failed, "
                        f"{batch_results['statistics']['total_time']:.1f}s total")
        
        return batch_results
    
    def _save_batch_progress(self, batch_results: Dict[str, Any]):
        """Save batch progress to file."""
        progress_file = self.output_dir / 'batch_progress.yaml'
        with open(progress_file, 'w') as f:
            yaml.dump(batch_results, f, default_flow_style=False)
    
    def _save_batch_results(self, batch_results: Dict[str, Any]):
        """Save final batch results."""
        results_file = self.output_dir / 'batch_results.yaml'
        with open(results_file, 'w') as f:
            yaml.dump(batch_results, f, default_flow_style=False)
        
        # Also save as compressed HDF5 for large datasets
        self._save_batch_results_hdf5(batch_results)
    
    def _save_batch_results_hdf5(self, batch_results: Dict[str, Any]):
        """Save batch results in HDF5 format for efficient access."""
        results_file = self.output_dir / 'batch_results.h5'
        
        with h5py.File(results_file, 'w') as f:
            # Metadata
            meta_group = f.create_group('metadata')
            for key, value in batch_results['batch_metadata'].items():
                meta_group.attrs[key] = str(value)
            
            # Statistics
            stats_group = f.create_group('statistics')
            for key, value in batch_results['statistics'].items():
                stats_group.attrs[key] = value
            
            # Parameter combinations and results
            runs_group = f.create_group('runs')
            
            # Extract parameter arrays for efficient storage
            if batch_results['run_results']:
                # Get parameter names from first successful run
                param_names = list(batch_results['run_results'][0]['parameters'].keys())
                n_runs = len(batch_results['run_results'])
                
                # Create datasets for each parameter
                for param_name in param_names:
                    param_values = [
                        result['parameters'].get(param_name, np.nan) 
                        for result in batch_results['run_results']
                    ]
                    runs_group.create_dataset(f'param_{param_name}', data=param_values)
                
                # Create datasets for run metadata
                run_ids = [result['run_id'] for result in batch_results['run_results']]
                statuses = [result['status'] for result in batch_results['run_results']]
                run_times = [result['run_time'] for result in batch_results['run_results']]
                
                runs_group.create_dataset('run_id', data=run_ids)
                runs_group.create_dataset('status', data=statuses)
                runs_group.create_dataset('run_time', data=run_times)


def run_single_simulation_worker(base_config_path: str, batch_config: Dict[str, Any], 
                                run_id: int, parameter_overrides: Dict[str, Any], 
                                output_dir: str) -> Dict[str, Any]:
    """
    Worker function for running single simulation in parallel process.
    
    This function is defined at module level to be pickle-able for multiprocessing.
    """
    import time
    from pathlib import Path
    
    # Setup minimal logging for worker
    logging.basicConfig(level=logging.ERROR)
    
    start_time = time.time()
    
    try:
        # Recreate config manager in worker process
        config_manager = ConfigManager(base_config_path)
        config = config_manager.create_config(parameter_overrides)
        
        # Check radiance calculation settings from batch config
        calculate_radiance_during_sim = batch_config.get('output_settings', {}).get('calculate_radiance_during_simulation', True)
        save_radiance_data = batch_config.get('output_settings', {}).get('save_radiance_data', True)
        
        # Run simulation with appropriate radiance settings
        simulator = Simulator(config)
        thermal_results = simulator.run(calculate_radiance=calculate_radiance_during_sim)
        
        # Save thermal results
        output_dir_path = Path(output_dir)
        thermal_output_path = output_dir_path / 'thermal_results' / f'thermal_run_{run_id:06d}.h5'
        simulation_data = create_simulation_data_dict(simulator)
        
        writer = ThermalResultsWriter(thermal_output_path, config)
        writer.save_thermal_results(simulation_data)
        
        # Handle radiance calculation and saving
        radiance_output_path = None
        if save_radiance_data and not calculate_radiance_during_sim:
            # Post-process radiance from saved thermal data
            radiance_output_path = output_dir_path / 'radiance_results' / f'radiance_run_{run_id:06d}.h5'
            try:
                calculate_radiance_from_file(
                    str(thermal_output_path), 
                    output_file_path=str(radiance_output_path)
                )
            except Exception as e:
                print(f"Run {run_id} radiance post-processing failed: {e}")
                radiance_output_path = None
        elif save_radiance_data and calculate_radiance_during_sim:
            # Radiance was calculated during simulation
            radiance_output_path = f"calculated_during_simulation_run_{run_id:06d}"
        
        run_time = time.time() - start_time
        
        # Return success with file paths
        return {
            'run_id': run_id,
            'status': 'success', 
            'parameters': parameter_overrides,
            'run_time': run_time,
            'config_hash': hash(str(sorted(parameter_overrides.items()))),
            'thermal_output_path': str(thermal_output_path),
            'radiance_output_path': str(radiance_output_path) if radiance_output_path else None
        }
        
    except Exception as e:
        run_time = time.time() - start_time
        import traceback
        return {
            'run_id': run_id,
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'parameters': parameter_overrides,
            'run_time': run_time
        }


def create_ml_dataset_from_batch(batch_output_dir: str, 
                               dataset_output_path: str,
                               train_fraction: float = 0.8,
                               val_fraction: float = 0.1,
                               test_fraction: float = 0.1,
                               random_seed: int = 42) -> Dict[str, str]:
    """
    Create ML-ready dataset from batch simulation results.
    
    Args:
        batch_output_dir: Directory containing batch simulation results
        dataset_output_path: Path to save ML dataset
        train_fraction: Fraction of data for training
        val_fraction: Fraction of data for validation
        test_fraction: Fraction of data for testing
        random_seed: Random seed for reproducible splits
        
    Returns:
        Dictionary with paths to train/val/test datasets
    """
    # Load batch results
    batch_dir = Path(batch_output_dir)
    batch_results_file = batch_dir / 'batch_results.h5'
    
    if not batch_results_file.exists():
        raise FileNotFoundError(f"Batch results not found: {batch_results_file}")
    
    # Load successful runs
    with h5py.File(batch_results_file, 'r') as f:
        runs_group = f['runs']
        run_ids = runs_group['run_id'][:]
        statuses = runs_group['status'][:]
        
        # Filter successful runs
        success_mask = [status == b'success' for status in statuses]
        successful_run_ids = run_ids[success_mask]
    
    # Split into train/val/test
    np.random.seed(random_seed)
    n_success = len(successful_run_ids)
    indices = np.random.permutation(n_success)
    
    n_train = int(n_success * train_fraction)
    n_val = int(n_success * val_fraction)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create dataset splits
    dataset_splits = {
        'train': successful_run_ids[train_indices],
        'validation': successful_run_ids[val_indices], 
        'test': successful_run_ids[test_indices]
    }
    
    # Save dataset metadata
    dataset_metadata = {
        'batch_output_dir': str(batch_output_dir),
        'dataset_creation_time': time.time(),
        'n_total_runs': len(run_ids),
        'n_successful_runs': n_success,
        'split_fractions': {
            'train': train_fraction,
            'validation': val_fraction,
            'test': test_fraction
        },
        'split_sizes': {
            'train': len(train_indices),
            'validation': len(val_indices),
            'test': len(test_indices)
        },
        'random_seed': random_seed
    }
    
    # Save dataset
    dataset_output_path = Path(dataset_output_path)
    with h5py.File(dataset_output_path, 'w') as f:
        # Save metadata
        meta_group = f.create_group('metadata')
        meta_group.attrs['dataset_creation_time'] = dataset_metadata['dataset_creation_time']
        meta_group.attrs['n_total_runs'] = dataset_metadata['n_total_runs']
        meta_group.attrs['random_seed'] = dataset_metadata['random_seed']
        
        # Save splits
        for split_name, run_ids in dataset_splits.items():
            f.create_dataset(f'splits/{split_name}', data=run_ids)
            
        # Save metadata as YAML string
        metadata_yaml = yaml.dump(dataset_metadata, default_flow_style=False)
        meta_group.create_dataset('metadata_yaml', data=metadata_yaml.encode('utf-8'))
    
    print(f"ML dataset created: {dataset_output_path}")
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    return {
        'dataset_path': str(dataset_output_path),
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'test_size': len(test_indices)
    }


if __name__ == "__main__":
    # Example usage
    base_config = "configs/base_configs/bennu.yaml"
    batch_config = "configs/batch_configs/ml_dataset_example.yaml"
    output_dir = "batch_outputs/ml_dataset_001"
    
    if len(sys.argv) > 1:
        # Allow command line override
        base_config = sys.argv[1]
        batch_config = sys.argv[2] if len(sys.argv) > 2 else batch_config
        output_dir = sys.argv[3] if len(sys.argv) > 3 else output_dir
    
    print(f"Running batch simulation:")
    print(f"  Base config: {base_config}")
    print(f"  Batch config: {batch_config}")
    print(f"  Output dir: {output_dir}")
    
    # Run batch simulation
    runner = BatchSimulationRunner(base_config, batch_config, output_dir)
    results = runner.run_batch(max_runs=10)  # Test with 10 runs
    
    print(f"Batch completed: {results['statistics']}")