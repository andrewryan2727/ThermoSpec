#!/usr/bin/env python3
"""
Command-line interface for thermal model batch processing.

Provides easy access to thermal simulation, radiance post-processing,
and ML dataset generation workflows.
"""

import argparse
import sys
from pathlib import Path
import yaml
import time

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from core.config_manager import ConfigManager, create_default_base_config
from core.thermal_simulation import run_thermal_simulation_with_config_file
from batch.batch_runner import BatchSimulationRunner, create_ml_dataset_from_batch
from postprocessing.radiance_calculator import calculate_radiance_from_file


def create_config_command(args):
    """Create a default configuration file."""
    create_default_base_config(args.output, args.planetary_body)
    print(f"Default configuration created: {args.output}")


def single_run_command(args):
    """Run a single thermal simulation."""
    print(f"Running single thermal simulation...")
    print(f"  Config file: {args.config}")
    
    # Parse parameter overrides if provided
    parameter_overrides = {}
    if args.params:
        for param_str in args.params:
            try:
                key, value = param_str.split('=')
                # Try to convert to float, int, or bool
                if value.lower() in ['true', 'false']:
                    parameter_overrides[key] = value.lower() == 'true'
                elif '.' in value:
                    parameter_overrides[key] = float(value)
                else:
                    try:
                        parameter_overrides[key] = int(value)
                    except ValueError:
                        parameter_overrides[key] = value  # Keep as string
            except ValueError:
                print(f"Warning: Invalid parameter format '{param_str}', expected 'key=value'")
    
    if parameter_overrides:
        print(f"  Parameter overrides: {parameter_overrides}")
    
    # Run simulation
    start_time = time.time()
    results = run_thermal_simulation_with_config_file(
        args.config,
        parameter_overrides=parameter_overrides,
        save_thermal_data=args.save_thermal,
        calculate_radiance=args.calculate_radiance
    )
    run_time = time.time() - start_time
    
    print(f"Simulation completed in {run_time:.1f} seconds")
    
    if results['thermal_results']['thermal_file_path']:
        print(f"Thermal data saved: {results['thermal_results']['thermal_file_path']}")
    
    if results['radiance_file_path']:
        print(f"Radiance data saved: {results['radiance_file_path']}")
    
    if results['config_warnings']:
        print("Configuration warnings:")
        for warning in results['config_warnings']:
            print(f"  - {warning}")


def batch_run_command(args):
    """Run batch simulations for ML dataset generation."""
    print(f"Running batch simulations...")
    print(f"  Base config: {args.base_config}")
    print(f"  Batch config: {args.batch_config}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Max runs: {args.max_runs if args.max_runs else 'all'}")
    print(f"  Workers: {args.workers}")
    
    # Create batch runner
    runner = BatchSimulationRunner(
        args.base_config,
        args.batch_config, 
        args.output_dir,
        n_workers=args.workers
    )
    
    # Run batch
    start_time = time.time()
    results = runner.run_batch(max_runs=args.max_runs)
    total_time = time.time() - start_time
    
    print(f"Batch execution completed in {total_time:.1f} seconds")
    print(f"Results: {results['statistics']['completed']} successful, {results['statistics']['failed']} failed")
    
    # Create ML dataset if requested
    if args.create_dataset:
        print("Creating ML dataset...")
        dataset_path = Path(args.output_dir) / "ml_dataset.h5"
        dataset_info = create_ml_dataset_from_batch(
            args.output_dir,
            str(dataset_path),
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            random_seed=args.random_seed
        )
        print(f"ML dataset created: {dataset_info}")


def postprocess_command(args):
    """Run radiance post-processing on saved thermal data."""
    print(f"Running radiance post-processing...")
    print(f"  Thermal file: {args.thermal_file}")
    
    output_path = args.output if args.output else None
    
    start_time = time.time()
    calculate_radiance_from_file(
        args.thermal_file,
        time_indices=args.time_indices,
        output_file_path=output_path
    )
    run_time = time.time() - start_time
    
    print(f"Radiance post-processing completed in {run_time:.1f} seconds")
    if output_path:
        print(f"Results saved: {output_path}")


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Thermal Model Batch Processing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create default configuration
  python run_thermal_batch.py create-config --output configs/my_config.yaml --planetary-body Mars
  
  # Run single simulation with parameter overrides
  python run_thermal_batch.py single --config configs/my_config.yaml --params Et=50000 radius=10e-6
  
  # Run batch simulations for ML dataset
  python run_thermal_batch.py batch --base-config configs/bennu.yaml --batch-config configs/ml_batch.yaml --output-dir results/dataset_001
  
  # Post-process thermal data to calculate radiance
  python run_thermal_batch.py postprocess --thermal-file results/thermal_sim_12345678.h5 --output radiance_results.h5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Create default configuration file')
    config_parser.add_argument('--output', '-o', required=True, help='Output configuration file path')
    config_parser.add_argument('--planetary-body', default='generic', help='Planetary body name for defaults')
    config_parser.set_defaults(func=create_config_command)
    
    # Single run command
    single_parser = subparsers.add_parser('single', help='Run single thermal simulation')
    single_parser.add_argument('--config', '-c', required=True, help='Configuration file path')
    single_parser.add_argument('--params', '-p', nargs='*', help='Parameter overrides (key=value)')
    single_parser.add_argument('--save-thermal', action='store_true', default=True, help='Save thermal data')
    single_parser.add_argument('--no-save-thermal', dest='save_thermal', action='store_false', help='Do not save thermal data')
    single_parser.add_argument('--calculate-radiance', action='store_true', default=True, help='Calculate radiance')
    single_parser.add_argument('--no-radiance', dest='calculate_radiance', action='store_false', help='Skip radiance calculation')
    single_parser.set_defaults(func=single_run_command)
    
    # Batch run command
    batch_parser = subparsers.add_parser('batch', help='Run batch simulations')
    batch_parser.add_argument('--base-config', '-b', required=True, help='Base configuration file')
    batch_parser.add_argument('--batch-config', '-B', required=True, help='Batch parameter configuration file')  
    batch_parser.add_argument('--output-dir', '-o', required=True, help='Output directory for batch results')
    batch_parser.add_argument('--max-runs', '-n', type=int, help='Maximum number of runs (default: all)')
    batch_parser.add_argument('--workers', '-w', type=int, help='Number of parallel workers (default: auto)')
    batch_parser.add_argument('--create-dataset', action='store_true', help='Create ML dataset after batch completion')
    batch_parser.add_argument('--train-fraction', type=float, default=0.8, help='Training set fraction')
    batch_parser.add_argument('--val-fraction', type=float, default=0.1, help='Validation set fraction')
    batch_parser.add_argument('--random-seed', type=int, default=42, help='Random seed for dataset splits')
    batch_parser.set_defaults(func=batch_run_command)
    
    # Post-processing command
    postprocess_parser = subparsers.add_parser('postprocess', help='Run radiance post-processing')
    postprocess_parser.add_argument('--thermal-file', '-t', required=True, help='Thermal simulation results file')
    postprocess_parser.add_argument('--output', '-o', help='Output file for radiance results')
    postprocess_parser.add_argument('--time-indices', nargs='*', type=int, help='Specific time indices to process')
    postprocess_parser.set_defaults(func=postprocess_command)
    
    # Parse arguments and run command
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}")
        if args.command == 'batch':
            print("Check that all configuration files exist and are valid.")
        sys.exit(1)


if __name__ == "__main__":
    main()