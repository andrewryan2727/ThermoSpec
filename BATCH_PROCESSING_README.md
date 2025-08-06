# Thermal Model Batch Processing System

This document describes the new modular architecture for generating ML training datasets from thermal simulations.

## Overview

The refactored system separates thermal evolution from post-processing, enabling:
- **Efficient batch processing** for ML dataset generation
- **Independent radiance calculations** from saved thermal data
- **Flexible configuration management** with YAML files
- **Parallel execution** for large parameter sweeps
- **Persistent data storage** in HDF5 format

## Architecture

```
thermal_model/
├── core/                      # Core simulation and data management
│   ├── config_manager.py      # YAML configuration loading
│   ├── data_persistence.py    # HDF5 data saving/loading  
│   └── thermal_simulation.py  # Enhanced thermal simulator
├── postprocessing/            # Independent post-processing
│   └── radiance_calculator.py # Standalone radiance calculations
├── batch/                     # Batch processing and ML datasets
│   └── batch_runner.py        # Parallel batch execution
├── configs/                   # Configuration files
│   ├── base_configs/          # Planet-specific base configurations
│   └── batch_configs/         # Parameter sweep definitions
└── run_thermal_batch.py       # Command-line interface
```

## Quick Start

### 1. Create Configuration Files

Generate a base configuration file:
```bash
python run_thermal_batch.py create-config --output configs/bennu.yaml --planetary-body Bennu
```

Create a batch parameter sweep configuration (see `configs/batch_configs/ml_dataset_example.yaml`).

### 2. Run Single Simulation

```bash
# Basic run
python run_thermal_batch.py single --config configs/bennu.yaml

# With parameter overrides
python run_thermal_batch.py single --config configs/bennu.yaml --params Et=50000 radius=10e-6 ndays=2

# Thermal-only (skip radiance calculations)
python run_thermal_batch.py single --config configs/bennu.yaml --no-radiance
```

### 3. Run Batch Simulations

```bash
# Full batch for ML dataset generation
python run_thermal_batch.py batch \
    --base-config configs/bennu.yaml \
    --batch-config configs/batch_configs/ml_dataset_example.yaml \
    --output-dir results/ml_dataset_001 \
    --create-dataset

# Test run with limited simulations
python run_thermal_batch.py batch \
    --base-config configs/bennu.yaml \
    --batch-config configs/batch_configs/ml_dataset_example.yaml \
    --output-dir results/test_run \
    --max-runs 10 \
    --workers 4
```

### 4. Post-Process Saved Data

```bash
# Calculate radiance from saved thermal data
python run_thermal_batch.py postprocess \
    --thermal-file results/thermal_sim_12345678.h5 \
    --output radiance_results.h5

# Process specific time points
python run_thermal_batch.py postprocess \
    --thermal-file results/thermal_sim_12345678.h5 \
    --time-indices 0 6 12 18
```

## Configuration System

### Base Configuration Files (YAML)

Base configurations define all simulation parameters for a specific planetary body:

```yaml
# configs/base_configs/bennu.yaml
planetary_body:
  name: "Bennu"
  R: 1.0                    # Heliocentric distance (AU)
  P: 15450                  # Rotation period (s)
  
material_properties:
  k_dust: 4.2e-4           # Thermal conductivity (W/m/K)
  rho_dust: 1100           # Bulk density (kg/m³)
  Et: 90000                # Extinction coefficient (m⁻¹)
  
simulation_settings:
  use_RTE: true
  crater: true
  diurnal: true
  # ... all other parameters
```

### Batch Parameter Sweeps

Batch configurations define parameter spaces to explore:

```yaml
# configs/batch_configs/ml_dataset_example.yaml
parameter_sweeps:
  - name: "extinction_coefficient"
    parameter: "Et"
    type: "log_uniform"
    min: 10000.0
    max: 200000.0
    n_samples: 20
    
  - name: "particle_size"
    parameter: "radius" 
    type: "log_uniform"
    min: 5.0e-6
    max: 50.0e-6
    n_samples: 15

fixed_overrides:
  ndays: 1
  thermal_evolution_mode: "two_wave"  # Fast thermal evolution
  output_radiance_mode: "hybrid"     # Detailed spectral output
```

## Data Persistence

All simulation data is saved in HDF5 format for efficient storage and access:

### Thermal Results (`thermal_sim_*.h5`)
```
/thermal_data/
├── T_out[depth, time]           # Subsurface temperatures
├── T_surf_out[time]             # Surface temperatures  
├── T_crater_out[depth, facets, time]  # Crater temperatures
└── t_out[time]                  # Time array

/grid_data/
├── x_grid, x_boundaries         # Spatial grids
└── layer_structure              # Dust/rock layer info

/metadata/
├── config_used.yaml            # Full configuration
└── simulation_info             # Runtime metadata
```

### Radiance Results (`radiance_*.h5`)
```
/smooth_surface/
├── radiance_out[waves, time]    # Surface radiance spectra
└── brightness_temperature[time] # Brightness temperatures

/crater/
├── observer_radiance_out[observers, waves, time]  # Observer radiances
└── observer_vectors[observers, 3]                 # Observer directions
```

## Workflow Examples

### 1. ML Dataset Generation Workflow

```bash
# 1. Create base configuration for your planetary body
python run_thermal_batch.py create-config --output configs/my_planet.yaml

# 2. Edit configs/my_planet.yaml with planet-specific parameters

# 3. Create batch configuration defining parameter space
# Edit configs/batch_configs/my_ml_dataset.yaml

# 4. Run batch simulations
python run_thermal_batch.py batch \
    --base-config configs/my_planet.yaml \
    --batch-config configs/batch_configs/my_ml_dataset.yaml \
    --output-dir datasets/my_planet_v1 \
    --create-dataset

# 5. Results will be in datasets/my_planet_v1/
#    - thermal_results/: Individual thermal simulations  
#    - radiance_results/: Individual radiance calculations
#    - ml_dataset.h5: Train/val/test splits for ML
#    - batch_results.yaml: Summary statistics
```

### 2. Two-Stage Processing (Thermal + Post-Processing)

For very large datasets, you can separate thermal evolution from radiance calculations:

```bash
# Stage 1: Run thermal-only simulations (fast)
python run_thermal_batch.py batch \
    --base-config configs/bennu.yaml \
    --batch-config configs/thermal_only_batch.yaml \
    --output-dir thermal_stage \
    --workers 8

# Stage 2: Post-process radiances as needed
for thermal_file in thermal_stage/thermal_results/*.h5; do
    python run_thermal_batch.py postprocess --thermal-file "$thermal_file"
done
```

### 3. Parameter Space Exploration

Use different sweep types to explore parameter spaces:

```yaml
parameter_sweeps:
  # Linear sweep
  - parameter: "ndays"
    type: "linear"
    min: 1
    max: 10
    n_samples: 10
    
  # Logarithmic sweep  
  - parameter: "Et"
    type: "log_uniform"
    min: 1000
    max: 100000
    n_samples: 50
    
  # Discrete values
  - parameter: "crater"
    type: "list"
    values: [true, false]
    
  # Random sampling
  - parameter: "latitude"
    type: "uniform"
    min: 0.0
    max: 1.57  # π/2
    n_samples: 20
```

## Integration with Existing Code

The new system is designed to be compatible with existing workflows:

### Option 1: Use Enhanced Simulator Directly
```python
from core.thermal_simulation import ThermalSimulator
from config import SimulationConfig

config = SimulationConfig(crater=True, diurnal=True, ndays=1)
simulator = ThermalSimulator(config, save_thermal_data=True)
results = simulator.run()
```

### Option 2: Use Configuration Files
```python
from core.thermal_simulation import run_thermal_simulation_with_config_file

results = run_thermal_simulation_with_config_file(
    "configs/bennu.yaml",
    parameter_overrides={'Et': 50000, 'radius': 10e-6}
)
```

### Option 3: Post-Process Existing Data
```python
from postprocessing.radiance_calculator import calculate_radiance_from_file

calculate_radiance_from_file(
    "thermal_sim_results.h5",
    output_file_path="radiance_results.h5"
)
```

## Performance Considerations

- **Thermal-only mode**: Skip radiance calculations during batch runs for 5-10x speedup
- **Parallel execution**: Use multiple workers for batch processing
- **HDF5 compression**: Reduces file sizes by 50-80%
- **Memory management**: Large datasets are processed in chunks
- **Progress tracking**: Batch runs save progress periodically

## File Formats and Compatibility

- **Configuration files**: YAML format (human-readable, version controllable)
- **Simulation data**: HDF5 format (efficient, language-agnostic)
- **Metadata**: YAML embedded in HDF5 files
- **ML datasets**: HDF5 with train/val/test splits

## Cluster Computing

The batch system is designed for cluster environments:

```bash
# Submit batch job to SLURM
sbatch --array=1-1000 run_batch_array.sh

# Where run_batch_array.sh contains:
python run_thermal_batch.py single \
    --config configs/cluster_config.yaml \
    --params run_id=$SLURM_ARRAY_TASK_ID \
    --save-thermal \
    --no-radiance
```

## Future Extensions

The modular architecture supports easy extension:

- **New RTE solvers**: Add to `postprocessing/` 
- **Additional analysis**: Extend `postprocessing/spectral_analysis.py`
- **Different file formats**: Modify `core/data_persistence.py`
- **ML model integration**: Add to `batch/ml_interface.py`
- **Visualization tools**: Add to `postprocessing/visualization.py`

## Troubleshooting

### Common Issues

1. **File not found errors**: Check that all paths in configuration files are absolute or relative to the working directory.

2. **Memory errors**: Reduce `n_workers` or process smaller batches.

3. **Configuration validation warnings**: Check parameter ranges and file paths.

4. **HDF5 compatibility**: Ensure h5py is installed (`pip install h5py`).

### Debug Mode

Add debugging to any command:
```bash
python -v run_thermal_batch.py batch ... 2>&1 | tee debug.log
```

### Validation

Test your setup with a small batch:
```bash
python run_thermal_batch.py batch \
    --base-config configs/bennu.yaml \
    --batch-config configs/test_batch.yaml \
    --output-dir test_validation \
    --max-runs 3 \
    --workers 1
```