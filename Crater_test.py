
import sys
import os
from pathlib import Path

# Enable interactive widgets
#import ipywidgets as widgets
#from IPython.display import display
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Import thermal model components
from core.config_manager import ConfigManager, create_default_base_config
from core.thermal_simulation import ThermalSimulator
from modelmain import Simulator
from config import SimulationConfig
from modelmain import fit_blackbody_wn_banded, max_btemp_blackbody, emissionT

from radiance_processor import calculate_radiances_from_results, recompute_radiance_with_angles


# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['lines.linewidth'] = 2

print("Imports completed successfully!")


# Option 1: Create a new default configuration
config_path = "../configs/base_configs/analysis_template.yaml"

# Create config directory if it doesn't exist
os.makedirs(os.path.dirname(config_path), exist_ok=True)

# Create default configuration for analysis
create_default_base_config(config_path)

# Load the configuration
config_manager = ConfigManager(config_path)

# Create base configuration
base_config = config_manager.create_config()

print(f"Configuration created at: {config_path}")
print(f"Loaded configuration.")


# Modify base configuration for our analysis
settings = {
    'diurnal': True,                       # Steady state simulation
    'sun': True,                           # Include solar heating
    'T_fixed': False,                        # Temperature is fixed, prevents thermal evolution
    'enable_diurnal_convergence': False,
    'thermal_evolution_mode': 'two_wave',   #Run thermal evolution later with broadband vis (turned off) and broadband thermal. 
    'RTE_solver': 'disort',
    'output_radiance_mode': 'two_wave',       #Compute spectral radiance in thermal only. 
    'depth_dependent_properties': False,
    'temperature_dependent_properties': False,
    'temp_dependent_cp': True,
    'temp_dependent_k': False,      #Only use for non-RTE model. Temperature-dependence of conductivity comes naturally in RTE model. 
    'temp_change_threshold': 1.0,
    'P': 15450,
    'dtfac': 100,
    'minsteps': 5000,
    'ndays': 2,
    'observer_mu': 1.0,                     # Observer zenith angle (0 for nadir)
    'Et': 500.0,                          # Mean extinction coefficient. For phi=0.37, 50300. For phi=0.60, 81600. 
    'eta': 1.0,                             #Vis/thermal extinction coefficient ratio. 
    'ssalb_therm': 0.1,                     # Single scattering albedo for thermal radiation (average from mie code)
    'g_therm': 0.0,                         # Asymmetry parameter for thermal radiation (average from mie code)
    'ssalb_vis': 0.50,
    'g_vis': 0.0,
    'R_base': 0.0,                          # Global reflectivity value for substrate     
    'disort_space_temp': 0.0,              # Cold shroud temperature
    'single_layer': True,                   # Use single-layer model
    'dust_thickness': 5.0,                 # 10 cm
    'T_bottom': 260,                        # Sample base fixed at 500 K
    'bottom_bc': 'dirichlet',               # Bottom boundary condition 
    'crater': True,
    'nstr_out': 16,                # Number of streams for output
    'nmom_out': 16,                # Number of moments for output
    'nstr': 4,
    'nmom': 4,
    'fill_frac': 0.37,                      #Fill fraction for particles. 
    'radius': 14.e-6,                       #Particle radius in meters.
    'mie_file_out': '/Users/ryan/Research/RT_models/RT_thermal_model/Optical_props/enst_300K_mie_combined.txt',  # Mie file for output
    'wn_bounds_out': '/Users/ryan/Research/RT_models/RT_thermal_model/Optical_props/enst_300k_wn_bounds.txt',  # Wavenumber bounds for output files,
    'nstr_out': 16,                # Number of streams for output
    'nmom_out': 16,                # Number of moments for output
    'scale_Et': False
}

# Create configuration with overrides
config1 = config_manager.create_config(settings)

print("Running baseline simulation...")
print(f"Configuration: Et={config1.Et}, k_dust={config1.k_dust}, thickness={config1.dust_thickness}m")

# Run simulation
sim1 = Simulator(config1)
T_out1, phi_vis1, phi_therm1, T_surf1, t_out1 = sim1.run()


print(f"Simulation completed! Output shape: {T_out1.shape}")
print(f"Time range: {t_out1[0]:.0f} to {t_out1[-1]:.0f} seconds")
print(f"Surface temperature range: {T_surf1.min():.1f} to {T_surf1.max():.1f} K")


from matplotlib.ticker import FuncFormatter, LogLocator

# Plot temperature profiles vs depth (log scale) with a second y-axis for tau (opacity, non-log)
fig, ax1 = plt.subplots()
time_fracs = np.linspace(0, 1, len(sim1.t_out))
plot_fracs = np.linspace(0, 0.9, 10)
for frac in plot_fracs:
    idx = np.argmin(np.abs(time_fracs - frac))
    if hasattr(sim1.cfg.Et,'shape'):
        x_m = sim1.grid.x[1:] / sim1.cfg.Et[1:]
    else:
        x_m = sim1.grid.x[1:] / sim1.cfg.Et
    ax1.plot(T_out1[1:, idx], x_m, label=f't+{frac:.1f}P')

ax1.set_ylabel('Depth')
ax1.set_xlabel('Temperature (K)')
ax1.legend()
ax1.invert_yaxis()

# Set custom ticks for more human-readable units
def depth_formatter(x, pos):
    if x < 1e-3:
        return f"{x*1e6:.0f} µm"
    elif x < 1:
        return f"{x*1e3:.0f} mm"
    else:
        return f"{x:.0f} m"

ax1.set_yscale('log')
ax1.yaxis.set_major_locator(LogLocator(base=10))
ax1.yaxis.set_major_formatter(FuncFormatter(depth_formatter))

# Add a second y-axis for tau (opacity)
def tau_formatter(tau, pos):
    return f"{tau:.2f}"

# Create the secondary axis
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim())
# Convert the log-scale x_m limits back to tau
ylim = ax1.get_ylim()
if hasattr(sim1.cfg.Et,'shape'):
    tau_min = ylim[0] * sim1.cfg.Et[1]
    tau_max = ylim[1] * sim1.cfg.Et[-1]
else:
    tau_min = ylim[0] * sim1.cfg.Et
    tau_max = ylim[1] * sim1.cfg.Et
ax2.set_yscale('log')
ax2.set_ylim(tau_min, tau_max)
ax2.yaxis.set_major_locator(LogLocator(base=10))
ax2.yaxis.set_major_formatter(FuncFormatter(tau_formatter))
ax2.set_ylabel('Optical Depth (tau)')

plt.tight_layout()
plt.show()

from modelmain import interactive_crater_temp_viewer
interactive_crater_temp_viewer(sim1.crater_mesh, sim1.T_surf_crater_out, sim1.grid.dt)

