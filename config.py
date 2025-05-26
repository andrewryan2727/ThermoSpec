from dataclasses import dataclass, field
import numpy as np

# -----------------------------------------------------------------------------
# File: config.py
# Description: Planetary regolith thermal model, including Hapke's 1996 radiative transfer 2-stream approximation. .
# Author: Andrew J. Ryan, 2025
#
# This code is free to use, modify, and distribute for any purpose.
# Please contact the author (ajryan4@arizona.edu) to discuss applications or if you
# use this code in your research or projects.
# -----------------------------------------------------------------------------

#Pick your poison!

@dataclass
class SimulationConfig:
    # Radiative properties
    ssalb_vis: float = 0.10          # single-scattering visible albedo
    gamma_therm: float = 0.9         # thermal albedo factor
    Et: float = 7000.0               # thermal extinction coefficient (m^-1)
    eta: float = 1.0                 # visible/thermal extinction ratio 
    em: float = 0.95                 # thermal emissivity, ONLY USED FOR NON-RTE MODELS
    albedo: float = 0.02              # surface albedo, ONLY USED FOR NON-RTE MODELS   

    # Orbital & rotational parameters
    R: float = 1.0                   # heliocentric distance (AU)
    latitude: float = 55.0 * np.pi/180.0  # latitude (rad)
    P: float = 4.293057 * 3600.0        #Rotational period in s

    # Dust (or top layer) material properties
    k_dust: float = 5.5e-4           # dust thermal conductivity (W/m/K). If using RTE model, this should just be phonon conduction. 
    rho_dust: float = 1100.0         # dust bulk density (kg/m^3)
    cp_dust: float = 775.0           # dust specific heat (J/kg/K)
    k_dust_auto: bool = True  # use auto-calculated dust thermal conductivity for non-RTE models. If False, use k_dust value directly.

    # Rock (or substrate) material properties
    k_rock: float = 0.5              # rock thermal conductivity (W/m/K)
    rho_rock: float = 1700.0         # rock density (kg/m^3)
    cp_rock: float = 775.0           # rock heat capacity (J/kg/K)

    # Boundary & layer settings
    T_bottom: float = 260.         # bottom boundary and global initialization temperature (K)
    dust_thickness: float = 0.2    # dust column total thickness (m)
    rock_thickness: float = 0.45     # rock substrate column total thickness (m)
    dust_lthick: float = 0.02        # dust node spacing (tau units, i.e., optical opacity!)
    rock_lthick: float = 0.0005      # rock node spacing (m)
    geometric_spacing: bool = True  # Node spacing increases by factor spacing_factor, otherwise constant thickness. Only applies to single layer scenario. 
    spacing_factor: float = 1.05     # layer thickness increase factor for geometric spacing. Only applies to single layer scenario!

    # Simulation flags and convergence settings
    single_layer: bool = True        # use single-layer model instead of two-layer
    use_RTE: bool = True             # use radiative transfer model
    bottom_bc: str = 'neumann'     # bottom boundary condition choices: "neumann" (zero‐flux), "dirichlet" (fixed T_bottom)
    sun: bool = True                 # include solar input
    diurnal: bool = True             # include diurnal variation
    custom_bvp: bool = True          # use the custom written bvp solver for RTE. Otherwise, reverts to scipy.solve_bvp
    bvp_tol: float = 1.0e-8          # tolerance for BVP solver
    bvp_max_iter: float = 1000       # max iterations for BVP solver
    T_surf_tol: float = 1.0e-4      # tolerance for surface temperature convergence
    T_surf_max_iter: int = 50       # max iterations for surface temperature convergence

    # Time-stepping parameters
    tsteps_day: int = 8000          # time steps per day
    ndays: int = 5                  # total simulation days


    # Physical constants
    sigma: float = 5.670374e-8       # Stefan-Boltzmann constant (W/m^2/K^4)

    # Computed fields
    gamma_vis: float = field(init=False)
    J: float = field(init=False)
    q: float = field(init=False)
    #rock_skin_depth: float = field(init=False)

    def __post_init__(self):
        # Derived radiative parameter
        self.gamma_vis = np.sqrt(1.0 - self.ssalb_vis)
        # Solar irradiance at distance R (W/m^2)
        self.J = 1366.0 / (self.R ** 2.0)
        # Radiative ratio q
        self.q = 1.0 / (self.k_dust * self.Et)
        if(self.k_dust_auto and not self.use_RTE):
            #Add the estimation for the radiative term from Hapke's book, equation 16.31. 
            self.k_dust += (4.0/self.Et)*self.sigma*self.T_bottom**3.
            print(f"Using auto-calculated dust thermal conductivity: {self.k_dust:.2e} W/m/K")
        # Rock skin depth (m)
        #self.rock_skin_depth = np.sqrt(
        #    self.k_rock * self.P / (self.rho_rock * self.cp_rock * np.pi)
        #)
