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
    em: float = 0.90                 # thermal emissivity, ONLY USED FOR NON-RTE MODELS
    albedo: float = 0.0178              # surface albedo, ONLY USED FOR NON-RTE MODELS   

    # Orbital & rotational parameters
    R: float = 1.2447                   # heliocentric distance (AU)
    latitude: float = 34.6 * np.pi/180.0  # latitude (rad)
    P: float = 7.632622 * 3600.0        #Rotational period in s

    # Dust (or top layer) material properties
    k_dust: float = 0.0025           # dust thermal conductivity (W/m/K). If using RTE model, this should just be phonon conduction. 5.5e-4
    rho_dust: float = 366.         # dust bulk density (kg/m^3)
    cp_dust: float = 700.0           # dust specific heat (J/kg/K)
    k_dust_auto: bool = False        # use auto-calculated dust thermal conductivity for non-RTE models. If False, use k_dust value directly.

    # Rock (or substrate) material properties
    k_rock: float = 1.0              # rock thermal conductivity (W/m/K)
    rho_rock: float = 2000.0         # rock density (kg/m^3)
    cp_rock: float = 700.0           # rock heat capacity (J/kg/K)

    # Boundary & layer settings
    T_bottom: float = 250.           # bottom boundary and global initialization temperature (K)
    dust_thickness: float = 0.001  # dust column total thickness (m)
    rock_thickness: float = 0.50     # rock substrate column total thickness (m)
    auto_thickness: bool = True      # auto-calculate dust and rock layer thicknesses based on thermal skin depth
    flay: float = 0.10               # First layer thickness (fraction of skin depth) if using auto thickness.
    geometric_spacing: bool = True   # Node spacing increases by factor spacing_factor, otherwise constant thickness. Only applies to single layer scenario. 
    spacing_factor: float = 1.05     # layer thickness increase factor for geometric spacing. Only applies to single layer scenario!

    dust_lthick: float = 0.02        # dust node spacing (tau units), only used if auto_thickness is False.
    rock_lthick: float = 0.0025      # rock node spacing (m), only used if auto_thickness is False.

    # Simulation flags and convergence settings
    single_layer: bool = False        # use single-layer model instead of two-layer
    use_RTE: bool = True             # use radiative transfer model
    bottom_bc: str = 'neumann'       # bottom boundary condition choices: "neumann" (zero‐flux), "dirichlet" (fixed T_bottom)
    sun: bool = True                 # include solar input
    diurnal: bool = True             # include diurnal variation. If false, model is steady-state. 


    # Time-stepping parameters
    ndays: int = 5                   # total simulation days (diurnal cycles)
    auto_dt: bool = True            # auto-calculate time step based on thermal skin depth
    freq_out: int = 100              # Number of outputs per diurnal cycle. 
    last_day: bool = True            # If True, only output last day of simulation. Otherwise, output all days.
    tsteps_day: int = 16000          # time steps per day, only used of auto_dt is False. 


    # Advanced times stepping and numerical accuracy prameters
    dtfac: float = 20               # Define minimum time step as dt = tfac*min(dx/K). Higher number increases speed at risk of reduced accuracy. 
    minsteps: int = 2000                # Minimum number of time steps per day. If auto_dt is True, this is the minimum number of time steps per day.
    min_nlay_dust: int = 12         # Minimum number of grid points within dust column. 
    rock_lthick_fac: float = 0.25  # Factor to multiply auto-calculated rock layer thickness by. This is used to ensure that the rock layer is not too thick compared to the dust layer.
    dust_rte_max_lthick: float = 0.02  # Maximum dust layer thickness for RTE model (in tau units, i.e., optical opacity).

    custom_bvp: bool = True          # use the custom written bvp solver for RTE. Otherwise, reverts to scipy.solve_bvp
    bvp_tol: float = 1.0e-8          # tolerance for BVP solver
    bvp_max_iter: float = 1000       # max iterations for BVP solver
    T_surf_tol: float = 1.0e-4       # tolerance for surface temperature convergence
    T_surf_max_iter: int = 50        # max iterations for surface temperature convergence


    # Physical constants
    sigma: float = 5.670374e-8       # Stefan-Boltzmann constant (W/m^2/K^4)

    # Computed fields
    gamma_vis: float = field(init=False)
    J: float = field(init=False)
    q: float = field(init=False)
    rock_skin_depth: float = field(init=False)
    dust_skin_depth: float = field(init=False)

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
        self.rock_skin_depth = np.sqrt(
            self.k_rock * self.Et**2. * self.P / (self.rho_rock * self.cp_rock * np.pi)
        )
        if(self.use_RTE):
            #If we are using the RTE model, we need to add the approximation for the radiative term from Hapke's book, equation 16.31.
            k_dust_approx = self.k_dust + (4.0/self.Et)*self.sigma*self.T_bottom**3.
        else:
            k_dust_approx = self.k_dust
        self.dust_skin_depth = np.sqrt(
            k_dust_approx * self.Et**2. * self.P / (self.rho_dust * self.cp_dust * np.pi)
        )
