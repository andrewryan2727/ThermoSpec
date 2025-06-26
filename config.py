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


@dataclass
class SimulationConfig:
    # Radiative properties
    ssalb_vis: float = 0.1          # single-scattering visible albedo
    gamma_therm: float = 0.9         # thermal albedo factor
    Et: float = 3000.0               # thermal extinction coefficient (m^-1)
    eta: float = 1.0                 # visible/thermal extinction ratio 
    em: float = 0.90                 # thermal emissivity, ONLY USED FOR NON-RTE MODELS
    albedo: float = 0.0178              # surface albedo, ONLY USED FOR NON-RTE MODELS

    # Orbital & rotational parameters
    R: float = 1.2447                   # heliocentric distance (AU)
    latitude: float = 0.0 * np.pi/180.0  # latitude (rad)
    P: float = 4.296 * 3600.0        #Rotational period in s

    # Dust (or top layer) material properties
    single_layer: bool = False        # use single-layer model instead of two-layer. If single layer, only dust properties are used. 
    k_dust: float = 5.5e-4           # dust thermal conductivity (W/m/K). If using RTE model, this should just be phonon conduction. 5.5e-4
    rho_dust: float = 366.         # dust bulk density (kg/m^3)
    cp_dust: float = 700.0           # dust specific heat (J/kg/K)
    k_dust_auto: bool = False        # use auto-calculated dust thermal conductivity for non-RTE models. If False, use k_dust value directly.

    # Rock (or substrate) material properties
    k_rock: float = 1.0              # rock thermal conductivity (W/m/K)
    rho_rock: float = 2000.0         # rock density (kg/m^3)
    cp_rock: float = 700.0           # rock heat capacity (J/kg/K)

    # Boundary & layer settings
    T_bottom: float = 260.           # bottom boundary temperature (when Dirichlet) and global initialization temperature (K)
    dust_thickness: float = 20.0e-6  # dust column total thickness (m)
    rock_thickness: float = 1.0     # rock substrate column total thickness (m)
    auto_thickness: bool = True      # auto-calculate dust and rock layer thicknesses based on thermal skin depth
    flay: float = 0.10               # First layer thickness (fraction of skin depth) if using auto thickness.
    geometric_spacing: bool = True   # Node spacing increases by factor spacing_factor, otherwise constant thickness. Only applies to single layer scenario. 
    spacing_factor: float = 1.05     # layer thickness increase factor for geometric spacing. Only applies to single layer scenario!

    # layer thickness values only used for manual spacing (auto_thickness=False)
    dust_lthick: float = 0.02        # dust node spacing (tau units), only used if auto_thickness is False.
    rock_lthick: float = 0.0025      # rock node spacing (m), only used if auto_thickness is False.

    # Simulation flags and convergence settings
    use_RTE: bool = True             # use radiative transfer model
    RTE_solver: str = 'disort'       # Options are disort or hapke
    bottom_bc: str = 'neumann'       # bottom boundary condition choices: "neumann" (zero‐flux), "dirichlet" (fixed T_bottom)
    sun: bool = False                 # include solar input
    diurnal: bool = False             # include diurnal variation. If false, model is steady-state. 


    # Time-stepping parameters
    ndays: int = 3                   # total simulation days (diurnal cycles)
    auto_dt: bool = True             # auto-calculate time step based on thermal skin depth
    freq_out: int = 50              # Number of outputs per diurnal cycle. 
    last_day: bool = True            # If True, only output last day of simulation. Otherwise, output all days.
    # Manual time-stepping options:
    tsteps_day: int = 16000          # Number of model calculation time steps per day, only used of auto_dt is False. 


    # Advanced times stepping and numerical accuracy prameters
    dtfac: float = 10               # Define time step as dt = dtfac*min(dx/K). Higher number increases speed at risk of reduced accuracy. Default=10
    minsteps: int = 2000            # Minimum number of time steps per day. Used if auto_dt is True. Default=2000
    min_nlay_dust: int = 20         # Minimum number of grid points within dust column for two-layer scenario. Default=12. 
    rock_lthick_fac: float = 0.1   # Factor by which to multiply auto-calculated rock layer thickness. This is used to ensure that the rock layer is not too thick compared to the dust layer. Default=0.25. 
    dust_rte_max_lthick: float = 0.05  # Maximum first grid layer thickness for RTE model (in tau units, i.e., optical opacity). Default=0.025

    custom_bvp: bool = True          # use the custom written bvp solver for the hapke RTE. Otherwise, reverts to scipy.solve_bvp (which may not actually work anymore)
    bvp_tol: float = 1.0e-8          # tolerance for BVP solver
    bvp_max_iter: float = 200        # max iterations for BVP solver
    T_surf_tol: float = 1.0e-4       # tolerance for surface temperature convergence
    T_surf_max_iter: int = 50        # max iterations for surface temperature convergence

    ##################################
    #DISORT radiative transfer options. 
    ##################################
    # Note that nmom and nstr are usuall equal. 
    nmom: int = 4            # Number of moments to phase function. But be >= nstr. Recommended ≥4
    nstr: int = 4            # Number of streams for disort discrete ordinate method. Recommended ≥4
    g: float = 0.5            # Scattering assymetry parameter for non-multi_wave scenarios. 

    #DISORT multi-wavelength options. Note that Cext in files is assumed to be in units of µm^2, as it is likely that the user defined particle size and wavelength in µm. 
    multi_wave: bool = True  # Use multiple wavelengths, using wavelength-dependent optical properties from file (extinction coefficient, ssalb, scattering matrix moments)
    T_fixed: bool = False    # Use initialization temperature to calculate radiance and emissivity spectra. No thermal evolution. Only valid if diurnal=False. 
    #folder: str = "/Users/ryan/Research/RT_models/RT_thermal_model/optical_constants/Quartz_5micron_30wns/pack_frac_0.35/output" #path to scattering table files
    mie_file: str = "/Users/ryan/Research/RT_models/RT_thermal_model/Preprocessing/serp85_mag5_dol5_graph5_5um_32.txt"   #Table of values from Mie code.  
    solar_spectrum_file: str = '/Users/ryan/Research/RT_models/RT_thermal_model/Preprocessing/solar_integrated_32.txt'  #integrates solar spectrum. Must have same spectral sampling as scattering matrix. 
    substrate_spectrum: str = '/Users/ryan/Research/RT_models/RT_thermal_model/Preprocessing/Bennu_hummocky_32wns.txt' #Bennu emissivity spectrum for substrate
    wn_bounds: str = '/Users/ryan/Research/RT_models/RT_thermal_model/Preprocessing/wn_bounds_32.txt' #Wavenumber bounds for input files.
    use_spec: bool = True     #Use emissivity spectrum for substrate. If false, uses global reflectivity value R_base
    R_base: float = 0.0       # Substrate reflectivity for 2-layer model. Used if multi_wave=False and/or use_spec=False. 
    fill_frac: float = 0.05   #Fill fraction for particles. 
    radius: float = 5.0e-6    #Particle radius in meters. 
    
    #Output settings. Choose files with desired multiwave spectral sampling for calculating radiance output. 
    mie_file_out: str = "/Users/ryan/Research/RT_models/RT_thermal_model/Preprocessing/serp85_mag5_dol5_graph5_5um.txt"    
    solar_spectrum_file_out: str = '/Users/ryan/Research/RT_models/RT_thermal_model/Preprocessing/solar_integrated_216.txt'  #integrates solar spectrum. Must have same spectral sampling as scattering matrix. 
    substrate_spectrum_out: str = '/Users/ryan/Research/RT_models/RT_thermal_model/Preprocessing/Bennu_hummocky_216wns.txt' #Bennu emissivity spectrum for substrate
    otesT1_out: str = '/Users/ryan/Research/RT_models/RT_thermal_model/Preprocessing/Bennu_Type1_216wns.txt' #Output file for OTES T2 radiance outputs.
    otesT2_out: str = '/Users/ryan/Research/RT_models/RT_thermal_model/Preprocessing/Bennu_Type2_216wns.txt' #Output file for OTES T2 radiance outputs.
    wn_bounds_out: str = '/Users/ryan/Research/RT_models/RT_thermal_model/Preprocessing/wn_bounds_216.txt' #Wavenumber bounds for output files.
    nstr_out: int = 16        #Number of streams for calculating radiance outputs. ≥16 recommended. 
    nmom_out: int = 16        #Number of scattering moments for radiance outputs. Must be ≥ nstr_out

    #DISORT depth-dependent options. NOT YET IMPLEMENTED. 
    depth_dependent: bool = False #Depth-dependent optical properties from file (extinction coefficient, ssalb, scattering matrix moments)


    # Physical constants
    sigma: float = 5.670373e-8       # Stefan-Boltzmann constant (W/m^2/K^4)

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
