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
    # Solver settings
    use_RTE: bool = True             # use radiative transfer model, otherwise runs traditional thermal model. 
    RTE_solver: str = 'disort'         # RTE core solver. Options are 'disort' or 'hapke'
    diurnal: bool = True             # Diurnal simulation? If false, model is steady-state with fixed sun position, or no sun. 
    sun: bool = True                 # Include solar input?    
    T_fixed: bool = False             # Use initialization temperature to calculate radiance and emissivity spectra. No thermal evolution. Only valid if diurnal=False.

    # Output settings
    freq_out: int = 48              # Number of outputs per diurnal cycle. 
    last_day: bool = True            # If True, only output last day of simulation. Otherwise, output data from all days.
    compute_observer_radiance: bool = False  # Calculate observer radiance outputs
    observer_mu: int = np.cos(np.radians(0.0)) #Observer cosine angle for flat model radiance outputs. 
    
    # DISORT Spectral Mode Configuration
    thermal_evolution_mode: str = 'two_wave'    # Thermal evolution solver mode: 'two_wave', 'multi_wave', 'hybrid'
    output_radiance_mode: str = 'hybrid'      # Output radiance calculation mode: 'two_wave', 'multi_wave', 'hybrid'

    # Basic optical properties for RTE
    Et: float = 1000.0              # thermal extinction coefficient (m^-1)
    eta: float = 1.0                 # visible/thermal extinction coefficient ratio     
    ssalb_vis: float = 0.0           # single-scattering visible albedo
    ssalb_therm: float = 0.0         # single-scattering visible albedo. Influences the overall effective emissivity of the regolith bed. ssalb_therm=0.1 leads to emissivity ~0.95. 
    g_vis: float = 0.0               # disort Scattering assymetry parameter for thermal spectrum in disort (not used for multi-wave)
    g_therm: float = 0.0             # disort Scattering assymetry parameter for visible spectrum in disort (not used for multi-wave)
    R_base: float = 0.0              # Substrate reflectivity for 2-layer model (DISORT only). Overridden by substrate_spectrum if use_spec=True in non-broadband cases. 

    
    #Non-RTE optical properties
    em: float = 1.0                 # Surface thermal emissivity, ONLY USED FOR NON-RTE MODELS, 0.90
    albedo: float = 0.0             # Surface bond albedo, ONLY USED FOR NON-RTE MODELS, 0.0178

    # Planetary properties
    R: float = 1.0                   # heliocentric distance (AU)
    S: float = 1366.0                 #Solar constant. 
    latitude: float = np.radians(0.0) # latitude (rad)
    dec: float = np.radians(0.0)     #Solar declination angle (rad)
    P: float = 15450        #Rotational period in s
    steady_state_mu: float = 1.0 #Solar incidence angle cosine for steady steate runs with the sun turned on.

    #Surface roughness options. 
    crater: bool = False            #Run hemispherical crater approximation alongside smooth model. Much slower! 
    illum_freq: int = 1            #Frequency at which to recompute rays for crater illumination/shadowing. I.e., every N time steps.     
    compute_crater_radiance: bool = False  # Calculate crater radiance as seen by observers. Computationally expensive!
    #observer_vectors: list = field(default_factory=lambda: [[0, 0, 1],[0.5,0,1],[0.7,0,1], [-0.5,0,1], [-0.7,0,1]])  # List of [x,y,z] observer direction vectors (default: overhead)
    observer_vectors: list = field(default_factory=lambda: [[0, 0, 1],[1,0,1],[0,1,1]])

    # Dust (or top layer) material properties
    single_layer: bool = True        # use single-layer model instead of two-layer. If single layer, only dust properties are used. 
    dust_thickness: float = 1.0  # dust column total thickness (m)
    k_dust: float = 5.5e-4           # dust thermal conductivity (W/m/K). If using RTE model, this should just be phonon conduction. 5.5e-4 
    rho_dust: float = 1100.         # dust bulk density (kg/m^3) 366
    cp_dust: float = 825.0           # dust specific heat (J/kg/K) 700
    k_dust_auto: bool = False        # use auto-calculated dust thermal conductivity for non-RTE models. If False, use k_dust value directly.

    # Rock (or substrate) material properties, only used if single_layer=False
    rock_thickness: float = 1.0     # rock substrate column total thickness (m)
    k_rock: float = 1.0              # rock thermal conductivity (W/m/K)
    rho_rock: float = 2000.0         # rock density (kg/m^3)
    cp_rock: float = 700.0           # rock heat capacity (J/kg/K)


    # Boundary conditions and initialization
    T_bottom: float = 300.           # bottom boundary temperature (when Dirichlet) and global initialization temperature (K)
    bottom_bc: str = 'neumann'       # bottom boundary condition choices: "neumann" (zero‐flux), "dirichlet" (fixed T_bottom)

    # Grid settings
    auto_thickness: bool = True      # auto-calculate dust and rock layer grid spacing thicknesses based on thermal skin depth
    flay: float = 0.1               # First layer thickness (fraction of skin depth) if using auto thickness.
    geometric_spacing: bool = True   # Node spacing increases by factor spacing_factor, otherwise constant thickness. Only applies to single layer scenario. 
    spacing_factor: float = 1.05     # layer thickness increase factor for geometric spacing. Only applies to single layer scenario!
    min_nlay_dust: int = 12          # Minimum number of grid points within dust column for two-layer scenario. Default=12. 
    rock_lthick_fac: float = 0.2     # Factor by which to multiply auto-calculated rock layer thickness. This is used to ensure that the rock layer is not too thick compared to the dust layer. Default=0.25. 
    dust_rte_max_lthick: float = 0.05  # Maximum first grid layer thickness for RTE model (in tau units, i.e., optical opacity). Default=0.025

    # Time-stepping parameters
    ndays: int = 3                   # total simulation days (diurnal cycles)
    auto_dt: bool = True             # auto-calculate time step based on thermal skin depth
    # Manual time-stepping options:
    tsteps_day: int = 16000          # Number of model calculation time steps per day, only used of auto_dt is False. 
    # Advanced times stepping and numerical accuracy prameters
    dtfac: float = 50.0               # Define time step as dt = dtfac*min(dx/K). Higher number increases speed at risk of reduced accuracy. Default=10
    minsteps: int = 10000             # Minimum number of time steps per day. Used if auto_dt is True. Default=5000. Try setting to a higher number if you think your model is inaccurate.  


    ##################################
    #DISORT radiative transfer options. 
    ##################################
    # Note that nmom and nstr are usuall equal. 
    nmom: int = 4            # Number of moments to phase function. But be >= nstr. Recommended ≥4
    nstr: int = 4            # Number of streams for disort discrete ordinate method. Recommended ≥4
    hybrid_wavelength_cutoff: float = 3.33     # Cutoff wavelength (μm) between thermal (multi-wave) and visible (broadband) in hybrid mode 
    #folder: str = "/Users/ryan/Research/RT_models/RT_thermal_model/optical_constants/Quartz_5micron_30wns/pack_frac_0.35/output" #path to scattering table files
    mie_file: str = "/Users/ryan/Research/RT_models/RT_thermal_model/Optical_props/quartz_spitzer_combined_15um_mie.txt"   #Table of values from Mie code.  
    solar_spectrum_file: str = '/Users/ryan/Research/RT_models/RT_thermal_model/Optical_props/solar_integrated_32.txt'  #integrates solar spectrum. Must have same spectral sampling as scattering matrix. 
    wn_bounds: str = '/Users/ryan/Research/RT_models/RT_thermal_model/Optical_props/quartz_spitzer_wn_bounds.txt' #Wavenumber bounds for input files.
    substrate_spectrum: str = '/Users/ryan/Research/RT_models/RT_thermal_model/Optical_props/Orgueil_P11442_VEH_32wns.txt' #Bennu emissivity spectrum for substrate
    use_spec: bool = False     #Use emissivity spectrum for substrate. If false, uses global reflectivity value R_base
    fill_frac: float = 0.63   #Fill fraction for particles. 
    radius: float = 15.e-6    #Particle radius in meters. 
    
    #Output settings. Choose files with desired multiwave spectral sampling for calculating radiance output. 
    mie_file_out: str = "/Users/ryan/Research/RT_models/RT_thermal_model/Optical_props/quartz_spitzer_combined_15um_mie.txt"    
    solar_spectrum_file_out: str = '/Users/ryan/Research/RT_models/RT_thermal_model/Optical_props/solar_integrated_216.txt'  #integrates solar spectrum. Must have same spectral sampling as scattering matrix. 
    wn_bounds_out: str = '/Users/ryan/Research/RT_models/RT_thermal_model/Optical_props/quartz_spitzer_wn_bounds.txt' #Wavenumber bounds for output files.
    substrate_spectrum_out: str = '/Users/ryan/Research/RT_models/RT_thermal_model/Optical_props/sabel_enstatite.txt' #Bennu emissivity spectrum for substrate
    otesT1_out: str = '/Users/ryan/Research/RT_models/RT_thermal_model/Optical_props/Bennu_Type1_216wns.txt' #Output file for OTES T2 radiance outputs.
    otesT2_out: str = '/Users/ryan/Research/RT_models/RT_thermal_model/Optical_props/Bennu_Type2_216wns.txt' #Output file for OTES T2 radiance outputs.
    nstr_out: int = 16        #Number of streams for calculating radiance outputs. ≥16 recommended. 
    nmom_out: int = 16        #Number of scattering moments for radiance outputs. Must be ≥ nstr_out

    #DISORT optical properties for visible portion of the spectrum
    force_vis_disort: bool = False  # Force visible optical properties to be used for visible portion of the spectrum. If False, uses optical propereties loaded from files above. 

    #DISORT depth-dependent options. 
    # NOT YET IMPLEMENTED. 
    depth_dependent: bool = False #Depth-dependent optical properties from file (extinction coefficient, ssalb, scattering matrix moments)


    # Physical constants
    sigma: float = 5.670373e-8       # Stefan-Boltzmann constant (W/m^2/K^4)

    #Advanced solver settings.
    steady_tol: float = 1.0e-4       # tolerance for steady-state convergence
    
    # Diurnal convergence settings
    enable_diurnal_convergence: bool = False  # Master enable/disable flag for automatic diurnal convergence
    diurnal_convergence_method: str = 'temperature'  # Convergence method: 'energy', 'temperature', or 'both'
    diurnal_convergence_temp_tol: float = 1.0e-4  # Temperature change tolerance (K) - Kieffer 2013 method
    diurnal_convergence_energy_tol: float = 0.05  # Energy balance tolerance (fractional) - Rozitis & Green 2011 method
    diurnal_convergence_min_cycles: int = 2  # Minimum number of cycles before checking convergence
    diurnal_convergence_check_interval: int = 1  # Check convergence every N cycles
    diurnal_convergence_window: int = 3  # Number of previous cycles to average over for convergence check
    bvp_tol: float = 1.0e-8          # tolerance for BVP solver
    bvp_max_iter: float = 300        # max iterations for BVP solver
    T_surf_tol: float = 1.0e-4       # tolerance for surface temperature convergence
    T_surf_max_iter: int = 50        # max iterations for surface temperature convergence
    disort_space_temp: float = 0.0 # Background temperature (space, cold shroud) for disort. 
    # layer thickness values only used for manual spacing (auto_thickness=False)
    dust_lthick: float = 0.02        # dust node spacing (tau units), only used if auto_thickness is False.
    rock_lthick: float = 0.0025      # rock node spacing (m), only used if auto_thickness is False.

    # Computed fields
    gamma_vis: float = field(init=False)
    gamma_therm: float = field(init=False)
    J: float = field(init=False)
    q: float = field(init=False)
    rock_skin_depth: float = field(init=False)
    dust_skin_depth: float = field(init=False)


    def __post_init__(self):
        # Validation of spectral mode combinations
        valid_modes = ['two_wave', 'multi_wave', 'hybrid']
        if self.thermal_evolution_mode not in valid_modes:
            raise ValueError(f"thermal_evolution_mode must be one of {valid_modes}")
        if self.output_radiance_mode not in valid_modes:
            raise ValueError(f"output_radiance_mode must be one of {valid_modes}")
            
        # Derived radiative parameter
        self.gamma_vis = np.sqrt(1.0 - self.ssalb_vis)
        self.gamma_therm = np.sqrt(1.0 - self.ssalb_therm)
        # Solar irradiance at distance R (W/m^2)
        self.J = self.S / (self.R ** 2.0)
        # Radiative ratio q
        self.q = 1.0 / (self.k_dust * self.Et)
        if(self.k_dust_auto and not self.use_RTE):
            #Add the estimation for the radiative term from Hapke's book, equation 16.31. 
            self.k_dust += (4.0/self.Et)*self.sigma*self.T_bottom**3.
            print(f"Using auto-calculated dust thermal conductivity: {self.k_dust:.2e} W/m/K")
            print(f"Thermal inertia is {np.sqrt(self.k_dust*self.rho_dust*self.cp_dust)}")
        # Rock skin depth (in tau units)
        self.rock_skin_depth = np.sqrt(
            self.k_rock * self.Et**2. * self.P / (self.rho_rock * self.cp_rock * np.pi)
        )
        if(self.use_RTE):
            #If we are using the RTE model, we need to add the approximation for the radiative term from Hapke's book, equation 16.31.
            k_dust_approx = self.k_dust + (4.0/self.Et)*self.sigma*self.T_bottom**3.
        else:
            k_dust_approx = self.k_dust
        #Dust (or top) layer skin depth in tau units. 
        self.dust_skin_depth = np.sqrt(
            k_dust_approx * self.Et**2. * self.P / (self.rho_dust * self.cp_dust * np.pi)
        )
