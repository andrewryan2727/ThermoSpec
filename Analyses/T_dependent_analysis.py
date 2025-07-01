import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modelmain import Simulator
from config import SimulationConfig
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline, interp1d


def build_single_layer_lookup(k_dust_values, base_config=None):
    """Build lookup table of temperature profiles for different k_dust values."""
    from copy import deepcopy
    
    if base_config is None:
        base_config = SimulationConfig()
        base_config.single_layer = True
        base_config.use_RTE = False
    
    results = {}
    
    for k_dust in k_dust_values:
        # Create a fresh copy of the configuration for each run
        cfg = deepcopy(base_config)
        cfg.k_dust = k_dust
        print(f"Building single-layer lookup for k_dust={k_dust:.4e} W/m/K")
        # Force recalculation of derived values by calling post_init
        cfg.__post_init__()
        
        # Create new simulator with fresh config
        sim = Simulator(cfg)
        T_out, _, phi_th, T_surf, t_out = sim.run()
        
        # Calculate emission temperature if using RTE
        if cfg.use_RTE:
            # x_RTE is already in tau units (optical depth)
            tau = sim.grid.x_RTE
            
            # Calculate emission temperature at each time point
            n_times = T_out.shape[1]
            emissT = np.zeros(n_times)
            for i in range(n_times):
                # Use temperature profile in dust layers only
                T_profile = T_out[1:sim.grid.nlay_dust+1, i]
                emissT[i] = emissionT(T_profile, tau,mu=np.cos(cfg.latitude))
        else:
            emissT = None
            
        results[k_dust] = {
            'T_out': T_out,
            'phi_therm_out': phi_th,
            'T_surf': T_surf,
            'emissT': emissT,  # Will be None for non-RTE cases
            'lut_times': t_out
        }
    return results

def phi_T_surf(phi):
    #phi is phi_therm_out[0,:]
    #Calculate effective surface temperature from radiative flux. 
    # Hale and Hapke, 2002 eq. 15
    sigma = 5.670e-8
    return ((2.0*phi*np.pi/sigma)**0.25)

def interpolate_temps(k_dust_lookup, temps_key='T_surf'):
    """Interpolate temperature profiles for a specific k_dust value."""
    k_dust_values = 10**k_dust_lookup['k_dust']
    temps = np.array([k_dust_lookup[k][temps_key] for k in k_dust_values])
    
    # Find where k_dust fits in the array
    interpolator = CubicSpline(np.log10(k_dust_values), temps, axis=0)
    times = np.linspace(0.0, 1.0, temps.shape[1])
    interpolator2 = CubicSpline(times, temps, axis=1)
    return interpolator, interpolator2

def find_max_time(T,times):
	poly_fit = np.polyfit(times,T,5)
	p = np.poly1d(poly_fit)
	result = minimize_scalar(-p,bounds=(times.min(),times.max()),method='bounded')
	return(result.x)

def maxT_fit(modelT,single_layer_lookup):
    #Find maximum temperature and corresponding k_dust value using interpolation.
    #modelT and model_time are arrays of temperature and time values from the model result that we're comparing to the LUT. 
    model_time = single_layer_lookup['lut_times'] #currently assuming that the lut and the model have the same time array.
    modelT_interp = interp1d(model_time,-modelT,fill_value='extrapolate',kind='quadratic')
    result = minimize_scalar(modelT_interp,bounds=(0.3,0.7),method='bounded')
    modelT_max = -modelT_interp(result.x)
    lut_interp = interp1d(single_layer_lookup['max_T'],single_layer_lookup['k_dust'],fill_value='extrapolate',kind='quadratic')
    return((modelT_max,lut_interp(modelT_max)))

def minT_fit(modelT,single_layer_lookup):
    #Find minimum temperature and corresponding k_dust value using interpolation.
    #modelT and model_time are arrays of temperature and time values from the model result that we're comparing to the LUT.   
    model_time = single_layer_lookup['lut_times'] #currently assuming that the lut and the model have the same time array.    
    modelT_interp = interp1d(model_time,modelT,fill_value='extrapolate',kind='quadratic')
    result = minimize_scalar(modelT_interp,bounds=(0.0,0.3),method='bounded')
    modelT_min = modelT_interp(result.x)
    lut_interp = interp1d(single_layer_lookup['min_T'],single_layer_lookup['k_dust'],fill_value='extrapolate',kind='quadratic')
    return((modelT_min,lut_interp(modelT_min))) 

def rmserr_many_time_scalar(k_val,modelT,model_time,times,single_layer_lookup):
    #k_val = guess for conductivity
    #modelT = temperatures from the model we are trying to fit (e.g., two layer model output)
    #model_time = time array from the model we are trying to fit
    #times = times at which are are doing the fitting. 
    #single_layer_lookup = lookup table of temperature profiles for different k_dust values
	lut_curve = single_layer_lookup['lut_k_interp'](k_val) #Give Temperature vs time from our lookup table for the current guess at k_dust. 
	lut_time_interp = interp1d(single_layer_lookup['lut_times'],lut_curve,fill_value='extrapolate',kind='cubic')
	lut_curve_resampled = lut_time_interp(times)
	modelT_interp = interp1d(model_time,modelT,fill_value='extrapolate',kind='cubic')
	modelT_resampled = modelT_interp(times)
	residuals = np.sum((modelT_resampled - lut_curve_resampled)**2.0)
	return(residuals)

def find_best_fit_k_dust(modelT, single_layer_lookup, temps_key='T_surf', start_idx=-8000):
    """Find best-fitting k_dust value using interpolation lookup table.
    
    Uses amplitude matching for initial guess, then optimizes using RMS error.
    """
    k_dust_values = single_layer_lookup['k_dust'] #values are already log10(k_dust)
    #log_k_dust = np.log10(k_dust_values)
    lut_times = single_layer_lookup['lut_times']

    max_T,max_T_k = maxT_fit(modelT,single_layer_lookup)
    min_T,min_T_k = minT_fit(modelT,single_layer_lookup)

    avg = (max_T_k + min_T_k)/2.0
    min_bound = max(min(k_dust_values),avg-1.)
    max_bound = min(max(k_dust_values),avg+1.)
    if(min_bound >= max_bound):
        min_bound = max_bound-1.0
    
    result = minimize_scalar(rmserr_many_time_scalar,
        args=(modelT,lut_times, lut_times, single_layer_lookup),
        bounds=(min_bound,max_bound),method='bounded')

    # # Extract temperature profiles for last day
    # lookup_temps = np.array([k_dust_lookup[k][temps_key][start_idx:] for k in k_dust_values])
    # target = target_temps[start_idx:]
    
    # # Get initial guess using amplitude matching in log space
    # target_amp = np.max(target) - np.min(target)
    # lookup_amps = np.max(lookup_temps, axis=1) - np.min(lookup_temps, axis=1)
    # amp_errors = np.abs(lookup_amps - target_amp) / target_amp
    # initial_log_k = log_k_dust[np.argmin(amp_errors)]
    
    # # Set bounds in log space, allowing one order of magnitude in each direction
    # min_bound = max(np.log10(k_dust_values[0]), initial_log_k - 0.25)
    # max_bound = min(np.log10(k_dust_values[-1]), initial_log_k + 0.25)
    # log_k_bounds = (min_bound, max_bound)
    
    # # Define RMS error function for optimization in log space
    # def rms_error(log_k):
    #     k_dust = 10**log_k
    #     temps = interpolate_temps(k_dust, k_dust_lookup, temps_key)[start_idx:]
    #     return np.sum((temps - target)**2)
    
    # # Optimize using RMS error in log space, starting from amplitude-based guess
    # result = minimize_scalar(
    #     rms_error,
    #     bounds=log_k_bounds,
    #     method='bounded',
    #     options={'xatol': 0.01}  # Tolerance in log space (about 2.3% in linear space)
    # )
    
    # Convert best log_k back to k_dust and get temperatures
    best_k_dust = 10**result.x
    best_temps = single_layer_lookup['lut_k_interp'](result.x)  # Interpolated temperatures for best k_dust
    #best_temps = interpolate_temps(best_k_dust, k_dust_lookup, temps_key)
    
    return best_k_dust, rmserr_many_time_scalar(result.x,modelT,lut_times, lut_times, single_layer_lookup), best_temps

def emissionT(T,tau,mu=1.0):
    # Calculate effective emission temperature from radiative flux.
    # T is the temperature profile
    # tau is the optical depth profile (already in tau units)
    # mu is the cosine of the emission angle (observer)
    T_calc = 0.0
    wt_calc = 0.0
    for i in np.arange(len(T)):
        T_calc += (T[i]**4.0)*np.exp(-tau[i]/mu)
        wt_calc += np.exp(-tau[i]/mu)
    T_calc = T_calc/wt_calc
    return(T_calc**0.25)

def analyze_heliocentric_distance(R_values, k_dust_values=None, lut_temps_key=None, model_temps_key=None):
    """Analyze equivalent single-layer k_dust for different heliocentric distances.
    
    Args:
        R_values: Array of heliocentric distances (AU) to analyze
        k_dust_values: Array of k_dust values for lookup table (optional)
        lut_temps_key: Temperature type to use from lookup table ('T_surf' or 'emissT')
        model_temps_key: Temperature type to use from two-layer model ('T_surf' or 'emissT')
    """
    if k_dust_values is None:
        k_dust_values = np.logspace(-4, -1, 20)  # 20 values from 1e-4 to 1e-1
    
    # Set up configurations
    target_cfg = SimulationConfig()
    ref_cfg = SimulationConfig()

    # Configure RTE modes based on temperature types
    if model_temps_key == 'emissT':
        target_cfg.use_RTE = True
        target_cfg.RTE_solver = 'hapke'
        target_cfg.multi_wave = False
    elif model_temps_key == 'T_surf':
        target_cfg.use_RTE = False
    else:
        # Default to RTE off if not specified
        target_cfg.use_RTE = False
        model_temps_key = 'T_surf'

    if lut_temps_key == 'emissT':
        ref_cfg.use_RTE = True
        ref_cfg.RTE_solver = 'hapke'
        ref_cfg.multi_wave = False
    elif lut_temps_key == 'T_surf':
        ref_cfg.use_RTE = False
    else:
        # Match two-layer configuration if not specified
        ref_cfg.use_RTE = target_cfg.use_RTE
        lut_temps_key = model_temps_key

    # Basic configuration
    target_cfg.single_layer = True
    target_cfg.ndays = 5
    target_cfg.dust_thickness = 1.0  # m, Default thickness for single-layer model
    target_cfg.rho_dust= 600.0
    target_cfg.cp_dust = 700.0
    target_cfg.k_dust = 5.5e-4
    target_cfg.Et = 90.0 #1 mm particles, fill factor of 0.3
    target_cfg.last_day = True
    target_cfg.ssalb_vis = 0.06
    target_cfg.gamma_therm = 0.95

    target_cfg.latitude = np.radians(0.0)
    
    ref_cfg.single_layer = True
    ref_cfg.ndays = target_cfg.ndays
    ref_cfg.dust_thickness = 1.0  # m, Default thickness for single-layer model
    ref_cfg.k_dust_auto = False  # Use fixed k_dust values
    ref_cfg.rho_dust = target_cfg.rho_dust  # Use same density as two-layer model
    ref_cfg.cp_dust = target_cfg.cp_dust  # Use same specific heat as two-layer model
    ref_cfg.last_day = True
    ref_cfg.latitude = target_cfg.latitude
    ref_cfg.albedo = 0.04

    results = {}
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages('thermal_model_fits.pdf') as pdf:
        fig = plt.figure(figsize=(15, 10))
        for i, R in enumerate(R_values):
            print(f"Processing heliocentric distance: {R:.2f} AU")



            # Build reference LUT for this R (use_RTE=False)
            ref_cfg.R = R
            ref_cfg.__post_init__()
            ref_cfg.use_RTE = False
            ref_cfg.k_dust = np.max(k_dust_values)
            #Figure out what base temperature to use by iterating the reference model
            # using the mean surface temperature from the previous iteration as the initialization temp
            # for the next iteration. 
            T_bottom_guess = 275.0*R**(-1/2)
            ref_cfg.T_bottom = T_bottom_guess
            tolerance = 3.0  # K
            max_iter = 10
            for _ in range(max_iter):
                sim_ref = Simulator(ref_cfg)
                _, _, _, T_surf_ref, _ = sim_ref.run()
                avg_T_surf = np.mean(T_surf_ref)
                print(avg_T_surf)
                if abs(avg_T_surf - ref_cfg.T_bottom) < tolerance:
                    break
                ref_cfg.T_bottom = avg_T_surf
            ref_cfg.T_bottom = avg_T_surf
            print(f"[R={R:.2f} AU] Final T_bottom used for reference LUT: {ref_cfg.T_bottom:.2f} K")
            ref_lookup = build_single_layer_lookup(k_dust_values, ref_cfg)
            ref_lookup['k_dust'] = np.log10(k_dust_values)



            # Prepare interpolation for this R
            temps = [ref_lookup[k][lut_temps_key] for k in k_dust_values]
            temps = np.array(temps)
            lut_times = ref_lookup[k_dust_values[0]]['lut_times']
            lut_times = (lut_times - np.min(lut_times)) / (np.max(lut_times) - np.min(lut_times))
            lut_k_interp, lut_time_interp = interpolate_temps(ref_lookup, temps_key=lut_temps_key)
            ref_lookup['lut_times'] = lut_times
            ref_lookup['lut_k_interp'] = lut_k_interp
            ref_lookup['lut_time_interp'] = lut_time_interp

            # Pre-calculate extrema
            nsolns = len(k_dust_values)
            lut_max_time = np.zeros(nsolns)
            lut_max_T = np.zeros(nsolns)
            lut_min_T = np.zeros(nsolns)
            idx1 = np.argmin(np.abs(lut_times-0.3))
            idx2 = np.argmin(np.abs(lut_times-0.7))
            for j in np.arange(nsolns):
                lut_max_time[j] = find_max_time(temps[j,idx1:idx2],lut_times[idx1:idx2])
                lut_max_T[j] = lut_time_interp(lut_max_time[j])[j]
                lut_interp2 = interp1d(lut_times,temps[j,:],axis=0,fill_value='extrapolate',kind='cubic')
                result = minimize_scalar(lut_interp2,bounds=(0.0,0.3),method='bounded')
                lut_min_T[j] = lut_interp2(result.x)
            ref_lookup.update({
                'max_time': lut_max_time,
                'max_T': lut_max_T,
                'min_T': lut_min_T
            })

            # Run target model (use_RTE=True)
            target_cfg.R = R
            target_cfg.T_bottom = ref_cfg.T_bottom
            target_cfg.__post_init__()
            sim = Simulator(target_cfg)
            T_out, _, phi_th, T_surf, t_out = sim.run()

            # Calculate target modelT as before
            if model_temps_key == 'emissT':
                if not target_cfg.use_RTE:
                    raise ValueError("Cannot get emission temperatures without RTE enabled")
                tau = sim.grid.x_RTE
                n_times = T_out.shape[1]
                modelT = np.zeros(n_times)
                for j in range(n_times):
                    T_profile = T_out[1:sim.grid.nlay_dust+1, j]
                    modelT[j] = emissionT(T_profile, tau,mu=np.cos(target_cfg.latitude))
            else:
                modelT = T_surf.copy()

            # Find best fit
            best_k_dust, error, best_temps = find_best_fit_k_dust(modelT, ref_lookup, temps_key=lut_temps_key)

            results[R] = {
                'best_k_dust': best_k_dust,
                'error': error,
                'thermal_inertia': np.sqrt(best_k_dust * ref_cfg.rho_dust * ref_cfg.cp_dust),
                'temperatures': best_temps,
                'model_temps': modelT
            }

            # Plot comparison
            plt.subplot(3, 4, (i % 12) + 1)
            temp_type = 'Emission' if model_temps_key == 'emissT' else 'Surface'
            plt.plot(sim.t_out / 3600, modelT, 'b-', label=f'R={R:.1f} AU {temp_type} T')
            plt.plot(sim.t_out / 3600, best_temps, 'r--', label=f'No RTE fit (k={best_k_dust:.2e})')
            plt.title(f'R={R:.1f} AU\nTI={results[R]["thermal_inertia"]:.1f}')
            if i % 12 == 0:
                plt.legend()
            if (i + 1) % 12 == 0 or i == len(R_values) - 1:
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
                if i < len(R_values) - 1:
                    fig = plt.figure(figsize=(15, 10))
    return results

if __name__ == "__main__":
    R_values = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0])
    #R_values = np.array([3.0, 4.0, 5.0])
    k_dust_values = np.logspace(-5, 0.5, 25)  # 25 values from 1e-5 to 1e0
    
    results = analyze_heliocentric_distance(R_values, k_dust_values, 
                                          lut_temps_key='T_surf', model_temps_key='emissT')
    
    # Plot summary of results
    import matplotlib.pyplot as plt
    R_distances = np.array(list(results.keys()))
    thermal_inertias = np.array([results[R]['thermal_inertia'] for R in R_distances])
    plt.figure(figsize=(10, 6))
    plt.loglog(R_distances, thermal_inertias, 'o-', label='Model Results')

    # Power-law fit: TI = A * R^alpha
    logR = np.log(R_distances)
    logTI = np.log(thermal_inertias)
    coeffs = np.polyfit(logR, logTI, 1)
    alpha = coeffs[0]
    A = np.exp(coeffs[1])
    TI_fit = A * R_distances**alpha
    plt.loglog(R_distances, TI_fit, 'r--', label=f'Power-law fit ($R^{{{alpha:.2f}}}$)')

    # Reference line for alpha = -0.75
    TI_ref = thermal_inertias[0] * (R_distances / R_distances[0])**(-0.75)
    plt.loglog(R_distances, TI_ref, 'k:', label=r'Ref: $R^{-0.75}$')

    plt.xlabel('Heliocentric Distance (AU)')
    plt.ylabel('Equivalent Thermal Inertia (J/m²/K/s$^{0.5}$)')
    plt.title(f'Equivalent Thermal Inertia vs Heliocentric Distance\nPower-law exponent $\\alpha$ = {alpha:.2f}')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.savefig('thermal_inertia_vs_distance.pdf')