from modelmain import Simulator
from config import SimulationConfig
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline, interp1d


def build_single_layer_lookup(k_dust_values, base_config=None):
    """Build lookup table of temperature profiles for different k_dust values."""
    if base_config is None:
        base_config = SimulationConfig()
        base_config.single_layer = True
        base_config.use_RTE = False
    
    results = {}
    # Reuse simulator instance to avoid reconstruction overhead
    sim = Simulator(base_config)
    
    for k_dust in k_dust_values:
        sim = Simulator(base_config)
        sim.cfg.k_dust = k_dust
        # Reinitialize necessary components without full reconstruction
        #sim._init_state()
        T_out, _, phi_th, T_surf = sim.run()
        tstart = int(sim.cfg.tsteps_day*(sim.cfg.ndays-1))
        results[k_dust] = {
            'T_out': T_out[:,tstart:],
            'phi_therm_out': phi_th[:,tstart:],
            'T_surf': T_surf[tstart:]
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

    #
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

def analyze_two_layer_equivalence(dust_thickness_values, k_dust_values=None, temps_key='T_surf'):
    """Analyze equivalent single-layer k_dust for different two-layer dust thicknesses."""
    if k_dust_values is None:
        k_dust_values = np.logspace(-4, -1, 20)  # 20 values from 1e-4 to 1e-1
    
    # Set up configurations
    two_layer_cfg = SimulationConfig()
    two_layer_cfg.use_RTE = False
    two_layer_cfg.single_layer = False
    two_layer_cfg.ndays = 5
    
    single_layer_cfg = SimulationConfig()
    single_layer_cfg.use_RTE = False
    single_layer_cfg.single_layer = True
    single_layer_cfg.dust_thickness = 0.50  # m, Default thickness for single-layer model
    single_layer_cfg.ndays = 5
    single_layer_cfg.k_dust_auto = False  # Use fixed k_dust values
    single_layer_cfg.rho_dust = two_layer_cfg.rho_rock  # Use same density as two-layer model
    single_layer_cfg.cp_dust = two_layer_cfg.cp_rock  # Use same specific heat as two-layer model
    
    # Build single-layer lookup table
    print("Building single-layer lookup table...")
    #This returns a table of temperature profiles for different k_dust values (last diurnal cycle only)
    single_layer_lookup = build_single_layer_lookup(k_dust_values, single_layer_cfg)
    single_layer_lookup['k_dust'] = np.log10(k_dust_values)
    temps = np.array([single_layer_lookup[k][temps_key] for k in k_dust_values])
    #Make interpolator function. 
    # lut_k_interp accepts log10(k_dust), outputs T_surf vs time. 
    # lut_time_interp accepts time fraction (0.0 to 1.0), outputs T_surf values for all k_dust_values.
    lut_k_interp, lut_time_interp  = interpolate_temps(single_layer_lookup, temps_key='T_surf')

    #Pre-calculate LUT max and min temperature values and time at which max temperature occurs. 
    nsolns = len(k_dust_values)
    lut_times = np.linspace(0.0, 1.0, single_layer_cfg.tsteps_day)
    lut_max_time = np.zeros(nsolns)
    lut_max_T = np.zeros(nsolns)
    lut_min_T = np.zeros(nsolns)
    idx1 = np.argmin(np.abs(lut_times-0.3))
    idx2 = np.argmin(np.abs(lut_times-0.7))
    for i in np.arange(nsolns):
        lut_max_time[i] = find_max_time(temps[i,idx1:idx2],lut_times[idx1:idx2])
        lut_max_T[i] = lut_time_interp(lut_max_time[i])[i]
        #lut_interp2 = interpolate.interp1d(lut_times,smooth_lut[i,0,:]*-1.0,axis=0,fill_value='extrapolate',kind='cubic')
        #result = optimize.minimize_scalar(lut_interp2,bounds=(0.3,0.7),method='bounded')
        #lut_max_time[i] = result.x
        #lut_max_T[i] = -lut_interp2(result.x)
        lut_interp2 = interp1d(lut_times,temps[i,:],axis=0,fill_value='extrapolate',kind='cubic')
        result = minimize_scalar(lut_interp2,bounds=(0.0,0.3),method='bounded')
        lut_min_T[i] = lut_interp2(result.x)
    
    # Add lut_max_time, lut_max_T, and lut_min_T to dictionary
    single_layer_lookup['max_time'] = lut_max_time
    single_layer_lookup['max_T'] = lut_max_T
    single_layer_lookup['min_T'] = lut_min_T
    single_layer_lookup['lut_times'] = lut_times
    single_layer_lookup['lut_k_interp'] = lut_k_interp
    single_layer_lookup['lut_time_interp'] = lut_time_interp

    results = {}
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages('thermal_model_fits.pdf') as pdf:
        fig = plt.figure(figsize=(15, 10))
        for i, dust_thickness in enumerate(dust_thickness_values):
            print(f"Processing dust thickness: {dust_thickness:.2e} m")
            
            # Run two-layer model
            two_layer_cfg.dust_thickness = dust_thickness
            sim = Simulator(two_layer_cfg)
            T_out, _, phi_th, T_surf = sim.run()

            if(temps_key == 'T_surf'):
                modelT = T_surf.copy()
            else:
                modelT = phi_T_surf(phi_th[0,:])

            tstart = int(sim.cfg.tsteps_day*(sim.cfg.ndays-1))
            modelT = modelT[tstart:]  # Use last day temperatures

            # Find best fit
            best_k_dust, error, best_temps = find_best_fit_k_dust(modelT, single_layer_lookup)
            
            results[dust_thickness] = {
                'best_k_dust': best_k_dust,
                'error': error,
                'thermal_inertia': np.sqrt(best_k_dust * single_layer_cfg.rho_dust * single_layer_cfg.cp_dust),
                'temperatures': best_temps
            }
            
            # Plot comparison
            plt.subplot(3, 4, (i % 12) + 1)
            plt.plot(sim.t[tstart:] / 3600, modelT, 'b-', label='Two-layer')
            plt.plot(sim.t[tstart:] / 3600, best_temps, 'r--', 
                    label=f'Single-layer (k={best_k_dust:.2e})')
            plt.title(f'd={dust_thickness:.2e}m')
            if i % 12 == 0:
                plt.legend()
            
            if (i + 1) % 12 == 0 or i == len(dust_thickness_values) - 1:
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
                if i < len(dust_thickness_values) - 1:
                    fig = plt.figure(figsize=(15, 10))
    
    return results

if __name__ == "__main__":
    dust_thickness_values = np.array([10.0e-6, 30.0e-6, 50.0e-6, 100.0e-6, 200.0e-6, 
                                    500.0e-6, 0.001, 0.002, 0.005, 0.01, 0.02])
    k_dust_values = np.logspace(-5, 0, 25)  # 25 values from 1e-5 to 1e0
    
    temps_key = 'T_surf'  # Key for temperature profiles in results
    results = analyze_two_layer_equivalence(dust_thickness_values, k_dust_values, temps_key=temps_key)
    
    # Plot summary of results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    dust_thicknesses = list(results.keys())
    thermal_inertias = [results[d]['thermal_inertia'] for d in dust_thicknesses]
    plt.semilogx(dust_thicknesses, thermal_inertias, 'o-')
    plt.xlabel('Dust Thickness (m)')
    plt.ylabel('Equivalent Single-Layer Thermal Inertia (J/m²/K/s^0.5)')
    plt.grid(True)
    plt.savefig('thermal_inertia_vs_thickness.pdf')