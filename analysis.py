from modelmain import Simulator
from config import SimulationConfig
from itertools import product
import numpy as np
from scipy.optimize import minimize


def run_parameter_sweep(param_grid, cfg=None):
    import matplotlib.pyplot as plt
    names = list(param_grid.keys())
    results = {}
    if not cfg:
        cfg = SimulationConfig()
    for combo in product(*(param_grid[n] for n in names)):
        # Set the parameter combinations
        for name, val in zip(names, combo):
            setattr(cfg, name, val)
        sim = Simulator(cfg)
        T_out, _, phi_th, T_surf = sim.run()
        results[combo] = {
            'T_out': T_out,
            'phi_therm_out': phi_th,
            'T_surf': T_surf
        }
        plt.plot(sim.t / 3600, T_surf, label='Surface Temperature (no RTE)')
        plt.show()
    return results

def phi_T_surf(phi):
    #phi is phi_therm_out[0,:]
    #Calculate effective surface temperature from radiative flux. 
    # Hale and Hapke, 2002 eq. 15
    sigma = 5.670e-8
    return ((2.0*phi*np.pi/sigma)**0.25)

def tau_T_surf(T,tau,mu):
	T_calc = 0.0
	wt_calc = 0.0
	for i in np.arange(len(T)):
		T_calc += (T[i]**4.0)*np.exp(-tau[i]/mu)
		wt_calc += np.exp(-tau[i]/mu)
	T_calc = T_calc/wt_calc
	return(T_calc**0.25)



def find_equivalent_param(results, target_combo, var_param_name, bounds, cfg=None):
    """Find parameter value that gives equivalent temperature profile to target."""
    if not cfg:
        cfg = SimulationConfig()
    
    # Get target temperatures from last day
    tstart = int(cfg.tsteps_day*(cfg.ndays-1))
    target_T = results[target_combo]['T_surf'][tstart:]
    
    def f(var_val):
        # Set parameter and run simulation
        setattr(cfg, var_param_name, var_val)
        sim = Simulator(cfg)
        T_out, _, phi_th, T_surf = sim.run()
        
        # Calculate RMS difference for last day
        T_diff = T_surf[tstart:] - target_T
        return np.sqrt(np.sum(T_diff**2))
    result = minimize(
        f, 
        x0=[(bounds[0] + bounds[1])/2],  # start in middle of bounds
        bounds=[bounds],
        method='L-BFGS-B'
    )
    
    if not result.success:
        print(f"Warning: Optimization failed: {result.message}")
    
    return result.x[0]


if __name__ == "__main__":
    # Example usage
    grid = {
        'dust_thickness': [0.02]  # dust thickness in meters
        #'Et':             [5000, 7000, 9000]
    }
    base_cfg = SimulationConfig()
    base_cfg.use_RTE = False 
    base_cfg.single_layer = False 
    base_cfg.ndays = 5
    base_cfg.rock_thickness = 0.40
    lookup = run_parameter_sweep(grid,cfg=base_cfg)

    #closest = fit_equivalent_configs(lookup, target_combo=(5e-5,7000))
    #print("Closest:", closest[1])
    fit_cfg = SimulationConfig()
    fit_cfg.dust_thickness = 0.40
    fit_cfg.use_RTE = False 
    fit_cfg.single_layer = True 
    fit_cfg.ndays = 5
    fit_cfg.k_dust_auto = False
    k_eq = find_equivalent_param(
        lookup,
        target_combo=(0.02,),
        var_param_name='k_dust',
        bounds=(0.001, 0.1),
        cfg = fit_cfg
    )
    print("Thermal inertia", np.sqrt(k_eq*fit_cfg.rho_dust*fit_cfg.cp_dust))
    import matplotlib.pyplot as plt
    fit_cfg.k_dust = k_eq
    sim = Simulator(fit_cfg)
    T_out, _, phi_th, T_surf = sim.run()
    tstart = int(sim.cfg.tsteps_day*(sim.cfg.ndays-1))
    plt.plot(sim.t / 3600, T_surf, label='Best fit single layer model')
    plt.plot(sim.t / 3600, lookup[(0.02,)]['T_surf'], label='2-layer model')
    plt.legend(loc='lower left')


