import time
import numpy as np
from config import SimulationConfig
from grid import LayerGrid
from rte_hapke import RadiativeTransfer
from rte_disort import DisortRTESolver
from scipy.linalg import solve_banded
import scipy.optimize
import scipy.integrate
from scipy.optimize import nnls
from scipy.optimize import differential_evolution
import torch

# -----------------------------------------------------------------------------
# File: modelmain.py
# Description: Planetary regolith thermal model, 
# including Hapke's 1996 radiative transfer 2-stream approximation and the DISORT atmospheric radiative transfer N-stream solver. 
# Author: Andrew J. Ryan, 2025
#
# This code is free to use, modify, and distribute for any purpose.
# Please contact the author (ajryan4@arizona.edu) to discuss applications or if you
# use this code in your research or projects.
# -----------------------------------------------------------------------------

class Simulator:
	"""
	High-level simulator that ties together configuration, grid setup,
	radiative transfer, and heat diffusion time-stepping.
	"""
	def __init__(self, config: SimulationConfig = None):
		# Initialize configuration
		self.cfg = config or SimulationConfig()
		# Build spatial grid and FD matrix
		self.grid = LayerGrid(self.cfg)
		#Initialize crater geometry files for roughness model
		if self.cfg.crater:
			from crater import CraterMesh, SelfHeatingList, ShadowTester, CraterRadiativeTransfer
			# File paths can be set in config or hardcoded for now
			self.crater_mesh = CraterMesh('new_crater2.txt')
			self.crater_selfheating = SelfHeatingList('new_crater2_selfheating_list.txt')
			self.crater_shadowtester = ShadowTester(self.crater_mesh)
			self.crater_radtrans = CraterRadiativeTransfer(
				self.crater_mesh, self.crater_selfheating)
			
			# Initialize observer radiance calculator if enabled
			if self.cfg.compute_observer_radiance:
				from observer_radiance import ObserverRadianceCalculator
				self.observer_radiance_calc = ObserverRadianceCalculator(
					self.cfg, self.grid, self.cfg.observer_vectors)
		# Initialize radiative-transfer solver
		if(self.cfg.use_RTE and self.cfg.RTE_solver == 'hapke'):
			self.rte_hapke  = RadiativeTransfer(self.cfg, self.grid)
		if(self.cfg.use_RTE and self.cfg.RTE_solver == 'disort'):
			self.rte_disort = DisortRTESolver(self.cfg, self.grid,planck=True) #Used for thermal only in two-wave scenario. Used for all wavelengths in multi-wave scenario. 
			self.rte_disort_vis = DisortRTESolver(self.cfg, self.grid,planck=False) #Only used in two-wave scenario for the visible spectrum. Planck=False turns off thermal emission. 
			if self.cfg.crater:
				#Set up multi-column pydisort instances for running all crater facets in parallel. 
				self.rte_disort_crater = DisortRTESolver(self.cfg, self.grid,n_cols=len(self.crater_mesh.normals), planck=True) 
				self.rte_disort_crater_vis = DisortRTESolver(self.cfg, self.grid,n_cols=len(self.crater_mesh.normals),planck=False) 
		# Precompute time arrays and insolation flags
		self._setup_time_arrays()
		# Initialize state variables and output arrays
		self._init_state()


	def _init_state(self):
		"""Initialize state variables and output arrays."""
		# Initial temperature field (K)
		self.T = np.zeros(self.grid.x_num) + self.cfg.T_bottom
		# Initialize surface temp for non-RTE models
		self.T_surf = self.cfg.T_bottom
		
		# Storage for current time step
		self.current_time = 0.0
		self.current_step = 0
		
		# Output arrays sized for interpolated output points
		n_out = len(self.t_out)
		self.T_out = np.zeros((self.grid.x_num, n_out))
		self.phi_vis_out = np.zeros((self.grid.nlay_dust, n_out))
		self.phi_therm_out = np.zeros((self.grid.nlay_dust, n_out))
		self.T_surf_out = np.zeros(n_out)

		# Arrays for storing integration step results for interpolation
		self.T_history = []
		self.phi_vis_history = []
		self.phi_therm_history = []
		self.T_surf_history = []
		self.t_history = []

		if self.cfg.use_RTE and self.cfg.RTE_solver=='hapke':
			self.phi_therm_prev = np.zeros(self.grid.nlay_dust)
			self.phi_vis_prev = np.zeros(self.grid.nlay_dust)

		if self.cfg.crater:
			n_facets = len(self.crater_mesh.normals)
			n_out = len(self.t_out)
			self.T_crater = np.zeros((self.grid.x_num,n_facets)) + self.cfg.T_bottom  # [depth, facets]
			self.T_surf_crater = np.zeros(n_facets) + self.cfg.T_bottom
			self.flux_therm_crater = np.zeros(n_facets)
			self.illuminated = np.zeros(n_facets)
			self.T_brightness_crater = np.zeros(n_facets) + self.cfg.T_bottom
			self.T_crater_history = []  # [n_facets, depth] at each step
			self.T_surf_crater_history = []
			self.F_crater_obs_history = []  # [n_facets] at each step
			self.T_crater_out = np.zeros((self.grid.x_num,n_facets, n_out))  # surface temp for each facet at each output time
			self.T_surf_crater_out = np.zeros((n_facets,n_out))
			self.F_crater_obs_out = np.zeros((n_facets, n_out))  # observed flux for each facet at each output time
			self.n_facets = n_facets
			
			# Initialize solar angle arrays for crater facets
			self.mu_solar_facets = np.zeros(n_facets)
			self.phi_solar_facets = np.zeros(n_facets)
			
			# Observer radiance arrays
			if self.cfg.compute_observer_radiance:
				n_observers = len(self.cfg.observer_vectors)
				if self.cfg.use_RTE and self.cfg.RTE_solver == 'disort' and self.cfg.multi_wave:
					# Multi-wave case: need to determine number of wavelengths
					# This will be set later when DISORT is initialized
					self.observer_radiance_out = None  # Will be initialized in _make_outputs
				else:
					# Two-wave or non-RTE case: single radiance value per observer
					self.observer_radiance_out = np.zeros((n_observers, n_out))
			if(self.cfg.RTE_solver=='hapke'):
				self.phi_therm_prev_crater = np.zeros((self.grid.nlay_dust,n_facets))
				self.phi_vis_prev_crater = np.zeros((self.grid.nlay_dust,n_facets))

	def _setup_time_arrays(self):
		"""Set up time arrays for integration and output. Optimized for non-diurnal cases."""
		P = self.cfg.P

		if not self.cfg.diurnal:
			# Non-diurnal: two time steps (j=0 for initialization, j=1+ for calculation)
			# Use proper dt for numerical stability
			if self.cfg.auto_dt:
				initial_t_num = self.grid.steps_per_day * self.cfg.ndays
				dt = self.grid.dt
			else:
				initial_t_num = self.cfg.tsteps_day * self.cfg.ndays
				dt = P * self.cfg.ndays / initial_t_num
			
			self.t_num = 2  # Will be extended in main loop for steady-state convergence
			self.t = np.array([0.0, dt])
			
			# Single output time point
			self.t_out = np.array([0.0])
			
			# Store non-diurnal conditions as scalars (not arrays)
			if self.cfg.sun:
				self.mu_steady = self.cfg.steady_state_mu
				self.F_steady = 1.0
			else:
				self.mu_steady = -1.0
				self.F_steady = 0.0
			
			# Store dt for dynamic time extension
			self.dt_steady = dt
				
			# Don't create mu_array and F_array for non-diurnal
		else:
			# Diurnal: full time arrays
			# Set up integration time steps
			if self.cfg.auto_dt:
				self.t_num = self.grid.steps_per_day * self.cfg.ndays
				self.t = np.arange(self.t_num) * self.grid.dt
			else:
				self.t_num = self.cfg.tsteps_day * self.cfg.ndays
				self.t = np.linspace(0, P * self.cfg.ndays, self.t_num)

			# Set up output time points
			if self.cfg.last_day:
				# Only output points from the final day
				out_start = (self.cfg.ndays - 1) * P
				out_end = self.cfg.ndays * P
			else:
				# Output points for all days
				out_start = 0
				out_end = self.cfg.ndays * P
			
			# Create output time points at exact intervals
			points_per_day = self.cfg.freq_out
			self.t_out = np.linspace(out_start, out_end, 
									points_per_day * (1 if self.cfg.last_day else self.cfg.ndays))
			
			# Solar angles for integration timesteps
			hour_angle = (np.pi / (P / 2)) * (self.t - (P / 2))
			mu = np.sin(self.cfg.dec)*np.sin(self.cfg.latitude) + np.cos(self.cfg.latitude) * np.cos(hour_angle) * np.cos(self.cfg.dec)
			sun_up = mu > 0.001
			F = sun_up.astype(float)
			
			# Set arrays for diurnal case
			self.mu_array = mu
			self.F_array = F
		
		# Calculate 3D sun vectors for crater thermal model
		if(self.cfg.crater):
			# Simplified equations: +y is east, +x is south
			# Can handle signed latitude, non-zero declination, and zero latitude
			# sun_x is equivalent to sun north value
			# sun_y is equivalent to sun east value
			if(self.cfg.diurnal):
				hour_angle = (np.pi / (P / 2)) * (self.t - (P / 2))
				self.sun_x = -np.sin(self.cfg.dec)*np.cos(self.cfg.latitude) + np.cos(self.cfg.dec)*np.cos(hour_angle)*np.sin(self.cfg.latitude)
				self.sun_y = -np.cos(self.cfg.dec)*np.sin(hour_angle)
				self.sun_z = mu.copy()
			else:
				# Non-diurnal: fixed sun position (stored as scalars)
				if(self.cfg.sun):
					# Calculate sun position from steady_state_mu
					sun_elevation = np.arccos(self.cfg.steady_state_mu)
					sun_azimuth = 0.0  # Can be made configurable via config
					self.sun_x_steady = np.sin(sun_elevation) * np.cos(sun_azimuth)
					self.sun_y_steady = np.sin(sun_elevation) * np.sin(sun_azimuth)
					self.sun_z_steady = self.cfg.steady_state_mu
				else:
					# No sun
					self.sun_x_steady = 0.0
					self.sun_y_steady = 0.0
					self.sun_z_steady = -1.0


	def _bc(self, T, T_surf=0.0, Q = None):
		
		
		if self.cfg.use_RTE:
			# Neumann boundary condition for surface, as required by RTE. 
			# Flux term Q not included here because it was already accounte for in the RTE solver. 
			T[0]  = T[1]
		else:
			# Non-RTE boundary conditions, solves for surface temperature based on the heat flux balance with Newton's method. 
			T, T_surf = self._bc_noRTE(T,T_surf, Q)
		#Bottom boundary condition:
		if self.cfg.bottom_bc == "neumann":
			T[-1] = T[-2]
		elif self.cfg.bottom_bc == "dirichlet":
			T[-1] = self.cfg.T_bottom
		else:
			raise ValueError(f"Invalid bottom boundary condition: {self.cfg.bottom_bc}. Choose 'neumann' or 'dirichlet'.")			
		return T, T_surf

	def _bc_noRTE(self, T, T_surf, Q):
		"""Upper boundary for non-RTE mode, plus bottom according to bottom_bc."""
		#Compute T_surf
		T_surf = self._T_surf_calc(T,T_surf, Q)
		#Set virtual node (x[0]) to enforce the correct flux
		T[0] = (T[1] - T_surf)*(self.grid.x[0]/self.grid.x[1]) + T_surf
		return T, T_surf

	def _T_surf_calc(self, T, T_surf, Q):
		"""Newton‐solve for the surface temperature."""
		#T_surf argument is starting guess, from previous time step. Can be singular value or array.  
		#Q argument should be incoming flux from self-heating, direct, and indirect sunlight. 
		# if Q is not supplied, direct solar heating only is assumed and calculated here. 
		T1 = T[1]
		if(Q is None):
			Q = self.F * self.cfg.J * self.mu * (1 - self.cfg.albedo)
		se = self.cfg.sigma * self.cfg.em
		#distance from first node to the surface, which occurs halfway between this node and virtual node. Converting x units from tau to m here. 
		dx = ((self.grid.x[1] - self.grid.x[0])/2.)/self.cfg.Et 
		k_dx = self.cfg.k_dust/dx
		for it in range(self.cfg.T_surf_max_iter):
			#Calculate surface temperature T_surf with Newton's method.
			W =  (Q + k_dx*(T1 - T_surf) - se*T_surf**4.)
			dWdT = (-k_dx - 4.*se*T_surf**3.)
			dT = W/dWdT
			if np.all(np.abs(dT) < self.cfg.T_surf_tol):
				break
			T_surf -= dT
		else:
			print("Warning: T_surf solver exceeded max_iter")
		return T_surf

	def _fd1d_heat_implicit_diag(self, T, source_term=0):
		"""
		Implicit heat solver (banded) for one time step:
		solves diag * U_new = U_old + dt * source_term.
		"""

		b = T + self.grid.dt * source_term
		if(not self.cfg.single_layer and self.cfg.use_RTE):
			#the boundary flux value is stored in the source_term array in position nlay_dust+1
			i = self.grid.nlay_dust
			b[i+1] -= self.grid.dt * source_term[i+1] #clear the source term for the interface node, which is not a real node in the grid.
			#dz_rho_cp_i   = self.grid.l_thick[i] * self.grid.dens[i] * self.grid.heat[i]/self.cfg.Et
			dz_rho_cp_ip1 = self.grid.l_thick[i+1] * self.grid.dens[i+1] * self.grid.heat[i+1]/self.cfg.Et
			#Choose one of the two following options, modifying b[i] or b[i+1]. Unclear if one is better than the other.
			#b[i] += self.grid.dt * source_term[i+1] / dz_rho_cp_i
			b[i+1] += self.grid.dt * source_term[i+1] / dz_rho_cp_ip1
		# Solve banded system
		U_new = solve_banded((1, 1), self.grid.diag, b)
		return U_new
		# Apply boundary conditions


	def _make_outputs(self):
		"""Interpolate integration results to desired output times, or just use final step for non-diurnal. Always compute DISORT radiance if needed. Also handles crater outputs."""
		from scipy.interpolate import interp1d
		non_diurnal = not self.cfg.diurnal
		if non_diurnal:
			self.T_out = np.expand_dims(self.T_history[-1], axis=1)
			self.T_surf_out = np.array([self.T_surf_history[-1]])
			if self.cfg.use_RTE and self.cfg.RTE_solver == 'hapke':
				self.phi_vis_out = np.expand_dims(self.phi_vis_history[-1], axis=1)
				self.phi_therm_out = np.expand_dims(self.phi_therm_history[-1], axis=1)
			self.t_out = np.array([self.t_history[-1]])
			self.mu_out = np.array([self.mu_steady])
			# Crater outputs (non-diurnal): just take last state
			if self.cfg.crater:
				self.T_crater_out = np.expand_dims(self.T_crater.copy(), axis=2)[:, :, 0]  # [depth, facets, 1] -> [depth, facets]
				self.T_surf_crater_out = np.expand_dims(self.T_surf_crater.copy(), axis=1)  # [facets] -> [facets, 1]
		else:
			# Interpolate outputs to the desired user frequency, across whole sim or just in final day. 
			t_hist = np.array(self.t_history)
			T_hist = np.stack(self.T_history, axis=1)
			if(self.cfg.use_RTE and self.cfg.RTE_solver == 'hapke'):
				phi_vis_hist = np.stack(self.phi_vis_history, axis=1)
				phi_therm_hist = np.stack(self.phi_therm_history, axis=1)
			T_surf_hist = np.array(self.T_surf_history)
			# Crater histories
			if self.cfg.crater and len(self.T_crater_history) > 0:
				T_crater_hist = np.stack(self.T_crater_history, axis=2)  # [depth, facets, time]
				T_surf_crater_hist = np.stack(self.T_surf_crater_history, axis=1)  # [facets, time]
			# Ensure output times are within the simulation time range with a small buffer
			t_min = t_hist[0]
			t_max = t_hist[-1]
			dt = np.mean(np.diff(t_hist))  # Average time step
			eps = dt * 1e-10  # Small fraction of a time step
			t_out_clipped = np.clip(self.t_out, t_min + eps, t_max - eps)

			try:
				# Create interpolators with cubic splines for smoother results
				T_interp = interp1d(t_hist, T_hist, axis=1, kind='cubic', 
								bounds_error=True, assume_sorted=True)
				if(self.cfg.use_RTE and self.cfg.RTE_solver == 'hapke'):
					phi_vis_interp = interp1d(t_hist, phi_vis_hist, axis=1, kind='cubic',
									bounds_error=True, assume_sorted=True)
					phi_therm_interp = interp1d(t_hist, phi_therm_hist, axis=1, kind='cubic',
										bounds_error=True, assume_sorted=True)
				T_surf_interp = interp1d(t_hist, T_surf_hist, kind='cubic',
									bounds_error=True, assume_sorted=True)
				mu_interp = interp1d(t_hist, self.mu_array, kind='linear',
									bounds_error=True, assume_sorted=True)
				# Interpolate to clipped output times
				self.T_out = T_interp(t_out_clipped)
				if(self.cfg.use_RTE and self.cfg.RTE_solver == 'hapke'):
					self.phi_vis_out = phi_vis_interp(t_out_clipped)
					self.phi_therm_out = phi_therm_interp(t_out_clipped)
				self.T_surf_out = T_surf_interp(t_out_clipped)
				self.mu_out = mu_interp(t_out_clipped)
				# Crater outputs: interpolate for each facet and depth
				if self.cfg.crater and len(self.T_crater_history) > 0:
					n_depth, n_facets, n_hist = T_crater_hist.shape
					n_out = len(self.t_out)
					self.T_crater_out = np.zeros((n_depth, n_facets, n_out))
					for i in range(n_facets):
						for d in range(n_depth):
							interp_func = interp1d(t_hist, T_crater_hist[d, i, :], kind='cubic',
												   bounds_error=True, assume_sorted=True)
							self.T_crater_out[d, i, :] = interp_func(t_out_clipped)
					# Surface temperature for each facet
					self.T_surf_crater_out = np.zeros((n_facets, n_out))
					for i in range(n_facets):
						interp_func = interp1d(t_hist, T_surf_crater_hist[i, :], kind='cubic',
											   bounds_error=True, assume_sorted=True)
						self.T_surf_crater_out[i, :] = interp_func(t_out_clipped)
			except ValueError as e:
				print(f"Warning in interpolation: {e}")
				# Fall back to linear interpolation if cubic fails
				self.T_out = T_hist[:, -len(self.t_out):]
				if(self.cfg.use_RTE and self.cfg.RTE_solver == 'hapke'):
					self.phi_vis_out = phi_vis_hist[:, -len(self.t_out):]
					self.phi_therm_out = phi_therm_hist[:, -len(self.t_out):]
				self.T_surf_out = T_surf_hist[-len(self.t_out):]
				if self.cfg.diurnal:
					self.mu_out = self.mu_array[-len(self.t_out):]
				else:
					self.mu_out = np.full(len(self.t_out), self.mu_steady)
				# Crater outputs: fallback to last available
				if self.cfg.crater and len(self.T_crater_history) > 0:
					self.T_crater_out = T_crater_hist[:, :, -len(self.t_out):]
					self.T_surf_crater_out = T_surf_crater_hist[:, -len(self.t_out):]

		if(self.cfg.use_RTE and self.cfg.RTE_solver=='disort'):
			#Reinitialize disort to get observer radiance values at output times. 
			#The calculation of radiances is slower, hence we only do it at the end with our output temperature profiles. 
			#This will trigger the loading of new optical constants files for multi-wave, which optionally can 
			# be at a higher spectral resolution. 
			self.rte_disort = DisortRTESolver(self.cfg, self.grid,planck=True,output_radiance=True,observer_mu = self.cfg.observer_mu)
			self.rte_disort_vis = DisortRTESolver(self.cfg, self.grid,planck=False,output_radiance=True,observer_mu = self.cfg.observer_mu)
			if(self.cfg.multi_wave):
				nwave = len(self.rte_disort.wavenumbers)
				self.radiance_out = np.zeros((nwave,self.T_out.shape[1]))
			else:
				self.radiance_out = np.zeros(self.T_out.shape[1])
				self.radiance_out_therm = np.zeros(self.T_out.shape[1])
				self.radiance_out_vis = np.zeros(self.T_out.shape[1])
			self.rad_T_out = np.zeros(self.T_out.shape[1])  #Temperature computed from radiance blackbody fit. 
			wn_bounds = np.sort(np.loadtxt(self.cfg.wn_bounds_out))
			print("Computing DISORT radiance spectra for output.")
			for idx in range(self.T_out.shape[1]):
				if non_diurnal:
					F = self.F_steady
				else:
					t_hist = np.array(self.t_history)
					t = self.t_out[idx]
					F_idx = np.argmin(np.abs(t_hist - t))
					F = self.F_array[F_idx] #get nearest value for F. Don't want to interpolate and get a value that isn't 0 or 1. 
				rad = self.rte_disort.disort_run(self.T_out[:,idx],self.mu_out[idx],F) #Returns thermal radiance for two-wave case, and entire radiance for multi-wave. 
				if(self.cfg.multi_wave):
					self.radiance_out[:,idx] = rad.numpy()[:,0]
				else:
					rad_vis = self.rte_disort_vis.disort_run(self.T_out[:,idx],self.mu_out[idx],F)
					self.radiance_out[idx] = (rad+rad_vis).numpy()
					self.radiance_out_therm[idx] = rad.numpy()
					self.radiance_out_vis[idx] = rad_vis.numpy()
				#Compute effective blackbodies that fit the results. 
				if(self.cfg.multi_wave):
					self.rad_T_out[idx], _, _, _ = fit_blackbody_wn_banded(self,wn_bounds, rad.numpy()[:,0],idx=idx)
				else:
					#self.rad_T_out[idx] = fit_blackbody_broadband(self,rad.numpy(),idx=idx)
					self.rad_T_out[idx] = (rad.numpy()*np.pi/self.cfg.sigma)**0.25
			if self.cfg.multi_wave:
				#Run disort again with spectral features removed to produce a smooth radiance spectrum for emissivity division. 
				self.disort_emissivity()
		
		# Calculate observer radiance for crater if enabled
		if (self.cfg.crater and self.cfg.compute_observer_radiance and 
			hasattr(self, 'observer_radiance_calc')):
			print("Computing crater observer radiance.")
			
			# Initialize observer radiance output array if not done yet (multi-wave case)
			if self.observer_radiance_out is None:
				n_observers = len(self.cfg.observer_vectors)
				n_out = len(self.t_out)
				if self.cfg.use_RTE and self.cfg.RTE_solver == 'disort' and self.cfg.multi_wave:
					n_waves = len(self.rte_disort.wavenumbers)
					self.observer_radiance_out = np.zeros((n_observers, n_waves, n_out))
				else:
					self.observer_radiance_out = np.zeros((n_observers, n_out))
			
			# Calculate observer radiance for each output time
			for idx in range(len(self.t_out)):
				print(f"Time {idx+1} of {len(self.t_out)}...")
				if non_diurnal:
					F = self.F_steady
					mu_sun = self.mu_steady
				else:
					t_hist = np.array(self.t_history)
					t = self.t_out[idx]
					F_idx = np.argmin(np.abs(t_hist - t))
					F = self.F_array[F_idx]
					mu_sun = self.mu_array[F_idx]
				
				# Get crater temperature for this time step
				if non_diurnal:
					T_crater_time = self.T_crater_out.copy()  # [depth, n_facets]
				else:
					T_crater_time = self.T_crater_out[:, :, idx]  # [depth, n_facets]
				
				# Calculate scattered energy parameters for observer radiance calculation
				sun_vec = None
				crater_radtrans = None
				therm_flux_facets = None
				illuminated = None
				
				# Get sun vector if available
				if not self.cfg.diurnal and hasattr(self, 'sun_x_steady') and hasattr(self, 'sun_y_steady') and hasattr(self, 'sun_z_steady'):
					sun_vec = np.array([self.sun_x_steady, self.sun_y_steady, self.sun_z_steady])
				elif self.cfg.diurnal and hasattr(self, 'sun_x') and hasattr(self, 'sun_y') and hasattr(self, 'sun_z'):
					# For diurnal case, use F_idx from above
					if 'F_idx' in locals() and F_idx < len(self.sun_x):
						sun_vec = np.array([self.sun_x[F_idx], self.sun_y[F_idx], self.sun_z[F_idx]])
				
				# Use crater radiative transfer object if available
				if hasattr(self, 'crater_radtrans'):
					crater_radtrans = self.crater_radtrans
					
				# Use thermal flux if available
				if hasattr(self, 'flux_therm_crater'):
					therm_flux_facets = self.flux_therm_crater
					
				# Use illuminated facets if available  
				if hasattr(self, 'illuminated'):
					illuminated = self.illuminated
				
				# Calculate radiance for all observers
				observer_radiances = self.observer_radiance_calc.compute_all_observers(
					T_crater_time, self.crater_mesh, self.crater_shadowtester, mu_sun, F,
					sun_vec, crater_radtrans, therm_flux_facets, illuminated, 
					self.crater_albedo, self.crater_emissivity
				)
				
				# Store results
				if self.cfg.use_RTE and self.cfg.RTE_solver == 'disort' and self.cfg.multi_wave:
					# observer_radiances shape: [n_observers, n_waves, n_times]
					self.observer_radiance_out[:, :, idx] = observer_radiances
				else:
					# observer_radiances shape: [n_observers, n_times]  
					self.observer_radiance_out[:, idx] = observer_radiances


	def run(self):
		"""Execute the full time-stepping simulation."""
		start_time = time.time()
		
		# Steady-state convergence settings
		check_convergence = not self.cfg.diurnal
		n_check = getattr(self.cfg, 'steady_n_check', 200)  # How often to check
		window = getattr(self.cfg, 'steady_window', 100)     # How far back to look for temperature history polynomial fit. 
		tol = getattr(self.cfg, 'steady_tol', 2.0)         # Convergence threshold (K to extremum)
		converged = False

		source_term = np.zeros(self.grid.x_num)
		source_term_vis = np.zeros(self.grid.x_num)

		# Handle non-diurnal vs diurnal time stepping
		if not self.cfg.diurnal:
			# Non-diurnal: extend time arrays as needed for steady-state convergence
			j = 0
			dt = self.t[1] - self.t[0]  # Get dt from setup
			max_steps = 100000  # Safety limit for non-diurnal
		else:
			max_steps = self.t_num
			
		j = 0
		while j < max_steps:
			# For non-diurnal, extend time array if needed
			if not self.cfg.diurnal and j >= len(self.t):
				new_time = self.t[-1] + self.dt_steady
				self.t = np.append(self.t, new_time)
			
			# Set current time and step
			self.current_time = self.t[j]
			self.current_step = j
			
			# Set mu and F based on diurnal vs non-diurnal
			if not self.cfg.diurnal:
				# Use steady scalar values
				self.mu = self.mu_steady
				self.F = self.F_steady
			else:
				# Use time-varying arrays
				self.mu = self.mu_array[j]
				self.F = self.F_array[j]

			#Smooth surface model. 
			if j > 0:
				# Compute radiative source term (if RTE enabled, otherwise remains zero)
				if self.cfg.use_RTE:
					if(self.cfg.RTE_solver == 'hapke'):
						source_term,self.phi_vis_prev,self.phi_therm_prev = self.rte_hapke.compute_source(self.T,self.phi_vis_prev,self.phi_therm_prev, self.mu, self.F)
						source_term_vis = np.zeros_like(source_term) #Hapke model source term already includes vis. 
					elif(self.cfg.RTE_solver == 'disort'):
						source_term,_ = self.rte_disort.disort_run(self.T,self.mu,self.F)
						if(not self.cfg.multi_wave):
							#In the standard mode, run disort again for visible portion of the spectrum. 
							source_term_vis,_ = self.rte_disort_vis.disort_run(self.T,self.mu,self.F,Q=300/np.pi)
						else:
							source_term_vis = np.zeros_like(source_term)
					else:
						print("Error: Invalid RTE solver choice! Options are hapke or disort, or set use_RTE to False")
						return self.T_out, self.phi_vis_out, self.phi_therm_out, self.T_surf_out, self.t_out
				# Advance heat equation implicitly
				self.T = self._fd1d_heat_implicit_diag(self.T,source_term+source_term_vis)
				#Apply boundary conditions. 
				self.T, self.T_surf = self._bc(self.T, self.T_surf)


			# Store current state for interpolation
			self.T_history.append(self.T.copy())
			self.T_surf_history.append(self.T_surf)
			self.t_history.append(self.current_time)
			if(self.cfg.use_RTE and self.cfg.RTE_solver=='hapke'):
				self.phi_vis_history.append(self.phi_vis_prev.copy())
				self.phi_therm_history.append(self.phi_therm_prev.copy())

			#Run the rough surface model, if activated. 
			if self.cfg.crater:
				#Pre-calculate the effective bond albedo and emissivity of the smooth surface for scattering calcs later. 
				if j==0 and self.cfg.use_RTE:
					#Measure the effective bond albedo and emissivity of the regolith. 
					#Run the 1D model with the sun at 0 incidence angle to retrief the diffusve visible upwards visible and thermal flux. 
					F_sun = self.cfg.J
					nwaves = 1
					if(self.cfg.RTE_solver == 'hapke'):
						_,phi_vis_prev,phi_therm_prev = self.rte_hapke.compute_source(self.T,self.phi_vis_prev,self.phi_therm_prev, 1.0, 1.0)
						vis_flux_up = phi_vis_prev[0]*2*np.pi
						albedo = vis_flux_up/self.cfg.J
						therm_flux_up = phi_therm_prev[0]*2*np.pi
						emissivity = therm_flux_up / (self.cfg.sigma*self.cfg.T_bottom**4.)
						#Initialize initial upwards thermal flux for all crater facets used in self-heating calculations. 
						self.flux_therm_crater = np.full(self.n_facets,therm_flux_up)
					elif(self.cfg.RTE_solver == 'disort'):
						_,flux_up_therm = self.rte_disort.disort_run(self.T,0.,0.)
						if(self.cfg.sun):
							_,flux_up_vis = self.rte_disort_vis.disort_run(self.T,1.0,1.0)
						if self.cfg.multi_wave:
							#Determine wavelength-dependent vis reflectance and thermal emissivity 
							vis_range = self.rte_disort.wavenumbers>3333
							therm_range = self.rte_disort.wavenumbers<=3333
							nwaves = len(self.rte_disort.wavenumbers)
							if(self.cfg.sun):
								albedo = np.zeros_like(self.rte_disort.wavenumbers)
								albedo[vis_range] = (flux_up_vis[vis_range,0] / self.rte_disort.solar[vis_range])
							else:
								albedo = 0.0
							emissivity = np.zeros_like(self.rte_disort.wavenumbers)
							wn_bounds = np.sort(np.loadtxt(self.cfg.wn_bounds))[:sum(therm_range)+1]
							planck_btemp = np.zeros(np.sum(therm_range))
							for i in np.arange(np.sum(therm_range)):
								planck_btemp[i] = planck_wn_integrated(wn_bounds[i:i+2], self.cfg.T_bottom)
							emissivity[therm_range] = flux_up_therm[therm_range,0] / (planck_btemp*np.pi)
							emiss_avg = np.average(emissivity[therm_range])
							F_sun = np.loadtxt(self.cfg.solar_spectrum_file)[:,1]/self.cfg.R**2.
							#Upward therma flux for self heating. Shape [nfacets,nwave]
							self.flux_therm_crater = np.tile(flux_up_therm[:,0],(self.n_facets,1))
							self.flux_therm_crater[:,vis_range] = 0.0 
						else:
							if(self.cfg.sun):
								albedo = np.array(flux_up_vis) / self.cfg.J
							else:
								albedo = 0.0
							emissivity = np.array(flux_up_therm)/(self.cfg.sigma*self.cfg.T_bottom**4.)
							self.flux_up_therm = np.full(self.n_facets, flux_up_therm)
					# Store albedo and emissivity as instance variables for observer radiance calculation
					self.crater_albedo = albedo
					self.crater_emissivity = emissivity
					print("Crater effective albedo and emissivity: " + str(albedo) + " " + str(emissivity))
					#Configure flux_up arrays for calculating brigthness temperature
				elif j==0: 
					#non-rte case, use user-provided bond abledo for scattering calcs. 
					F_sun = self.cfg.J
					nwaves = 1
					albedo = self.cfg.albedo
					emissivity = self.cfg.em
					# Store albedo and emissivity as instance variables for observer radiance calculation
					self.crater_albedo = albedo
					self.crater_emissivity = emissivity
					self.flux_up_therm = np.full(self.n_facets,emissivity*self.cfg.sigma*self.T[0]**4.)
				if j > 0:
					#sun vector (pointing towards sun)
					if not self.cfg.diurnal:
						# Use steady scalar values
						sun_vec = np.array([self.sun_x_steady, self.sun_y_steady, self.sun_z_steady])
					else:
						# Use time-varying arrays
						sun_vec = np.array([self.sun_x[j], self.sun_y[j], self.sun_z[j]])
					#Calculate which crater facets are illuminated by the sun and solar angles
					if(self.F>0):
						if j==1 or j%self.cfg.illum_freq==0:
							self.illuminated = self.crater_shadowtester.illuminated_facets(sun_vec)
							
							# Calculate solar mu and phi for all facets using precomputed coordinates
							from crater import compute_solar_angles_all_facets
							self.mu_solar_facets, self.phi_solar_facets = compute_solar_angles_all_facets(
								self.crater_mesh, sun_vec)
					else:
						self.illuminated = np.zeros_like(self.T_surf_crater)
						# No solar illumination - set solar angles to zero
						self.mu_solar_facets = np.zeros(self.n_facets)
						self.phi_solar_facets = np.zeros(self.n_facets)
					#Get direct visible flux, scattered visible flux, and self heating thermal flux for all crater facets. 
					#Note that Q_dir and Q_scat have already been multiplied by (1-albedo)
					#TO DO: For multi-wave, we need to pass the solar spectrum for each band. 
					# Likewise, albedo and brightness temperature for each band. 
					Q_dir, Q_scat, Q_selfheat, cosines = self.crater_radtrans.compute_fluxes(
						sun_vec, self.illuminated, self.flux_therm_crater,albedo, emissivity, F_sun,nwaves,multiple_scatter=True
						)
					# if self.cfg.diurnal and j % max(100, self.t_num//20) == 0:
					# 	print(f"Q_dir, Q_scat, Q_selfheat {Q_dir[50]}/{Q_scat[50]}/{Q_selfheat[i]}")
					#Q_dir is already multipled by 1-albedo, illum fraction, and the cosine of the incidence angle, so it is absorbed energy. 
					#Q_scat is incident energy, so it is NOT multiplied by 1-albedo
					#Q_selfheat is likewise not multplied by the absorptivity of the surface. 
					#Both are divided by pi to convert to intesity, as needed for hapke and disort:
					Q_scat /= np.pi
					Q_selfheat /= np.pi
					if self.cfg.use_RTE and self.cfg.RTE_solver == 'disort':
						if(self.cfg.multi_wave):
							# Use proper solar mu and phi for each facet
							source_term, flux_up = self.rte_disort_crater.disort_run(
								self.T_crater, self.mu_solar_facets, self.illuminated, 
								Q=Q_selfheat+Q_scat, phi=self.phi_solar_facets)
							source_term_vis = np.zeros_like(source_term)
							self.flux_therm_crater = flux_up.swapaxes(0,1)
							self.flux_therm_crater[:,vis_range] = 0.0
						else:
							#In the standard two-wave mode, run disort for thermal and visible portions
							# Thermal: no direct solar input (mu=0), only self-heating
							source_term, self.flux_therm_crater = self.rte_disort_crater.disort_run(
								self.T_crater, np.zeros(self.n_facets), np.zeros(self.n_facets), 
								Q=Q_selfheat, phi=np.zeros(self.n_facets))
							
							if(np.any(Q_scat>1.0e-2)):
								# Visible: use proper solar angles for scattered sunlight
								source_term_vis,_ = self.rte_disort_crater_vis.disort_run(
									self.T_crater, self.mu_solar_facets, self.illuminated, 
									Q=Q_scat, phi=self.phi_solar_facets)
							else:
								source_term_vis = np.zeros_like(source_term)
						for i in np.arange(len(self.T_surf_crater)):
							self.T_crater[:,i] = self._fd1d_heat_implicit_diag(self.T_crater[:,i],source_term[i]+source_term_vis[i])
					else:
						for i in np.arange(len(self.T_surf_crater)):
							if self.cfg.use_RTE:
								if(self.cfg.RTE_solver == 'hapke'):
									# Use proper solar incidence angle for this facet
									source_term, self.phi_vis_prev_crater[:,i], self.phi_therm_prev_crater[:,i] = self.rte_hapke.compute_source(
										self.T_crater[:,i], self.phi_vis_prev_crater[:,i], self.phi_therm_prev_crater[:,i],
										self.mu_solar_facets[i], self.illuminated[i], Q_therm=Q_selfheat[i], Q_vis=Q_scat[i])
									#Calculate flux up from equation that phi = (up+down)/2, where down is the Q_therm from above. Multiply by pi as usual to convert from Hapke convention to flux in W/m2
									self.flux_therm_crater[i] = (self.phi_therm_prev_crater[0,i]*2 - Q_selfheat[i])*np.pi
							# Advance heat equation with solve_banded (must be looped)
							self.T_crater[:,i] = self._fd1d_heat_implicit_diag(self.T_crater[:,i],source_term+source_term_vis)
					#Apply boundary conditions, which is vectorized and doesn't require a loop. 
					self.T_crater, self.T_surf_crater = self._bc(self.T_crater, self.T_surf_crater,Q_dir+Q_scat*(1-albedo)*np.pi+Q_selfheat*emissivity*np.pi)
					if self.cfg.use_RTE:
						#Calculate T_surf_crater as an effective brightness temperature from upwards flux for outputs. 
						if(self.cfg.RTE_solver == 'hapke'):
							self.T_surf_crater = (self.flux_therm_crater/self.cfg.sigma/emissivity)**0.25
							#self.T_surf_crater = emissionT(self.T_crater[1:self.grid.nlay_dust+1],self.grid.x_boundaries,0.0,1.0)
						elif(self.cfg.RTE_solver == 'disort'):
							if(self.cfg.multi_wave):
								#Calculate the overall brightness temperature from the sum of all wavelengths. 
								self.T_surf_crater = (np.sum(self.flux_therm_crater,axis=1)/self.cfg.sigma/emiss_avg)**0.25
							else:
								self.T_surf_crater = (self.flux_therm_crater/self.cfg.sigma/emissivity)**0.25
					else:
						#Non-RTE model, calculate upwards thermal emission for self-heating in next time step. 
						self.flux_therm_crater = emissivity*self.cfg.sigma*self.T_surf_crater**4.
				self.T_crater_history.append(self.T_crater.copy())
				self.T_surf_crater_history.append(self.T_surf_crater.copy())


			# Terminate here for fixed temperature run, saving outputs at initialization temperature. Useful for spectroscopy. 
			if(self.cfg.T_fixed and not self.cfg.diurnal and j>0):
				break

			# Steady-state convergence check (only if diurnal=False)
			if check_convergence and (j % n_check == 0) and (len(self.T_history) >= window):
				T_hist_fields = self.T_history[-window:]
				T_hist_times = self.t_history[-window:]
				T_arr = np.stack(T_hist_fields, axis=1)  # shape: (x_num, window)
				times = np.array(T_hist_times)
				max_dist = 0.0
				for i in range(self.grid.x_num):
					p = np.polyfit(times, T_arr[i], 2)  # 2nd order polynomial fit
					# Vertex of parabola: t_ext = -b/(2a)
					a, b, c = p
					if a == 0:
						t_ext = times[-1]  # Linear, fallback to last time
					else:
						t_ext = -b / (2 * a)
					# Only accept extremum if it is within or just beyond the window
					t0, t1 = times[0], times[-1]
					if t_ext < t0:
						t_ext = t0
					elif t_ext > t1 + (t1-t0)/window:  # allow a small extrapolation
						t_ext = t1
					T_ext = a * t_ext**2 + b * t_ext + c
					dist = abs(self.T[i] - T_ext)
					max_dist = max(max_dist, dist)
				print(f"[Steady-state check] step {j}, max |T - T_ext| = {max_dist:.3e} K")
				if max_dist < tol:
					print(f"Converged to steady-state (all |T - T_ext| < {tol}) at step {j}, t={self.current_time:.2f}s.")
					converged = True
					break

			# Optional progress updates
			if self.cfg.diurnal and j % max(100, self.t_num//20) == 0:
				print(f"Time step {j}/{self.t_num}")
			elif not self.cfg.diurnal and j % max(100, 1000) == 0:
				print(f"Non-diurnal step {j}, time={self.current_time:.2f}s")
			
			# Increment loop counter
			j += 1
			
			# For diurnal runs, break when we've completed all planned time steps
			if self.cfg.diurnal and j >= self.t_num:
				break

		# Interpolate results to desired output times
		self._make_outputs()

		elapsed = time.time() - start_time
		print(f"Simulation completed in {elapsed:.2f} s")

		return self.T_out, self.phi_vis_out, self.phi_therm_out, self.T_surf_out, self.t_out
	
	def disort_emissivity(self):
		#Runs disort with uniform_props flag, which averages out all spectral properties (extinction and scattering properties)
		# for the purpose of producing an equivalent but spectrally smooth radiance spectrum for the emissivity division. 
		rte_disort = DisortRTESolver(self.cfg, self.grid,planck=True,output_radiance=True, uniform_props = True,observer_mu = self.cfg.observer_mu)
		if(self.cfg.multi_wave):
			nwave = len(rte_disort.wavenumbers)
			self.radiance_out_uniform = np.zeros((nwave,self.T_out.shape[1]))
		else:
			self.radiance_out_uniform = np.zeros(self.T_out.shape[1])
		#Need to get interpolated T_v_depth array, mu, and F as inputs. (F should be nearest value, not interp)
		print("Computing DISORT emissivity spectra for output.")
		for idx in range(self.T_out.shape[1]):
			if not self.cfg.diurnal:
				F = self.F_steady
			else:
				t_hist = np.array(self.t_history)
				t = self.t_out[idx]
				F_idx = np.argmin(np.abs(t_hist - t))
				F = self.F_array[F_idx] #get nearest value for F. Don't want to interpolate and get a value that isn't 0 or 1. 
			rad = rte_disort.disort_run(self.T_out[:,idx],self.mu_out[idx],F)
			if(self.cfg.multi_wave):
				self.radiance_out_uniform[:,idx] = rad[:,0]
			else:
				self.radiance_out_uniform[idx] = rad

def emissionT(T,tau_edges,T_interface,mu):
	if(len(T.shape)>1):
		T_calc = np.zeros(T.shape[1])
	else:
		T_calc = 0.0
	for i in np.arange(len(T)):
		T_calc += (T[i]**4.0)*(np.exp(-tau_edges[i]/mu) - np.exp(-tau_edges[i+1]/mu))
	T_calc += T_interface**4. * np.exp(-tau_edges[-1]/mu)
	return(T_calc**0.25)

def planck_wn_integrated(wn_edges, T):
	"""
	Integrate the Planck function over each wavenumber bin (edges in cm^-1).
	Returns band-integrated radiance (W/m^2/sr per band).
	"""
	h = 6.62607015e-34  # Planck constant (J s)
	c = 2.99792458e8    # Speed of light (m/s)
	k = 1.380649e-23    # Boltzmann constant (J/K)
	def planck_wn(wn, T):
		wn_m = wn * 100.0  # Convert from cm^-1 to m^-1
		return (2 * h * c**2 * wn_m**3) / (np.exp(h * c * wn_m / (k * T)) - 1)
	B_bands = np.zeros(len(wn_edges)-1)
	for i in range(len(B_bands)):
		# Integrate over each bin
		B_bands[i], _ = scipy.integrate.quad(planck_wn, wn_edges[i], wn_edges[i+1], args=(T,), limit=100)
	# Removed division by bin width to match DISORT's band-integrated output
	return B_bands*100

def fit_blackbody_wn_banded(sim,wn_edges, radiance,idx=-1):
	"""
	Fit a blackbody spectrum (integrated over each wavenumber band) to the given radiance spectrum.
	wn_edges: array of bin edges (cm^-1)
	radiance: array of band-integrated radiances (same length as len(wn_edges)-1)
	Returns best-fit temperature and the fitted blackbody band-integrated spectrum.
	"""
	wn_cutoff = 1500  # cm^-1, cutoff for fitting
	indx = np.argmin(np.abs(sim.rte_disort.wavenumbers - wn_cutoff))
	wn_edges = wn_edges[:indx+1]  # Use only up to the cutoff wavenumber
	radiance = radiance[:indx]  # Corresponding radiance values
	def loss(T):
		B = planck_wn_integrated(wn_edges, T[0])
		mask = (radiance > 0) & (B > 0)
		return np.sum((np.log(radiance[mask]) - np.log(B[mask]))**2)
	T0 = sim.T_out[1,idx]
	if hasattr(sim, 'T_crater_out'):
		if np.ndim(sim.T_crater_out) ==2:
			#steady state run. T_crater_out doesn't have a third dimension, which otherwise would be time. 
			minbound = sim.T_crater_out.min()-5
			maxbound = sim.T_crater_out.max()+5
		else:
			minbound = sim.T_crater_out[:,:,idx].min()-5
			maxbound = sim.T_crater_out[:,:,idx].max()+5
	else:
		minbound = sim.T_out[:,idx].min()-5
		maxbound = sim.T_out[:,idx].max()+5
	res = scipy.optimize.minimize(loss, [T0], bounds=[(minbound, maxbound)],)
	T_fit = res.x[0]
	B_fit = planck_wn_integrated(wn_edges, T_fit)
	return T_fit, B_fit, radiance/B_fit, sim.rte_disort.wavenumbers[:indx]


def calculate_interface_T(T,i,alpha,beta):
	return((alpha*T[i] + beta*T[i+1])/(alpha + beta))

# --- Crater effective brightness temperature calculation and plot ---
def crater_brightness_temperature(sim, obs_vec=None, time_idx=None, plot=True):
	"""
	Calculate effective brightness temperature of the crater as seen by an observer.
	Args:
		sim: Simulator object
		obs_vec: observer direction (3,), default is [0,0,1] (overhead)
		time_idx: if not None, only compute for this time index
		plot: if True, plot the time series
	Returns:
		Tb_time: array of brightness temperature vs time
	"""
	from scipy.constants import sigma
	if obs_vec is None:
		obs_vec = np.array([0,0,1])
	obs_vec = np.array(obs_vec) / np.linalg.norm(obs_vec)
	mesh = sim.crater_mesh
	shadowtester = sim.crater_shadowtester
	Tsurf = sim.T_surf_crater_out  # shape: (n_facets, n_out)
	n_facets, n_out = Tsurf.shape
	# Get facet normals and areas
	normals = mesh.normals  # (n_facets, 3)
	areas = mesh.areas     # (n_facets,)
	# Projected area for each facet as seen by observer
	proj = np.dot(normals, obs_vec)
	proj[proj < 0] = 0.0  # Only facets facing observer
	# For each time, get visibility (fractional, e.g. 0 or 1 for now)
	# Use shadowtester to get which facets are visible to observer (0 to 1)
	visible = shadowtester.illuminated_facets(obs_vec)
	Tb_time = np.zeros(n_out)
	for t in range(n_out):
		# If illuminated_facets returns bool, convert to float
		#visible = np.asarray(visible, dtype=float)
		# Mesh-area-weighted sum of emission (Stefan-Boltzmann law)
		# Only include visible facets and projected area
		numer = np.sum(areas * proj * visible * (Tsurf[:,t]**4))
		denom = np.sum(areas * proj * visible)
		if denom > 0:
			Tb_time[t] = numer/denom
			Tb_time[t] = Tb_time[t]**0.25
		else:
			Tb_time[t] = np.nan
	if plot:
		import matplotlib.pyplot as plt
		plt.figure()
		plt.plot(sim.t_out/3600, Tb_time, label='Crater Brightness Temp (observer)')
		plt.xlabel('Time (hours)')
		plt.ylabel('Brightness Temperature (K)')
		plt.title('Crater Effective Brightness Temperature vs Time')
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		plt.show()
	return(Tb_time)


def interactive_crater_temp_viewer(mesh, T_history, dt):
	"""
	Interactive 3D viewer for crater temperature time series.
	Args:
		mesh: your CraterMesh object (with .vertices, .faces)
		T_history: array [n_facets, n_time] surface temperature
		dt: timestep in seconds
	"""
	import numpy as np
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d.art3d import Poly3DCollection
	from matplotlib.widgets import Slider
	verts = mesh.vertices
	faces = mesh.faces
	n_times = T_history.shape[1]
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection='3d')

	# Initial plot
	time_idx = 0
	temp = T_history[:, time_idx]
	norm = plt.Normalize(np.min(T_history), np.max(T_history))
	facecolors = plt.cm.inferno(norm(temp))
	poly3d = [verts[face] for face in faces]
	pc = Poly3DCollection(poly3d, facecolors=facecolors, edgecolor='k', linewidths=0.05)
	meshplot = ax.add_collection3d(pc)

	ax.set_xlim([verts[:,0].min(), verts[:,0].max()])
	ax.set_ylim([verts[:,1].min(), verts[:,1].max()])
	ax.set_zlim([verts[:,2].min(), verts[:,2].max()])
	ax.set_box_aspect([2,2,1])
	ax.set_title(f"Surface Temperature - Time = {time_idx*dt:.1f} s")
	mappable = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
	cbar = plt.colorbar(mappable, ax=ax, shrink=0.6)
	cbar.set_label("Surface Temperature [K]")

	# Slider
	axcolor = 'lightgoldenrodyellow'
	ax_slider = plt.axes([0.2, 0.01, 0.6, 0.03], facecolor=axcolor)
	slider = Slider(ax_slider, 'Time Step', 0, n_times-1, valinit=0, valfmt='%d')

	def update(val):
		tidx = int(slider.val)
		temp = T_history[:, tidx]
		# Update color
		new_facecolors = plt.cm.inferno(norm(temp))
		pc.set_facecolor(new_facecolors)
		ax.set_title(f"Surface Temperature - Time = {tidx*dt:.1f} s")
		fig.canvas.draw_idle()

	slider.on_changed(update)
	plt.show()

def interactive_crater_radiance_viewer(mesh, observer_radiance_out, observer_vectors, t_out, cfg):
	"""
	Interactive 3D viewer for crater radiance as seen by different observers.
	Args:
		mesh: CraterMesh object
		observer_radiance_out: observer radiance array
		observer_vectors: list of observer direction vectors
		t_out: output time array
		cfg: simulation configuration
	"""
	import numpy as np
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d.art3d import Poly3DCollection
	from matplotlib.widgets import Slider
	
	# Get radiance data for visualization
	if cfg.use_RTE and cfg.RTE_solver == 'disort' and cfg.multi_wave:
		# Multi-wave case: observer_radiance_out shape [n_observers, n_waves, n_time]
		# Use first wavelength for visualization
		radiance_data = observer_radiance_out[:, 0, :]  # [n_observers, n_time]
		title_suffix = " (First Wavelength)"
	else:
		# Two-wave case: observer_radiance_out shape [n_observers, n_time]  
		radiance_data = observer_radiance_out
		title_suffix = ""
	
	n_observers = len(observer_vectors)
	n_times = len(t_out)
	
	# Create a synthetic radiance field by using observer vectors to determine facet visibility
	# This is a simplified approach - in reality each facet would have different radiance
	# based on its local temperature and viewing angle
	verts = mesh.vertices
	faces = mesh.faces
	normals = mesh.normals
	
	fig = plt.figure(figsize=(12, 8))
	ax = fig.add_subplot(111, projection='3d')
	
	# Initial state
	time_idx = 0
	observer_idx = 0
	
	# Calculate facet radiance based on visibility from current observer
	observer_vec = np.array(observer_vectors[observer_idx]) / np.linalg.norm(observer_vectors[observer_idx])
	facet_mu = np.dot(normals, observer_vec)
	facet_mu[facet_mu < 0] = 0.0  # Only facets facing observer
	
	# Use the global radiance value scaled by facet visibility
	global_radiance = radiance_data[observer_idx, time_idx]
	facet_radiance = facet_mu * global_radiance
	
	# Set up normalization across all data
	max_radiance = np.max(radiance_data)
	min_radiance = np.min(radiance_data[radiance_data > 0])  # Avoid zero values in log scale
	if min_radiance <= 0:
		min_radiance = max_radiance * 1e-6
	
	# Use log normalization for radiance
	from matplotlib.colors import LogNorm
	norm = LogNorm(vmin=min_radiance, vmax=max_radiance)
	
	# Handle zero/negative radiance values
	plot_radiance = np.maximum(facet_radiance, min_radiance)
	
	facecolors = plt.cm.plasma(norm(plot_radiance))
	poly3d = [verts[face] for face in faces]
	pc = Poly3DCollection(poly3d, facecolors=facecolors, edgecolor='k', linewidths=0.05)
	meshplot = ax.add_collection3d(pc)

	ax.set_xlim([verts[:,0].min(), verts[:,0].max()])
	ax.set_ylim([verts[:,1].min(), verts[:,1].max()])
	ax.set_zlim([verts[:,2].min(), verts[:,2].max()])
	ax.set_box_aspect([2,2,1])
	
	# Format observer vector for display
	obs_str = f"[{observer_vec[0]:.1f}, {observer_vec[1]:.1f}, {observer_vec[2]:.1f}]"
	ax.set_title(f"Crater Radiance{title_suffix}\nObserver: {obs_str}, Time: {t_out[time_idx]:.1f} s")
	
	mappable = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
	cbar = plt.colorbar(mappable, ax=ax, shrink=0.6)
	cbar.set_label("Radiance [W/m²/sr]")

	# Two sliders: time and observer
	axcolor = 'lightgoldenrodyellow'
	ax_time_slider = plt.axes([0.2, 0.02, 0.5, 0.03], facecolor=axcolor)
	ax_obs_slider = plt.axes([0.2, 0.06, 0.5, 0.03], facecolor=axcolor)
	
	time_slider = Slider(ax_time_slider, 'Time', 0, n_times-1, valinit=0, valfmt='%d')
	obs_slider = Slider(ax_obs_slider, 'Observer', 0, n_observers-1, valinit=0, valfmt='%d')

	def update_plot(time_idx, observer_idx):
		# Get current observer vector and radiance
		observer_vec = np.array(observer_vectors[observer_idx]) / np.linalg.norm(observer_vectors[observer_idx])
		facet_mu = np.dot(normals, observer_vec)
		facet_mu[facet_mu < 0] = 0.0
		
		# Scale global radiance by facet visibility
		global_radiance = radiance_data[observer_idx, time_idx]
		facet_radiance = facet_mu * global_radiance
		plot_radiance = np.maximum(facet_radiance, min_radiance)
		
		# Update colors
		new_facecolors = plt.cm.plasma(norm(plot_radiance))
		pc.set_facecolor(new_facecolors)
		
		# Update title
		obs_str = f"[{observer_vec[0]:.1f}, {observer_vec[1]:.1f}, {observer_vec[2]:.1f}]"
		ax.set_title(f"Crater Radiance{title_suffix}\nObserver: {obs_str}, Time: {t_out[time_idx]:.1f} s")
		fig.canvas.draw_idle()

	def update_time(val):
		tidx = int(time_slider.val)
		oidx = int(obs_slider.val)
		update_plot(tidx, oidx)
	
	def update_observer(val):
		tidx = int(time_slider.val)
		oidx = int(obs_slider.val)
		update_plot(tidx, oidx)

	time_slider.on_changed(update_time)
	obs_slider.on_changed(update_observer)
	plt.show()


def plot_observer_radiance_time_series(observer_radiance_out, observer_vectors, t_out, cfg):
	"""
	Plot integrated radiance over time for each observer vector (non-multi-wave diurnal cases).
	
	Args:
		observer_radiance_out: radiance array [n_observers, n_time]
		observer_vectors: list of observer direction vectors
		t_out: time array
		cfg: simulation configuration
	"""
	plt.figure(figsize=(12, 6))
	
	n_observers = len(observer_vectors)
	colors = plt.cm.tab10(np.linspace(0, 1, n_observers))
	
	for i, (observer_vec, color) in enumerate(zip(observer_vectors, colors)):
		# Create observer label from vector
		obs_label = f"Observer [{observer_vec[0]:.1f}, {observer_vec[1]:.1f}, {observer_vec[2]:.1f}]"
		
		# Plot radiance time series for this observer
		plt.plot(t_out / 3600, observer_radiance_out[i, :], 
				 label=obs_label, color=color, linewidth=2)
	
	plt.xlabel('Time (hours)')
	plt.ylabel('Integrated Radiance (W/m²/sr)')
	plt.title('Crater Observer Radiance vs Time')
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.show()


def plot_observer_emissivity_spectra_interactive(observer_radiance_out, observer_vectors, t_out, cfg, sim):
	"""
	Interactive emissivity spectrum plot with time slider for multi-wave cases.
	Shows emissivity spectrum for each observer angle at each time step.
	
	Args:
		observer_radiance_out: radiance array [n_observers, n_waves, n_time]
		observer_vectors: list of observer direction vectors  
		t_out: time array
		cfg: simulation configuration
		sim: simulator object (for wavelength info)
	"""
	from matplotlib.widgets import Slider
	
	# Get wavelength information from the simulator
	if hasattr(sim, 'rte_disort') and hasattr(sim.rte_disort, 'wavenumbers'):
		wavenumbers = sim.rte_disort.wavenumbers
		wavelengths = 10000.0 / wavenumbers  # Convert cm^-1 to microns
	else:
		print("Warning: Could not get wavelength information from simulator")
		return
	
	n_observers, n_waves, n_times = observer_radiance_out.shape
	
	wn_cutoff = 1500  # cm^-1, cutoff for fitting blackbody
	indx = np.argmin(np.abs(wavenumbers - wn_cutoff))

	# Calculate emissivity by dividing by blackbody radiance at surface temperature
	emissivity_data = np.zeros_like(observer_radiance_out)
	
	for t_idx in range(n_times):
		for obs_idx in range(n_observers):
			# Read bin edges from config
			wn_bounds = np.sort(np.loadtxt(sim.cfg.wn_bounds_out))
			T_fit, B_fit, emiss_spec, wn_BB = fit_blackbody_wn_banded(sim,wn_bounds, observer_radiance_out[obs_idx, :, t_idx],idx=t_idx)
			idx1 = np.argmin(np.abs(wavenumbers - 900))
			idx2 = np.argmin(np.abs(wavenumbers - 1200))
			cf_emis = emiss_spec[idx1:idx2].max()
			emissivity_data[obs_idx, :indx, t_idx] = emiss_spec + 1 - cf_emis
	
	# Create the interactive plot
	fig, ax = plt.subplots(figsize=(12, 8))
	plt.subplots_adjust(bottom=0.25)
	
	# Initial time index
	time_idx = 0
	
	# Plot initial spectra for all observers
	colors = plt.cm.tab10(np.linspace(0, 1, n_observers))
	lines = []
	
	for obs_idx, (observer_vec, color) in enumerate(zip(observer_vectors, colors)):
		obs_label = f"Observer [{observer_vec[0]:.1f}, {observer_vec[1]:.1f}, {observer_vec[2]:.1f}]"
		line, = ax.plot(wavenumbers[:indx], emissivity_data[obs_idx, :indx, time_idx], 
						label=obs_label, color=color, linewidth=2)
		lines.append(line)
	
	ax.set_xlabel('Wavenumber (cm-1)')
	ax.set_ylabel('Emissivity')
	ax.set_title(f'Crater Observer Emissivity Spectra - Time: {t_out[time_idx]/3600:.2f} hours')
	ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	ax.grid(True, alpha=0.3)
	ax.set_ylim([0, 1.2])
	ax.set_xlim([wavenumbers.min(), wavenumbers[:indx].max()])
	
	# Add time slider
	axcolor = 'lightgoldenrodyellow'
	ax_time_slider = plt.axes([0.2, 0.1, 0.5, 0.03], facecolor=axcolor)
	time_slider = Slider(ax_time_slider, 'Time', 0, n_times-1, valinit=0, valfmt='%d')
	
	def update_plot(val):
		tidx = int(time_slider.val)
		
		# Update all observer lines
		for obs_idx, line in enumerate(lines):
			line.set_ydata(emissivity_data[obs_idx, :indx, tidx])
		
		# Update title
		ax.set_title(f'Crater Observer Emissivity Spectra - Time: {t_out[tidx]/3600:.2f} hours')
		fig.canvas.draw_idle()
	
	time_slider.on_changed(update_plot)
	plt.tight_layout()
	plt.gca().invert_xaxis()
	plt.show()


if __name__ == "__main__":
	sim = Simulator()
	T_out, phi_vis, phi_therm, T_surf_out, t_out = sim.run()
	import matplotlib.pyplot as plt

	# Plot temperature at the surface (first grid point) over time
	if(sim.cfg.diurnal):
		plt.figure(figsize=(10, 5))
		if(sim.cfg.use_RTE):
			#plt.plot(sim.t_out / 3600, (2.0*phi_therm[0,:]*np.pi/5.670e-8)**0.25, label='Phi temperature')
			plt.plot(sim.t_out / 3600, T_out[1,:], label='Surface Temperature')
			#emissT = emissionT(T_out[1:sim.grid.nlay_dust+1,:],sim.grid.x[1:sim.grid.nlay_dust+1],1.0)
			emissT = np.zeros(len(sim.t_out))
			T_interface = 0.0
			for j in np.arange(len(sim.t_out)):
				if not sim.cfg.single_layer:
					T_interface = calculate_interface_T(T_out[:,j], sim.grid.nlay_dust, sim.grid.alpha, sim.grid.beta)
				emissT[j] = emissionT(T_out[1:sim.grid.nlay_dust+1,j],sim.grid.x_boundaries[:sim.grid.nlay_dust+1],T_interface,1.0)
			plt.plot(sim.t_out / 3600, emissT, label='Emission Temperature')
			if(sim.cfg.RTE_solver == 'disort'):
				plt.plot(sim.t_out / 3600, sim.rad_T_out, label='Radiance Fit Temperature')
			if(sim.cfg.RTE_solver== 'hapke'):
				plt.plot(sim.t_out / 3600, (sim.phi_therm_out[0,:]*2*np.pi/sim.cfg.sigma)**0.25,label="phi therm temperature")
		else:
			plt.plot(sim.t_out / 3600, T_surf_out, label='Surface Temperature (no RTE)')
		if sim.cfg.crater and hasattr(sim, 'T_surf_crater_out'):
			btemp = crater_brightness_temperature(sim, obs_vec=[0,0,1],plot=False)
			plt.plot(sim.t_out / 3600, btemp,label='Crater brightness temperature')
		plt.xlabel('Time (hours)')
		plt.ylabel('Temperature (K)')
		plt.title('Surface Temperature vs Time')
		plt.legend()
		plt.tight_layout()
		plt.show()


	# Plot temperature vs depth profiles for final day
	plt.figure(figsize=(10, 6))
	Et = sim.cfg.Et
	x = sim.grid.x
	
	# Calculate time fractions for the final day
	if sim.cfg.last_day:
		# If only the last day was output, use all points
		time_fracs = np.linspace(0, 1, len(sim.t_out))
	else:
		# If all days were output, select points from the final day
		pts_per_day = sim.cfg.freq_out
		time_fracs = np.linspace(0, 1, pts_per_day)
		T_out = T_out[:, -pts_per_day:]  # Select final day's worth of points
		phi_vis = phi_vis[:, -pts_per_day:]
		phi_therm = phi_therm[:, -pts_per_day:]
	
	# Plot profiles at 10 times throughout the day
	plot_fracs = np.linspace(0, 0.9, 10)  # Plot at 0.0, 0.1, ..., 0.9 of the day
	for frac in plot_fracs:
		# Find nearest output time to desired fraction
		idx = np.argmin(np.abs(time_fracs - frac))
		plt.semilogx(x[1:]/Et, T_out[1:,idx], 
					 label=f't+{frac:.1f}P')
		if not sim.cfg.diurnal: break
	
	plt.xlabel('Depth into medium [m]')
	plt.ylabel('Kinetic Temperature [K]')
	plt.legend(loc='upper right')
	plt.grid(True)
	plt.show()
	# if(sim.cfg.use_RTE and sim.cfg.RTE_solver=='disort'):
	# 	if(sim.cfg.multi_wave):
	# 		plt.plot(sim.rte_disort.wavenumbers,sim.radiance_out[:,-1])
	# 		plt.xlabel('Wavenumber [cm-1]')
	# 		plt.ylabel('Radiance')
	# 		plt.show()
	# 	else:
	# 		plt.plot(sim.t_out / 3600, sim.radiance_out)
	# 		plt.xlabel('Time')
	# 		plt.ylabel('Radiance')
	# 		plt.show()

	if (sim.cfg.use_RTE and sim.cfg.RTE_solver == 'hapke'):
		# For RTE quantities (phi), only plot dust layer values
		x_dust = x[1:sim.grid.nlay_dust+1]  # Exclude virtual nodes, only dust layers
		
		# Plot phi_therm in dust layer
		plt.figure(figsize=(10, 6))
		for frac in plot_fracs:
			idx = np.argmin(np.abs(time_fracs - frac))
			plt.semilogx(x_dust/Et, phi_therm[:,idx], 
						label=f't+{frac:.1f}P')
			if not sim.cfg.diurnal: break
		
		plt.xlabel('Depth into medium [m]')
		plt.ylabel('phi_therm')
		plt.legend(loc='upper right')
		plt.grid(True)
		plt.show()

		# Plot phi_vis in dust layer
		plt.figure(figsize=(10, 6))
		for frac in plot_fracs:
			idx = np.argmin(np.abs(time_fracs - frac))
			plt.semilogx(x_dust/Et, phi_vis[:,idx], 
						label=f't+{frac:.1f}P')
			if not sim.cfg.diurnal: break
		
		plt.xlabel('Depth into medium [m]')
		plt.ylabel('phi_vis')
		plt.legend(loc='upper right')
		plt.grid(True)
		plt.show()

	# Plot final emissivity spectrum for non-diurnal, DISORT case
	if (not sim.cfg.diurnal) and sim.cfg.use_RTE and sim.cfg.RTE_solver == 'disort' and sim.cfg.multi_wave:
		wn = sim.rte_disort.wavenumbers  # [cm^-1], band centers
		final_rad = sim.radiance_out[:, -1]  # Final time step
		ir_cutoff = np.argmin(np.abs(wn - 1500))  # cm^-1, cutoff for IR bands
		# Read bin edges from config
		wn_bounds = np.sort(np.loadtxt(sim.cfg.wn_bounds_out))
		T_fit, B_fit, emiss_spec, wn_BB = fit_blackbody_wn_banded(sim,wn_bounds, final_rad)
		substrate = np.loadtxt(sim.cfg.substrate_spectrum_out)
		#otesT1 = np.loadtxt(sim.cfg.otesT1_out)
		#otesT2 = np.loadtxt(sim.cfg.otesT2_out)
		idx1 = np.argmin(np.abs(wn - 900))
		idx2 = np.argmin(np.abs(wn - 1200))
		#otes_cf_emis = otesT1[idx1:idx2,1].max()
		#bbfit_cf_emis = emiss_spec[idx1:idx2].max()
		ds_cf_emis = (final_rad/sim.radiance_out_uniform[:, -1])[idx1:idx2].max()
		#samp_cf_emis = otes_samp[idx1:idx2,1].max()
		idx1 = np.argmin(np.abs(substrate[:,0] - 1200))
		idx2 = np.argmin(np.abs(substrate[:,0] - 900))
		substrate_cf = substrate[idx1:idx2,1].max()
		plt.figure()
		#plt.plot(wn_BB, emiss_spec + otes_cf_emis - bbfit_cf_emis, label=f'Emissivity (T_fit={T_fit:.1f} K)')
		#plt.plot(multi_wn_BB, multi_emiss_spec, label=f'Mixture Emissivity (T_fit={multi_T_fit})')
		plt.plot(wn[:ir_cutoff], (final_rad/sim.radiance_out_uniform[:, -1])[:ir_cutoff]+1-ds_cf_emis, label='DISORT emissivity',linewidth=2)
		plt.plot(substrate[:,0], substrate[:,1]+1-substrate_cf, label='Sabel Enstatite',linewidth=1)
		if(sim.cfg.crater):
			final_rad_crater = sim.observer_radiance_out[0,:,0] #[observation_angle, wavelengths, time]
			T_fit, B_fit, emiss_spec, wn_BB = fit_blackbody_wn_banded(sim,wn_bounds, final_rad_crater,idx=0)
			print(T_fit)
			idx1 = np.argmin(np.abs(wn_BB - 900))
			idx2 = np.argmin(np.abs(wn_BB - 1200))
			ds_cf_emis = emiss_spec[idx1:idx2].max()
			plt.plot(wn_BB, emiss_spec+1-ds_cf_emis, label='DISORT crater emissivity',linewidth=2)
		#plt.plot(wn[:ir_cutoff], otesT1[:ir_cutoff,1], label='OTES T1 (fewer fines)')
		#plt.plot(wn[:ir_cutoff], otesT2[:ir_cutoff,1], label='OTES T2 (more fines)')
		plt.xlabel('Wavenumber [cm$^{-1}$]')
		plt.ylabel('Emissivity')
		plt.title('Final Emissivity Spectrum (Radiance / Best-fit Blackbody, band-integrated)')
		plt.legend()
		plt.grid(True)
		plt.gca().invert_xaxis()
		plt.show()

	# --- Crater 3D surface temperature visualization ---
	if sim.cfg.crater and hasattr(sim, 'T_surf_crater_out'):
		print("Displaying interactive crater temperature viewer...")
		interactive_crater_temp_viewer(sim.crater_mesh, sim.T_surf_crater_out, sim.grid.dt)
		
		# Show observer radiance plots if available
		if (sim.cfg.compute_observer_radiance and hasattr(sim, 'observer_radiance_out') 
			and sim.observer_radiance_out is not None):
			
			if sim.cfg.multi_wave and sim.cfg.diurnal:
				# Multi-wave case: interactive emissivity spectrum plot with time slider
				print("Displaying interactive crater observer emissivity spectra...")
				plot_observer_emissivity_spectra_interactive(sim.observer_radiance_out, 
														   sim.cfg.observer_vectors, 
														   sim.t_out, sim.cfg, sim)
			elif sim.cfg.diurnal and not sim.cfg.multi_wave:
				# Non-multi-wave diurnal case: time series plot for each observer
				print("Displaying crater observer radiance time series...")
				plot_observer_radiance_time_series(sim.observer_radiance_out, 
												  sim.cfg.observer_vectors, 
												  sim.t_out, sim.cfg)