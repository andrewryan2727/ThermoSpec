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

# -----------------------------------------------------------------------------
# File: modelmain.py
# Description: Planetary regolith thermal model, including Hapke's 1996 radiative transfer 2-stream approximation. .
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
		# Initialize radiative-transfer solver
		if(self.cfg.use_RTE and self.cfg.RTE_solver == 'hapke'):
			self.rte_hapke  = RadiativeTransfer(self.cfg, self.grid)
		if(self.cfg.use_RTE and self.cfg.RTE_solver == 'disort'):
			self.rte_disort = DisortRTESolver(self.cfg, self.grid)
		# Precompute time arrays and insolation flags
		self._setup_time_arrays()
		# Initialize state variables and output arrays
		self._init_state()


	def _setup_time_arrays(self):
		"""Set up time arrays for integration and output."""
		P = self.cfg.P

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
		mu = np.cos(self.cfg.latitude) * np.cos(hour_angle)
		sun_up = mu > 0.001
		F = sun_up.astype(float)
		
		# Handle non-diurnal cases
		if not self.cfg.diurnal:
			if self.cfg.sun:
				mu = np.ones_like(mu)
				F = np.ones_like(F)
			else:
				mu = -np.ones_like(mu)
				F = np.zeros_like(F)
		
		self.mu_array = mu
		self.F_array = F

	def _bc(self):
		if not self.cfg.use_RTE:
			# Non-RTE boundary conditions, solves for surface temperature based on the heat flux balance.
			self._bc_noRTE()
		else:
			# Neumann boundary condition for surface, as required by RTE.
			self.T[0]  = self.T[1]
			#Bottom boundary condition:
			if self.cfg.bottom_bc == "neumann":
				self.T[-1] = self.T[-2]
			elif self.cfg.bottom_bc == "dirichlet":
				self.T[-1] = self.cfg.T_bottom
			else:
				raise ValueError(f"Invalid bottom boundary condition: {self.cfg.bottom_bc}. Choose 'neumann' or 'dirichlet'.")

	def _bc_noRTE(self):
		"""Upper boundary for non-RTE mode, plus bottom according to bottom_bc."""
		#Compute T_surf
		self._T_surf_calc()
		#Set virtual node (x[0]) to enforce the correct flux
		self.T[0] = (self.T[1] - self.T_surf)*(self.grid.x[0]/self.grid.x[1]) + self.T_surf

		#bottom BC:
		if self.cfg.bottom_bc == "neumann":
			self.T[-1] = self.T[-2]
		elif self.cfg.bottom_bc == "dirichlet":
			self.T[-1] = self.cfg.T_bottom
		else:
			raise ValueError(f"Invalid bottom boundary condition: {self.cfg.bottom_bc}. Choose 'neumann' or 'dirichlet'.")

	def _T_surf_calc(self):
		"""Newton‐solve for the surface temperature."""
		T1 = self.T[1]
		S = self.F * self.cfg.J * self.mu * (1 - self.cfg.albedo)
		se = self.cfg.sigma * self.cfg.em
		#distance from first node to the surface, which occurs halfway between this node and virtual node. Converting x units from tau to m here. 
		dx = ((self.grid.x[1] - self.grid.x[0])/2.)/self.cfg.Et 
		k_dx = self.cfg.k_dust/dx
		for it in range(self.cfg.T_surf_max_iter):
			#Calculate surface temperature T_surf with Newton's method.
			W =  (S + k_dx*(T1 - self.T_surf) - se*self.T_surf**4.)
			dWdT = (-k_dx - 4.*se*self.T_surf**3.)
			dT = W/dWdT
			if np.abs(dT) < self.cfg.T_surf_tol:
				break
			self.T_surf -= dT
		else:
			print("Warning: T_surf solver exceeded max_iter")  

	def _fd1d_heat_implicit_diag(self):
		"""
		Implicit heat solver (banded) for one time step:
		solves diag * U_new = U_old + dt * source_term.
		"""
		b = self.T + self.grid.dt * self.source_term
		if(not self.cfg.single_layer and self.cfg.use_RTE):
			#the boundary flux value is stored in the source_term array in position nlay_dust+1
			i = self.grid.nlay_dust
			b[i+1] -= self.grid.dt * self.source_term[i+1] #clear the source term for the interface node, which is not a real node in the grid.
			dz_rho_cp_i   = self.grid.l_thick[i] * self.grid.dens[i] * self.grid.heat[i]/self.cfg.Et
			dz_rho_cp_ip1 = self.grid.l_thick[i+1] * self.grid.dens[i+1] * self.grid.heat[i+1]/self.cfg.Et
			#Choose one of the two following options, modifying b[i] or b[i+1]. Unclear if one is better than the other.
			#b[i] += self.grid.dt * self.source_term[i+1] / dz_rho_cp_i
			b[i+1] += self.grid.dt * self.source_term[i+1] / dz_rho_cp_ip1
		# Solve banded system
		U_new = solve_banded((1, 1), self.grid.diag, b)
		self.T = U_new
		# Apply boundary conditions


	def _init_state(self):
		"""Initialize state variables and output arrays."""
		# Initial temperature field (K)
		self.T = np.zeros(self.grid.x_num) + self.cfg.T_bottom
		# Radiative source vector
		self.source_term = np.zeros(self.grid.x_num)
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

	def _make_outputs(self):
		"""Interpolate integration results to desired output times, or just use final step for non-diurnal. Always compute DISORT radiance if needed."""
		from scipy.interpolate import interp1d
		non_diurnal = not self.cfg.diurnal
		if non_diurnal:
			self.T_out = np.expand_dims(self.T_history[-1], axis=1)
			self.T_surf_out = np.array([self.T_surf_history[-1]])
			if self.cfg.use_RTE and self.cfg.RTE_solver == 'hapke':
				self.phi_vis_out = np.expand_dims(self.phi_vis_history[-1], axis=1)
				self.phi_therm_out = np.expand_dims(self.phi_therm_history[-1], axis=1)
			self.t_out = np.array([self.t_history[-1]])
			self.mu_out = np.array([self.mu_array[-1]])
		else:
			#Interpolate outputs to the desired user frequency, across whole sim or just in final day. 
			# Convert history lists to arrays for interpolation
			t_hist = np.array(self.t_history)
			T_hist = np.stack(self.T_history, axis=1)
			if(self.cfg.use_RTE and self.cfg.RTE_solver == 'hapke'):
				phi_vis_hist = np.stack(self.phi_vis_history, axis=1)
				phi_therm_hist = np.stack(self.phi_therm_history, axis=1)
			T_surf_hist = np.array(self.T_surf_history)
			
			# Ensure output times are within the simulation time range with a small buffer
			# to avoid floating point edge cases
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
			except ValueError as e:
				print(f"Warning in interpolation: {e}")
				# Fall back to linear interpolation if cubic fails
				self.T_out = T_hist[:, -len(self.t_out):]
				if(self.cfg.use_RTE and self.cfg.RTE_solver == 'hapke'):
					self.phi_vis_out = phi_vis_hist[:, -len(self.t_out):]
					self.phi_therm_out = phi_therm_hist[:, -len(self.t_out):]
				self.T_surf_out = T_surf_hist[-len(self.t_out):]
				self.mu_out = self.mu_array[-len(self.t_out):]

		
		if(self.cfg.use_RTE and self.cfg.RTE_solver=='disort'):
			#Reinitialize disort to get observer radiance values at output times. 
			#This will trigger the loading of new optical constants files for multi-wave, which optionally can 
			# be at a higher spectral resolution. 
			self.rte_disort = DisortRTESolver(self.cfg, self.grid,output_radiance=True)
			if(self.cfg.multi_wave):
				nwave = len(self.rte_disort.wavenumbers)
				self.radiance_out = np.zeros((nwave,self.T_out.shape[1]))
			else:
				self.radiance_out = np.zeros(self.T_out.shape[1])
			self.rad_T_out = np.zeros(self.T_out.shape[1])  #Temperature computed from radiance blackbody fit. 
			wn_bounds = np.loadtxt(self.cfg.wn_bounds_out)
			print("Computing DISORT radiance spectra for output.")
			for idx in range(self.T_out.shape[1]):
				if non_diurnal:
					F = self.F_array[-1]
				else:
					t_hist = np.array(self.t_history)
					t = self.t_out[idx]
					F_idx = np.argmin(np.abs(t_hist - t))
					F = self.F_array[F_idx] #get nearest value for F. Don't want to interpolate and get a value that isn't 0 or 1. 
				rad = self.rte_disort.disort_run(self.T_out[:,idx],self.mu_out[idx],F)
				if(self.cfg.multi_wave):
					self.radiance_out[:,idx] = rad.numpy()
				else:
					self.radiance_out[idx] = rad.numpy()
				if(self.cfg.multi_wave):
					self.rad_T_out[idx], _, _, _ = fit_blackbody_wn_banded(self,wn_bounds, rad.numpy(),idx=idx)
				else:
					self.rad_T_out[idx] = fit_blackbody_broadband(self,rad.numpy(),idx=idx)
			if self.cfg.multi_wave:
				#Run disort again with spectral features removed to produce a smooth radiance spectrum for emissivity division. 
				self.disort_emissivity()


	def run(self):
		"""Execute the full time-stepping simulation."""
		start_time = time.time()
		
		# Steady-state convergence settings
		check_convergence = not self.cfg.diurnal
		n_check = getattr(self.cfg, 'steady_n_check', 200)  # How often to check
		window = getattr(self.cfg, 'steady_window', 100)     # How far back to look for temperature history polynomial fit. 
		tol = getattr(self.cfg, 'steady_tol', 0.2)         # Convergence threshold (K to extremum)
		converged = False

		for j in range(self.t_num):
			self.current_time = self.t[j]
			self.current_step = j
			self.mu = self.mu_array[j]
			self.F = self.F_array[j]

			if j > 0:
				# Compute radiative source term (if RTE enabled, otherwise remains zero)
				if self.cfg.use_RTE:
					if(self.cfg.RTE_solver == 'hapke'):
						self.source_term = self.rte_hapke.compute_source(self.T, self.mu, self.F)
					elif(self.cfg.RTE_solver == 'disort'):
						self.source_term = self.rte_disort.disort_run(self.T,self.mu,self.F)
					else:
						print("Error: Invalid RTE solver choice! Options are hapke or disort, or set use_RTE to False")
						return self.T_out, self.phi_vis_out, self.phi_therm_out, self.T_surf_out, self.t_out
				# Advance heat equation implicitly
				self._fd1d_heat_implicit_diag()
				self._bc()

			# Store current state for interpolation
			self.T_history.append(self.T.copy())
			self.T_surf_history.append(self.T_surf)
			self.t_history.append(self.current_time)
			if(self.cfg.use_RTE and self.cfg.RTE_solver=='hapke'):
				self.phi_vis_history.append(self.rte_hapke.phi_vis_prev.copy())
				self.phi_therm_history.append(self.rte_hapke.phi_therm_prev.copy())

			# Terminate here for fixed temperature run, saving outputs at initialization temperature. 
			if(self.cfg.T_fixed and not self.cfg.diurnal):
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
				

		# Interpolate results to desired output times
		self._make_outputs()

		elapsed = time.time() - start_time
		print(f"Simulation completed in {elapsed:.2f} s")

		return self.T_out, self.phi_vis_out, self.phi_therm_out, self.T_surf_out, self.t_out
	
	def disort_emissivity(self):
		#Runs disort with uniform_props flag, which averages out all spectral properties (extinction and scattering properties)
		# for the purpose of producing an equivalent but spectrally smooth radiance spectrum for the emissivity division. 
		rte_disort = DisortRTESolver(self.cfg, self.grid,output_radiance=True, uniform_props = True)
		if(self.cfg.multi_wave):
			nwave = len(rte_disort.wavenumbers)
			self.radiance_out_uniform = np.zeros((nwave,self.T_out.shape[1]))
		else:
			self.radiance_out_uniform = np.zeros(self.T_out.shape[1])
		#Need to get interpolated T_v_depth array, mu, and F as inputs. (F should be nearest value, not interp)
		print("Computing DISORT emissivity spectra for output.")
		for idx in range(self.T_out.shape[1]):
			if not self.cfg.diurnal:
				F = self.F_array[-1]
			else:
				t_hist = np.array(self.t_history)
				t = self.t_out[idx]
				F_idx = np.argmin(np.abs(t_hist - t))
				F = self.F_array[F_idx] #get nearest value for F. Don't want to interpolate and get a value that isn't 0 or 1. 
			rad = rte_disort.disort_run(self.T_out[:,idx],self.mu_out[idx],F)
			if(self.cfg.multi_wave):
				self.radiance_out_uniform[:,idx] = rad
			else:
				self.radiance_out_uniform[idx] = rad

def emissionT(T,tau_edges,T_interface,mu):
	T_calc = 0.0
	wt_calc = 0.0
	for i in np.arange(len(T)):
		T_calc += (T[i]**4.0)*(np.exp(-tau_edges[i]/mu) - np.exp(-tau_edges[i+1]))
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
	minbound = sim.T_out[:,idx].min()-5
	maxbound = sim.T_out[:,idx].max()+5
	res = scipy.optimize.minimize(loss, [T0], bounds=[(minbound, maxbound)],)
	T_fit = res.x[0]
	B_fit = planck_wn_integrated(wn_edges, T_fit)
	return T_fit, B_fit, radiance/B_fit, sim.rte_disort.wavenumbers[:idx]

def fit_blackbody_broadband(sim,radiance,idx=-1):
	"""
	Fit a blackbody spectrum to the given broadband radiance.
	radiance: array of band-integrated radiances (same length as sim.rte_disort.wavenumbers)
	Returns best-fit temperature and the fitted blackbody spectrum.
	"""
	def loss(T):
		B = sim.cfg.sigma*(T[0])**4.0
		return np.sum((radiance*np.pi - B)**2)
	T0 = sim.T_out[1,idx]
	minbound = sim.T_out[:, idx].min()-5
	maxbound = sim.T_out[:, idx].max()+5
	res = scipy.optimize.minimize(loss, [T0], bounds=[(minbound, maxbound)],)
	T_fit = res.x[0]
	return T_fit

def calculate_interface_T(T,i,alpha,beta):
    return((alpha*T[i] + beta*T[i+1])/(alpha + beta))

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
		else:
			plt.plot(sim.t_out / 3600, T_surf_out, label='Surface Temperature (no RTE)')
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
		wn_bounds = np.loadtxt(sim.cfg.wn_bounds_out)
		T_fit, B_fit, emiss_spec, wn_BB = fit_blackbody_wn_banded(sim,wn_bounds, final_rad)
		#multi_T_fit, abund, multi_B_fit, multi_emiss_spec, multi_wn_BB= fit_blackbody_mixture_wn_banded(sim,wn_bounds, final_rad)
		otes_samp = np.loadtxt(sim.cfg.substrate_spectrum_out)
		otesT1 = np.loadtxt(sim.cfg.otesT1_out)
		otesT2 = np.loadtxt(sim.cfg.otesT2_out)
		idx1 = np.argmin(np.abs(otesT1[:,0] - 900))
		idx2 = np.argmin(np.abs(otesT1[:,0] - 1200))
		otes_cf_emis = otesT1[idx1:idx2,1].max()
		bbfit_cf_emis = emiss_spec[idx1:idx2].max()
		ds_cf_emis = (final_rad/sim.radiance_out_uniform[:, -1])[idx1:idx2].max()
		samp_cf_emis = otes_samp[idx1:idx2,1].max()
		plt.figure()
		#plt.plot(wn_BB, emiss_spec + otes_cf_emis - bbfit_cf_emis, label=f'Emissivity (T_fit={T_fit:.1f} K)')
		#plt.plot(multi_wn_BB, multi_emiss_spec, label=f'Mixture Emissivity (T_fit={multi_T_fit})')
		plt.plot(wn[:ir_cutoff], (final_rad/sim.radiance_out_uniform[:, -1])[:ir_cutoff]+otes_cf_emis - ds_cf_emis, label='DISORT emissivity',linewidth=3)
		plt.plot(wn[:ir_cutoff], otes_samp[:ir_cutoff,1]+otes_cf_emis - samp_cf_emis, label='Substrate spectrum (Orgueil)',linewidth=1)
		plt.plot(wn[:ir_cutoff], otesT1[:ir_cutoff,1], label='OTES T1 (fewer fines)')
		plt.plot(wn[:ir_cutoff], otesT2[:ir_cutoff,1], label='OTES T2 (more fines)')
		plt.xlabel('Wavenumber [cm$^{-1}$]')
		plt.ylabel('Emissivity')
		plt.title('Final Emissivity Spectrum (Radiance / Best-fit Blackbody, band-integrated)')
		plt.legend()
		plt.grid(True)
		plt.gca().invert_xaxis()
		plt.show()