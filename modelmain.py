import time
import numpy as np
from config import SimulationConfig
from grid import LayerGrid
from rte import RadiativeTransfer
from scipy.linalg import solve_banded
from numba import jit, njit


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
		self.rte  = RadiativeTransfer(self.cfg, self.grid)
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

	def _interpolate_outputs(self):
		"""Interpolate integration results to desired output times."""
		from scipy.interpolate import interp1d
		
		# Convert history lists to arrays for interpolation
		t_hist = np.array(self.t_history)
		T_hist = np.stack(self.T_history, axis=1)
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
			phi_vis_interp = interp1d(t_hist, phi_vis_hist, axis=1, kind='cubic',
							 bounds_error=True, assume_sorted=True)
			phi_therm_interp = interp1d(t_hist, phi_therm_hist, axis=1, kind='cubic',
								   bounds_error=True, assume_sorted=True)
			T_surf_interp = interp1d(t_hist, T_surf_hist, kind='cubic',
								bounds_error=True, assume_sorted=True)
			
			# Interpolate to clipped output times
			self.T_out = T_interp(t_out_clipped)
			self.phi_vis_out = phi_vis_interp(t_out_clipped)
			self.phi_therm_out = phi_therm_interp(t_out_clipped)
			self.T_surf_out = T_surf_interp(t_out_clipped)
		except ValueError as e:
			print(f"Warning in interpolation: {e}")
			# Fall back to linear interpolation if cubic fails
			self.T_out = T_hist[:, -len(self.t_out):]
			self.phi_vis_out = phi_vis_hist[:, -len(self.t_out):]
			self.phi_therm_out = phi_therm_hist[:, -len(self.t_out):]
			self.T_surf_out = T_surf_hist[-len(self.t_out):]

	def run(self):
		"""Execute the full time-stepping simulation."""
		start_time = time.time()
		
		for j in range(self.t_num):
			self.current_time = self.t[j]
			self.current_step = j
			self.mu = self.mu_array[j]
			self.F = self.F_array[j]
			
			if j > 0:
				# Compute radiative source term (if RTE enabled, otherwise remains zero)
				if self.cfg.use_RTE:
					self.source_term = self.rte.compute_source(self.T, self.mu, self.F)
				# Advance heat equation implicitly
				self._fd1d_heat_implicit_diag()
				# Apply boundary conditions
				self._bc()
			
			# Store current state for interpolation
			self.T_history.append(self.T.copy())
			self.phi_vis_history.append(self.rte.phi_vis_prev.copy())
			self.phi_therm_history.append(self.rte.phi_therm_prev.copy())
			self.T_surf_history.append(self.T_surf)
			self.t_history.append(self.current_time)
			
			# Optional progress updates
			if j % max(100, self.t_num//20) == 0:
				print(f"Time step {j}/{self.t_num}")
		
		# Interpolate results to desired output times
		self._interpolate_outputs()
		
		elapsed = time.time() - start_time
		print(f"Simulation completed in {elapsed:.2f} s")
		
		return self.T_out, self.phi_vis_out, self.phi_therm_out, self.T_surf_out, self.t_out

if __name__ == "__main__":
	sim = Simulator()
	T_out, phi_vis, phi_therm, T_surf_out, t_out = sim.run()
	import matplotlib.pyplot as plt

	# Plot temperature at the surface (first grid point) over time
	plt.figure(figsize=(10, 5))
	if(sim.cfg.use_RTE):
		#plt.plot(sim.t_out / 3600, (2.0*phi_therm[0,:]*np.pi/5.670e-8)**0.25, label='Surface Temperature')
		plt.plot(sim.t_out / 3600, T_out[0,:], label='Surface Temperature')
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