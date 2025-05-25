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
		P = self.cfg.P
		t_num = self.cfg.tsteps_day * self.cfg.ndays
		# Time vector (s)
		self.t = np.linspace(0, P * self.cfg.ndays, t_num)
		# Solar hour angle
		hour_angle = (np.pi / (P / 2)) * (self.t - (P / 2))
		# Cosine of solar zenith
		mu = np.cos(self.cfg.latitude) * np.cos(hour_angle)
		# Boolean array for sun-up; avoid near-zero mu
		sun_up = mu > 0.001
		F = sun_up.astype(float)
		# Handle non-diurnal or always-sun/no-sun cases
		if not self.cfg.diurnal:
			if self.cfg.sun:
				mu = np.ones_like(mu)
				F  = np.ones_like(F)
			else:
				mu = -np.ones_like(mu)
				F  = np.zeros_like(F)
		self.mu_array = mu
		self.F_array  = F
		self.t_num    = t_num

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
		# Initial temperature field (K)
		self.T = np.zeros(self.grid.x_num) + self.cfg.T_bottom
		# Radiative source vector
		self.source_term = np.zeros(self.grid.x_num)
		# Initialize surface temp for non-RTE models
		self.T_surf = self.cfg.T_bottom
		# Output storage
		self.T_out         = np.zeros((self.grid.x_num, self.t_num))
		self.phi_vis_out   = np.zeros((self.grid.nlay_dust, self.t_num))
		self.phi_therm_out = np.zeros((self.grid.nlay_dust, self.t_num))
		self.T_surf_out   = np.zeros(self.t_num)

	def run(self):
		"""Execute the full time-stepping simulation."""
		start_time = time.time()
		for j in range(self.t_num):
			self.mu = self.mu_array[j]
			self.F  = self.F_array[j]
			if j > 0:
				# Compute radiative source term (if RTE enabled)
				if self.cfg.use_RTE:
					#Calculate source term using RTE. Otherwise, this array remains zero.
					self.source_term = self.rte.compute_source(self.T, self.mu, self.F)
				# Advance heat equation implicitly
				self._fd1d_heat_implicit_diag()
				# Apply boundary conditions
				self._bc()
			# Store outputs
			self.T_out[:, j]         = self.T.copy()
			self.phi_vis_out[:, j]   = self.rte.phi_vis_prev.copy()
			self.phi_therm_out[:, j] = self.rte.phi_therm_prev.copy()
			self.T_surf_out[j] = self.T_surf
			# Progress update
			if j % 1000 == 0:
				print(f"Time step {j}/{self.t_num}")
		elapsed = time.time() - start_time
		print(f"Simulation completed in {elapsed:.2f} s")
		return self.T_out, self.phi_vis_out, self.phi_therm_out, self.T_surf_out

if __name__ == "__main__":
	sim = Simulator()
	T_out, phi_vis, phi_therm, T_surf_out = sim.run()
	import matplotlib.pyplot as plt

	# Plot temperature at the surface (first grid point) over time
	plt.figure(figsize=(10, 5))
	plt.plot(sim.t / 3600, T_out[1, :], label='Surface Temperature')
	plt.plot(sim.t / 3600, T_surf_out, label='Surface Temperature (no RTE)')
	plt.xlabel('Time (hours)')
	plt.ylabel('Temperature (K)')
	plt.title('Surface Temperature vs Time')
	plt.legend()
	plt.tight_layout()
	plt.show()

	day = sim.cfg.ndays - 1  # Last day for plotting
	tstepsday = sim.cfg.tsteps_day
	Et = sim.cfg.Et
	x = sim.grid.x
	plt.semilogx(x[1:]/Et,T_out[1:,int(tstepsday*0.0) + day*tstepsday],label="t=0")
	plt.semilogx(x[1:]/Et,T_out[1:,int(tstepsday*0.1) + day*tstepsday],label="t+0.1")
	plt.semilogx(x[1:]/Et,T_out[1:,int(tstepsday*0.2) + day*tstepsday],label="t+0.2")
	plt.semilogx(x[1:]/Et,T_out[1:,int(tstepsday*0.3) + day*tstepsday],label="t+0.3")
	plt.semilogx(x[1:]/Et,T_out[1:,int(tstepsday*0.4) + day*tstepsday],label="t+0.4")
	plt.semilogx(x[1:]/Et,T_out[1:,int(tstepsday*0.5) + day*tstepsday],label="t+0.5")
	plt.semilogx(x[1:]/Et,T_out[1:,int(tstepsday*0.6) + day*tstepsday],label="t+0.6")
	plt.semilogx(x[1:]/Et,T_out[1:,int(tstepsday*0.7) + day*tstepsday],label="t+0.7")
	plt.semilogx(x[1:]/Et,T_out[1:,int(tstepsday*0.8) + day*tstepsday],label="t+0.8")
	plt.semilogx(x[1:]/Et,T_out[1:,int(tstepsday*0.9) + day*tstepsday],label="t+0.9")
	plt.xlabel('Depth into medium [m]')
	plt.ylabel('Kinetic Temperature [K]')
	plt.legend(loc='upper right')
	plt.show()