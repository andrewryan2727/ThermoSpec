import numpy as np
from pydisort import Disort, DisortOptions, scattering_moments
import torch


torch.set_default_dtype(torch.float64)

n_layers = 81
T = np.linspace(250,100, n_layers+1)      # [K], should have dimensions of nlayers+1
mu0=np.cos(np.radians(0.0))  # cosine solar zenith
ssalb = 0.10
nmom = 16
S = 1366.0  # Solar constant [W/m^2]
g = 0.5  # Asymmetry parameter for isotropic scattering
nwave = 1

sigma = 5.670374419e-8  # Stefan–Boltzmann constant

tau_max = 5.0

dtau = tau_max / n_layers  # Δτ
tau_layer = torch.full((n_layers,), dtau)  # Optical depth for each layer
ssa_layer = torch.full((n_layers,), ssalb)
moms = scattering_moments(nmom, "henyey-greenstein", g)

moments = torch.stack(
    [torch.full((n_layers,), moms[i]) for i in range(nmom)],
    dim=-1
)  # shape: (n_layers, nmom)


# Stack all optical properties into (nlyr, nprop = 6)
optical_props = torch.cat([
    tau_layer.unsqueeze(-1),     # (nlyr, 1)
    ssa_layer.unsqueeze(-1),     # (nlyr, 1)
    moments                      # (nlyr, 4)
], dim=-1)  # shape: (nlyr, 6)

# Expand to (nwave, ncol, nlyr, nprop)
prop = optical_props.unsqueeze(0).unsqueeze(0)  # (1, 1, nlyr, 6)


op = DisortOptions().header("Test")
op.flags(
    "lamber,quiet,planck,"
    "print-input"
)

op.wave_lower([20]) # Set lower wavenumber for planck function, in cm^-1
op.wave_upper([4000])  # Set wavenumber for planck function, in cm^-1
op.nwave(nwave)  # Number of wavelength ranges

op.ds().nlyr = n_layers
op.ds().nmom = nmom
op.ds().nstr = nmom  # Number of streams, matching number of moments for now. 
op.ds().nphase = nmom # Number of phase angles, matching number of moments. 


ds = Disort(op)

# set boundary conditions
bc = {
    "umu0": torch.tensor([mu0]),  # Cosine of solar zenith angle
    "phi0": torch.tensor([0.0]),    # Solar azimuth angle
    "albedo": torch.tensor([0.0]),  # Albedo of bottom surface, assumed lambertian based on lamber flag
    "btemp": torch.tensor([T[-1]]),  # Temperature at bottom boundary [K]" 
    "ttemp": torch.tensor([3.0]),  # Temperature at top boundary [K], temperature of space. 
    "temis": torch.tensor([1.0]),  # Emissivity of top boundary
    "fisot": torch.tensor([0.0]),
}

bc["fbeam"] = torch.tensor(S)


temf = torch.tensor(T).unsqueeze(0)  # Temperature profile at layer centers [K]

result = ds.forward(prop, '',temf, **bc)  # Forward pass with optical properties and temperature profile
#Returns upward and downward fluxes. 

F_net = result.squeeze(0).squeeze(0)[:,1] - result.squeeze(0).squeeze(0)[:,0]  # Net flux (down - up) at each layer
Q_rad = -np.diff(F_net)/ tau_layer[0]  # Radiative heating rate (W/m^2) between layers


Q_rad_exact = ds.gather_flx()[0,0,:,3]


