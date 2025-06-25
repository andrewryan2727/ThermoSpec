import numpy as np
from scipy.special import legendre
import matplotlib.pyplot as plt

def compute_phase_moments_from_F11(theta_deg_array, F11_array, nmom_desired):
    theta_rad = np.deg2rad(theta_deg_array)
    mu_array = np.cos(theta_rad)
    #
    sort_idx = np.argsort(mu_array)
    mu_sorted = mu_array[sort_idx]
    F11_sorted = F11_array[sort_idx]
    #
    dmu = np.diff(mu_sorted)
    F_mid = 0.5 * (F11_sorted[:-1] + F11_sorted[1:])
    integral_F11 = np.sum(F_mid * dmu)
    #
    P_mu = F11_sorted / integral_F11
    #
    beta_moments = []
    for l in range(nmom_desired):
        Pl_mu = legendre(l)(mu_sorted)
        integrand = P_mu * Pl_mu
        integrand_mid = 0.5 * (integrand[:-1] + integrand[1:])
        beta_l = 0.5 * np.sum(integrand_mid * dmu)
        beta_moments.append(beta_l)
    #
    return np.array(np.array(beta_moments)*2.0)

def parse_F11_block(lines, start_idx):
    theta_deg_array = []
    F11_array = []
    #
    for line in lines[start_idx+1:]:
        if line.strip() == '' or line.startswith('*') or line.startswith('<'):
            break  # End of block or start of new block
        tokens = line.strip().split()
        if len(tokens) < 2:
            continue
        theta = float(tokens[0])
        F11 = float(tokens[1])
        theta_deg_array.append(theta)
        F11_array.append(F11)
    #
    return np.array(theta_deg_array), np.array(F11_array)

def compute_beta_moments_from_mie_file(filename, nmom_desired):
    with open(filename, 'r') as f:
        lines = f.readlines()
    #
    # Find the start of the F11 block
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith('<') and 'F11' in line:
            start_idx = idx
            break
    #
    if start_idx is None:
        raise RuntimeError("Could not find F11 block in the file!")
    #
    # Parse the F11 block
    theta_deg_array, F11_array = parse_F11_block(lines, start_idx)
    #
    # Compute Beta moments
    beta_moments = compute_phase_moments_from_F11(theta_deg_array, F11_array, nmom_desired)
    #
    return beta_moments, theta_deg_array, F11_array



def plot_phase_function_and_legendre_fit(theta_deg_array, F11_array, beta_moments):
    # Prepare mu = cos(theta)
    theta_rad = np.deg2rad(theta_deg_array)
    mu_array = np.cos(theta_rad)
    #
    # Sort for plotting
    sort_idx = np.argsort(mu_array)
    mu_sorted = mu_array[sort_idx]
    F11_sorted = F11_array[sort_idx]
    theta_deg_array_sorted = theta_deg_array[sort_idx]
    #
    # Normalize F11 to get P(mu)
    #dmu = np.diff(mu_sorted)
    #F_mid = 0.5 * (F11_sorted[:-1] + F11_sorted[1:])
    #integral_F11 = np.sum(F_mid * dmu)
    #P_mu = F11_sorted / integral_F11
    P_mu = F11_sorted
    #
    # Reconstruct P(mu) from beta moments
    P_mu_legendre = np.zeros_like(mu_sorted)
    for l, beta_l in enumerate(beta_moments):
        P_mu_legendre += (2 * l + 1) * beta_l * legendre(l)(mu_sorted)
    #P_mu_legendre *= 0.5  # consistent normalization
    #
    # Plot
    plt.figure(figsize=(8, 6))
    plt.semilogy(theta_deg_array_sorted, P_mu, label='Original $P(\\mu)$ from F11', linewidth=2)
    plt.semilogy(theta_deg_array_sorted, P_mu_legendre, '--', label='Legendre expansion', linewidth=2)
    plt.xlabel('$\\mu = \\cos(\\Theta)$', fontsize=14)
    plt.ylabel('$P(\\mu)$', fontsize=14)
    plt.title('Phase Function and Legendre Expansion', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def generate_HG_phase_function(theta_deg_array, g):
    theta_rad = np.deg2rad(theta_deg_array)
    mu_array = np.cos(theta_rad)
    #
    # Henyey-Greenstein formula for P(mu)
    P_mu = (1 - g**2) / (1 + g**2 - 2 * g * mu_array)**1.5
    #
    # Normalize to integral = 1 over mu
    # (note: original HG is normalized to 2, but our compute_phase_moments_from_F11 expects normalization to 1)
    mu_sorted = np.sort(mu_array)
    dmu = np.diff(mu_sorted)
    P_sorted = P_mu[np.argsort(mu_array)]
    F_mid = 0.5 * (P_sorted[:-1] + P_sorted[1:])
    integral_P = np.sum(F_mid * dmu)
    P_mu /= integral_P
    #
    # Return F11(θ) → we just define F11(θ) ∝ P(μ)
    # For testing, we can pretend F11(θ) = P(μ)
    F11_array = P_mu.copy()
    #
    return F11_array

def compute_asymmetry_parameter(theta_deg_array, F11_array):
    # Convert to mu = cos(theta)
    theta_rad = np.deg2rad(theta_deg_array)
    mu_array = np.cos(theta_rad)
    #
    # Sort for proper integration
    sort_idx = np.argsort(mu_array)
    mu_sorted = mu_array[sort_idx]
    F11_sorted = F11_array[sort_idx]
    #
    # Normalize F11 → P(mu)
    dmu = np.diff(mu_sorted)
    F_mid = 0.5 * (F11_sorted[:-1] + F11_sorted[1:])
    integral_F11 = np.sum(F_mid * dmu)
    #
    P_mu = F11_sorted / integral_F11
    #
    # Now compute g = <cos(theta)> = ∫ mu * P(mu) dmu
    integrand = mu_sorted * P_mu
    integrand_mid = 0.5 * (integrand[:-1] + integrand[1:])
    g = np.sum(integrand_mid * dmu)
    #
    return g


#Misc tests below. 
filename = '/Users/ryan/Research/RT_models/RT_thermal_model/optical_constants/Quartz_5micron_30wns/pack_frac_0.35/output/PackedMatrix2380.95.txt'
filename2 = '/Users/ryan/Research/RT_models/RT_thermal_model/optical_constants/Quartz_5micron_30wns/spherFiles/spher2380.95.print'

nmom_desired = 30

beta_moments, theta_deg_array, F11_array = compute_beta_moments_from_mie_file(filename, nmom_desired)
beta_moments2, theta_deg_array2, F11_array2 = compute_beta_moments_from_mie_file(filename2, nmom_desired)

g = compute_asymmetry_parameter(theta_deg_array, F11_array)
g2 = compute_asymmetry_parameter(theta_deg_array2, F11_array2)


plot_phase_function_and_legendre_fit(theta_deg_array, F11_array, beta_moments)
plot_phase_function_and_legendre_fit(theta_deg_array2, F11_array2, beta_moments2)


#Code validation with Henyey-Greenstein function
#In comparing to pydisort, normalization adjusted. 
g_test = 0.4
nmom_desired = 20
theta_deg_array = np.linspace(0, 180, 181)  # 1 degree steps

# Generate synthetic HG F11
F11_array = generate_HG_phase_function(theta_deg_array, g_test)

g_verify = compute_asymmetry_parameter(theta_deg_array, F11_array)

# Compute phase moments from F11
beta_moments = compute_phase_moments_from_F11(theta_deg_array, F11_array, nmom_desired)

# Compute analytic HG moments from PyDISORT
#beta_moments_expected = scattering_moments(nmom_desired, "henyey-greenstein", g_test).numpy()

