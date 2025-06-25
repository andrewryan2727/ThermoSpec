import numpy as np
import os
import matplotlib.pyplot as plt

def bruggeman(host_nk, inclusion_nk, f, max_iter=100, tol=1e-6):
    # Convert to complex dielectric constants
    eps_host = (host_nk[0] + 1j * host_nk[1]) ** 2
    eps_incl = (inclusion_nk[0] + 1j * inclusion_nk[1]) ** 2
    # Initial guess for effective dielectric constant
    eps_eff = eps_host * (1 - f) + eps_incl * f
    for _ in range(max_iter):
        num = f * (eps_incl - eps_eff) / (eps_incl + 2 * eps_eff) + \
              (1 - f) * (eps_host - eps_eff) / (eps_host + 2 * eps_eff)
        eps_eff_new = eps_eff + num * eps_eff
        if np.abs(eps_eff_new - eps_eff) < tol:
            break
        eps_eff = eps_eff_new

    n_eff = np.sqrt(eps_eff)
    return n_eff.real, n_eff.imag

def maxwell_garnett(host_nk, inclusion_nk, f_incl):
    """Compute Maxwell Garnett effective medium."""
    eps_h = (host_nk[0] + 1j * host_nk[1]) ** 2
    eps_i = (inclusion_nk[0] + 1j * inclusion_nk[1]) ** 2
    eps_eff = eps_h * ((eps_i + 2 * eps_h) + 2 * f_incl * (eps_i - eps_h)) / ((eps_i + 2 * eps_h) - f_incl * (eps_i - eps_h))
    #factor_numer = 2 * (1 - f_incl) * eps_h + (1 + 2 * f_incl) * eps_i
    #factor_denom = (2 + f_incl) * eps_h + (1 - f_incl) * eps_i
    #eps_eff = eps_h * factor_numer / factor_denom
    n_eff = np.real(np.sqrt(eps_eff))
    k_eff = np.imag(np.sqrt(eps_eff))
    return n_eff, k_eff

def effective_medium_wavelength_dependent(wavenumbers, carbon_black_nk, f_incl,
                                          host_nk=None, host_components=None, host_fractions=None,model='maxwell_garnett'):
    n_eff_arr = []
    k_eff_arr = []
    for i in range(len(wavenumbers)):
        host_n, host_k = host_nk[0][i], host_nk[1][i]
        host = (host_n, host_k)
        inclusion = (carbon_black_nk[0][i], carbon_black_nk[1][i])
        if(model=='maxwell_garnett'):
            n_eff, k_eff = maxwell_garnett(host, inclusion, f_incl)
        else:
            n_eff, k_eff = bruggeman(host, inclusion, f_incl)
        n_eff_arr.append(n_eff)
        k_eff_arr.append(k_eff)
    return np.array(n_eff_arr), np.array(k_eff_arr)

def read_nk_file(n_file, k_file):
    n_data = np.loadtxt(n_file)
    k_data = np.loadtxt(k_file)
    wn_n, n = n_data[:,0], n_data[:,1]
    wn_k, k = k_data[:,0], k_data[:,1]
    assert np.allclose(wn_n, wn_k), "Wavenumber grids do not match!"
    return wn_n, n, k

def main():
    import argparse
    # parser = argparse.ArgumentParser(description="Compute effective medium optical constants.")
    # parser.add_argument('--host_n', required=True, help='Host n file')
    # parser.add_argument('--host_k', required=True, help='Host k file')
    # parser.add_argument('--incl_n', required=True, help='Inclusion n file')
    # parser.add_argument('--incl_k', required=True, help='Inclusion k file')
    # parser.add_argument('--f_incl', type=float, required=True, help='Volume fraction of inclusion')
    # parser.add_argument('--output_prefix', default='Effective', help='Prefix for output files')
    # args = parser.parse_args()
    host_n = 'Preprocessing/serpentine_n_216wns.txt'
    host_k = 'Preprocessing/serpentine_k_216wns.txt'
    incl_n = 'Preprocessing/graphite_n_216wns.txt'
    incl_k = 'Preprocessing/graphite_k_216wns.txt'
    f_incl = 0.3
    output_prefix = 'Preprocessing/Effective'
    mixing_model = 'bruggeman'

    # Read host and inclusion
    wn, host_n, host_k = read_nk_file(host_n, host_k)
    _, incl_n, incl_k = read_nk_file(incl_n, incl_k)
    wavelengths = 1e4 / wn  # microns


    n_eff, k_eff = effective_medium_wavelength_dependent(
        wn,
        carbon_black_nk=(incl_n, incl_k),
        f_incl=f_incl,
        host_nk=(host_n, host_k),model= mixing_model
    )


    # Compute effective medium
    # n_eff, k_eff = effective_medium_wavelength_dependent(
    #     wn,
    #     carbon_black_nk=(incl_n, incl_k),
    #     f_incl=f_incl,
    #     host_nk=(host_n, host_k)
    # )
    plt.plot(wavelengths,host_n)
    plt.plot(wavelengths,n_eff)
    plt.plot(wavelengths,host_k)
    plt.plot(wavelengths,k_eff)
    plt.plot(wavelengths,incl_n)
    plt.plot(wavelengths,incl_k)
    plt.show()
    
    # Output files
    np.savetxt(f"{output_prefix}_n.txt", np.column_stack([wn, n_eff]), fmt='%.6e', delimiter='\t')
    np.savetxt(f"{output_prefix}_k.txt", np.column_stack([wn, k_eff]), fmt='%.6e', delimiter='\t')
    np.savetxt(f"wn_wl.txt", np.column_stack([wn, wavelengths]), fmt='%.6e', delimiter='\t')
    print(f"Wrote {output_prefix}_n.txt, {output_prefix}_k.txt, wn_wl.txt")

if __name__ == "__main__":
    main()

""" python effective_medium.py \
  --host_n serpentine_n_216wns.txt \
  --host_k serpentine_k_216wns.txt \
  --incl_n graphite_n_216wns.txt \
  --incl_k graphite_k_216wns.txt \
  --f_incl 0.01 \
  --output_prefix Effective """