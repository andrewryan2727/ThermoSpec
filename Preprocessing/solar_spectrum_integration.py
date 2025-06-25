import numpy as np
import matplotlib.pyplot as plt

def compute_bin_bounds_lists(wn_centers):
    wn_centers = np.array(wn_centers)
    n_bins = len(wn_centers)
    #
    edges = np.zeros(n_bins + 1)
    edges[1:-1] = 0.5 * (wn_centers[:-1] + wn_centers[1:])
    first_spacing = wn_centers[1] - wn_centers[0]
    last_spacing = wn_centers[-1] - wn_centers[-2]
    edges[0] = wn_centers[0]
    edges[-1] = wn_centers[-1] 
    #
    lower_bounds = edges[:-1].tolist()
    upper_bounds = edges[1:].tolist()
    #
    return lower_bounds, upper_bounds

import numpy as np

def integrate_solar_spectrum(solar_file, bin_bounds_file, bin_centers_file,output_file,rescale=None):
    # Load solar spectrum: columns [wavenumber, flux]
    solar = np.loadtxt(solar_file,usecols=(0,1))
    nm_solar = solar[:,0]
    flux_solar = solar[:,1]

    # Load bin edges
    bin_edges = np.loadtxt(bin_bounds_file) #wavenumbers, cm-1
    bin_edges = (1e7 / bin_edges)  # Convert cm^-1 to nm
    n_bins = len(bin_edges) - 1
    integrated_flux = np.zeros(n_bins)
    nm_centers = np.zeros(n_bins)

    for i in range(n_bins):
        upper = bin_edges[i]
        lower = bin_edges[i+1]
        mask = (nm_solar >= lower) & (nm_solar < upper)
        if np.any(mask):
            nm_bin = nm_solar[mask]
            flux_bin = flux_solar[mask]
            # Trapezoidal integration
            integrated_flux[i] = np.trapezoid(flux_bin, nm_bin)
            nm_centers[i] = 0.5 * (lower + upper)
        else:
            integrated_flux[i] = 0.0
            nm_centers[i] = 0.5 * (lower + upper)

    if(rescale):
       print("Integrated solar flux rescaled from " + str(np.sum(integrated_flux)) + " to " + str(rescale) + " W/m2")
       integrated_flux *= rescale/np.sum(integrated_flux) 

    #plt.scatter(nm_centers,integrated_flux)
    #plt.scatter(nm_solar,flux_solar*50, s=1, alpha=0.5, label='Original Solar Spectrum')
    #plt.show()
    wn_centers = np.loadtxt(bin_centers_file)[:,0]  # Load wavenumber centers, cm^-1
    # Save output: columns [wn_center, integrated_flux]
    np.savetxt(output_file, np.column_stack([wn_centers, integrated_flux]), fmt='%.6e', delimiter='\t')
    print(f"Saved integrated solar flux to {output_file}")

# Example usage:
if __name__ == "__main__":
    integrate_solar_spectrum(
        solar_file='./Preprocessing/solar_spectrum.txt',
        bin_bounds_file='./Preprocessing/wn_bounds_216.txt',
        bin_centers_file='./Preprocessing/serpentine_n_216wns.txt',
        output_file='./Preprocessing/solar_integrated_216.txt',
        rescale=1366.0
    )

