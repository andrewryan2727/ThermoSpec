import os
import numpy as np
import argparse

def convert_optical_constants(
    input_csv,
    output_txt,
    output_binbounds_txt,
    wl_min_vis,
    wl_max_vis,      # microns
    wl_step_vis,     # microns
    wn_min_ir,       # cm^-1
    wn_step_ir       # cm^-1
):
    # Load CSV
    data = np.loadtxt(input_csv)

    # Extract columns
    wn_input = data[:,0]  # wavenumbers in cm-1
    n_values = data[:,1]


    # Sort by increasing wavenumber
    sort_idx = np.argsort(wn_input)
    wn_input_sorted = wn_input[sort_idx]
    n_sorted = n_values[sort_idx]

    # Build visible λ grid
    wl_vis = np.arange(wl_min_vis, wl_max_vis, wl_step_vis)
    wn_vis_target = 1e4 / wl_vis

    # Build IR wavenumber grid
    wn_max_ir = 1e4 / wl_max_vis
    wn_ir_target = np.arange(wn_min_ir, wn_max_ir, wn_step_ir)

    # Combine target grid
    wn_target = np.concatenate([wn_ir_target,np.sort(wn_vis_target)])
    n_target = np.zeros_like(wn_target)

    # Compute bin boundaries
    bin_edges = np.zeros(len(wn_target) + 1)
    bin_edges[1:-1] = 0.5 * (wn_target[:-1] + wn_target[1:])
    bin_edges[0] = wn_target[0]
    bin_edges[-1] = wn_target[-1]

    # Bin average
    for i in range(len(wn_target)):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        mask = (wn_input_sorted >= lower) & (wn_input_sorted < upper)
        n_in_bin = n_sorted[mask]

        if len(n_in_bin) > 0:
            n_avg = np.mean(n_in_bin)
        else:
            idx_nearest = np.argmin(np.abs(wn_input_sorted - wn_target[i]))
            n_avg = n_sorted[idx_nearest]

        n_target[i] = n_avg

    # Save output: tab-delimited, no header
    output_data = np.column_stack([wn_target, n_target])
    np.savetxt(output_txt, output_data, fmt='%.6e', delimiter='\t')
    np.savetxt(output_binbounds_txt, bin_edges, fmt='%.6e', delimiter='\t')
    print(f"Saved output to {output_txt} and bin bounds to {output_binbounds_txt}")

def main():
    files = [
        '/Users/ryan/Research/RT_models/RT_thermal_model/Preprocessing/serpentine_n.txt',
        '/Users/ryan/Research/RT_models/RT_thermal_model/Preprocessing/serpentine_k.txt',
        '/Users/ryan/Research/RT_models/RT_thermal_model/Preprocessing/graphite_n.txt',
        '/Users/ryan/Research/RT_models/RT_thermal_model/Preprocessing/graphite_k.txt',
        '/Users/ryan/Research/RT_models/RT_thermal_model/Preprocessing/Bennu_Type1.txt',
        '/Users/ryan/Research/RT_models/RT_thermal_model/Preprocessing/Bennu_Type2.txt',
    ]
    wl_min_vis = 0.2
    wl_max_vis = 5.0
    wl_step_vis = 0.050 #0.050, 0.400
    wn_min_ir = 80
    wn_step_ir = 16 #16, 100
    for fname in files:
        base = os.path.splitext(os.path.basename(fname))[0]
        path = os.path.dirname(fname)
        # Temporarily run to get wn_target
        data = np.loadtxt(fname)
        wl_um = data[:,0]
        wn_input = 1e4 / wl_um
        wl_vis = np.arange(wl_min_vis, wl_max_vis, wl_step_vis)
        wn_vis_target = 1e4 / wl_vis
        wn_max_ir = 1e4 / wl_max_vis
        wn_ir_target = np.arange(wn_min_ir, wn_max_ir, wn_step_ir)
        wn_target = np.concatenate([wn_ir_target,np.sort(wn_vis_target)])
        n_wns = len(wn_target)
        output_txt = f"{path}/{base}_{n_wns}wns.txt"
        output_binbounds_txt = f"{path}/wn_bounds_{n_wns}.txt"
        convert_optical_constants(
            fname,
            output_txt,
            output_binbounds_txt,
            wl_min_vis,
            wl_max_vis,
            wl_step_vis,
            wn_min_ir,
            wn_step_ir
        )

if __name__ == "__main__":
    main()