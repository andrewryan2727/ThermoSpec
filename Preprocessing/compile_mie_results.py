import glob
import re
import numpy as np
import argparse
import os

def parse_mie_print_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    lam = None
    cos_theta = None
    cext = None
    csca = None
    albedo = None

    for line in lines:
        # LAM line
        if 'LAM' in line and 'MRR' in line:
            match = re.search(r'LAM\s*=\s*([0-9.D+-]+)', line)
            if match:
                lam = float(match.group(1).replace('D', 'E'))

        # <COS> line
        elif '<COS>' in line:
            match = re.search(r'<COS>\s*=\s*([0-9.D+-]+)', line)
            if match:
                cos_theta = float(match.group(1).replace('D', 'E'))

        # CEXT / CSCA / ALBEDO line
        elif 'CEXT' in line and 'CSCA' in line and 'ALBEDO' in line:
            match_cext = re.search(r'CEXT\s*=\s*([0-9.D+-]+)', line)
            match_csca = re.search(r'CSCA\s*=\s*([0-9.D+-]+)', line)
            match_albedo = re.search(r'ALBEDO\s*=\s*([0-9.D+-]+)', line)
            if match_cext and match_csca and match_albedo:
                cext = float(match_cext.group(1).replace('D', 'E'))
                csca = float(match_csca.group(1).replace('D', 'E'))
                albedo = float(match_albedo.group(1).replace('D', 'E'))

    # Only return if all values were found
    if None not in (lam, cos_theta, cext, csca, albedo):
        return [lam, cos_theta, cext, csca, albedo]
    else:
        print(f"Warning: Incomplete data in file {filepath}, skipping.")
        return None


def parse_all_print_files(input_folder, output_filename):
    files = glob.glob(os.path.join(input_folder, '*.print'))

    if not files:
        print("No .print files found in the folder!")
        return

    all_data = []

    for filepath in files:
        result = parse_mie_print_file(filepath)
        if result:
            all_data.append(result)

    if not all_data:
        print("No valid data extracted!")
        return

    # Convert to array and sort by LAM (column 0)
    data_array = np.array(all_data)
    data_array = data_array[data_array[:,0].argsort()]

    # Write to output file
    np.savetxt(output_filename, data_array, fmt="%.6f", delimiter='\t')

    print(f"Saved parsed data to {output_filename}.")

def main():
    parser = argparse.ArgumentParser(description='Parse Mie .print files and extract LAM, <COS>, CEXT, CSCA, ALBEDO.')
    parser.add_argument('input_folder', type=str, help='Input folder containing .print files')
    parser.add_argument('output_filename', type=str, help='Output text filename')

    args = parser.parse_args()

    parse_all_print_files(args.input_folder, args.output_filename)

if __name__ == '__main__':
    main()
