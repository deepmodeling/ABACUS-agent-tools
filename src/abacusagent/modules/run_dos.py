import os
import re
import sys
import numpy as np
import shutil
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
from abacustest.lib_prepare.abacus import ReadInput, WriteInput
from abacusagent.init_mcp import mcp

@mcp.tool()
def run_dos(abacus_path: Path,
    test_mode: bool = False,
) -> str:
    """
    Run the DOS and PDOS calculation in the path called 'abacus_path'.
    The function assumes that the calculation files for normal SCF calculations are ready in the directory
    called 'abacus_path'. It generates a new input file for the DOS and PDOS calculation, runs the normal SCF
    calculation, and then runs the DOS and PDOS calculation using the ABACUS code.
    This function returns the string of the path for the DOS and PDOS plots if the calculation is successful,
    otherwise it returns an error message.
    
    Args:
        abacus_path: the string for the path of the directary to run ABACUS DOS calcualtions
        test_mode: the boolen-type function to determine whether this is run in a unit test or not,
                   in the unit-test mode, skip the abacus calculations. Default: false.
    Returns:
        str: The path to the DOS and PDOS plots if successful, otherwise an error code.
    Raises:
        FileNotFoundError: If the required input files are not found in the specified path.
        ValueError: The first SCF calculation does not end normally.
    """

    # Change to the abacus_path directory
    os.chdir(abacus_path)

    # Hard-coded file names
    ab_input = 'INPUT'
    ab_stru = 'STRU'

    # Check if these files exists
    inpfiles = [ ab_input, ab_stru ]
    if not all(os.path.isfile(file) 
               for file in inpfiles):
        raise FileNotFoundError(
            f"Required input files in {inpfiles} not found in the directory {abacus_path}."
        )

    # Save input info to variables
    input_param = ReadInput(ab_input)
    input_param_scf = ReadInput(ab_input)
    input_param_dos = ReadInput(ab_input)
    out_dir = "OUT." + input_param[ "suffix" ]

    # Generate the input params for the SCF calculation
    dos_keys = [ "calculation", "out_chg" ]
    dos_vals = [ "scf"        , 1         ]
    for key, value in zip(dos_keys, dos_vals):
        input_param_scf[key] = value
    # Generate the input params for the DOS and PDOS calculation
    dos_keys = [ "calculation", "read_file_dir", "init_chg", "out_dos", "dos_sigma" ]
    dos_vals = [ "nscf"       , f"./{out_dir}" , "file"    , 2        , 0.07        ]
    for key, value in zip(dos_keys, dos_vals):
        input_param_dos[key] = value

    #run abacus SCF calculation
    WriteInput(input_param_scf, ab_input)
    if (not test_mode):
        subprocess.run(["mpirun", "-np", "2", "abacus"], stdout=open("scf.log", "w"), stderr=subprocess.STDOUT)
    line = pygrep("charge density convergence is achieved", f"{out_dir}/running_scf.log")
    if not line:
        # restore INPUT file for debugging purpose
        WriteInput(input_param, ab_input)
        raise ValueError("SCF calculation did not end normally.")

    # run abacus DOS and PDOS calculation again with the same command
    WriteInput(input_param_dos, ab_input)
    if (not test_mode):
        subprocess.run(["mpirun", "-np", "2", "abacus"], stdout=open("dos.log", "w"), stderr=subprocess.STDOUT)

    # restore 'original INPUT'
    WriteInput(input_param, ab_input)

    # Generate the DOS and PDOS plots
    plot_paths = plot_dos_pdos(out_dir, '.')

    return plot_paths


def pygrep(pattern: str, filename: Path) -> str:
    """
    Check if a pattern exists in a file.
    
    Parameters:
    pattern (str): The pattern to search for.
    filename (str): The name of the file to search in.
    Returns:
    str: The pattern if found, otherwise an empty string.
    """
    with open(filename, 'r') as file:
        for line in file:
            if pattern in line:
                return line
    return ""


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python run_dos.py <abacus_path>")
        sys.exit(1)
    
    abacus_path = sys.argv[1]
    try:
        plot_paths = run_dos(abacus_path)
        if abacus_path == '.':
            print("DOS and PDOS calculation completed successfully in the current directory")
        else:
            print(f"DOS and PDOS calculation completed successfully in {abacus_path}.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def parse_pdos_file(file_path):
    """Parse the PDOS file and extract energy values and orbital data."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    energy_match = re.search(r'<energy_values\s+units="eV">(.*?)</energy_values>', content, re.DOTALL)
    if not energy_match:
        raise ValueError("Energy values not found in the file.")
    
    energy_text = energy_match.group(1)
    energy_values = np.array([float(line.strip()) for line in energy_text.strip().split()])
    
    orbital_pattern = re.compile(r'<orbital\s+index="\s*(\d+)"\s+atom_index="\s*(\d+)"\s+species="(\w+)"\s+l="\s*(\d+)"\s+m="\s*(\d+)"\s+z="\s*(\d+)"\s*>(.*?)</orbital>', re.DOTALL)
    orbitals = []
    
    for match in orbital_pattern.finditer(content):
        index, atom_index, species, l, m, z, orbital_content = match.groups()
        
        data_match = re.search(r'<data>(.*?)</data>', orbital_content, re.DOTALL)
        if data_match:
            data_text = data_match.group(1)
            data_values = np.array([float(line.strip()) for line in data_text.strip().split()])
            
            orbitals.append({
                'index': int(index),
                'atom_index': int(atom_index),
                'species': species,
                'l': int(l),
                'm': int(m),
                'z': int(z),
                'data': data_values
            })
    
    return energy_values, orbitals

def parse_log_file(file_path):
    """Parse Fermi energy from log file and convert to eV."""
    ry_to_ev = 13.605698066
    fermi_energy = None
    
    with open(file_path, 'r') as f:
        for line in f:
            if "Fermi energy is" in line:
                match = re.search(r'Fermi energy is\s*([\d.-]+)', line)
                if match:
                    fermi_energy = float(match.group(1))
    
    if fermi_energy is None:
        raise ValueError("Fermi energy not found in log file")
    
    return fermi_energy * ry_to_ev

def parse_basref_file(file_path):
    """Parse basref file to create mapping for custom labels."""
    label_map = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if len(parts) >= 6:
                # Add 1 to atom_index as per requirement
                atom_index = int(parts[0]) + 1
                species = parts[1]
                l = int(parts[2])
                m = int(parts[3])
                z = int(parts[4])
                symbol = parts[5]
                
                key = (atom_index, species, l, m, z)
                label_map[key] = f'{species}{atom_index}({symbol})'
    
    return label_map

def plot_pdos(energy_values, orbitals, fermi_level, label_map, output_dir):
    """Plot PDOS data separated by atom/species with custom labels."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Shift energy values by Fermi level
    shifted_energy = energy_values - fermi_level
    
    # Group orbitals by atom_index and species
    atom_species_groups = {}
    for orbital in orbitals:
        key = (orbital['atom_index'], orbital['species'])
        if key not in atom_species_groups:
            atom_species_groups[key] = []
        atom_species_groups[key].append(orbital)
    
    plot_files = []
    
    # Generate plots for each atom/species group
    for (atom_index, species), group_orbitals in atom_species_groups.items():
        # Get the symbol from the first orbital's key in label_map
        first_orbital = group_orbitals[0]
        key = (atom_index, species, first_orbital['l'], first_orbital['m'], first_orbital['z'])
        base_label = label_map.get(key, f"{species}{atom_index}")
        
        # Group orbitals by l and m quantum numbers
        lm_groups = {}
        for orbital in group_orbitals:
            lm_key = (orbital['l'], orbital['m'])
            if lm_key not in lm_groups:
                lm_groups[lm_key] = []
            lm_groups[lm_key].append(orbital)
        
        # Create a figure with subplots for each l,m group
        n_subplots = len(lm_groups)
        fig, axes = plt.subplots(n_subplots, 1, figsize=(10, 4 * n_subplots), sharex=True)
        
        if n_subplots == 1:
            axes = [axes]  # Ensure axes is always a list
        
        # Determine global y limits for consistent scaling
        all_data = []
        for lm_key, orbitals_list in lm_groups.items():
            l, m = lm_key
            mask = (shifted_energy >= -fermi_level) & (shifted_energy <= fermi_level)
            for orbital in orbitals_list:
                all_data.extend(orbital['data'][mask])
        
        if not all_data:
            y_min, y_max = 0, 1
        else:
            y_min = -0.1 * max(all_data)
            y_max = 1.1 * max(all_data)
        
        # Plot each l,m group in a subplot
        for i, ((l, m), orbitals_list) in enumerate(lm_groups.items()):
            ax = axes[i]
            
            for orbital in orbitals_list:
                z = orbital['z']
                ax.plot(shifted_energy, orbital['data'], label=f'z={z}')
            
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel('PDOS')
            ax.set_ylim(y_min, y_max)
            ax.legend(loc='best')
            
            # Get symbol from label_map
            key = (atom_index, species, l, m, orbitals_list[0]['z'])
            symbol = label_map.get(key, '').split('(')[-1].split(')')[0]
            ax.set_title(f'Projected Density of States for {species}{atom_index}({symbol})')
        
        axes[-1].set_xlabel('Energy (eV)')
        axes[-1].set_xlim(-fermi_level, fermi_level)
        
        plt.tight_layout()
        
        # Save plot with proper naming
        output_file = os.path.join(output_dir, f"{species}{atom_index}_pdos.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plot_files.append(os.path.abspath(output_file))
        plt.close()
    
    return plot_files

def plot_dos(file_path, fermi_level, output_file):
    """Plot total DOS from DOS1_smearing.dat file."""
    # Read first two columns from file
    data = np.loadtxt(file_path, usecols=(0, 1))
    energy = data[:, 0] - fermi_level  # Shift by Fermi level
    dos = data[:, 1]
    
    # Determine y limits based on data within x range
    x_min, x_max = -fermi_level, fermi_level
    mask = (energy >= x_min) & (energy <= x_max)
    
    if not any(mask):
        y_min, y_max = 0, 1
    else:
        y_min = -0.1 * np.max(dos[mask])
        y_max = 1.1 * np.max(dos[mask])
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(energy, dos)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Fermi Level')
    plt.xlabel('Energy (eV)')
    plt.ylabel('DOS')
    plt.title('Density of States')
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    
    # Save plot
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return os.path.abspath(output_file)

def plot_dos_pdos(input_dir, output_dir):
    
    # Construct file paths based on input directory
    input_file = os.path.join(input_dir, "PDOS")
    log_file = os.path.join(input_dir, "running_nscf.log")
    basref_file = os.path.join(input_dir, "Orbital")
    dos_file = os.path.join(input_dir, "DOS1_smearing.dat")
    dos_output = os.path.join(output_dir, "DOS.png")
    
    # Validate input files exist
    for file_path in [input_file, log_file, basref_file, dos_file]:
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            sys.exit(1)
    
    try:
        energy_values, orbitals = parse_pdos_file(input_file)
        fermi_level = parse_log_file(log_file)
        label_map = parse_basref_file(basref_file)
        
        # Plot DOS and get file path
        dos_plot_file = plot_dos(dos_file, fermi_level, dos_output)
        
        # Plot PDOS and get file paths
        pdos_plot_files = plot_pdos(energy_values, orbitals, fermi_level, label_map, output_dir)
        
        # Combine file paths into a single list
        all_plot_files = [dos_plot_file] + pdos_plot_files
        
        print("Plots generated:")
        for file in all_plot_files:
            print(f"- {file}")

        return all_plot_files

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
