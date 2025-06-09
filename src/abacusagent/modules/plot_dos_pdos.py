import sys
import re
import os
import numpy as np
import matplotlib.pyplot as plt

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
