import os
import re
import sys
import numpy as np
import shutil
import subprocess
import matplotlib.pyplot as plt
from abacustest.lib_prepare.abacus import ReadInput, WriteInput
from abacustest.lib_collectdata.collectdata import RESULT

from pathlib import Path
from typing import Dict, Any, List, Literal

from abacusagent.init_mcp import mcp

from abacusagent.modules.util.comm import generate_work_path, link_abacusjob, run_abacus, has_chgfile

@mcp.tool()
def abacus_dos_run(
    abacus_inputs_path: Path,
    dos_edelta_ev: float = None,
    dos_sigma: float = None,
    dos_scale: float = None,
    dos_emin_ev: float = None,
    dos_emax_ev: float = None,
    dos_nche: int = None,
) -> Dict[str, Any]:
    """Run the DOS and PDOS calculation.
    
    This function will firstly run a SCF calculation with out_chg set to 1, 
    then run a NSCF calculation with init_chg set to 'file' and out_dos set to 1 or 2.
    If the INPUT parameter "basis" is "PW", then out_dos will be set to 1, and only DOS will be calculated.
    If the INPUT parameter "basis" is "LCAO", then out_dos will be set to 2, and both DOS and PDOS will be calculated.
    
    Args:
        abacus_inputs_path: Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        dos_edelta_ev: Step size in writing Density of States (DOS) in eV.
        dos_sigma: Width of the Gaussian factor when obtaining smeared Density of States (DOS) in eV. 
        dos_scale: Defines the energy range of DOS output as (emax-emin)*(1+dos_scale), centered at (emax+emin)/2. 
                   This parameter will be used when dos_emin_ev and dos_emax_ev are not set.
        dos_emin_ev: Minimal range for Density of States (DOS) in eV.
        dos_emax_ev: Maximal range for Density of States (DOS) in eV.
        dos_nche: The order of Chebyshev expansions when using Stochastic Density Functional Theory (SDFT) to calculate DOS.
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - results_scf: Results of the SCF calculation, including: work path, normal end status, SCF steps, convergence status, and energies.
            - results_nscf: Results of the NSCF calculation, including work path and normal end status.
            - fig_paths: List of paths to the generated figures for DOS and PDOS. DOS will be saved as "DOS.png" and PDOS will be saved as "species_atom_index_pdos.png" in the output directory.
    """
    metrics_scf = abacus_dos_run_scf(abacus_inputs_path)
    metrics_nscf = abacus_dos_run_nscf(metrics_scf["scf_work_path"],
                                       dos_edelta_ev=dos_edelta_ev,
                                       dos_sigma=dos_sigma,
                                       dos_scale=dos_scale, 
                                       dos_emin_ev=dos_emin_ev,
                                       dos_emax_ev=dos_emax_ev,
                                       dos_nche=dos_nche)
    
    fig_paths = plot_dos_pdos(metrics_scf["scf_work_path"], metrics_nscf["nscf_work_path"])
    fig_paths = [p for p in fig_paths]

    return_dict = {
        "dos_picture": fig_paths[0],
        'pdos_fig_paths': fig_paths[1:],
    }

    return_dict.update(metrics_scf)
    return_dict.update(metrics_nscf)

    return return_dict

def abacus_dos_run_scf(abacus_inputs_path: Path,
                       force_run: bool = False) -> Dict[str, Any]:
    """
    Run the SCF calculation to generate the charge density file.
    If the charge file already exists, it will skip the SCF calculation.
    
    Args:
        abacus_inputs_path: Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        force_run: If True, it will run the SCF calculation even if the charge file already exists.
    
    Returns:
        Dict[str, Any]: A dictionary containing the work path, normal end status, SCF steps, convergence status, and energies.
    """
    
    input_param = ReadInput(os.path.join(abacus_inputs_path, "INPUT"))
    # check if charge file has been generated
    if has_chgfile(abacus_inputs_path) and not force_run:
        print("Charge file already exists, skipping SCF calculation.")
        work_path = abacus_inputs_path
    else:
        work_path = generate_work_path()
        link_abacusjob(src=abacus_inputs_path,
                       dst=work_path,
                       copy_files=["INPUT"])

        input_param = ReadInput(os.path.join(work_path, "INPUT"))
        input_param["calculation"] = "scf"
        input_param["out_chg"] = 1
        WriteInput(input_param, os.path.join(work_path, "INPUT"))

        run_abacus(work_path)

    rs = RESULT(path=work_path, fmt="abacus")
    
    return {
        "scf_work_path": Path(work_path).absolute(),
        "scf_normal_end": rs["normal_end"],
        "scf_steps": rs["scf_steps"],
        "scf_converge": rs["converge"],
        "scf_energies": rs["energies"]
    }

def abacus_dos_run_nscf(abacus_inputs_path: Path,
                        dos_edelta_ev: float = None,
                        dos_sigma: float = None,
                        dos_scale: float = None,
                        dos_emin_ev: float = None,
                        dos_emax_ev: float = None,
                        dos_nche: int = None,) -> Dict[str, Any]:
    
    work_path = generate_work_path()
    link_abacusjob(src=abacus_inputs_path,
                   dst=work_path,
                   copy_files=["INPUT"])
    
    input_param = ReadInput(os.path.join(work_path, "INPUT"))
    input_param["calculation"] = "nscf"
    input_param["init_chg"] = "file"
    if input_param.get("basis_type", "pw") == "lcao":
        input_param["out_dos"] = 2 # only for LCAO basis, and will output DOS and PDOS
    else:
        input_param["out_dos"] = 1
    
    for dos_param, value in {
        "dos_edelta_ev": dos_edelta_ev,
        "dos_sigma": dos_sigma,
        "dos_scale": dos_scale,
        "dos_emin_ev": dos_emin_ev,
        "dos_emax_ev": dos_emax_ev,
        "dos_nche": dos_nche
    }.items():
        if value is not None:
            input_param[dos_param] = value
    
    
    WriteInput(input_param, os.path.join(work_path, "INPUT"))
    
    run_abacus(work_path)
    
    rs = RESULT(path=work_path, fmt="abacus")
    
    return {
        "nscf_work_path": Path(work_path).absolute(),
        "nscf_normal_end": rs["normal_end"]
    }

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

def plot_pdos(energy_values, orbitals, fermi_level, label_map, output_dir, dpi=300):
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
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plot_files.append(os.path.abspath(output_file))
        plt.close()
    
    return plot_files

def plot_dos(file_path, fermi_level, output_file, dpi=300):
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
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return os.path.abspath(output_file)

def plot_dos_pdos(nscf_job_path: Path, 
                  output_dir: Path,
                  dpi=300) -> List[str]:
    """Plot DOS and PDOS from the NSCF job path.
    
    Args:
        nscf_job_path (Path): Path to the NSCF job directory containing the OUT.* files.
        output_dir (Path): Directory where the output plots will be saved.
        dpi (int): Dots per inch for the saved plots.
    
    Returns:
        List[str]: List of paths to the generated plot files.
    
    """
    input_param = ReadInput(os.path.join(nscf_job_path, "INPUT"))
    input_dir = os.path.join(nscf_job_path, "OUT." + input_param.get("suffix","ABACUS"))

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
            raise FileNotFoundError(f"Required file not found: {file_path}")
    

    energy_values, orbitals = parse_pdos_file(input_file)
    fermi_level = parse_log_file(log_file)
    label_map = parse_basref_file(basref_file)
    
    # Plot DOS and get file path
    dos_plot_file = plot_dos(dos_file, fermi_level, dos_output, dpi)
    
    # Plot PDOS and get file paths
    pdos_plot_files = plot_pdos(energy_values, orbitals, fermi_level, label_map, output_dir, dpi)
    
    # Combine file paths into a single list
    all_plot_files = [dos_plot_file] + pdos_plot_files
    
    print("Plots generated:")
    for file in all_plot_files:
        print(f"- {file}")
        
    return all_plot_files

