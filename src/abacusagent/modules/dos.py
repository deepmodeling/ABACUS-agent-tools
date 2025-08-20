import os
import re
import sys
import numpy as np
import shutil
import matplotlib.pyplot as plt
from abacustest.lib_prepare.abacus import ReadInput, WriteInput
from abacustest.lib_collectdata.collectdata import RESULT
from abacustest.lib_model.comm import check_abacus_inputs

from pathlib import Path
from typing import Dict, Any, List, Literal

from abacusagent.init_mcp import mcp
from abacusagent.modules.util.comm import generate_work_path, link_abacusjob, run_abacus, has_chgfile, collect_metrics



angular_momentum_map = ['s', 'p', 'd', 'f', 'g']
color_map = {
    's': '#FF5733',
    'p': '#33FF57',
    'd': '#3357FF',
    'f': '#F033FF',
    'g': '#33FFF0' 
}

orbital_rep_map = {
    's': 's',
    'px': r'$p_x$',
    'py': r'$p_y$',
    'pz': r'$p_z$',
    'dz^2': r'$d_{z^2}$',
    'dxz': r'$d_{xz}$',
    'dyz': r'$d_{yz}$',
    'dxy': r'$d_{xy}$',
    'dx^2-y^2': r'$d_{x^2-y^2}$',
    'fz^3': r'$f_{z^3}$',
    'fxz^2': r'$f_{xz^2}$',
    'fyx^2': r'$f_{yx^2}$',
    'fzx^2-zy^2': r'$f_{zx^2-zy^2}$',
    'fxyz': r'$f_{xyz}$',
    'fx^3-3*xy^2': r'$f_{x^3-3xy^2}$',
    'f3yx^2-y^3': r'$f_{3yx^2-y^3}$'
}

@mcp.tool()
def abacus_dos_run(
    abacus_inputs_dir: Path,
    pdos_mode: Literal['species', 'species+shell', 'species+orbital'] = 'species+shell',
    dos_edelta_ev: float = 0.01,
    dos_sigma: float = 0.07,
    dos_scale: float = 0.01,
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
        abacus_inputs_dir: Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        pdos_mode: Mode of plotted PDOS file.
            - "species": Total PDOS of any species will be plotted in a picture.
            - "species+shell": PDOS for any shell (s, p, d, f, g,...) of any species will be plotted. PDOS of a shell of a species willbe plotted in a subplot.
            - “species+orbital": Orbital-resolved PDOS will be plotted. PDOS of orbitals in the same shell of a species will be plotted in a subplot.
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
    try:
        is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
        if not is_valid:
            raise RuntimeError(f"Invalid ABACUS input files: {msg}")
        
        input_file = os.path.join(abacus_inputs_dir, "INPUT")
        input_params = ReadInput(input_file)
        nspin = input_params.get("nspin", 1)
        if nspin in [4]:
            raise ValueError("Currently DOS calculation can only be plotted using for nspin=1 and nspin=2")

        metrics_scf = abacus_dos_run_scf(abacus_inputs_dir)
        metrics_nscf = abacus_dos_run_nscf(metrics_scf["scf_work_path"],
                                           dos_edelta_ev=dos_edelta_ev,
                                           dos_sigma=dos_sigma,
                                           dos_scale=dos_scale, 
                                           dos_emin_ev=dos_emin_ev,
                                           dos_emax_ev=dos_emax_ev,
                                           dos_nche=dos_nche)

        fig_paths = plot_dos_pdos(metrics_scf["scf_work_path"],
                                  metrics_nscf["nscf_work_path"],
                                  metrics_nscf["nscf_work_path"],
                                  nspin,
                                  pdos_mode)

        return_dict = {
            "dos_fig_path": fig_paths[0],
            'pdos_fig_path': fig_paths[1],
        }

        return_dict.update(metrics_scf)
        return_dict.update(metrics_nscf)

        return return_dict
    except Exception as e:
        return {"message": f"Calculating DOS and PDOS failed: {e}"}

def abacus_dos_run_scf(abacus_inputs_dir: Path,
                       force_run: bool = False) -> Dict[str, Any]:
    """
    Run the SCF calculation to generate the charge density file.
    If the charge file already exists, it will skip the SCF calculation.
    
    Args:
        abacus_inputs_dir: Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        force_run: If True, it will run the SCF calculation even if the charge file already exists.
    
    Returns:
        Dict[str, Any]: A dictionary containing the work path, normal end status, SCF steps, convergence status, and energies.
    """
    
    input_param = ReadInput(os.path.join(abacus_inputs_dir, "INPUT"))
    # check if charge file has been generated
    if has_chgfile(abacus_inputs_dir) and not force_run:
        print("Charge file already exists, skipping SCF calculation.")
        work_path = abacus_inputs_dir
    else:
        work_path = generate_work_path()
        link_abacusjob(src=abacus_inputs_dir,
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

def abacus_dos_run_nscf(abacus_inputs_dir: Path,
                        dos_edelta_ev: float = None,
                        dos_sigma: float = None,
                        dos_scale: float = None,
                        dos_emin_ev: float = None,
                        dos_emax_ev: float = None,
                        dos_nche: int = None,) -> Dict[str, Any]:
    
    work_path = generate_work_path()
    link_abacusjob(src=abacus_inputs_dir,
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

def plot_pdos(energy_values, orbitals, fermi_level, label_map, output_dir, nspin, mode, dpi=300):
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
    
    if mode == "species":
        pdos_pic_file = plot_pdos_species(shifted_energy, orbitals, output_dir, nspin, dpi)
    elif mode == "species+shell":
        pdos_pic_file = plot_pdos_species_shell(shifted_energy, orbitals, output_dir, nspin, dpi)
    elif mode == "species+orbital":
        pdos_pic_file = plot_pdos_species_orbital(shifted_energy, orbitals, output_dir, nspin, label_map, dpi)
    else:
        raise ValueError(f"Not allowed mode {mode}")
    
    return pdos_pic_file

def plot_pdos_species(shifted_energy, orbitals, output_dir, nspin, dpi):
    species = {}
    for orbital in orbitals:
        species_one = orbital['species']
        if species_one not in species.keys():
            species[species_one] = orbital['data']
        else:
            species[species_one] += orbital['data']
    
    num_species = len(species)
    plt.plot(figsize=(10, 6))
    for species_name, pdos_data in species.items():
        if nspin == 1:
            plt.plot(shifted_energy, pdos_data, label=species_name, linewidth=1.0)
        elif nspin == 2:
            plt.plot(shifted_energy, pdos_data[::2], label=f'{species_name} ' + r'$\uparrow$', linestyle='-', linewidth=1.0)
            plt.plot(shifted_energy, -pdos_data[1::2], label=f'{species_name} ' + r'$\downarrow$', linestyle='--', linewidth=1.0)
    
    plt.axvline(x=0, color='black', linestyle=':', linewidth=1.0)
    plt.xlabel('Energy/eV', fontsize=10)
    plt.ylabel(r'States/ev${^{-1}}$', fontsize=10)
    plt.xlim(max(min(shifted_energy), -20), min(20, max(shifted_energy)))
    if nspin == 1:
        plt.ylim(bottom=0)
    plt.legend(fontsize=8, ncol=nspin)
    plt.grid(alpha=0.3)
    plt.title('Projected density of States of different species')

    pdos_pic_file = os.path.join(output_dir, 'PDOS.png')
    plt.savefig(pdos_pic_file, dpi=dpi)
    plt.close()

    return Path(pdos_pic_file).absolute()

def plot_pdos_species_shell(shifted_energy, orbitals, output_dir, nspin, dpi):
    species_shells = {}
    for orbital in orbitals:
        species = orbital['species']
        if species not in species_shells.keys():
            species_shells[species] = {}  # Initialize species kind
        
        angular_momentum = angular_momentum_map[orbital['l']]
        if angular_momentum not in species_shells[species].keys():
            species_shells[species][angular_momentum] = orbital['data'] # Initialize DOS for angular momentum of a species
        else:
            species_shells[species][angular_momentum] += orbital['data'] # Add DOS of a angular momentum of a species
    
    # Plot PDOS for each species and each shell
    num_species = len(species_shells)
    fig, axes = plt.subplots(nrows=num_species, ncols=1, figsize=(8, 4*num_species))
    if num_species == 1:
        axes = [axes]

    for species_idx, (species, pdos_data_dict) in enumerate(species_shells.items()):
        ax = axes[species_idx]
        
        for l, pdos_data in pdos_data_dict.items():
            if nspin == 1:
                ax.plot(shifted_energy, pdos_data, color=color_map[l], label=f'{species}-{l}', linewidth=1.0)
            elif nspin == 2:
                ax.plot(shifted_energy, pdos_data[::2], color=color_map[l], label=f'{species}-{l}' + r' $\uparrow$', linestyle='-', linewidth=1.0)
                ax.plot(shifted_energy, -pdos_data[1::2], color=color_map[l], label=f'{species}-{l}' + r' $\downarrow$', linestyle='--', linewidth=1.0)
        
        ax.axvline(x=0, color='black', linestyle=':', linewidth=1.0)
        ax.set_title(f'PDOS for {species}', fontsize=12, pad=10)
        ax.set_ylabel(r'States/ev${^{-1}}$', fontsize=10)
        ax.set_xlim(max(min(shifted_energy), -20), min(20, max(shifted_energy)))
        #if nspin == 1:
        #    ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, ncol=nspin)
        ax.grid(alpha=0.3)
        
        #ax.set_ylim(bottom=0)
    
    axes[-1].set_xlabel('Energy/', fontsize=10)

    plt.tight_layout()
    pdos_pic_file = os.path.join(output_dir, 'PDOS.png')
    plt.savefig(pdos_pic_file, dpi=dpi, bbox_inches='tight')
    plt.close()

    return Path(pdos_pic_file).absolute()

def plot_pdos_species_orbital(shifted_energy, orbitals, output_dir, nspin, label_map, dpi):

    plt.rcParams["text.usetex"] = False
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)

    orbital_label = {}
    for (atom_index, species, l, m, z), full_label in label_map.items():
        if species not in orbital_label.keys():
            orbital_label[species] = {}
        if str(l) not in orbital_label[species].keys():
            orbital_label[species][str(l)] = {}
        if str(m) not in orbital_label[species][str(l)].keys():
            orbital_name = full_label.split('(')[1].split(')')[0]
            if orbital_name in orbital_rep_map.keys():
                orbital_label[species][str(l)][str(m)] = orbital_rep_map[orbital_name]
            else:
                orbital_label[species][str(l)][str(m)] = orbital_name
        else:
            pass

    species_orbitals = {}
    for orbital in orbitals:
        species = orbital['species']
        if species not in species_orbitals.keys():
            species_orbitals[species] = {}
        
        angular_momentum = angular_momentum_map[orbital['l']]
        if angular_momentum not in species_orbitals[species].keys():
            species_orbitals[species][angular_momentum] = {}
        
        mag_quantum_num = orbital['m']
        if mag_quantum_num not in species_orbitals[species][angular_momentum].keys():
            species_orbitals[species][angular_momentum][mag_quantum_num] = orbital['data']
        else:
            species_orbitals[species][angular_momentum][mag_quantum_num] += orbital['data']
    
    total_subplots = 0
    for species, species_pdos in species_orbitals.items():
        total_subplots += len(species_pdos)
    fig, axes = plt.subplots(nrows=total_subplots, ncols=1, figsize=(8, 4*total_subplots))

    subplot_count = 0
    for species, species_pdos in species_orbitals.items():
        for angular_momentum, species_shell_pdos in species_pdos.items():
            for m, species_orbital_pdos in species_shell_pdos.items():
                ax = axes[subplot_count]
                orbital_name = orbital_label[species][str(angular_momentum_map.index(angular_momentum))][str(m)]
                if nspin == 1:
                    ax.plot(shifted_energy, species_orbital_pdos, label=f'{orbital_name}', linewidth=1.0)
                elif nspin == 2:
                    ax.plot(shifted_energy, species_orbital_pdos[::2], label=f'{orbital_name} '+r'$\uparrow$', linestyle='-', linewidth=1.0)
                    ax.plot(shifted_energy, -species_orbital_pdos[1::2], label=f'{orbital_name} '+r'$\downarrow$', linestyle='--', linewidth=1.0)
                    
            ax.axvline(x=0, color='black', linestyle=':', linewidth=1.0)
            ax.set_title(f'PDOS for {species}-{angular_momentum}', fontsize=12, pad=10)
            ax.set_xlim(max(min(shifted_energy), -20), min(20, max(shifted_energy)))
            if nspin == 1:
                ax.set_ylim(bottom=0)
            ax.set_ylabel(r'States/ev${^{-1}}$', fontsize=10)
            ax.legend(fontsize=8, ncol=nspin)
            ax.grid(alpha=0.3)

            subplot_count += 1
        
    axes[-1].set_xlabel('Energy/eV', fontsize=10)

    plt.tight_layout()
    pdos_pic_file = os.path.join(output_dir, 'PDOS.png')
    plt.savefig(pdos_pic_file, dpi=dpi, bbox_inches='tight')
    plt.close()

    return Path(pdos_pic_file).absolute()

def plot_dos(file_path: List[Path],
             fermi_level: float, 
             output_file: str = 'DOS.png',
             nspin: Literal[1, 2] = 1,
             dpi: int=300):
    """Plot total DOS from DOS1_smearing.dat and DOS2_smearing (if nspin=2) file."""
    # Read first two columns from file
    data = np.loadtxt(file_path[0], usecols=(0, 1))
    energy = data[:, 0] - fermi_level  # Shift by Fermi level
    dos = data[:, 1]
    if nspin == 2:
        data = np.loadtxt(file_path[1], usecols=(0, 1))
        dos_dn = data[:, 1]
    
    # Determine energy limits based on data within x range
    x_min, x_max = max(min(energy), -20), min(20, max(energy))

    # Create plot
    plt.figure(figsize=(8, 6))
    if nspin == 1:
        plt.plot(energy, dos, linestyle='-')
    elif nspin == 2:
        plt.plot(energy, dos, linestyle='-', label='spin up')
        plt.plot(energy, -dos_dn, linestyle='--', label='spin down')
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Energy/eV')
    plt.ylabel(r'States/ev${^{-1}}$')
    plt.title('Density of States')
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min, x_max)
    #plt.ylim(y_min, y_max)
    #plt.legend()
    
    # Save plot
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return Path(output_file).absolute()

def plot_dos_pdos(scf_job_path: Path,
                  nscf_job_path: Path, 
                  output_dir: Path,
                  nspin: Literal[1, 2] = 1,
                  mode: Literal['species', 'species+shell', 'species+orbital'] = 'species+shell',
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
    dos_file = [os.path.join(input_dir, "DOS1_smearing.dat")]
    dos_output = os.path.join(output_dir, "DOS.png")
    if nspin == 2:
        dos_file += [os.path.join(input_dir, "DOS2_smearing.dat")]
    
    # Validate input files exist
    for file_path in [input_file, log_file, basref_file, dos_file[0]]:
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            raise FileNotFoundError(f"Required file not found: {file_path}")
    if nspin == 2:
        if not os.path.exists(dos_file[1]):
            print(f"Error: File not found - {dos_file[1]}")
            raise FileNotFoundError(f"Required file not found: {dos_file[1]}")
    

    energy_values, orbitals = parse_pdos_file(input_file)
    fermi_level = collect_metrics(scf_job_path, ['efermi'])['efermi']
    label_map = parse_basref_file(basref_file)
    
    # Plot DOS and get file path
    dos_plot_file = plot_dos(dos_file, fermi_level, dos_output, nspin, dpi)
    
    # Plot PDOS and get file paths
    pdos_plot_file = plot_pdos(energy_values, orbitals, fermi_level, label_map, output_dir, nspin, mode, dpi)
    
    # Combine file paths into a single list
    all_plot_files = [dos_plot_file, pdos_plot_file]
    
    print("Plots generated:")
    for file in all_plot_files:
        print(f"- {file}")
        
    return all_plot_files

