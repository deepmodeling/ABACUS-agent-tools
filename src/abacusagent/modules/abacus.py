import os
import json
from pathlib import Path
from typing import Literal, Optional, TypedDict, Dict, Any, List, Tuple, Union

import numpy as np

from abacustest.lib_model.model_013_inputs import PrepInput
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteInput
from abacustest.lib_collectdata.collectdata import RESULT
from abacustest.lib_model.comm import check_abacus_inputs

from abacusagent.init_mcp import mcp
from abacusagent.modules.util.comm import generate_work_path, run_abacus, collect_metrics


@mcp.tool()
def abacus_prepare(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    #pp_path: Optional[str] = None,
    #orb_path: Optional[str] = None,
    job_type: Literal["scf", "relax", "cell-relax", "md"] = "scf",
    lcao: bool = True,
    nspin: Literal[1, 2, 4] = 1,
    soc: bool = False,
    dftu: bool = False,
    dftu_param: Optional[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]]] = None,
    init_mag: Optional[Dict[str, float]] = None,
    afm: bool = False,
    extra_input: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Prepare input files for ABACUS calculation.
    
    Args:
        stru_file (Path): Structure file in cif, poscar, or abacus/stru format.
        stru_type (Literal["cif", "poscar", "abacus/stru"] = "cif"): Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        job_type (Literal["scf", "relax", "cell-relax", "md"] = "scf"): The type of job to be performed, can be:
            'scf': Self-consistent field calculation, which is the default. 
            'relax': Geometry relaxation calculation, which will relax the atomic position to the minimum energy configuration.
            'cell-relax': Cell relaxation calculation, which will relax the cell parameters and atomic positions to the minimum energy configuration.
            'md': Molecular dynamics calculation, which will perform molecular dynamics simulation.
        lcao (bool): Whether to use LCAO basis set, default is True. If True, the orbital library path must be provided.
        nspin (int): The number of spins, can be 1 (no spin), 2 (spin polarized), or 4 (non-collinear spin). Default is 1.
        soc (bool): Whether to use spin-orbit coupling, if True, nspin should be 4.
        dftu (bool): Whether to use DFT+U, default is False.
        dftu_param (dict): The DFT+U parameters, should be a dict like {"Fe": 4, "Ti": 1}, where the key is the element symbol and the value is the U value.
            Value can also be a list of two values, and the first value is the orbital (p, d, f) to apply DFT+U, and the second value is the U value.
            For example, {"Fe": ["d", 4], "O": ["p", 1]} means applying DFT+U to Fe 3d orbital with U=4 eV and O 2p orbital with U=1 eV.
        init_mag ( dict or None): The initial magnetic moment for magnetic elements, should be a dict like {"Fe": 4, "Ti": 1}, where the key is the element symbol and the value is the initial magnetic moment.
        afm (bool): Whether to use antiferromagnetic calculation, default is False. If True, half of the magnetic elements will be set to negative initial magnetic moment.
        extra_input: Extra input parameters for ABACUS. 
    
    Returns:
        A dictionary containing the job path.
        - 'abacus_inputs_dir': The absolute path to the generated ABACUS input directory, containing INPUT, STRU, pseudopotential and orbital files.
        - 'input_content': The content of the generated INPUT file.
    Raises:
        FileNotFoundError: If the structure file or pseudopotential path does not exist.
        ValueError: If LCAO basis set is selected but no orbital library path is provided.
        RuntimeError: If there is an error preparing input files.
    """
    try:
        stru_file = Path(stru_file).absolute()
        if not os.path.isfile(stru_file):
            raise FileNotFoundError(f"Structure file {stru_file} does not exist.")

        # Check if the pseudopotential path exists
        if soc:
            pp_path = os.environ.get("ABACUS_SOC_PP_PATH")
            orb_path = os.environ.get("ABACUS_SOC_ORB_PATH")
        else:
            pp_path = os.environ.get("ABACUS_PP_PATH")
            orb_path = os.environ.get("ABACUS_ORB_PATH")

        if not os.path.exists(pp_path):
            raise FileNotFoundError(f"Pseudopotential path {pp_path} does not exist.")

        if lcao and not os.path.exists(orb_path):
            raise FileNotFoundError(f"Orbital library path {orb_path} does not exist.")

        work_path = generate_work_path()
        pwd = os.getcwd()
        os.chdir(work_path)
        try:
            extra_input_file = None
            if extra_input is not None:
                # write extra input to the input file
                extra_input_file = Path("INPUT.tmp").absolute()
                WriteInput(extra_input, extra_input_file)

            _, abacus_inputs_dir = PrepInput(
                files=str(stru_file),
                filetype=stru_type,
                jobtype=job_type,
                pp_path=pp_path,
                orb_path=orb_path,
                input_file=extra_input_file,
                lcao=lcao,
                nspin=nspin,
                soc=soc,
                dftu=dftu,
                dftu_param=dftu_param,
                init_mag=init_mag,
                afm=afm,
                copy_pp_orb=True
            ).run()  
        except Exception as e:
            os.chdir(pwd)
            raise RuntimeError(f"Error preparing input files: {e}")

        if len(abacus_inputs_dir) == 0:
            os.chdir(pwd)
            raise RuntimeError("No job path returned from PrepInput.")

        input_content = ReadInput(os.path.join(abacus_inputs_dir[0], "INPUT"))
        input_files = os.listdir(abacus_inputs_dir[0])
        abacus_inputs_dir = Path(abacus_inputs_dir[0]).absolute()
        os.chdir(pwd)

        is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
        if not is_valid:
            raise RuntimeError(f"Invalid ABACUS input files: {msg}")
        
        return {"abacus_inputs_dir": abacus_inputs_dir,
                "input_content": input_content}
    except Exception as e:
        return {"abacus_inputs_dir": None,
                "input_content": None,
                "message": f"Prepare ABACUS input files from given structure failed: {e}"}

#@mcp.tool()
def get_file_content(
    filepath: Path
) -> Dict[str, str]:
    """
    Get content of a file.
    Args:
        filepath: Path to a file
    Returns:
        A string containing file content
    Raises:
        IOError: if read content of `filepath` failed
    """
    filepath = Path(filepath)
    file_content = ''
    try:
        with open(filepath) as fin:
            for lines in fin:
                file_content += lines
    except:
        raise IOError(f"Read content of {filepath} failed")
    
    max_length = 2000
    if len(file_content) > max_length:
        file_content = file_content[:max_length]
    return {'file_content': file_content}

@mcp.tool()
def abacus_modify_input(
    abacus_inputs_dir: Path,
    dft_plus_u_settings: Optional[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]]] = None,
    extra_input: Optional[Dict[str, Any]] = None,
    remove_input: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Modify keywords in ABACUS INPUT file.
    Args:
        abacus_inputs_dir (str): Path to the directory containing the ABACUS input files.
        dft_plus_u_setting: Dictionary specifying DFT+U settings.  
            - Key: Element symbol (e.g., 'Fe', 'Ni').  
            - Value: A list with one or two elements:  
                - One-element form: float, representing the Hubbard U value (orbital will be inferred).  
                - Two-element form: [orbital, U], where `orbital` is one of {'p', 'd', 'f'}, and `U` is a float.
        extra_input: Additional key-value pairs to update the INPUT file.
        remove_input: A list of param names to be removed in the INPUT file

    Returns:
        A dictionary containing:
        - modified_abacus_inputs_dir: the path of the modified INPUT file.
        - input_content: the content of the modified INPUT file as a dictionary.
    Raises:
        FileNotFoundError: If path of given INPUT file does not exist
        RuntimeError: If write modified INPUT file failed
    """
    try:
        input_file = os.path.join(abacus_inputs_dir, "INPUT")
        if dft_plus_u_settings is not None:
            stru_file = os.path.join(abacus_inputs_dir, "STRU")
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"INPUT file {input_file} does not exist.")

        # Update simple keys and their values
        input_param = ReadInput(input_file)
        if extra_input is not None:
            for key, value in extra_input.items():
                input_param[key] = value
    
        # Remove keys
        if remove_input is not None:
            for param in remove_input:
                input_param.pop(param,None)

        # DFT+U settings
        main_group_elements = [
        "H", "He", 
        "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
        "K", "Ca", "Ga", "Ge", "As", "Se", "Br", "Kr",
        "Rb", "Sr", "In", "Sn", "Sb", "Te", "I", "Xe",
        "Cs", "Ba", "Tl", "Pb", "Bi", "Po", "At", "Rn",
        "Fr", "Ra", "Nh", "Fl", "Mc", "Lv", "Ts", "Og" ]
        transition_metals = [
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn"]
        lanthanides_and_acnitides = [
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
        "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]

        orbital_corr_map = {'p': 1, 'd': 2, 'f': 3}
        if dft_plus_u_settings is not None:
            input_param['dft_plus_u'] = 1

            stru = AbacusStru.ReadStru(stru_file)
            elements = stru.get_element(number=False,total=False)

            orbital_corr_param, hubbard_u_param = '', ''
            for element in elements:
                if element not in dft_plus_u_settings:
                    orbital_corr_param += ' -1 '
                    hubbard_u_param += ' 0 '
                else:
                    if type(dft_plus_u_settings[element]) is not float: # orbital_corr and hubbard_u are provided
                        orbital_corr = orbital_corr_map[dft_plus_u_settings[element][0]]
                        orbital_corr_param += f" {orbital_corr} "
                        hubbard_u_param += f" {dft_plus_u_settings[element][1]} "
                    else: #Only hubbard_u is provided, use default orbital_corr
                        if element in main_group_elements:
                            default_orb_corr = 1
                        elif element in transition_metals:
                            default_orb_corr = 2
                        elif element in lanthanides_and_acnitides:
                            default_orb_corr = 3

                        orbital_corr_param += f" {default_orb_corr} "
                        hubbard_u_param += f" {dft_plus_u_settings[element]} "

            input_param['orbital_corr'] = orbital_corr_param.strip()
            input_param['hubbard_u'] = hubbard_u_param.strip()

        WriteInput(input_param, input_file)

        is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
        if not is_valid:
            raise RuntimeError(f"Invalid ABACUS input files: {msg}")
        
        return {'modified_abacus_inputs_dir': abacus_inputs_dir,
                'input_content': input_param}
    except Exception as e:
        return {'message': f"Modify ABACUS INPUT file failed: {e}"}

@mcp.tool()
def abacus_modify_stru(
    abacus_inputs_dir: Path,
    pp: Optional[Dict[str, str]] = None,
    orb: Optional[Dict[str, str]] = None,
    fix_atoms_idx: Optional[List[int]] = None,
    cell: Optional[List[List[float]]] = None,
    coord_change_type: Literal['scale', 'original'] = 'scale',
    movable_coords: Optional[List[bool]] = None,
    initial_magmoms: Optional[List[float]] = None,
    angle1: Optional[List[float]] = None,
    angle2: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Modify pseudopotential, orbital, atom fixation, initial magnetic moments and initial velocities in ABACUS STRU file.
    Args:
        abacus_inputs_dir (str): Path to the directory containing the ABACUS input files.
        pp: Dictionary mapping element names to pseudopotential file paths.
            If not provided, the pseudopotentials from the original STRU file are retained.
        orb: Dictionary mapping element names to numerical orbital file paths.
            If not provided, the orbitals from the original STRU file are retained.
        fix_atoms_idx: List of indices of atoms to be fixed.
        cell: New cell parameters to be set in the STRU file. Should be a list of 3 lists, each containing 3 floats.
        coord_change_type: Type of coordinate change to apply.
            - 'scale': Scale the coordinates by the cell parameters. Suitable for most cases.
            - 'original': Use the original coordinates without scaling. Suitable for single atom or molecule in a large cell.
        movable_coords: For each fixed atom, specify which coordinates are allowed to move.
            Each entry is a list of 3 integers (0 or 1), where 1 means the corresponding coordinate (x/y/z) can move.
            Example: if `fix_atoms_idx = [1]` and `movable_coords = [[0, 1, 1]]`, the x-coordinate of atom 1 will be fixed.
        initial_magmoms: Initial magnetic moments for atoms.
            - For collinear calculations: a list of floats, shape (natom).
            - For non-collinear using Cartesian components: a list of 3-element lists, shape (natom, 3).
            - For non-collinear using angles: a list of floats, shape (natom), one magnetude of magnetic moment per atom.
        angle1: in non-colinear case, specify the angle between z-axis and real spin, in angle measure instead of radian measure
        angle2: in non-colinear case, specify angle between x-axis and real spin in projection in xy-plane , in angle measure instead of radian measure

    Returns:
        A dictionary containing:
        - modified_abacus_inputs_dir: the path of the modified ABACUS STRU file
        - stru_content: the content of the modified ABACUS STRU file as a string.
    Raises:
        ValueError: If `stru_file` is not path of a file, or dimension of initial_magmoms, angle1 or angle2 is not equal with number of atoms,
          or length of fixed_atoms_idx and movable_coords are not equal, or element in movable_coords are not a list with 3 bool elements
        KeyError: If pseudopotential or orbital are not provided for a element
    """
    try:
        stru_file = os.path.join(abacus_inputs_dir, "STRU")
        if os.path.isfile(stru_file):
            stru = AbacusStru.ReadStru(stru_file)
        else:
            raise ValueError(f"{stru_file} is not path of a file")

        # Set pp and orb
        elements = stru.get_element(number=False,total=False)
        if pp is not None:
            pplist = []
            for element in elements:
                if element in pp:
                    pplist.append(pp[element])
                else:
                    raise KeyError(f"Pseudopotential for element {element} is not provided")

            stru.set_pp(pplist)

        if orb is not None:
            orb_list = []
            for element in elements:
                if element in orb:
                    orb_list.append(orb[element])
                else:
                    raise KeyError(f"Orbital for element {element} is not provided")

            stru.set_orb(orb_list)

        # Set cell
        if cell is not None:
            if len(cell) != 3 or any(len(c) != 3 for c in cell):
                raise ValueError("Cell should be a list of 3 lists, each containing 3 floats")

            if np.allclose(np.linalg.det(np.array(cell)), 0) is True:
                raise ValueError("Cell cannot be a singular matrix, please provide a valid cell")
            if coord_change_type == "scale":
                stru.set_cell(cell, bohr=False)
            elif coord_change_type == "original":
                stru.set_cell(cell, bohr=False, change_coord=False)
            else:
                raise ValueError("coord_change_type should be 'scale' or 'original'")

        # Set atomic magmom for every atom
        natoms = len(stru.get_coord())
        if initial_magmoms is not None:
            if len(initial_magmoms) == natoms:
                stru.set_atommag(initial_magmoms)
            else:
                raise ValueError("The dimension of given initial magmoms is not equal with number of atoms")
        if angle1 is not None and angle2 is not None:
            if len(initial_magmoms) == natoms:
                stru.set_angle1(angle1)
            else:
                raise ValueError("The dimension of given angle1 of initial magmoms is not equal with number of atoms")

            if len(initial_magmoms) == natoms:
                stru.set_angle2(angle2)
            else:
                raise ValueError("The dimension of given angle2 of initial magmoms is not equal with number of atoms")

        # Set atom fixations
        # Atom fixations in fix_atoms and movable_coors will be applied to original atom fixation
        if fix_atoms_idx is not None:
            atom_move = stru.get_move()
            for fix_idx, atom_idx in enumerate(fix_atoms_idx):
                if fix_idx < 0 or fix_idx >= natoms:
                    raise ValueError("Given index of atoms to be fixed is not a integer >= 0 or < natoms")

                if len(fix_atoms_idx) == len(movable_coords):
                    if len(movable_coords[fix_idx]) == 3:
                        atom_move[atom_idx] = movable_coords[fix_idx]
                    else:
                        raise ValueError("Elements of movable_coords should be a list with 3 bool elements")
                else:
                    raise ValueError("Length of fix_atoms_idx and movable_coords should be equal")

            stru._move = atom_move

        stru.write(stru_file)
        stru_content = Path(stru_file).read_text(encoding='utf-8')

        is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
        if not is_valid:
            raise RuntimeError(f"Invalid ABACUS input files: {msg}")
        
        return {'modified_abacus_inputs_dir': Path(abacus_inputs_dir).absolute(),
                'stru_content': stru_content}
    except Exception as e:
        return {'message': f"Modify ABACUS STRU file failed: {e}"}

@mcp.tool()
def abacus_collect_data(
    abacus_outputs_dir: Path,
    metrics: List[Literal["version", "ncore", "omp_num", "normal_end", "INPUT", "kpt", "fft_grid",
                          "nbase", "nbands", "nkstot", "ibzk", "natom", "nelec", "nelec_dict", "point_group",
                          "point_group_in_space_group", "converge", "total_mag", "absolute_mag", "energy", 
                          "energy_ks", "energies", "volume", "efermi", "energy_per_atom", "force", "forces", 
                          "stress", "virial", "pressure", "stresses", "virials", "pressures", "largest_gradient", "largest_gradient_stress",
                          "band", "band_weight", "band_plot", "band_gap", "total_time", "stress_time", "force_time", 
                          "scf_time", "scf_time_each_step", "step1_time", "scf_steps", "atom_mags", "atom_mag", 
                          "atom_elec", "atom_orb_elec", "atom_mag_u", "atom_elec_u", "drho", "drho_last", 
                          "denergy", "denergy_last", "denergy_womix", "denergy_womix_last", "lattice_constant", 
                          "lattice_constants", "cell", "cells", "cell_init", "coordinate", "coordinate_init", 
                          "element", "label", "element_list", "atomlabel_list", "pdos", "charge", "charge_spd", 
                          "atom_mag_spd", "relax_converge", "relax_steps", "ds_lambda_step", "ds_lambda_rms", 
                          "ds_mag", "ds_mag_force", "ds_time", "mem_vkb", "mem_psipw"]]
                          = ["normal_end", "converge", "energy", "total_time"]
) -> Dict[str, Any]:
    """
    Collect results after ABACUS calculation and dump to a json file.
    Args:
        abacus_outputs_dir (str): Path to the directory containing the ABACUS job output files.
        metrics (List[str]): List of metric names to collect.  
                  metric_name  description
                      version: the version of ABACUS
                        ncore: the mpi cores
                      omp_num: the omp cores
                   normal_end: if the job is normal ending
                        INPUT: a dict to store the setting in OUT.xxx/INPUT, see manual of ABACUS INPUT file
                          kpt: list, the K POINTS setting in KPT file
                     fft_grid: fft grid for charge/potential
                        nbase: number of basis in LCAO
                       nbands: number of bands
                       nkstot: total K point number
                         ibzk: irreducible K point number
                        natom: total atom number
                        nelec: total electron number
                   nelec_dict: dict of electron number of each species
                  point_group: point group
   point_group_in_space_group: point group in space group
                     converge: if the SCF is converged
                    total_mag: total magnetism (Bohr mag/cell)
                 absolute_mag: absolute magnetism (Bohr mag/cell)
                       energy: the total energy (eV)
                    energy_ks: the E_KohnSham, unit in eV
                     energies: list of total energy of each ION step
                       volume: the volume of cell, in A^3
                       efermi: the fermi energy (eV). If has set nupdown, this will be a list of two values. The first is up, the second is down.
              energy_per_atom: the total energy divided by natom, (eV)
                        force: list[3*natoms], force of the system, if is MD or RELAX calculation, this is the last one
                       forces: list of force, the force of each ION step. Dimension is [nstep,3*natom]
                       stress: list[9], stress of the system, if is MD or RELAX calculation, this is the last one
                       virial: list[9], virial of the system, = stress * volume, and is the last one.
                     pressure: the pressure of the system, unit in kbar.
                     stresses: list of stress, the stress of each ION step. Dimension is [nstep,9]
                      virials: list of virial, the virial of each ION step. Dimension is [nstep,9]
                    pressures: list of pressure, the pressure of each ION step.
             largest_gradient: list, the largest gradient of each ION step. Unit in eV/Angstrom
      largest_gradient_stress: list, the largest stress of each ION step. Unit in kbar
                         band: Band of system. Dimension is [nspin,nk,nband].
                  band_weight: Band weight of system. Dimension is [nspin,nk,nband].
                    band_plot: Will plot the band structure. Return the file name of the plot.
                     band_gap: band gap of the system
                   total_time: the total time of the job
                  stress_time: the time to do the calculation of stress
                   force_time: the time to do the calculation of force
                     scf_time: the time to do SCF
           scf_time_each_step: list, the time of each step of SCF
                   step1_time: the time of 1st SCF step
                    scf_steps: the steps of SCF
                    atom_mags: list of list, the magnization of each atom of each ion step.
                     atom_mag: list, the magnization of each atom. Only the last ION step.
                    atom_elec: list of list of each atom. Each atom list is a list of each orbital, and each orbital is a list of each spin
                atom_orb_elec: list of list of each atom. Each atom list is a list of each orbital, and each orbital is a list of each spin
                   atom_mag_u: list of a dict, the magnization of each atom calculated by occupation number. Only the last SCF step.
                  atom_elec_u: list of a dict with keys are atom index, atom label, and electron of U orbital.
                         drho: [], drho of each scf step
                    drho_last: drho of the last scf step
                      denergy: [], denergy of each scf step
                 denergy_last: denergy of the last scf step
                denergy_womix: [], denergy (calculated by rho without mixed) of each scf step
           denergy_womix_last: float, denergy (calculated by rho without mixed) of last scf step
             lattice_constant: a list of six float which is a/b/c,alpha,beta,gamma of cell. If has more than one ION step, will output the last one.
            lattice_constants: a list of list of six float which is a/b/c,alpha,beta,gamma of cell
                         cell: [[],[],[]], two-dimension list, unit in Angstrom. If is relax or md, will output the last one.
                        cells: a list of [[],[],[]], which is a two-dimension list of cell vector, unit in Angstrom.
                    cell_init: [[],[],[]], two-dimension list, unit in Angstrom. The initial cell
                   coordinate: [[],..], two dimension list, is a cartesian type, unit in Angstrom. If is relax or md, will output the last one
              coordinate_init: [[],..], two dimension list, is a cartesian type, unit in Angstrom. The initial coordinate
                      element: list[], a list of the element name of all atoms
                        label: list[], a list of atom label of all atoms
                 element_list: same as element
               atomlabel_list: same as label
                         pdos: a dict, keys are 'energy' and 'orbitals', and 'orbitals' is a list of dict which is (index,species,l,m,z,data), dimension of data is nspin*ne
                       charge: list, the charge of each atom.
                   charge_spd: list of list, the charge of each atom spd orbital.
                 atom_mag_spd: list of list, the magnization of each atom spd orbital.
               relax_converge: if the relax is converged
                  relax_steps: the total ION steps
               ds_lambda_step: a list of DeltaSpin converge step in each SCF step
                ds_lambda_rms: a list of DeltaSpin RMS in each SCF step
                       ds_mag: a list of list, each element list is for each atom. Unit in uB
                 ds_mag_force: a list of list, each element list is for each atom. Unit in eV/uB
                      ds_time: a list of the total time of inner loop in deltaspin for each scf step.
                      mem_vkb: the memory of VNL::vkb, unit it MB
                    mem_psipw: the memory of PsiPW, unit it MB

    Returns:
        A dictionary containing all collected metrics
    Raises:
        IOError: If read abacus result failed
        RuntimeError: If error occured during collectring data using abacustest
    """
    try:
        collected_metrics = collect_metrics(abacusjob=abacus_outputs_dir, 
                                            metrics_names=metrics,)

        return {'collected_metrics': collected_metrics}
    except Exception as e:
        return {'message': f'Collectiong results from ABACUS output files failed: {e}'}

#@mcp.tool()
def run_abacus_onejob(
    abacus_inputs_dir: Path,
) -> Dict[str, Any]:
    """
    Run one ABACUS job and collect data.
    Args:
        abacus_inputs_dir (str): Path to the directory containing the ABACUS input files.
    Returns:
        the collected metrics from the ABACUS job.
    """
    try:
        run_abacus(abacus_inputs_dir)

        return {'abacus_inputs_dir_dir': abacus_inputs_dir,
                'metrics': abacus_collect_data(abacus_inputs_dir)}
    except Exception as e:
        return {'message': f"Run ABACUS using given input file failed: {e}"}

