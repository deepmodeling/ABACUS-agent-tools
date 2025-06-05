import os
import json
from pathlib import Path
from typing import Literal, Optional, TypedDict, Dict, Any, List, Tuple, Union
from abacustest.lib_model.model_013_inputs import PrepInput
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteInput
from abacustest.lib_collectdata.collectdata import RESULT

from abacusagent.init_mcp import mcp

@mcp.tool()
def abacus_prepare(
    stru_file: str,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    pp_path: Optional[str] = None,
    orb_path: Optional[str] = None,
    job_type: Literal["scf", "relax", "cell-relax", "md"] = "scf",
    lcao: bool = True,
    extra_input: Optional[Dict[str, Any]] = None,
) -> TypedDict("results",{"job_path": str}):
    """
    Prepare input files for ABACUS calculation.
    Args:
        stru_file: Structure file in cif, poscar, or abacus/stru format.
        stru_type: Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        pp_path: The pseudopotential library directory, if is None, will use the value of environment variable ABACUS_PP_PATH.
        orb_path: The orbital library directory, if is None, will use the value of environment variable ABACUS_ORB_PATH.
        job_type: The type of job to be performed, can be 'scf', 'relax', 'cell-relax', or 'md'. 'scf' is the default.
        lcao: Whether to use LCAO basis set, default is True. If True, the orbital library path must be provided.
        extra_input: Extra input parameters for ABACUS. 
    
    Returns:
        A dictionary containing the job path.
    Raises:
        FileNotFoundError: If the structure file or pseudopotential path does not exist.
        ValueError: If LCAO basis set is selected but no orbital library path is provided.
        RuntimeError: If there is an error preparing input files.
    """
    data_dir = Path("/mnt/e/temp")
    stru_file = data_dir / stru_file
    if not os.path.isfile(stru_file):
        raise FileNotFoundError(f"Structure file {stru_file} does not exist.")
    
    # Check if the pseudopotential path exists
    pp_path = data_dir / pp_path if pp_path is not None else os.getenv("ABACUS_PP_PATH")
    if pp_path is None or not os.path.exists(pp_path):
        raise FileNotFoundError(f"Pseudopotential path {pp_path} does not exist.")
    
    orb_path = data_dir / orb_path if orb_path is not None else os.getenv("ABACUS_ORB_PATH")
    if orb_path is None and os.getenv("ABACUS_ORB_PATH") is not None:
        orb_path = os.getenv("ABACUS_ORB_PATH")
    
    if lcao and orb_path is None:
        raise ValueError("LCAO basis set is selected but no orbital library path is provided.")
    
    extra_input_file = None
    if extra_input is not None:
        # write extra input to the input file
        extra_input_file = "INPUT.tmp"
        WriteInput(extra_input, extra_input_file)
    
    try:
        _, job_path = PrepInput(
            files=str(stru_file),
            filetype=stru_type,
            jobtype=job_type,
            pp_path=pp_path,
            orb_path=orb_path,
            input_file=extra_input_file,
            lcao=lcao
        ).run()
    except Exception as e:
        raise RuntimeError(f"Error preparing input files: {e}")

    if len(job_path) == 0:
        raise RuntimeError("No job path returned from PrepInput.")
    
    return {"job_path": str(Path(job_path[0]).absolute())}

def abacus_get_input(
    input_file: str
) -> TypedDict("results", {"input_params": Dict[str, Any]}):
    """
    Get parameters and their values from INPUT.
    Args:
        input_file: Path to the original ABACUS INPUT file.
    Returns:
        A dictionary containing all parameters in INPUT file.
    Raises:
        IOError: if `input_file` is not a ABACUS INPUT file
    """
    input_file = Path(input_file)
    input_params = ReadInput(input_file)
    return {"input_param": input_params}

@mcp.tool()
def get_file_content(
    filepath: str
) -> TypedDict("results", {"file_content": str}):
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
    
    return {'file_content': file_content}

@mcp.tool()
def abacus_modify_input(
    input_file: str,
    stru_file: Optional[str] = None,
    dft_plus_u_settings: Optional[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]]] = None,
    extra_input: Optional[Dict[str, Any]] = None,
    remove_input: Optional[List[str]] = None
) -> TypedDict("results",{"input_path": str}):
    """
    Modify keywords in ABACUS INPUT file.
    Args:
        input_file: Path to the original ABACUS INPUT file.
        stru_file: Path to the ABACUS STRU file, required for determining atom types in DFT+U settings.
        dft_plus_u_setting: Dictionary specifying DFT+U settings.  
            - Key: Element symbol (e.g., 'Fe', 'Ni').  
            - Value: A list with one or two elements:  
                - One-element form: float, representing the Hubbard U value (orbital will be inferred).  
                - Two-element form: [orbital, U], where `orbital` is one of {'p', 'd', 'f'}, and `U` is a float.
        extra_input: Additional key-value pairs to update the INPUT file.
        remove_input: A list of param names to be removed in the INPUT file

    Returns:
        A dictionary containing the path of the modified INPUT file under the key `'input_path'`.
    Raises:
        FileNotFoundError: If path of given INPUT file does not exist
        RuntimeError: If write modified INPUT file failed
    """
    input_file = Path(input_file)
    if dft_plus_u_settings is not None:
        stru_file = Path(stru_file)
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
            try:
                del input_param[param]
            except:
                raise KeyError(f"There's no {param} in the original INPUT file")
       
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

    try:
        WriteInput(input_param, input_file)
        return {'input_path': str(Path(input_file).absolute())}
    except Exception as e:
        raise RuntimeError("Error occured during writing modified INPUT file")

@mcp.tool()
def abacus_modify_stru(
    stru_file: str,
    pp: Optional[Dict[str, str]] = None,
    orb: Optional[Dict[str, str]] = None,
    fix_atoms_idx: Optional[List[int]] = None,
    movable_coords: Optional[List[bool]] = None,
    initial_magmoms: Optional[List[float]] = None,
    angle1: Optional[List[float]] = None,
    angle2: Optional[List[float]] = None
) -> TypedDict("results",{"stru_path": str}):
    """
    Modify pseudopotential, orbital, atom fixation, initial magnetic moments and initial velocities in ABACUS STRU file.
    Args:
        stru_file: Path to the original ABACUS STRU file.
        pp: Dictionary mapping element names to pseudopotential file paths.
            If not provided, the pseudopotentials from the original STRU file are retained.
        orb: Dictionary mapping element names to numerical orbital file paths.
            If not provided, the orbitals from the original STRU file are retained.
        fix_atoms_idx: List of indices of atoms to be fixed.
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
        A dictionary containing the path of the modified ABACUS STRU file under the key 'stru_path'.
    Raises:
        ValueError: If `stru_file` is not path of a file, or dimension of initial_magmoms, angle1 or angle2 is not equal with number of atoms,
          or length of fixed_atoms_idx and movable_coords are not equal, or element in movable_coords are not a list with 3 bool elements
        KeyError: If pseudopotential or orbital are not provided for a element
    """
    stru_file = Path(stru_file)
    if stru_file.is_file():
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
    
    return {'stru_path': str(stru_file.absolute())}

@mcp.tool()
def abacus_collect_data(
    abacusjob: str,
    metrics: List[Literal["version", "ncore", "omp_num", "normal_end", "INPUT", "kpt", "fft_grid",
                          "nbase", "nbands", "nkstot", "ibzk", "natom", "nelec", "nelec_dict", "point_group",
                          "point_group_in_space_group", "converge", "total_mag", "absolute_mag", "energy", 
                          "energy_ks", "energies", "volume", "efermi", "energy_per_atom", "force", "forces", 
                          "stress", "virial", "pressure", "stresses", "virials", "pressures", "largest_gradient", 
                          "band", "band_weight", "band_plot", "band_gap", "total_time", "stress_time", "force_time", 
                          "scf_time", "scf_time_each_step", "step1_time", "scf_steps", "atom_mags", "atom_mag", 
                          "atom_elec", "atom_orb_elec", "atom_mag_u", "atom_elec_u", "drho", "drho_last", 
                          "denergy", "denergy_last", "denergy_womix", "denergy_womix_last", "lattice_constant", 
                          "lattice_constants", "cell", "cells", "cell_init", "coordinate", "coordinate_init", 
                          "element", "label", "element_list", "atomlabel_list", "pdos", "charge", "charge_spd", 
                          "atom_mag_spd", "relax_converge", "relax_steps", "ds_lambda_step", "ds_lambda_rms", 
                          "ds_mag", "ds_mag_force", "ds_time", "mem_vkb", "mem_psipw"]]
                          = ["normal_end", "converge", "energy", "total_time"]
) -> TypedDict("results", {"metrics": Dict[str, Any]}):
    """
    Collect results after ABACUS calculation and dump to a json file.
    Args:
        abacusjob (str): Path to the directory containing the ABACUS job output files.
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
    abacusjob = Path(abacusjob)
    try:
        abacusresult = RESULT(fmt="abacus", path=abacusjob)
    except:
        raise IOError("Read abacus result failed")
    
    collected_metrics = {}
    for metric in metrics:
        try:
            collected_metrics[metric] = abacusresult[metric]
        except Exception as e:
            raise RuntimeError(f"Error during collecting {metric}")
    
    metric_file_path = abacusjob / "metrics.json"
    with open(metric_file_path, "w", encoding="UTF-8") as f:
        json.dump(collected_metrics, f, indent=4)
    
    return {'collected_metrics': collected_metrics}

@mcp.tool()
def run_abacus_local(
    abacusjob: str,
) -> TypedDict("results",{"normal_end": bool}):
    """
    Run abacus in local machine.
    Args:
        abacusjob (str): Path to the directory containing the ABACUS job output files.
    Returns:
        Whether the abacus job ends normally.
    
    Raises:
        RuntimeError: if the abacus calculations didn't end normally
    """
    abacusjob = Path(abacusjob)
    original_path = Path("./")
    os.chdir(Path(abacusjob))
    os.system("runabacus.sh")
    os.chdir(original_path)

    output = abacus_collect_data(str(abacusjob))
    with open("metrics.json") as fin:
        metrics = json.load(fin)
    if metrics['normal_end'] is not True:
        raise RuntimeError("the abacus job didn't end normally")

    return {'normal_end': metrics['normal_end']}

def get_pp(
    elements: List[str],
    pp_type: Literal['sg15-v1-std', 'sg15-v1-acc', 'sg15-v2', 'dojo'] = 'sg15-v2'
) -> TypedDict("result", {'pp': Dict[str, str]}):
    """
    Get pseudopotential filename of given type of pseudopotential collections for a list of elements 
    Args:
        elements: A list of elements to be assigned for a pseudopotential.
        pp_type: The name of pseudopotential collections.
    Returns: A dictionary contains assigned filename of pseudopotentials for each element
    Raises:
        KeyError: if a pseudopotential collection doesn't provide a pseudopotential for a element
    """
    pp_basis_dict = {
        'sg15-v2': {'pp': 'pp_sg15', 'basis': 'basis_sg15_v2'}, 
        'sg15-v1-std': {'pp': 'pp_sg15', 'basis': 'basis_sg15_std_v1'},
        'sg15-v1-act': {'pp': 'pp_sg15', 'basis': 'basis_sg15_act_v1'},
        'dojo': {'pp': 'pp_dojo', 'basis': 'basis_dojo'}
    }

    pp_dict = {'pp_dojo': {'Cu': 'Cu.upf',
                          'W': 'W.upf',
                          'Bi': 'Bi.upf',
                          'Br': 'Br.upf',
                          'Rn': 'Rn.upf',
                          'Se': 'Se.upf',
                          'Pb': 'Pb.upf',
                          'Sn': 'Sn.upf',
                          'O': 'O.upf',
                          'Y': 'Y.upf',
                          'Co': 'Co.upf',
                          'Si': 'Si.upf',
                          'Au': 'Au.upf',
                          'Re': 'Re.upf',
                          'Tc': 'Tc.upf',
                          'Pd': 'Pd.upf',
                          'Ge': 'Ge.upf',
                          'Ta': 'Ta.upf',
                          'N': 'N.upf',
                          'Po': 'Po.upf',
                          'Pt': 'Pt.upf',
                          'B': 'B.upf',
                          'Rb': 'Rb.upf',
                          'Ir': 'Ir.upf',
                          'Sr': 'Sr.upf',
                          'Ag': 'Ag.upf',
                          'Cl': 'Cl.upf',
                          'Xe': 'Xe.upf',
                          'In': 'In.upf',
                          'Sc': 'Sc.upf',
                          'As': 'As.upf',
                          'Be': 'Be.upf',
                          'Cd': 'Cd.upf',
                          'Ti': 'Ti.upf',
                          'Hf': 'Hf.upf',
                          'F': 'F.upf',
                          'P': 'P.upf',
                          'Rh': 'Rh.upf',
                          'Fe': 'Fe.upf',
                          'Na': 'Na.upf',
                          'C': 'C.upf',
                          'Cr': 'Cr.upf',
                          'Kr': 'Kr.upf',
                          'Nb': 'Nb.upf',
                          'Cs': 'Cs.upf',
                          'Ne': 'Ne.upf',
                          'He': 'He.upf',
                          'Ni': 'Ni.upf',
                          'Ca': 'Ca.upf',
                          'Ba': 'Ba.upf',
                          'H': 'H.upf',
                          'Te': 'Te.upf',
                          'Ru': 'Ru.upf',
                          'Sb': 'Sb.upf',
                          'Hg': 'Hg.upf',
                          'Os': 'Os.upf',
                          'Zn': 'Zn.upf',
                          'S': 'S.upf',
                          'Mo': 'Mo.upf',
                          'V': 'V.upf',
                          'Mg': 'Mg.upf',
                          'Zr': 'Zr.upf',
                          'I': 'I.upf',
                          'Tl': 'Tl.upf',
                          'Al': 'Al.upf',
                          'K': 'K.upf',
                          'Li': 'Li.upf',
                          'Mn': 'Mn.upf',
                          'Ga': 'Ga.upf',
                          'Ar': 'Ar.upf'},
              'pp_sg15': {'Br': 'Br_ONCV_PBE-1.0.upf',
                          'Li': 'Li_ONCV_PBE-1.0.upf',
                          'Re': 'Re_ONCV_PBE-1.0.upf',
                          'Zn': 'Zn_ONCV_PBE-1.0.upf',
                          'S': 'S_ONCV_PBE-1.0.upf',
                          'Tc': 'Tc_ONCV_PBE-1.0.upf',
                          'Cu': 'Cu_ONCV_PBE-1.0.upf',
                          'Sr': 'Sr_ONCV_PBE-1.0.upf',
                          'Si': 'Si_ONCV_PBE-1.0.upf',
                          'Cr': 'Cr_ONCV_PBE-1.0.upf',
                          'Hf': 'Hf_ONCV_PBE-1.0.upf',
                          'Rb': 'Rb_ONCV_PBE-1.0.upf',
                          'O': 'O_ONCV_PBE-1.0.upf',
                          'He': 'He_ONCV_PBE-1.0.upf',
                          'K': 'K_ONCV_PBE-1.0.upf',
                          'H': 'H_ONCV_PBE-1.0.upf',
                          'Fe': 'Fe_ONCV_PBE-1.0.upf',
                          'Sc': 'Sc_ONCV_PBE-1.0.upf',
                          'Pd': 'Pd_ONCV_PBE-1.0.upf',
                          'Mn': 'Mn_ONCV_PBE-1.0.upf',
                          'Ti': 'Ti_ONCV_PBE-1.0.upf',
                          'N': 'N_ONCV_PBE-1.0.upf',
                          'Sb': 'Sb_ONCV_PBE-1.0.upf',
                          'Cd': 'Cd_ONCV_PBE-1.0.upf',
                          'La': 'La_ONCV_PBE-1.0.upf',
                          'Na': 'Na_ONCV_PBE-1.0.upf',
                          'Sn': 'Sn_ONCV_PBE-1.0.upf',
                          'Hg': 'Hg_ONCV_PBE-1.0.upf',
                          'As': 'As_ONCV_PBE-1.0.upf',
                          'Co': 'Co_ONCV_PBE-1.0.upf',
                          'Rh': 'Rh_ONCV_PBE-1.0.upf',
                          'Ag': 'Ag_ONCV_PBE-1.0.upf',
                          'Ar': 'Ar_ONCV_PBE-1.0.upf',
                          'Ni': 'Ni_ONCV_PBE-1.0.upf',
                          'Zr': 'Zr_ONCV_PBE-1.0.upf',
                          'Mo': 'Mo_ONCV_PBE-1.0.upf',
                          'Ru': 'Ru_ONCV_PBE-1.0.upf',
                          'P': 'P_ONCV_PBE-1.0.upf',
                          'Be': 'Be_ONCV_PBE-1.0.upf',
                          'Mg': 'Mg_ONCV_PBE-1.0.upf',
                          'Te': 'Te_ONCV_PBE-1.0.upf',
                          'Se': 'Se_ONCV_PBE-1.0.upf',
                          'Os': 'Os_ONCV_PBE-1.0.upf',
                          'In': 'In_ONCV_PBE-1.0.upf',
                          'Ta': 'Ta_ONCV_PBE-1.0.upf',
                          'F': 'F_ONCV_PBE-1.0.upf',
                          'Ca': 'Ca_ONCV_PBE-1.0.upf',
                          'Bi': 'Bi_ONCV_PBE-1.0.upf',
                          'Ir': 'Ir_ONCV_PBE-1.0.upf',
                          'Nb': 'Nb_ONCV_PBE-1.0.upf',
                          'Ga': 'Ga_ONCV_PBE-1.0.upf',
                          'Ba': 'Ba_ONCV_PBE-1.0.upf',
                          'Cs': 'Cs_ONCV_PBE-1.0.upf',
                          'Au': 'Au_ONCV_PBE-1.0.upf',
                          'V': 'V_ONCV_PBE-1.0.upf',
                          'Al': 'Al_ONCV_PBE-1.0.upf',
                          'Ge': 'Ge_ONCV_PBE-1.0.upf',
                          'Ne': 'Ne_ONCV_PBE-1.0.upf',
                          'Y': 'Y_ONCV_PBE-1.0.upf',
                          'C': 'C_ONCV_PBE-1.0.upf',
                          'W': 'W_ONCV_PBE-1.0.upf',
                          'Kr': 'Kr_ONCV_PBE-1.0.upf',
                          'B': 'B_ONCV_PBE-1.0.upf',
                          'Tl': 'Tl_ONCV_PBE-1.0.upf',
                          'Pb': 'Pb_ONCV_PBE-1.0.upf',
                          'Pt': 'Pt_ONCV_PBE-1.0.upf',
                          'Cl': 'Cl_ONCV_PBE-1.0.upf',
                          'Xe': 'Xe_ONCV_PBE-1.0.upf',
                          'I': 'I_ONCV_PBE-1.0.upf'}
    }
    
    return_pps = {}
    for element in elements: 
        try:
            return_pps[element] = pp_dict[pp_basis_dict[pp_type]['pp']][element]
        except:
            raise KeyError(f"No pseudopotential for element {element} in {pp_type}")

    return {"result": return_pps}

def get_orb(
    elements: List[str],
    orb_type: Literal['sg15-v1-std', 'sg15-v1-acc', 'sg15-v2', 'dojo'] = 'sg15-v2'
) -> TypedDict("result", {'pp': Dict[str, str]}):
    """
    Get orbital filename of given type of orbital collections for a list of elements 
    Args:
        elements: A list of elements to be assigned for a orbital.
        orb_type: The name of orbital collections.
    Returns: A dictionary contains assigned filename of orbitals for each element
    Raises:
        KeyError: if a orbital collection doesn't provide a orbital for a element
    """
    pp_basis_dict = {
        'sg15-v2': {'pp': 'pp_sg15', 'basis': 'basis_sg15_v2'}, 
        'sg15-v1-std': {'pp': 'pp_sg15', 'basis': 'basis_sg15_std_v1'},
        'sg15-v1-act': {'pp': 'pp_sg15', 'basis': 'basis_sg15_act_v1'},
        'dojo': {'pp': 'pp_dojo', 'basis': 'basis_dojo'}
    }

    
    orb_dict = {'basis_sg15_v2': {'Be': 'Be_gga_7au_100Ry_4s1p.orb',
                                   'Fe': 'Fe_gga_8au_100Ry_4s2p2d1f.orb',
                                   'Ca': 'Ca_gga_9au_100Ry_4s2p1d.orb',
                                   'Cl': 'Cl_gga_7au_100Ry_2s2p1d.orb',
                                   'Xe': 'Xe_gga_8au_100Ry_2s2p2d1f.orb',
                                   'As': 'As_gga_7au_100Ry_2s2p1d.orb',
                                   'Br': 'Br_gga_7au_100Ry_2s2p1d.orb',
                                   'Sc': 'Sc_gga_8au_100Ry_4s2p2d1f.orb',
                                   'Tl': 'Tl_gga_7au_100Ry_2s2p2d1f.orb',
                                   'Te': 'Te_gga_7au_100Ry_2s2p2d1f.orb',
                                   'Ru': 'Ru_gga_7au_100Ry_4s2p2d1f.orb',
                                   'Os': 'Os_gga_7au_100Ry_4s2p2d1f.orb',
                                   'Zn': 'Zn_gga_8au_100Ry_4s2p2d1f.orb',
                                   'Ta': 'Ta_gga_8au_100Ry_4s2p2d2f1g.orb',
                                   'Ir': 'Ir_gga_7au_100Ry_4s2p2d1f.orb',
                                   'Ti': 'Ti_gga_8au_100Ry_4s2p2d1f.orb',
                                   'Ne': 'Ne_gga_6au_100Ry_2s2p1d.orb',
                                   'Sn': 'Sn_gga_7au_100Ry_2s2p2d1f.orb',
                                   'H': 'H_gga_6au_100Ry_2s1p.orb',
                                   'Mg': 'Mg_gga_8au_100Ry_4s2p1d.orb',
                                   'Mo': 'Mo_gga_7au_100Ry_4s2p2d1f.orb',
                                   'I': 'I_gga_7au_100Ry_2s2p2d1f.orb',
                                   'Li': 'Li_gga_7au_100Ry_4s1p.orb',
                                   'Cs': 'Cs_gga_10au_100Ry_4s2p1d.orb',
                                   'Sr': 'Sr_gga_9au_100Ry_4s2p1d.orb',
                                   'Co': 'Co_gga_8au_100Ry_4s2p2d1f.orb',
                                   'Si': 'Si_gga_7au_100Ry_2s2p1d.orb',
                                   'Ge': 'Ge_gga_8au_100Ry_2s2p2d1f.orb',
                                   'Nb': 'Nb_gga_8au_100Ry_4s2p2d1f.orb',
                                   'S': 'S_gga_7au_100Ry_2s2p1d.orb',
                                   'K': 'K_gga_9au_100Ry_4s2p1d.orb',
                                   'C': 'C_gga_7au_100Ry_2s2p1d.orb',
                                   'Al': 'Al_gga_7au_100Ry_4s4p1d.orb',
                                   'Cr': 'Cr_gga_8au_100Ry_4s2p2d1f.orb',
                                   'Ar': 'Ar_gga_7au_100Ry_2s2p1d.orb',
                                   'Tc': 'Tc_gga_7au_100Ry_4s2p2d1f.orb',
                                   'Kr': 'Kr_gga_7au_100Ry_2s2p1d.orb',
                                   'Pd': 'Pd_gga_7au_100Ry_4s2p2d1f.orb',
                                   'N': 'N_gga_7au_100Ry_2s2p1d.orb',
                                   'Rb': 'Rb_gga_10au_100Ry_4s2p1d.orb',
                                   'Cu': 'Cu_gga_8au_100Ry_4s2p2d1f.orb',
                                   'W': 'W_gga_8au_100Ry_4s2p2d2f1g.orb',
                                   'Ba': 'Ba_gga_10au_100Ry_4s2p2d1f.orb',
                                   'Cd': 'Cd_gga_7au_100Ry_4s2p2d1f.orb',
                                   'P': 'P_gga_7au_100Ry_2s2p1d.orb',
                                   'Zr': 'Zr_gga_8au_100Ry_4s2p2d1f.orb',
                                   'B': 'B_gga_8au_100Ry_2s2p1d.orb',
                                   'Mn': 'Mn_gga_8au_100Ry_4s2p2d1f.orb',
                                   'Ga': 'Ga_gga_8au_100Ry_2s2p2d1f.orb',
                                   'Hf': 'Hf_gga_7au_100Ry_4s2p2d2f1g.orb',
                                   'In': 'In_gga_7au_100Ry_2s2p2d1f.orb',
                                   'Sb': 'Sb_gga_7au_100Ry_2s2p2d1f.orb',
                                   'Pt': 'Pt_gga_7au_100Ry_4s2p2d1f.orb',
                                   'Au': 'Au_gga_7au_100Ry_4s2p2d1f.orb',
                                   'V': 'V_gga_8au_100Ry_4s2p2d1f.orb',
                                   'He': 'He_gga_6au_100Ry_2s1p.orb',
                                   'Pb': 'Pb_gga_7au_100Ry_2s2p2d1f.orb',
                                   'Hg': 'Hg_gga_9au_100Ry_4s2p2d1f.orb',
                                   'Rh': 'Rh_gga_7au_100Ry_4s2p2d1f.orb',
                                   'F': 'F_gga_7au_100Ry_2s2p1d.orb',
                                   'Na': 'Na_gga_8au_100Ry_4s2p1d.orb',
                                   'Se': 'Se_gga_8au_100Ry_2s2p1d.orb',
                                   'Ag': 'Ag_gga_7au_100Ry_4s2p2d1f.orb',
                                   'O': 'O_gga_7au_100Ry_2s2p1d.orb',
                                   'Y': 'Y_gga_8au_100Ry_4s2p2d1f.orb',
                                   'Ni': 'Ni_gga_8au_100Ry_4s2p2d1f.orb',
                                   'Bi': 'Bi_gga_7au_100Ry_2s2p2d1f.orb',
                                   'Re': 'Re_gga_7au_100Ry_4s2p2d1f.orb'},
                 'basis_sg15_std_v1': {'C': 'C_gga_8au_60Ry_2s2p1d.orb',
                                       'S': 'S_gga_8au_60Ry_2s2p1d.orb',
                                       'Ba': 'Ba_gga_11au_60Ry_4s2p2d.orb',
                                       'V': 'V_gga_9au_60Ry_4s2p2d1f.orb',
                                       'He': 'He_gga_6au_60Ry_2s1p.orb',
                                       'Al': 'Al_gga_9au_60Ry_4s4p1d.orb',
                                       'Pd': 'Pd_gga_9au_60Ry_4s2p2d1f.orb',
                                       'B': 'B_gga_8au_60Ry_2s2p1d.orb',
                                       'Sr': 'Sr_gga_10au_60Ry_4s2p1d.orb',
                                       'Ru': 'Ru_gga_9au_60Ry_4s2p2d1f.orb',
                                       'Ca': 'Ca_gga_9au_60Ry_4s2p1d.orb',
                                       'Ne': 'Ne_gga_6au_60Ry_2s2p1d.orb',
                                       'Co': 'Co_gga_9au_60Ry_4s2p2d1f.orb',
                                       'K': 'K_gga_9au_60Ry_4s2p1d.orb',
                                       'Mg': 'Mg_gga_9au_60Ry_4s2p1d.orb',
                                       'Nb': 'Nb_gga_9au_60Ry_4s2p2d1f.orb',
                                       'Au': 'Au_gga_9au_60Ry_4s2p2d1f.orb',
                                       'Si': 'Si_gga_8au_60Ry_2s2p1d.orb',
                                       'Cd': 'Cd_gga_9au_60Ry_4s2p2d1f.orb',
                                       'Os': 'Os_gga_9au_60Ry_4s2p2d1f.orb',
                                       'Fe': 'Fe_gga_9au_60Ry_4s2p2d1f.orb',
                                       'Rh': 'Rh_gga_9au_60Ry_4s2p2d1f.orb',
                                       'Ag': 'Ag_gga_9au_60Ry_4s2p2d1f.orb',
                                       'Y': 'Y_gga_9au_60Ry_4s2p2d1f.orb',
                                       'Zr': 'Zr_gga_9au_60Ry_4s2p2d1f.orb',
                                       'Be': 'Be_gga_8au_60Ry_4s1p.orb',
                                       'Ti': 'Ti_gga_9au_60Ry_4s2p2d1f.orb',
                                       'Bi': 'Bi_gga_9au_60Ry_2s2p2d.orb',
                                       'Cl': 'Cl_gga_8au_60Ry_2s2p1d.orb',
                                       'Ni': 'Ni_gga_9au_60Ry_4s2p2d1f.orb',
                                       'H': 'H_gga_8au_60Ry_2s1p.orb',
                                       'Mo': 'Mo_gga_9au_60Ry_4s2p2d1f.orb',
                                       'Ar': 'Ar_gga_7au_60Ry_2s2p1d.orb',
                                       'Pt': 'Pt_gga_9au_60Ry_4s2p2d1f.orb',
                                       'I': 'I_gga_8au_60Ry_2s2p2d.orb',
                                       'N': 'N_gga_8au_60Ry_2s2p1d.orb',
                                       'Hg': 'Hg_gga_9au_60Ry_4s2p2d1f.orb',
                                       'Ge': 'Ge_gga_8au_60Ry_2s2p2d.orb',
                                       'Ga': 'Ga_gga_9au_60Ry_2s2p2d.orb',
                                       'Ta': 'Ta_gga_10au_60Ry_4s2p2d2f.orb',
                                       'Re': 'Re_gga_10au_60Ry_4s2p2d1f.orb',
                                       'Rb': 'Rb_gga_10au_60Ry_4s2p1d.orb',
                                       'Na': 'Na_gga_10au_60Ry_4s2p1d.orb',
                                       'O': 'O_gga_7au_60Ry_2s2p1d.orb',
                                       'Cr': 'Cr_gga_9au_60Ry_4s2p2d1f.orb',
                                       'Kr': 'Kr_gga_7au_60Ry_2s2p1d.orb',
                                       'Sn': 'Sn_gga_9au_60Ry_2s2p2d.orb',
                                       'In': 'In_gga_9au_60Ry_2s2p2d.orb',
                                       'Ir': 'Ir_gga_9au_60Ry_4s2p2d1f.orb',
                                       'Se': 'Se_gga_8au_60Ry_2s2p1d.orb',
                                       'Pb': 'Pb_gga_9au_60Ry_2s2p2d.orb',
                                       'Cu': 'Cu_gga_9au_60Ry_4s2p2d1f.orb',
                                       'Xe': 'Xe_gga_8au_60Ry_2s2p2d.orb',
                                       'Te': 'Te_gga_9au_60Ry_2s2p2d.orb',
                                       'Zn': 'Zn_gga_9au_60Ry_4s2p2d1f.orb',
                                       'Mn': 'Mn_gga_9au_60Ry_4s2p2d1f.orb',
                                       'As': 'As_gga_8au_60Ry_2s2p1d.orb',
                                       'Tl': 'Tl_gga_9au_60Ry_2s2p2d.orb',
                                       'Sc': 'Sc_gga_9au_60Ry_4s2p2d1f.orb',
                                       'Cs': 'Cs_gga_11au_60Ry_4s2p1d.orb',
                                       'Br': 'Br_gga_8au_60Ry_2s2p1d.orb',
                                       'Sb': 'Sb_gga_9au_60Ry_2s2p2d.orb',
                                       'Tc': 'Tc_gga_9au_60Ry_4s2p2d1f.orb',
                                       'W': 'W_gga_10au_60Ry_4s2p2d2f.orb',
                                       'F': 'F_gga_7au_60Ry_2s2p1d.orb',
                                       'Li': 'Li_gga_9au_60Ry_4s1p.orb',
                                       'P': 'P_gga_8au_60Ry_2s2p1d.orb',
                                       'Hf': 'Hf_gga_10au_60Ry_4s2p2d2f.orb'},
                 'basis_sg15_act_v1': {'Ni': 'Ni_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Ca': 'Ca_gga_9au_100Ry_4s2p1d.orb',
                                       'Cs': 'Cs_gga_11au_100Ry_4s2p1d.orb',
                                       'S': 'S_gga_8au_100Ry_2s2p1d.orb',
                                       'Rh': 'Rh_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Tl': 'Tl_gga_9au_100Ry_2s2p2d.orb',
                                       'Cr': 'Cr_gga_9au_100Ry_4s2p2d1f.orb',
                                       'N': 'N_gga_8au_100Ry_2s2p1d.orb',
                                       'I': 'I_gga_8au_100Ry_2s2p2d.orb',
                                       'Ne': 'Ne_gga_6au_100Ry_2s2p1d.orb',
                                       'Zn': 'Zn_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Co': 'Co_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Fe': 'Fe_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Sc': 'Sc_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Sb': 'Sb_gga_9au_100Ry_2s2p2d.orb',
                                       'Pb': 'Pb_gga_9au_100Ry_2s2p2d.orb',
                                       'W': 'W_gga_10au_100Ry_4s2p2d2f.orb',
                                       'Au': 'Au_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Mo': 'Mo_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Ga': 'Ga_gga_9au_100Ry_2s2p2d.orb',
                                       'Be': 'Be_gga_8au_100Ry_4s1p.orb',
                                       'Tc': 'Tc_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Pd': 'Pd_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Br': 'Br_gga_8au_100Ry_2s2p1d.orb',
                                       'Ti': 'Ti_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Te': 'Te_gga_9au_100Ry_2s2p2d.orb',
                                       'K': 'K_gga_9au_100Ry_4s2p1d.orb',
                                       'Ar': 'Ar_gga_7au_100Ry_2s2p1d.orb',
                                       'Mg': 'Mg_gga_9au_100Ry_4s2p1d.orb',
                                       'Cu': 'Cu_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Kr': 'Kr_gga_7au_100Ry_2s2p1d.orb',
                                       'Bi': 'Bi_gga_9au_100Ry_2s2p2d.orb',
                                       'Os': 'Os_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Sn': 'Sn_gga_9au_100Ry_2s2p2d.orb',
                                       'Li': 'Li_gga_9au_100Ry_4s1p.orb',
                                       'Cd': 'Cd_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Rb': 'Rb_gga_10au_100Ry_4s2p1d.orb',
                                       'Ag': 'Ag_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Pt': 'Pt_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Sr': 'Sr_gga_10au_100Ry_4s2p1d.orb',
                                       'Si': 'Si_gga_8au_100Ry_2s2p1d.orb',
                                       'In': 'In_gga_9au_100Ry_2s2p2d.orb',
                                       'B': 'B_gga_8au_100Ry_2s2p1d.orb',
                                       'Xe': 'Xe_gga_7au_100Ry_2s2p2d.orb',
                                       'As': 'As_gga_8au_100Ry_2s2p1d.orb',
                                       'Ge': 'Ge_gga_8au_100Ry_2s2p2d.orb',
                                       'Ru': 'Ru_gga_9au_100Ry_4s2p2d1f.orb',
                                       'C': 'C_gga_8au_100Ry_2s2p1d.orb',
                                       'Ir': 'Ir_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Ba': 'Ba_gga_11au_100Ry_4s2p2d.orb',
                                       'He': 'He_gga_6au_100Ry_2s1p.orb',
                                       'Hg': 'Hg_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Ta': 'Ta_gga_10au_100Ry_4s2p2d2f.orb',
                                       'Y': 'Y_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Al': 'Al_gga_9au_100Ry_4s4p1d.orb',
                                       'V': 'V_gga_9au_100Ry_4s2p2d1f.orb',
                                       'P': 'P_gga_8au_100Ry_2s2p1d.orb',
                                       'Nb': 'Nb_gga_9au_100Ry_4s2p2d1f.orb',
                                       'F': 'F_gga_7au_100Ry_2s2p1d.orb',
                                       'Re': 'Re_gga_10au_100Ry_4s2p2d1f.orb',
                                       'Mn': 'Mn_gga_9au_100Ry_4s2p2d1f.orb',
                                       'Se': 'Se_gga_8au_100Ry_2s2p1d.orb',
                                       'H': 'H_gga_8au_100Ry_2s1p.orb',
                                       'Zr': 'Zr_gga_9au_100Ry_4s2p2d1f.orb',
                                       'O': 'O_gga_7au_100Ry_2s2p1d.orb',
                                       'Na': 'Na_gga_9au_100Ry_4s2p1d.orb',
                                       'Cl': 'Cl_gga_8au_100Ry_2s2p1d.orb',
                                       'Hf': 'Hf_gga_10au_100Ry_4s2p2d2f.orb'},
                 'basis_dojo': {'Be': 'Be_gga_7au_100Ry_4s1p.orb',
                                   'Fe': 'Fe_gga_8au_100Ry_4s2p2d1f.orb',
                                   'Ca': 'Ca_gga_9au_100Ry_4s2p1d.orb',
                                   'Cl': 'Cl_gga_7au_100Ry_2s2p1d.orb',
                                   'Xe': 'Xe_gga_8au_100Ry_2s2p2d1f.orb',
                                   'As': 'As_gga_7au_100Ry_2s2p1d.orb',
                                   'Br': 'Br_gga_7au_100Ry_2s2p1d.orb',
                                   'Sc': 'Sc_gga_8au_100Ry_4s2p2d1f.orb',
                                   'Tl': 'Tl_gga_7au_100Ry_2s2p2d1f.orb',
                                   'Te': 'Te_gga_7au_100Ry_2s2p2d1f.orb',
                                   'Ru': 'Ru_gga_7au_100Ry_4s2p2d1f.orb',
                                   'Os': 'Os_gga_7au_100Ry_4s2p2d1f.orb',
                                   'Zn': 'Zn_gga_8au_100Ry_4s2p2d1f.orb',
                                   'Ta': 'Ta_gga_8au_100Ry_4s2p2d2f1g.orb',
                                   'Ir': 'Ir_gga_7au_100Ry_4s2p2d1f.orb',
                                   'Ti': 'Ti_gga_8au_100Ry_4s2p2d1f.orb',
                                   'Ne': 'Ne_gga_6au_100Ry_2s2p1d.orb',
                                   'Sn': 'Sn_gga_7au_100Ry_2s2p2d1f.orb',
                                   'H': 'H_gga_6au_100Ry_2s1p.orb',
                                   'Mg': 'Mg_gga_8au_100Ry_4s2p1d.orb',
                                   'Mo': 'Mo_gga_7au_100Ry_4s2p2d1f.orb',
                                   'I': 'I_gga_7au_100Ry_2s2p2d1f.orb',
                                   'Li': 'Li_gga_7au_100Ry_4s1p.orb',
                                   'Cs': 'Cs_gga_10au_100Ry_4s2p1d.orb',
                                   'Sr': 'Sr_gga_9au_100Ry_4s2p1d.orb',
                                   'Co': 'Co_gga_8au_100Ry_4s2p2d1f.orb',
                                   'Si': 'Si_gga_7au_100Ry_2s2p1d.orb',
                                   'Ge': 'Ge_gga_8au_100Ry_2s2p2d1f.orb',
                                   'Nb': 'Nb_gga_8au_100Ry_4s2p2d1f.orb',
                                   'S': 'S_gga_7au_100Ry_2s2p1d.orb',
                                   'K': 'K_gga_9au_100Ry_4s2p1d.orb',
                                   'C': 'C_gga_7au_100Ry_2s2p1d.orb',
                                   'Al': 'Al_gga_7au_100Ry_4s4p1d.orb',
                                   'Cr': 'Cr_gga_8au_100Ry_4s2p2d1f.orb',
                                   'Ar': 'Ar_gga_7au_100Ry_2s2p1d.orb',
                                   'Tc': 'Tc_gga_7au_100Ry_4s2p2d1f.orb',
                                   'Kr': 'Kr_gga_7au_100Ry_2s2p1d.orb',
                                   'Pd': 'Pd_gga_7au_100Ry_4s2p2d1f.orb',
                                   'N': 'N_gga_7au_100Ry_2s2p1d.orb',
                                   'Rb': 'Rb_gga_10au_100Ry_4s2p1d.orb',
                                   'Cu': 'Cu_gga_8au_100Ry_4s2p2d1f.orb',
                                   'W': 'W_gga_8au_100Ry_4s2p2d2f1g.orb',
                                   'Ba': 'Ba_gga_10au_100Ry_4s2p2d1f.orb',
                                   'Cd': 'Cd_gga_7au_100Ry_4s2p2d1f.orb',
                                   'P': 'P_gga_7au_100Ry_2s2p1d.orb',
                                   'Zr': 'Zr_gga_8au_100Ry_4s2p2d1f.orb',
                                   'B': 'B_gga_8au_100Ry_2s2p1d.orb',
                                   'Mn': 'Mn_gga_8au_100Ry_4s2p2d1f.orb',
                                   'Ga': 'Ga_gga_8au_100Ry_2s2p2d1f.orb',
                                   'Hf': 'Hf_gga_7au_100Ry_4s2p2d2f1g.orb',
                                   'In': 'In_gga_7au_100Ry_2s2p2d1f.orb',
                                   'Sb': 'Sb_gga_7au_100Ry_2s2p2d1f.orb',
                                   'Pt': 'Pt_gga_7au_100Ry_4s2p2d1f.orb',
                                   'Au': 'Au_gga_7au_100Ry_4s2p2d1f.orb',
                                   'V': 'V_gga_8au_100Ry_4s2p2d1f.orb',
                                   'He': 'He_gga_6au_100Ry_2s1p.orb',
                                   'Pb': 'Pb_gga_7au_100Ry_2s2p2d1f.orb',
                                   'Hg': 'Hg_gga_9au_100Ry_4s2p2d1f.orb',
                                   'Rh': 'Rh_gga_7au_100Ry_4s2p2d1f.orb',
                                   'F': 'F_gga_7au_100Ry_2s2p1d.orb',
                                   'Na': 'Na_gga_8au_100Ry_4s2p1d.orb',
                                   'Se': 'Se_gga_8au_100Ry_2s2p1d.orb',
                                   'Ag': 'Ag_gga_7au_100Ry_4s2p2d1f.orb',
                                   'O': 'O_gga_7au_100Ry_2s2p1d.orb',
                                   'Y': 'Y_gga_8au_100Ry_4s2p2d1f.orb',
                                   'Ni': 'Ni_gga_8au_100Ry_4s2p2d1f.orb',
                                   'Bi': 'Bi_gga_7au_100Ry_2s2p2d1f.orb',
                                   'Re': 'Re_gga_7au_100Ry_4s2p2d1f.orb'},
    }
    
    return_orbs = {}
    for element in elements: 
        try:
            return_orbs[element] = orb_dict[pp_basis_dict[orb_type]['basis']][element]
        except:
            raise KeyError(f"No orbital for element {element} in {orb_type}")

    return {"result": return_orbs}

