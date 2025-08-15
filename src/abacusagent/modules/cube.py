"""
Functions about cube files. Currently including:
- Electron localization function (ELF)
- Charge density difference
"""
import os
from pathlib import Path
from typing import Literal, Optional, TypedDict, Dict, Any, List, Tuple, Union
from itertools import groupby

from ase.data import chemical_symbols
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteInput

from abacusagent.init_mcp import mcp
from abacusagent.modules.util.comm import run_abacus, generate_work_path, link_abacusjob
from abacusagent.modules.util.cube_manipulator import read_gaussian_cube, axpy, write_gaussian_cube

#@mcp.tool()
def abacus_cal_elf(abacusjob_dir: Path):
    """
    Calculate electron localization function (ELF) using ABACUS.
    
    Args:
        abacusjob_dir (Path): Path to the ABACUS job directory.
    
    Returns:
        Dict[str, Any]: A dictionary containing:
         - work_path: Path to the directory containing ABACUS input files and output files when calculating ELF.
         - elf_file: ELF file path (in .cube file format).
    
    Raises:
        ValueError: If the nspin in INPUT is not 1 or 2.
        FileNotFoundError: If the ELF file is not found in the output directory.
    """
    try:
        work_path = Path(generate_work_path()).absolute()
        link_abacusjob(src=abacusjob_dir, dst=work_path, copy_files=["INPUT"])

        input_params = ReadInput(os.path.join(work_path, "INPUT"))
        if input_params.get('nspin', 1) not in [1, 2]:
            raise ValueError("ELF calculation only supports nspin=1 or nspin=2.")

        input_params['calculation'] = 'scf'
        input_params['out_elf'] = 1
        WriteInput(input_params, os.path.join(work_path, "INPUT"))

        run_abacus(work_path)

        suffix = input_params.get('suffix', 'ABACUS')
        elf_file = os.path.join(work_path, f'OUT.{suffix}/ELF.cube')
        if not os.path.exists(elf_file):
            raise FileNotFoundError(f"ELF file not found in {work_path}")

        return {
            "work_path": Path(work_path).absolute(),
            "elf_file": Path(elf_file).absolute()
        }
    except Exception as e:
        return {
            "work_path": None,
            "elf_file": None,
            'message': f"Calculating electron localization function failed: {e}"
        }

def get_subsys_pp_orb(stru: AbacusStru,
                      subsys_atom_index: List[int]
                      ) -> Tuple[str, str]:
    """
    Get the pseudopotential and orbital files for a subsystem.
    
    Args:
        stru (AbacusStru): The structure of the full system.
        subsys_atom_index (List[int]): Atom indices of the subsystem.
    
    Returns:
        Tuple[str, str, str]: Paths to the pseudopotential and orbital files, and labels of different kinds for the subsystem.
    """
    pp_list, orb_list = stru.get_pp(), stru.get_orb()
    element_indices = [key for key, _ in groupby(stru.get_element())]
    elements = [chemical_symbols[i] for i in element_indices]
    pp_dict, orb_dict = dict(zip(elements, pp_list)), dict(zip(elements, orb_list))

    subsys_elements = [chemical_symbols[stru.get_element()[i]] for i in subsys_atom_index]
    subsys_pp, subsys_orb = [], []
    for element in subsys_elements:
        if pp_dict[element] not in subsys_pp:
            subsys_pp.append(pp_dict[element])
        if orb_dict[element] not in subsys_orb:
            subsys_orb.append(orb_dict[element])
    
    label_list = stru.get_label()
    subsys_label = [label_list[idx] for idx in subsys_atom_index]

    return subsys_pp, subsys_orb, subsys_label

def get_total_charge_density(abacusjob_dir: Path):
    """
    Get charge density for non spin-polarized case and total charge density for spin-polarized case.
    """
    input_params = ReadInput(os.path.join(abacusjob_dir, "INPUT"))
    nspin = input_params.get('nspin', 1)
    chg_file = os.path.join(abacusjob_dir, f"OUT.{input_params.get('suffix', 'ABACUS')}/SPIN1_CHG.cube")
    if nspin == 1:
        chg = read_gaussian_cube(str(Path(chg_file).absolute()))
    elif nspin == 2:
        chg_up = read_gaussian_cube(str(Path(chg_file).absolute()))
        chg_down_file = os.path.join(abacusjob_dir, f"OUT.{input_params.get('suffix', 'ABACUS')}/SPIN2_CHG.cube")
        chg_dn = read_gaussian_cube(str(Path(chg_down_file).absolute()))
        chg = read_gaussian_cube(str(Path(chg_file).absolute()))
        chg['data'] = axpy(chg_up['data'], chg_dn['data'])
    else:
        raise ValueError("Only nspin=1 and nspin=2 are supported now.")
    
    return chg

#@mcp.tool()
def abacus_cal_charge_density_difference(
    abacusjob_dir: Path,
    subsys1_atom_index: Optional[List[int]] = [],
) -> Dict[str, Any]:
    """
    Calculate charge density difference using ABACUS.
    
    Args:
        abacusjob_dir (Path): Path to the ABACUS job directory.
        reference_abacusjob_dir (Optional[Path]): Path to the reference ABACUS job directory. If not provided, 
            the charge density of the current job will be used as the reference.
        suffix (str): Suffix for the output files. Defaults to "ABACUS".
    
    Returns:
        Dict[str, Any]: A dictionary containing:
         - work_path: Path to the directory containing ABACUS input files and output files when calculating charge density difference.
         - charge_density_diff_file: Charge density difference file path (in .cube file format).
    
    Raises:
        FileNotFoundError: If the charge density difference file is not found in the output directory.
    """
    try:
        work_path = Path(generate_work_path()).absolute()
        full_system_jobpath = os.path.join(work_path, "full_system")
        subsys1_jobpath = os.path.join(work_path, 'subsys1')
        subsys2_jobpath = os.path.join(work_path, 'subsys2')
        link_abacusjob(src=abacusjob_dir, dst=full_system_jobpath, copy_files=["INPUT", "STRU"], exclude_directories=True)
        link_abacusjob(src=abacusjob_dir, dst=subsys1_jobpath, copy_files=["INPUT", "STRU"], exclude_directories=True)
        link_abacusjob(src=abacusjob_dir, dst=subsys2_jobpath, copy_files=["INPUT", "STRU"], exclude_directories=True)

        input_params = ReadInput(os.path.join(full_system_jobpath, "INPUT"))
        full_system_stru_file = os.path.join(full_system_jobpath, input_params.get('stru_file', 'STRU'))
        full_system_stru = AbacusStru.ReadStru(full_system_stru_file)

        # Prepare labels, coordinates, pp and orbital settings needed to generate STRU file for every subsystems
        subsys2_atom_index = [i for i in range(full_system_stru.get_natoms()) if i not in subsys1_atom_index]
        if len(subsys1_atom_index) is None:
            raise ValueError("Subsystem 1 have no atoms! Aborting calculating charge density difference")
        if len(subsys2_atom_index) is None:
            raise ValueError("Subsystem 2 have no atoms! Aborting calculating charge density difference")

        subsys1_stru_file = os.path.join(work_path, f"subsys1/{input_params.get('stru_file', 'STRU')}")
        subsys2_stru_file = os.path.join(work_path, f"subsys2/{input_params.get('stru_file', 'STRU')}")
        subsys1_pp, subsys1_orb, subsys1_label = get_subsys_pp_orb(full_system_stru, subsys1_atom_index)
        subsys2_pp, subsys2_orb, subsys2_label = get_subsys_pp_orb(full_system_stru, subsys2_atom_index)

        subsys1_coord, subsys2_coord = [], []
        full_system_stru_coord = full_system_stru.get_coord()
        for i in range(full_system_stru.get_natoms()):
            if i in subsys1_atom_index:
                subsys1_coord.append(full_system_stru_coord[i])
            elif i in subsys2_atom_index:
                subsys2_coord.append(full_system_stru_coord[i])
            else:
                raise ValueError(f"Atom {i} does not belong to neither subsystem1 nor subsystem2")

        subsys1_stru = AbacusStru(label=subsys1_label,
                                  cell=full_system_stru.get_cell(),
                                  coord=subsys1_coord,
                                  lattice_constant=full_system_stru.get_stru()['lat'],
                                  pp=subsys1_pp,
                                  orb=subsys1_orb,
                                  cartesian=True)
        subsys1_stru.write(subsys1_stru_file)

        subsys2_stru = AbacusStru(label=subsys2_label,
                                  cell=full_system_stru.get_cell(),
                                  coord=subsys2_coord,
                                  lattice_constant=full_system_stru.get_stru()['lat'],
                                  pp=subsys2_pp,
                                  orb=subsys2_orb,
                                  cartesian=True)
        subsys2_stru.write(subsys2_stru_file)

        # Modify INPUT file to output cube file needed for calculating charge density difference
        input_params['calculation'] = 'scf'
        input_params['out_chg'] = '1'

        WriteInput(input_params, os.path.join(full_system_jobpath, 'INPUT'))
        run_abacus(full_system_jobpath)
        WriteInput(input_params, os.path.join(subsys1_jobpath, 'INPUT'))
        run_abacus(subsys1_jobpath)
        WriteInput(input_params, os.path.join(subsys2_jobpath, 'INPUT'))
        run_abacus(subsys2_jobpath)

        # Generate cube file containing charge density difference
        full_system_chgfile = os.path.join(full_system_jobpath, f"OUT.{input_params.get('suffix', 'ABACUS')}/SPIN1_CHG.cube")

        full_system_chg = get_total_charge_density(full_system_jobpath)
        subsys1_chg = get_total_charge_density(subsys1_jobpath)
        subsys2_chg = get_total_charge_density(subsys2_jobpath)

        sum_subsys_chg = read_gaussian_cube(full_system_chgfile)  # Fetch the data structure. The volumeric data will be replaced later
        chg_density_difference = read_gaussian_cube(full_system_chgfile) # Fetch the data structure. The volumeric data will be replaced later
        sum_subsys_chg['data'] = axpy(subsys1_chg['data'], subsys2_chg['data'])
        chg_density_difference['data'] = axpy(full_system_chg['data'], sum_subsys_chg['data'], alpha=1.0, beta=-1.0)

        chg_dens_diff_cube_file = Path(os.path.join(work_path, 'chg_density_diff.cube')).absolute()
        write_gaussian_cube(chg_density_difference, chg_dens_diff_cube_file)

        return {'work_path': Path(work_path).absolute(),
                'charge_density_difference_cube_file': chg_dens_diff_cube_file}
    except Exception as e:
        return {'work_path': None,
                'charge_density_difference_cube_file': None,
                'message': f'Calculaing charge density difference failed: {e}'}

#@mcp.tool()
def abacus_cal_spin_density(
    abacusjob_dir: Path
) -> Dict[str, Any]:
    """
    Calculate the spin density for collinear spin-polarized system (nspin=2).

    Args:
        abacusjob_dir (Path): Path to the ABACUS job directory.
    
    Returns:
        A dictionary containing the following keys:
        - work_path (Path): Path to the ABACUS job directory calculating spin density.
        - spin_density (Path): Path to the cube file containing the spin density.
    
    Raises:
        ValueError: If nspin in INPUT file is not 2.
    """
    try:
        work_path = Path(generate_work_path()).absolute()
        link_abacusjob(src=abacusjob_dir,dst=work_path,copy_files=["INPUT", "STRU"], exclude_directories=True)
        input_params = ReadInput(os.path.join(work_path, 'INPUT'))
        if input_params.get('nspin', 1) not in [2]:
            raise ValueError('Only collinear spin-polarized calculation is supported for calculating spin density')

        input_params['calculation'] = 'scf'
        input_params['out_chg'] = '1'
        WriteInput(input_params, os.path.join(work_path, 'INPUT'))

        run_abacus(work_path)
        chg_up_file = os.path.join(work_path, f"OUT.{input_params.get('suffix', 'ABACUS')}/SPIN1_CHG.cube")
        chg_dn_file = os.path.join(work_path, f"OUT.{input_params.get('suffix', 'ABACUS')}/SPIN2_CHG.cube")

        chg_up, chg_dn = read_gaussian_cube(chg_up_file), read_gaussian_cube(chg_dn_file)
        spin_density = read_gaussian_cube(chg_up_file)
        spin_density['data'] = axpy(chg_up['data'], chg_dn['data'], alpha=1.0, beta=-1.0)

        spin_density_file = os.path.join(work_path, 'spin_density.cube')
        write_gaussian_cube(spin_density, spin_density_file)

        return {'work_path': Path(work_path).absolute(),
                'spin_density': Path(spin_density_file).absolute()}
    except Exception as e:
        return {'work_path': None,
                'spin_density': None,
                'message': f"Calculating spin density failed: {e}"}
