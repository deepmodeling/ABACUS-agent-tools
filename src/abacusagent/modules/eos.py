import os
from pathlib import Path
from typing import Literal, Optional, TypedDict, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteInput
from abacustest.lib_model.comm_eos import eos_fit

from abacusagent.init_mcp import mcp
from abacusagent.modules.abacus import abacus_modify_input, abacus_modify_stru, abacus_collect_data
from abacusagent.modules.util.comm import run_abacus, link_abacusjob, generate_work_path

def is_cubic(cell: List[List[float]]) -> bool:
    """
    Check if the cell is cubic.
    
    Args:
        cell (List[List[float]]): The cell vectors.

    Returns:
        bool: True if the cell is cubic, False otherwise.
    """
    a, b, c = np.array(cell[0]), np.array(cell[1]), np.array(cell[2])
    len_a, len_b, len_c = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
    alpha = np.arccos(np.dot(b, c) / (len_b * len_c))
    beta = np.arccos(np.dot(a, c) / (len_a * len_c))
    gamma = np.arccos(np.dot(a, b) / (len_a * len_b))
    if np.isclose(len_a, len_b) and np.isclose(len_b, len_c):
        if np.isclose(alpha, np.pi / 2) and np.isclose(beta, np.pi / 2) and np.isclose(gamma, np.pi / 2):
            return True
        else:
            return False
    else:
        return False

@mcp.tool()
def abacus_eos(
    abacus_inputs_path: Path,
    stru_scale_number: int = 3,
    stru_scale_type: Literal['percentage', 'angstrom'] = 'percentage',
    scale_stepsize: float = 0.02
):
    """
    Use Birch-Murnaghan equation of state (EOS) to calculate the EOS data.

    Args:
        abacus_inputs_path (Path): Path to the ABACUS inputs directory.
        stru_scale_number (int): Number of structures to generate for EOS calculation.
        stru_scale_type (Literal['percentage', 'angstrom']): Type of scaling for structures.
        scale_stepsize (float): Step size for scaling.
            - 'percentage' means percentage of the original cell size. Default is 0.02, which means 2% of the original cell size.
            - 'angstrom' means absolute angstrom value. The typical stepsize 0.1 angstrom is recommended for most system.

    Returns:
        Dict[str, Any]: A dictionary containing EOS calculation results:
            - "eos_work_path" (Path): Working directory for the EOS calculation.
            - "optimal_stru_abacusjob_dir" (Path): ABACUS input files directory with the lowest energy structure.
            - "eos_fig_path" (Path): Path to the EOS fitting plot (energy vs. volume).
            - "E0" (float): Minimum energy (in eV) from the EOS fit.
            - "V0" (float): Equilibrium volume (in Å³) corresponding to E0.
            - "B0" (float): Bulk modulus (in GPa) at equilibrium volume.
            - "B0_deriv" (float): Pressure derivative of the bulk modulus.
    """
    work_path = Path(generate_work_path()).absolute()

    input_params = ReadInput(os.path.join(abacus_inputs_path, "INPUT"))
    input_stru_file = input_params.get('stru_file', 'STRU')
    input_stru = AbacusStru.ReadStru(os.path.join(abacus_inputs_path, input_stru_file))
    if is_cubic(input_stru.get_cell()) is False:
        raise ValueError("The structure is not cubic. Implemented EOS calculation requires a cubic structure.")
    
    # Generated lattice parameters for EOS calculation
    original_cell_param = input_stru.get_cell()[0][0]  # Assuming cubic structure, take one cell parameter
    if stru_scale_type == 'percentage':
        scales = [1 + i * scale_stepsize for i in range(-stru_scale_number, stru_scale_number + 1)]
    elif stru_scale_type == 'angstrom':
        scales = [1 + i * scale_stepsize / original_cell_param for i in range(-stru_scale_number, stru_scale_number + 1)]
    else:
        raise ValueError("Invalid stru_scale_type. Use 'percentage' or 'angstrom'.")
    
    scaled_lat_params = [original_cell_param * scale for scale in scales]
    
    output = abacus_modify_input(abacus_inputs_path, extra_input={'calculation': 'scf'})

    scale_cell_job_dirs = []
    for i in range(len(scales)):
        dir_name = Path(os.path.join(work_path, f"scale_cell_{i}")).absolute()
        os.makedirs(dir_name, exist_ok=True)
        scale_cell_job_dirs.append(dir_name)

        link_abacusjob(
            src=abacus_inputs_path,
            dst=Path(dir_name).absolute(),
            copy_files=["INPUT", input_stru_file],
            exclude=["OUT.*", "*.log", "*.out", "*.json", "log"],
            exclude_directories=True
        )

        new_cell = (np.array(input_stru.get_cell()) * scales[i]).tolist()
        output = abacus_modify_stru(dir_name, cell=new_cell, coord_change_type='scale')
    
    run_abacus(scale_cell_job_dirs)

    energies = []
    for i, job_dir in enumerate(scale_cell_job_dirs):
        metrics = abacus_collect_data(job_dir)['collected_metrics']
        if metrics['normal_end'] is not True or metrics['converge'] is not True:
            raise RuntimeError(f"Job {i} did not end normally or did not converge. Please check the job directory: {job_dir}")
        energies.append(metrics['energy'])
    
    volumes = [x**3 for x in scaled_lat_params]
    V0, E0, fit_volume, fit_energy, B0, B0_deriv, residual0 = eos_fit(volumes, energies)
    lat_params = np.cbrt(np.array(fit_volume))

    plt.figure(figsize=(8, 6))
    plt.plot(lat_params, fit_energy, label='Fitted Birch-Murnaghan EOS', color='red')
    plt.scatter(scaled_lat_params, energies, label='Calculated Energies', color='blue')
    plt.xlabel('Lattice Parameter (Angstrom)')
    plt.ylabel('Energy (eV)')
    plt.title('Birch-Murnaghan EOS Fit')
    plt.legend()
    plt.grid()
    plt.savefig('birch_murnaghan_eos_fit.png', dpi=300)
    fig_path = Path('birch_murnaghan_eos_fit.png').absolute()

    optimal_stru_abacusjob_dir = Path(os.path.join(work_path, "optimal_stru_abacusjob_dir")).absolute()
    link_abacusjob(
            src=abacus_inputs_path,
            dst=optimal_stru_abacusjob_dir,
            copy_files=["INPUT", input_stru_file],
            exclude=["OUT.*", "*.log", "*.out", "*.json", "log"],
            exclude_directories=True
        )
    optimal_lat_param = V0 ** (1.0 / 3)
    optimal_cell = (np.array(input_stru.get_cell()) * optimal_lat_param / original_cell_param).tolist()
    output = abacus_modify_stru(optimal_stru_abacusjob_dir,
                                cell = optimal_cell,
                                coord_change_type = 'scale')

    return {
        "eos_work_path": work_path.absolute(),
        "optimal_stru_abacusjob_dir": Path(optimal_stru_abacusjob_dir).absolute(),
        "eos_fig_path": fig_path,
        "E0": E0,
        "V0": V0,
        "B0": B0,
        "B0_deriv": B0_deriv, }

