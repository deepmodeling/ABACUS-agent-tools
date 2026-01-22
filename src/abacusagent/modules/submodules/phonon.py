import os
from typing import Dict, List, Optional, Any, Literal, Union
from pathlib import Path

import numpy as np
import phonopy
from phonopy import Phonopy
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath, get_band_qpoints_and_path_connections
from abacustest.lib_prepare.abacus import ReadInput, WriteInput, AbacusStru
from abacustest.lib_model.comm import check_abacus_inputs

from abacusagent.init_mcp import mcp
from abacusagent.constant import THZ_TO_K
from abacusagent.modules.util.comm import run_abacus, generate_work_path, link_abacusjob, collect_metrics


def get_ph_atoms(stru: AbacusStru):
    # Get PhonopyAtoms instance
    return PhonopyAtoms(
        symbols=stru.get_element(number=False),
        cell=stru.get_cell(),
        scaled_positions=stru.get_coord(direct=True),
        magnetic_moments=stru.get_atommag()
    )

def initialize_phonopy_calc(stru: AbacusStru, supercell: Optional[List[int]] = None, min_supercell_length: float = 10.0):
    """
    Initialize phonopy instance
    """
    ph_atoms = get_ph_atoms(stru)

    # Determine supercell if not provided
    if supercell is None:
        a, b, c = stru.to_ase().get_cell().lengths()
        supercell = [int(np.ceil(min_supercell_length / a)),
                     int(np.ceil(min_supercell_length / b)),
                     int(np.ceil(min_supercell_length / c))]

    phonon = Phonopy(ph_atoms, supercell_matrix=supercell)

    return phonon

def prepare_phonon_dispersion(work_path: Path,
                              abacus_inputs_dir: Path,
                              supercell: Optional[List[int]] = None,
                              displacement_stepsize: float = 0.01,
                              min_supercell_length: float = 10.0):
    """
    Prepare the job directories for each displacement of phonon calculation.
    """
    input_params = ReadInput(os.path.join(abacus_inputs_dir, "INPUT"))
    stru_file = input_params.get('stru_file', "STRU")
    stru = AbacusStru.ReadStru(os.path.join(abacus_inputs_dir, stru_file))
    # Provide extra INPUT parameters necessary for calculating phonon dispersion
    extra_input_params = {'calculation': 'scf',
                          'cal_force': 1}
    if input_params.get('scf_thr', 1e-7) > 1e-7:
        extra_input_params['scf_thr'] = 1e-7
    for param_name, param_value in extra_input_params.items():
        input_params[param_name] = param_value


    phonon = initialize_phonopy_calc(stru, supercell, min_supercell_length)

    phonon.generate_displacements(distance=displacement_stepsize)
    print("Generated {} supercell structures with displacements.".format(len(phonon.supercells_with_displacements)) +
          " Doing SCF calculations for each supercell structure...")
    
    stru_supercell = stru.supercell(supercell)
    structure_index = 1
    displaced_job_dirs = []
    for sc in phonon.supercells_with_displacements:
        stru_supercell.set_cell(sc.cell, bohr=False)
        stru_supercell.set_coord(sc.positions, bohr=False)
        dir_name = os.path.join(work_path, f"disp-{structure_index}")
        os.makedirs(dir_name)
        link_abacusjob(abacus_inputs_dir, dir_name)
        stru_supercell.write(os.path.join(dir_name, stru_file))
        WriteInput(input_params, os.path.join(dir_name, "INPUT"))
        displaced_job_dirs.append(dir_name)
        structure_index += 1
    
    return displaced_job_dirs

def postprocess_phonon_dispersion(
    work_path: Path,
    abacus_inputs_dir: Path,
    displaced_job_dirs: List[Path],
    displacement_stepsize: float = 0.01,
    supercell: Optional[List[int]] = None,
    min_supercell_length: float = 10.0,
    temperature: Optional[float] = 298.15,
    qpath: Optional[Union[List[str], List[List[str]]]] = None,
    high_symm_points: Optional[Dict[str, List[float]]] = None
):
    # Postprocess ABACUS calculation results and obtain phonon dispersion results and related quantities
    
    input_params = ReadInput(os.path.join(abacus_inputs_dir, "INPUT"))
    stru_file = input_params.get('stru_file', "STRU")
    stru = AbacusStru.ReadStru(os.path.join(abacus_inputs_dir, stru_file))
    phonon = initialize_phonopy_calc(stru, supercell, min_supercell_length)
    phonon.generate_displacements(distance=displacement_stepsize) # Regenerate used structures

    force_sets = []
    for job_dir in displaced_job_dirs:
        metrics = collect_metrics(abacusjob = job_dir,
                                  metrics_names=['force', 'normal_end', 'converge'])
        if metrics['normal_end'] is not True:
            print(f"ABACUS calculation in {job_dir} didn't end normally")
        elif metrics['converge'] is not True:
            print(f"ABACUS calculation in {job_dir} didn't reached SCF convergence")
        else:
            pass

        force_sets.append(np.array(metrics['force']).reshape(-1, 3))

    phonon.forces = force_sets
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()

    phonon.run_mesh([20, 20, 20], with_eigenvectors=True, is_mesh_symmetry=False)
    phonon.run_thermal_properties(temperatures=[temperature])
    thermal = phonon.get_thermal_properties_dict()

    comm_q = get_commensurate_points(phonon.supercell_matrix)
    freqs = np.array([phonon.get_frequencies(q) for q in comm_q])
    
    # Calculate phonon DOS
    phonon.run_total_dos()

    # Calculate phonon dispersion
    if qpath is not None and high_symm_points is not None: # Use provided qpath
        # Convert provided qpath to phonopy format used by get_band_qpoints_and_path_connections
        # Labels of Gamma point are processed during this process
        qpath_phonopy = []
        labels = []
        if all(isinstance(item, str) for item in qpath): # A whole continous qpath:
            path = []
            for point in qpath:
                path.append(high_symm_points[point])
                if point == 'G' or point.lower() == 'gamma':
                    labels.append(r"$\Gamma$")
                else:
                    labels.append(point)
            qpath_phonopy.append(path)
        elif all(isinstance(item, list) for item in qpath): # Q-points line with uncontinous points
            for sub_path in qpath:
                path = []
                for point in sub_path:
                    path.append(high_symm_points[point])
                    if point == 'G' or point.lower() == 'gamma':
                        labels.append(r"$\Gamma$")
                    else:
                        labels.append(point)
                qpath_phonopy.append(path)

        qpoints, connections = get_band_qpoints_and_path_connections(qpath_phonopy, npoints=101)
        phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
    else:
        # Use automatically generated qpath
        ph_atoms = get_ph_atoms(stru)
        bands, labels, path_connections = get_band_qpoints_by_seekpath(ph_atoms, npoints=101, is_const_interval=True)
        phonon.run_band_structure(bands, path_connections=path_connections, labels=labels)
    
    # Plot phonon dispersion and DOS
    import matplotlib.pyplot as plt

    band_dos_plot_path = os.path.join(work_path, "phonon_dispersion_dos.png")
    band_dos_plot = phonon.plot_band_structure_and_dos()
    axes = plt.gcf().get_axes()
    axes[0].set_ylabel("Frequency (THz)") # Set ylabel of phonon dispersion plot
    band_dos_plot.savefig(band_dos_plot_path, dpi=300)

    return band_dos_plot_path, thermal, freqs

def abacus_phonon_dispersion(
    abacus_inputs_dir: Path,
    supercell: Optional[List[int]] = None,
    displacement_stepsize: float = 0.01,
    temperature: Optional[float] = 298.15,
    min_supercell_length: float = 10.0,
    qpath: Optional[Union[List[str], List[List[str]]]] = None,
    high_symm_points: Optional[Dict[str, List[float]]] = None
):
    """
    Calculate phonon dispersion with finite-difference method using Phonopy with ABACUS as the calculator. 
    This tool function is usually followed by a cell-relax calculation (`calculation` is set to `cell-relax`). 
    Args:
        abacus_inputs_dir (Path): Path to the directory containing ABACUS input files.
        supercell (List[int], optional): Supercell matrix for phonon calculations. If default value None are used,
            the supercell matrix will be determined by how large a supercell can have a length of lattice vector
            along all 3 directions larger than 10.0 Angstrom.
        displacement_stepsize (float, optional): Displacement step size for finite difference. Defaults to 0.01 Angstrom.
        temperature (float, optional): Temperature in Kelvin for thermal properties. Defaults to 298.15. Units in Kelvin.
        min_supercell_length (float): If supercell is not provided, the generated supercell will have a length of lattice vector
            along all 3 directions larger than min_supercell_length. Defaults to 10.0 Angstrom. Units in Angstrom.
        qpath (Tuple[List[str], List[List[str]]]): 
            A list of name of high symmetry points in the phonon dispersion path. Non-continuous line of high symmetry points are stored as seperate lists.
            For example, ['G', 'M', 'K', 'G'] and [['G', 'X', 'P', 'N', 'M', 'S'], ['S_0', 'G', R']] are both acceptable inputs.
            Default is None. If None, will use automatically generated q-point path.
            `kpath` must be used with `high_symm_points` to take effect.
        high_symm_points: A dictionary containing high symmetry points and their coordinates in the band path. All points in `qpath` should be included.
            For example, {'G': [0, 0, 0], 'M': [0.5, 0.0, 0.0], 'K': [0.33333333, 0.33333333, 0.0], 'G': [0, 0, 0]}.
            Default is None. If None, will use automatically generated high symmetry points.
    Returns:
        A dictionary containing:
            - phonon_work_path: Path to the directory containing phonon calculation results.
            - band_dos_plot: Path to the phonon dispersion plot.
            - entropy: Entropy at the specified temperature.
            - free_energy: Free energy at the specified temperature.
            - heat_capacity: Heat capacity at the specified temperature.
            - max_frequency_THz: Maximum phonon frequency in THz.
            - max_frequency_K: Maximum phonon frequency in Kelvin.
    """
    try:
        is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
        if not is_valid:
            raise RuntimeError(f"Invalid ABACUS input files: {msg}")
        
        work_path = Path(generate_work_path()).absolute()

        displaced_job_dirs = prepare_phonon_dispersion(work_path,
                                                       abacus_inputs_dir,
                                                       supercell=supercell,
                                                       displacement_stepsize=displacement_stepsize,
                                                       min_supercell_length=min_supercell_length)

        run_abacus(displaced_job_dirs)

        band_dos_plot_path, thermal, freqs = postprocess_phonon_dispersion(work_path,
                                                                           abacus_inputs_dir,
                                                                           displaced_job_dirs,
                                                                           displacement_stepsize,
                                                                           supercell,
                                                                           min_supercell_length,
                                                                           temperature,
                                                                           qpath=qpath,
                                                                           high_symm_points=high_symm_points)

        return {
            "phonon_work_path": Path(work_path).absolute(),
            "band_dos_plot": Path(band_dos_plot_path).absolute(),
            "entropy": float(thermal['entropy'][0]),
            "free_energy": float(thermal['free_energy'][0]),
            "heat_capacity": float(thermal['heat_capacity'][0]),
            "max_frequency_THz": float(np.max(freqs)),
            "max_frequency_K": float(np.max(freqs) * THZ_TO_K),
        }
    except Exception as e:
        return {"message": f"Calculating phonon spectrum failed: {e}"}
