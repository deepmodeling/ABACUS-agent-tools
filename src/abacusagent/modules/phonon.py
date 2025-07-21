import os
from typing import Dict, List, Optional, Any, Literal
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from phonopy import Phonopy
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.structure.atoms import PhonopyAtoms
from abacustest.lib_prepare.abacus import ReadInput

from abacusagent.init_mcp import mcp
from abacusagent.modules.util.comm import generate_work_path
from abacusagent.modules.vibration import set_ase_abacus_calculator

THz_TO_K = 47.9924

# Modified from calculate_phonon in https://github.com/deepmodeling/AI4S-agent-tools/blob/main/servers/DPACalculator/server.py
@mcp.tool()
def abacus_phonon_dispersion(
    abacus_inputs_path: Path,
    supercell: Optional[List[int]] = None,
    displacement_stepsize: float = 0.01,
    temperature: Optional[float] = 298.15,
):
    """
    Calculate phonon dispersion using Phonopy with ABACUS as the calculator.
    Args:
        abacus_inputs_path (Path): Path to the directory containing ABACUS input files.
        supercell (List[int], optional): Supercell matrix for phonon calculations. If default value None are used,
            the supercell matrix will be determined by how large a supercell can have a length of lattice vector
            along all 3 directions larger than 10.0 Angstrom.
        displacement_stepsize (float, optional): Displacement step size for finite difference. Defaults to 0.01 Angstrom.
        temperature (float, optional): Temperature in Kelvin for thermal properties. Defaults to 298.15. Units in Kelvin.
    Returns:
        A dictionary containing:
            - phonon_work_path: Path to the directory containing phonon calculation results.
            - band_plot: Path to the phonon dispersion plot.
            - entropy: Entropy at the specified temperature.
            - free_energy: Free energy at the specified temperature.
            - heat_capacity: Heat capacity at the specified temperature.
            - max_frequency_THz: Maximum phonon frequency in THz.
            - max_frequency_K: Maximum phonon frequency in Kelvin.
    """
    work_path = Path(generate_work_path()).absolute()
    
    input_params = ReadInput(os.path.join(abacus_inputs_path, "INPUT"))
    stru_file = input_params.get('stru_file', "STRU")
    stru = read(os.path.join(abacus_inputs_path, stru_file))
    # Provide extra INPUT parameters necessary for calculating phonon dispersion
    extra_input_params = {'calculation': 'scf',
                          'cal_force': 1,
                          'out_chg': 1,
                          'init_chg': 'auto'}
    if input_params.get('scf_thr', 1e-7) > 1e-7:
        extra_input_params['scf_thr'] = 1e-7
    calc = set_ase_abacus_calculator(abacus_inputs_path,
                                     work_path,
                                     extra_input_params)
    
    ph_atoms = PhonopyAtoms(
        symbols=stru.get_chemical_symbols(),
        cell=stru.get_cell(),
        scaled_positions=stru.get_scaled_positions(),
        magnetic_moments=stru.get_initial_magnetic_moments()
    )

    # Determine supercell if not provided
    if supercell is None:
        min_supercell_length = 6.0 # In Angstrom. A temporary value, should be verified in detail
        a, b, c = stru.get_cell().lengths()
        supercell = [int(np.ceil(min_supercell_length / a)),
                     int(np.ceil(min_supercell_length / b)),
                     int(np.ceil(min_supercell_length / c))]

    phonon = Phonopy(ph_atoms, supercell_matrix=supercell)
    phonon.generate_displacements(distance=displacement_stepsize)
    
    force_sets = []
    for sc in phonon.supercells_with_displacements:
        sc_atoms = Atoms(
            cell=sc.cell,
            symbols=sc.symbols,
            scaled_positions=sc.scaled_positions,
            magmoms=sc.magnetic_moments,
            pbc=True,
        )
        sc_atoms.calc = calc
        force = sc_atoms.get_forces()
        force_sets.append(force - np.mean(force, axis=0))

    phonon.forces = force_sets
    phonon.produce_force_constants()

    phonon.run_mesh([10, 10, 10])
    phonon.run_thermal_properties(temperatures=[temperature])
    thermal = phonon.get_thermal_properties_dict()

    comm_q = get_commensurate_points(phonon.supercell_matrix)
    freqs = np.array([phonon.get_frequencies(q) for q in comm_q])

    plot_path = os.path.join(work_path, "phonon_dispersion.png")
    yaml_path = os.path.join(work_path, "phonon_dispersion.yaml")
    phonon.auto_band_structure(
        npoints=101,
        write_yaml=True,
        filename=str(yaml_path)
    )

    plot = phonon.plot_band_structure()
    plot.savefig(plot_path, dpi=300)

    return {
        "phonon_work_path": Path(work_path).absolute(),
        "band_plot": Path(plot_path).absolute(),
        "entropy": float(thermal['entropy'][0]),
        "free_energy": float(thermal['free_energy'][0]),
        "heat_capacity": float(thermal['heat_capacity'][0]),
        "max_frequency_THz": float(np.max(freqs)),
        "max_frequency_K": float(np.max(freqs) * THz_TO_K),
    }
