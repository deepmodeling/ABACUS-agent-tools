import os
from typing import Dict, List, Optional, Any, Literal
from pathlib import Path
from itertools import groupby

import numpy as np
from ase.io import read
from ase.vibrations import Vibrations
from ase.calculators.abacus import Abacus, AbacusProfile
from ase.thermochemistry import HarmonicThermo
from ase.io.abacus import read_kpt
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput

from abacusagent.init_mcp import mcp
from abacusagent.modules.util.comm import get_physical_cores, generate_work_path

def set_ase_abacus_calculator(abacus_inputs_path: Path,
                              work_path: Path,
                              extra_input_params: Optional[Dict[str, Any]]) -> Abacus:
    """
    Set Abacus calculator using input files in ABACUS input directory. 
    ABACUS will be executed in pure MPI parallalized mode.
    """
    # Parallel settings
    os.environ["OMP_NUM_THREADS"] = "1"
    profile = AbacusProfile(command=f"mpirun -np {get_physical_cores()} abacus")
    out_directory = os.path.join(work_path, "SCF")

    # Read INPUT, STRU
    input_params = ReadInput(os.path.join(abacus_inputs_path, "INPUT"))
    input_params.update(extra_input_params)
    stru_file = input_params.get('stru_file', "STRU")
    stru = read(os.path.join(abacus_inputs_path, stru_file))
    abacus_stru = AbacusStru.ReadStru(os.path.join(abacus_inputs_path, stru_file))

    # Read KPT
    if 'gamma_only' in input_params.keys():
        kpts = {'gamma_only': input_params['gamma_only']}
    if 'kspacing' in input_params.keys():
        kpts = {'kspacing': input_params['kspacing']}
    else:
        kpt_file = input_params.get('kpt_file', 'KPT')
        kpt_info = read_kpt(os.path.join(abacus_inputs_path, kpt_file))
        # Set kpoint information required by `ase.calculators.calculator.kpts2sizeandoffsets`
        # used by ase-abacus
        kpts = {'size': kpt_info['kpts']}
        if kpt_info['mode'] in ['Gamma']:
            kpts['gamma'] = True

    # Get pp and orb from provided STRU file
    pseudo_dir = Path(abacus_inputs_path).absolute()
    orbital_dir = Path(abacus_inputs_path).absolute()
    pp_list, orb_list = abacus_stru.get_pp(), abacus_stru.get_orb()
    elements = [key for key, _ in groupby(stru.get_chemical_symbols())]
    pp = {element: ppfile for element, ppfile in zip(elements, pp_list)}
    basis = {element: orbfile for element, orbfile in zip(elements, orb_list)}

    input_params['pseudo_dir'] = pseudo_dir
    input_params['orbital_dir'] = orbital_dir

    calc = Abacus(profile=profile,
                  directory=out_directory,
                  pp=pp,
                  basis=basis,
                  kpts=kpts,
                  **input_params)
    
    return calc

def identify_complex_types(complex_array):
    real_part = np.real(complex_array)
    imag_part = np.imag(complex_array)

    is_real = np.isclose(imag_part, 0)
    is_pure_imag = np.isclose(real_part, 0) & ~np.isclose(imag_part, 0)
    is_general = ~is_real & ~is_pure_imag

    return is_real, is_pure_imag, is_general

@mcp.tool()
def abacus_vibration_analysis(abacus_inputs_path: Path,
                              selected_atoms: Optional[List[int]] = None,
                              stepsize: float = 0.01,
                              nfree: Literal[2, 4] = 2,
                              temperature: Optional[float] = 298.15):
    """
    Performing vibrational analysis using finite displacement method.
    This tool function is usually followed by a relax calculation (`calculation` is set to `relax`).
    Args:
        abacus_inputs_path (Path): Path to the ABACUS input files directory.
        selected_atoms (Optional[List[int]]): Indices of atoms included in the vibrational analysis. If this
            parameter are not given, all atoms in the structure will be included.
        stepsize (float): Step size to displace cartesian coordinates of atoms during the vibrational analysis.
            Units in Angstrom. The default value (0.01 Angstrom) is generally OK.
        nfree (int): Number of force calculations performed for each cartesian coordinate components of each 
            included atom. Allowed values are 2 and 4, where 2 represents calculating matrix element of force constant
            matrix using 3-point center difference and need 2 SCF calculations, and 4 means using 5-point center
            difference and need 4 SCF calculations. Generally `nfree=2` is accurate enough.
        temperature (float): Temperature used to calculate thermodynamic quantities. Units in Kelvin.
    Returns:
        A dictionary containing the following keys:
        - 'real_frequencies': List of real frequencies from vibrational analysis. Units in cm^{-1}.
        - 'imaginary_frequencies': Imaginary frequencies will be a string ended with 'i'. Units in cm^{-1}.
        - 'work_path': Path to directory performing vibrational analysis. Containing animation of normal modes 
            with non-zero frequency in ASE traj format and `vib` directory containing collected forces.
        - 'zero_point_energy': Zero-point energy summed over all modes. Units in eV.
        - 'vib_entropy': Vibrational entropy using harmonic approximation. Units in eV/K.
        - 'vib_free_energy': Vibrational Helmholtz free energy using harmonic approximation. Units in eV.
    """
    try:
        work_path = Path(generate_work_path()).absolute()

        input_params = ReadInput(os.path.join(abacus_inputs_path, "INPUT"))
        stru_file = input_params.get('stru_file', "STRU")
        stru = read(os.path.join(abacus_inputs_path, stru_file))
        # Provide extra INPUT parameters necessary for vibration analysis using finite difference
        extra_input_params = {'calculation': 'scf',
                              'cal_force': 1}
        stru.calc = set_ase_abacus_calculator(abacus_inputs_path,
                                              work_path,
                                              extra_input_params)

        if selected_atoms is None:
            selected_atoms = [i for i in range(stru.get_global_number_of_atoms())]

        vib = Vibrations(stru,
                         name=os.path.join(work_path, "vib"),
                         indices=selected_atoms,
                         delta=stepsize,
                         nfree=nfree)
        vib.run()
        vib.summary()
        # Generate list of frequencies in the return value
        frequencies = vib.get_frequencies()
        real_freq_mask, imag_freq_mask, complex_freq_mask = identify_complex_types(frequencies)
        real_freq, imag_freq = np.real(frequencies[real_freq_mask]).tolist(), frequencies[imag_freq_mask].tolist()
        for key, value in enumerate(imag_freq):
            imag_freq[key] = str(value).replace('j', 'i')
        freqs = imag_freq + real_freq

        # Write animations of normal modes in ASE traj format
        vib.write_mode()

        # Thermochemistry calculations
        # Vibrations.get_energies() gets `h \nu` for each mode, which is from the eigenvalues of force constant
        # matrix. The force constant matrix should be a real symmetric matrix mathematically, but due to numerical
        # errors during calculating its matrix element, it will deviate from symmetric matric slightly, and its eigenvalue
        # will have quite small imaginary parts. Magnitude of imaginary parts will decrease as the calculation accuracy
        # increases, and it's safe to use norm of the complex eigenvalue as vibration energy if the calculation is 
        # accurate enough.
        vib_energies = vib.get_energies()
        vib_energies_float = [float(np.linalg.norm(i)) for i in vib_energies]
        zero_point_energy = sum(vib_energies_float) / 2
        thermo = HarmonicThermo(vib_energies, ignore_imag_modes=True)
        entropy = thermo.get_entropy(temperature)
        free_energy = thermo.get_helmholtz_energy(temperature)

        return {'real_frequencies': real_freq,
                'imaginary_frequencies': imag_freq,
                'work_dir': Path(work_path).absolute(),
                'zero_point_energy': float(zero_point_energy),
                'vib_entropy': float(entropy),
                'vib_free_energy': float(free_energy)}
    except Exception as e:
        return {'real_frequencies': None,
                'imaginary_frequencies': None,
                'work_dir': None,
                'zero_point_energy': None,
                'vib_entropy': None,
                'vib_free_energy': None,
                'message': f"Doing vibration analysis failed: {e}"}

