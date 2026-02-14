import os
import shutil
from pathlib import Path
from typing import Literal, Optional, TypedDict, Dict, Any, List, Union
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteInput, WriteKpt

from abacusagent.modules.util.comm import run_abacus, run_pyatb, collect_metrics
from abacusagent.modules.util.pyatb import property_calculation_scf

def read_band_data(band_file: Path, efermi: float):
    """
    Read data in band file.
    Args:
        band_file (Path): Absolute path to the band file.
    Returns:
        A dictionary containing band data.
    Raises:
        RuntimeError: If read band data from BANDS_1.dat or BANDS_2.dat failed
    """
    bands, kline = [], []
    try:
        with open(band_file) as fin:
            for lines in fin:
                words = lines.split()
                nbands = len(words) - 2
                kline.append(float(words[1]))
                if len(bands) == 0:
                    for _ in range(nbands):
                        bands.append([])
            
                for i in range(nbands):
                    bands[i].append(float(words[i+2]) - efermi)
    except Exception as e:
        raise RuntimeError(f"Read data from {band_file} failed")
    
    return bands, kline, nbands
    
def split_array(array: List[Any], splits: List[int]):
    """
    Split band and kline by incontinuous points
    """
    splited_array = []
    start = 0

    for split_point in splits:
        splited_array.append(array[start:split_point])
        start = split_point
    
    splited_array.append(array[start:])
    return splited_array

def read_high_symmetry_labels(abacusjob_dir: Path):
    """
    Read high symmetry labels from KPT file
    """
    high_symm_labels = []
    band_point_nums = []
    band_point_num = 0
    with open(os.path.join(abacusjob_dir, "KPT")) as fin:
        for lines in fin:
            words = lines.split()
            if len(words) > 2:
                if words[-2] == '#':  # "# G" form
                    if words[-1] == 'G':
                        high_symm_labels.append(r'$\Gamma$')
                    else:
                        high_symm_labels.append(words[-1])
                    band_point_nums.append(band_point_num)
                    band_point_num += int(words[-3])
                elif words[-1].startswith("#"):  # "#G" form
                    label = words[-1][1:].split()[0]
                    if words[-1] == 'G':
                        high_symm_labels.append(r'$\Gamma$')
                    else:
                        high_symm_labels.append(label)
                    band_point_nums.append(band_point_num)
                    band_point_num += int(words[-2])
    
    return high_symm_labels, band_point_nums

def process_band_data(abacusjob_dir: Path, 
                      nspin: Literal[1, 2], 
                      efermi: float, 
                      kline: List[float],
                      bands: List[List[float]],  
                      bands_dw: Optional[List[List[float]]] = None):
    """
    Process band data, including properly process incontinous points and label high symmetry points
    """
    high_symm_labels, band_point_nums = read_high_symmetry_labels(abacusjob_dir)
    
    # Reduce extra kline length between incontinuous points
    modify_indexes = []
    for i in range(len(band_point_nums) - 1):
        if band_point_nums[i+1] - band_point_nums[i] == 1:
            reduce_length = kline[band_point_nums[i+1]] - kline[band_point_nums[i]]
            for j in range(band_point_nums[i+1], len(kline)):
                kline[j] -= reduce_length

            modify_indexes.append(i)
    
    # Modify incontinuous point labels
    high_symm_labels_old = high_symm_labels.copy()
    band_point_nums_old = band_point_nums.copy()
    high_symm_labels = []
    band_point_nums = []
    for i in range(len(high_symm_labels_old)):
        if i in modify_indexes:
            modified_tick = high_symm_labels_old[i] + "|" + high_symm_labels_old[i+1]
            high_symm_labels.append(modified_tick)
            band_point_nums.append(band_point_nums_old[i])
        elif i-1 in modify_indexes:
            pass
        else:
            band_point_nums.append(band_point_nums_old[i])
            high_symm_labels.append(high_symm_labels_old[i])
    
    # Split incontinuous bands to list of continous bands
    band_split_points = [band_point_nums_old[x]+1 for x in modify_indexes]
    kline_splited = split_array(kline, band_split_points)
    bands_splited = []
    for i in range(len(bands)):
        bands_splited.append(split_array(bands[i], band_split_points))
    if nspin == 2:
        bands_dw_splited = []
        for i in range(len(bands_dw)):
            bands_dw_splited.append(split_array(bands_dw[i], band_split_points))

    high_symm_poses = [kline[i] for i in band_point_nums]
    
    if nspin == 1:
        return high_symm_labels, high_symm_poses, kline_splited, bands_splited
    else:
        return high_symm_labels, high_symm_poses, kline_splited, bands_splited, bands_dw_splited

def abacus_plot_band_nscf(abacusjob_dir: Path,
                          energy_min: float = -10,
                          energy_max: float = 10
) -> Dict[str, Any]:
    """
    Plot band after ABACUS SCF and NSCF calculation.
    Args:
        abacusjob_dir (str): Absolute path to the ABACUS calculation directory.
        energy_min (float): Lower bound of $E - E_F$ in the plotted band.
        energy_max (float): Upper bound of $E - E_F$ in the plotted band.
    Returns:
        A dictionary containing band gap of the system and path to the plotted band.
    Raises:
        NotImplementedError: If band plot for an nspin=4 calculation is requested
        RuntimeError: If read band data from BANDS_1.dat or BANDS_2.dat failed
    """
    import matplotlib.pyplot as plt

    input_args = ReadInput(os.path.join(abacusjob_dir, "INPUT"))
    suffix = input_args.get('suffix', 'ABACUS')
    nspin = input_args.get('nspin', 1)
    if nspin not in (1, 2):
        raise NotImplementedError("Band plot for nspin=4 is not supported yet")
    
    metrics = collect_metrics(abacusjob_dir, ['efermi', 'nelec', 'band_gap'])
    efermi, band_gap = metrics['efermi'], float(metrics['band_gap'])
    band_file = os.path.join(abacusjob_dir, f"OUT.{suffix}/BANDS_1.dat")
    if nspin == 2:
        band_file_dw = os.path.join(abacusjob_dir, f"OUT.{suffix}/BANDS_2.dat")
    
    # Read band data
    bands, kline, nbands = read_band_data(band_file, efermi)
    if nspin == 2:
        bands_dw, _, _ = read_band_data(band_file_dw, efermi)
    
    # Process band data
    if nspin == 1:
        high_symm_labels, high_symm_poses, kline_splited, bands_splited = \
            process_band_data(abacusjob_dir, nspin, efermi, kline, bands)
    else:
        high_symm_labels, high_symm_poses, kline_splited, bands_splited, bands_dw_splited = \
            process_band_data(abacusjob_dir, nspin, efermi, kline, bands, bands_dw)
    
    # Final band plot
    for i in range(nbands):
        for j in range(len(kline_splited)):
            plt.plot(kline_splited[j], bands_splited[i][j], 'r-', linewidth=1.0)
    if nspin == 2:
        for i in range(nbands):
            for j in range(len(kline_splited)):
                plt.plot(kline_splited[j], bands_dw_splited[i][j], 'b--', linewidth=1.0)
    plt.xlim(kline[0], kline[-1])
    plt.ylim(energy_min, energy_max)
    plt.ylabel(r"$E-E_\text{F}$/eV")
    plt.xticks(high_symm_poses, high_symm_labels)
    plt.grid()
    plt.title(f"Band structure  (Gap = {band_gap:.2f} eV)")
    plt.savefig(os.path.join(abacusjob_dir, 'band.png'), dpi=300)
    plt.close()

    return {'band_gap': band_gap,
            'band_picture': Path(os.path.join(abacusjob_dir, 'band.png')).absolute()}

def write_pyatb_input(band_calc_path: Path):
    """
    Write Input file for PYATB
    """
    input_args = ReadInput(os.path.join(band_calc_path, "INPUT"))
    suffix = input_args.get('suffix', 'ABACUS')
    nspin = input_args.get('nspin', 1)
    metrics = collect_metrics(band_calc_path, ['efermi', 'cell', 'band_gap'])
    efermi, cell = metrics['efermi'], metrics['cell']

    input_parameters = {
        'nspin': nspin,
        'package': "ABACUS",
        'fermi_energy': efermi,
        'HR_route': f"OUT.{suffix}/data-HR-sparse_SPIN0.csr",
        'SR_route': f"OUT.{suffix}/data-SR-sparse_SPIN0.csr",
        'rR_route': f"OUT.{suffix}/data-rR-sparse.csr",
        "HR_unit":  "Ry",
        "rR_unit": "Bohr"
    }
    if nspin == 2:
        input_parameters['HR_route'] += f' OUT.{suffix}/data-HR-sparse_SPIN1.csr'
        input_parameters['SR_route'] += f' OUT.{suffix}/data-SR-sparse_SPIN1.csr'
    
    shutil.move(os.path.join(band_calc_path, "INPUT"), os.path.join(band_calc_path, "INPUT_scf"))
    shutil.move(os.path.join(band_calc_path, "KPT"),   os.path.join(band_calc_path, "KPT_scf"))
    pyatb_input_file = open(os.path.join(band_calc_path, "Input"), "w")
    
    pyatb_input_file.write("INPUT_PARAMETERS\n{\n")
    for key, value in input_parameters.items():
        pyatb_input_file.write(f"    {key}  {value}\n")
    pyatb_input_file.write("}\n\nLATTICE\n{\n")

    pyatb_input_file.write(f"    {'lattice_constant'}  {1.8897162}\n")
    pyatb_input_file.write(f"    {'lattice_constant_unit'}  {'Bohr'}\n    lattice_vector\n")
    for cell_vec in cell:
        pyatb_input_file.write(f"    {cell_vec[0]:.8f}  {cell_vec[1]:.8f}  {cell_vec[2]:.8f}\n")
    pyatb_input_file.write("}\n\nBAND_STRUCTURE\n{\n    kpoint_mode   line\n")

    # Get kline and write to pyatb Input file
    kpt_file = os.path.join(band_calc_path, "KPT_band")
    kpt_file_content = []
    with open(kpt_file) as fin:
        for lines in fin:
            words = lines.split()
            kpt_file_content.append(words)

    high_symm_nums = int(kpt_file_content[1][0])
    kpoint_label = ''
    for linenum in range(3, 3+high_symm_nums):
        kpoint_label += kpt_file_content[linenum][-1].split('#')[-1]
        if linenum < 2+high_symm_nums:
            kpoint_label += ", "
    pyatb_input_file.write(f"    kpoint_num    {high_symm_nums}\n")
    pyatb_input_file.write(f"    kpoint_label  {kpoint_label}\n    high_symmetry_kpoint\n")
    for linenum in range(3, 3+high_symm_nums):
        kpoint_coord = f"    {kpt_file_content[linenum][0]} {kpt_file_content[linenum][1]} {kpt_file_content[linenum][2]}"
        kline_num = f" {kpt_file_content[linenum][3]}\n"
        pyatb_input_file.write(kpoint_coord + kline_num)
    pyatb_input_file.write("}\n")

    pyatb_input_file.close()

    return True

def abacus_plot_band_pyatb(band_calc_path: Path,
                           energy_min: float = -10,
                           energy_max: float = 10,
) -> Dict[str, Any]:
    """
    Read result from self-consistent (scf) calculation of hybrid functional using uniform grid,
    and calculate and plot band using PYATB.  

    Currently supports only non-spin-polarized and collinear spin-polarized calculations.

    Args:
        band_calc_path (str): Absolute path to the band calculation directory.
        energy_min (float): Lower bound of $E - E_F$ in the plotted band.
        energy_max (float): Upper bound of $E - E_F$ in the plotted band.

    Returns:
        dict: A dictionary containing:
            - 'band_gap': Calculated band gap in eV. 
            - 'band_picture': Path to the saved band structure plot image file.
    Raises:
        NotImplementedError: If requestes to plot band structure for a collinear or SOC calculation
        RuntimeError: If read band gap from band_info.dat failed
    """
    input_args = ReadInput(os.path.join(band_calc_path, "INPUT"))
    nspin = input_args.get('nspin', 1)
    band_gap = float(collect_metrics(band_calc_path, ['band_gap'])['band_gap'])
    if nspin not in (1, 2):
        raise NotImplementedError("Band plot for nspin=4 is not supported yet")
    
    if write_pyatb_input(band_calc_path) is not True:
        raise RuntimeError("Failed to write pyatb input file")
    
    # Use pyatb to plot band
    run_pyatb(band_calc_path)

    # read band gap
    band_gaps = []
    try:
        with open(os.path.join(band_calc_path, "Out/Band_Structure/band_info.dat")) as fin:
            for lines in fin:
                if "Band gap" in lines:
                    band_gaps.append(float(lines.split()[-1]))
    except Exception as e:
        raise RuntimeError("band_info.dat not found!")
    
    # Modify auto generated plot_band.py and replot the band
    os.system(f'sed -i "16c y_min =  {energy_min} # eV" {band_calc_path}/Out/Band_Structure/plot_band.py')
    os.system(f'sed -i "17c y_max =  {energy_max} # eV" {band_calc_path}/Out/Band_Structure/plot_band.py')
    os.system(f'''sed -i "18c fig_name = os.path.join(work_path, \\"band.png\\")" "{band_calc_path}/Out/Band_Structure/plot_band.py"''')
    os.system(f'sed -i "91c plt.savefig(fig_name, dpi=300)" {band_calc_path}/Out/Band_Structure/plot_band.py')
    os.system(f"cd {band_calc_path}/Out/Band_Structure; python plot_band.py; cd ../../")
    
    # Copy plotted band.pdf to given directory
    band_picture = os.path.join(band_calc_path, "band.png")
    os.system(f"cp {os.path.join(band_calc_path, 'Out/Band_Structure/band.png')} {band_picture}")

    return {'band_gap': band_gap,
            'band_picture': Path(band_picture).absolute()}    

def abacus_cal_band(abacus_inputs_dir: Path,
                    mode: Literal["nscf", "pyatb", "auto"] = "auto",
                    kpath: Union[List[str], List[List[str]]] = None,
                    high_symm_points: Dict[str, List[float]] = None,
                    energy_min: float = -10,
                    energy_max: float = 10,
                    insert_point_nums: int = 30
) -> Dict[str, float|str]:
    """
    Calculate band using ABACUS based on prepared directory containing the INPUT, STRU, KPT, and pseudopotential or orbital files.
    PYATB or ABACUS NSCF calculation will be used according to parameters in INPUT.
    Args:
        abacus_inputs_dir (str): Absolute path to a directory containing the INPUT, STRU, KPT, and pseudopotential or orbital files.
        mode: Method used to plot band. Should be `auto`, `pyatb` or `nscf`. 
            - `nscf` means using `nscf` calculation in ABACUS to calculate and plot the band
            - `pyatb` means using PYATB to plot the band
            - `auto` means deciding use `nscf` or `pyatb` mode according to the `basis_type` in INPUT file and files included in `abacus_inputs_dir`.
                -- If charge files are in `abacus_input_dir`, `nscf` mode will be used.
                -- If matrix files are in `abacus_input_dir`, `pyatb` mode will be used.
                -- If no matrix file or charge file are in `abacus_input_dir`, will determine mode by `basis_type`. If `basis_type` is lcao, will use `pyatb` mode.
                    If `basis_type` is pw, will use `nscf` mode.
        kpath (Tuple[List[str], List[List[str]]]): 
                A list of name of high symmetry points in the band path. Non-continuous line of high symmetry points are stored as seperate lists.
                For example, ['G', 'M', 'K', 'G'] and [['G', 'X', 'P', 'N', 'M', 'S'], ['S_0', 'G', R']] are both acceptable inputs.
                Default is None. If None, will use automatically generated kpath.
                `kpath` must be used with `high_symm_points` to take effect.
        high_symm_points: A dictionary containing high symmetry points and their coordinates in the band path. All points in `kpath` should be included.
                For example, {'G': [0, 0, 0], 'M': [0.5, 0.0, 0.0], 'K': [0.33333333, 0.33333333, 0.0], 'G': [0, 0, 0]}.
                Default is None. If None, will use automatically generated high symmetry points.
        energy_min (float): Lower bound of $E - E_F$ in the plotted band.
        energy_max (float): Upper bound of $E - E_F$ in the plotted band.
        insert_point_nums (int): Number of points to insert between two high symmetry points. Default is 30.
    Returns:
        A dictionary containing band gap, path to the work directory for calculating band and path to the plotted band.
    Raises:
    """
    try:
        input_params = ReadInput(os.path.join(abacus_inputs_dir, "INPUT"))
        original_stru_file = os.path.join(abacus_inputs_dir, input_params.get('stru_file', "STRU"))
        original_stru = AbacusStru.ReadStru(original_stru_file)
        band_kpt_file = os.path.join(abacus_inputs_dir, "KPT_band")
        new_stru, point_coords, path, _ = original_stru.get_kline(point_number=30,
                                                                      new_stru_file=original_stru_file,
                                                                      kpt_file=band_kpt_file)
        
        if kpath is not None and high_symm_points is not None:
            kline = []
            if all(isinstance(item, str) for item in kpath): # A whole continous kline
                for idx, high_symm_point in enumerate(kpath):
                    if idx == len(kpath) - 1: # Treat tail of kline
                        kpoint = high_symm_points[high_symm_point] + [1, '# ' + high_symm_point]
                    else:
                        kpoint = high_symm_points[high_symm_point] + [insert_point_nums, '# ' + high_symm_point]
                    kline.append(kpoint)
            elif all(isinstance(item, list) for item in kpath): # kline with uncontinous points
                kline = []
                for sub_kpath in kpath:
                    for idx, high_symm_point in enumerate(sub_kpath):
                        if idx == len(sub_kpath) - 1: # Treat tail of kline
                            kpoint = high_symm_points[high_symm_point] + [1, '# ' + high_symm_point]
                        else:
                            kpoint = high_symm_points[high_symm_point] + [insert_point_nums, '# ' + high_symm_point]
                        kline.append(kpoint)
            
            WriteKpt(kline, band_kpt_file, model='line')
        elif kpath is not None or high_symm_points is not None:
            print("kpath and high_symm_points must be used together. Use auto-generated kpath and high_symm_points")
        
        force_run = True if original_stru.get_natoms() != new_stru.get_natoms() else False
        scf_output = property_calculation_scf(abacus_inputs_dir, mode, always_run=force_run)
        work_path, mode = scf_output["work_path"], scf_output["mode"]
        if mode == 'pyatb':
            # Obtain band using PYATB
            postprocess_output = abacus_plot_band_pyatb(work_path,
                                                        energy_min,
                                                        energy_max)

            return {'band_gap': postprocess_output['band_gap'],
                    'band_calc_dir': abacus_inputs_dir,
                    'band_picture': postprocess_output['band_picture'],
                    "message": "The band is calculated using PYATB after SCF calculation using ABACUS"}

        elif mode == 'nscf':
            input_params["calculation"] = "nscf"
            input_params["init_chg"] = "file"
            input_params["out_band"] = 1
            input_params["symmetry"] = 0
            input_params['kspacing'] = None
            WriteInput(input_params, os.path.join(work_path, "INPUT"))
            
            # Prepare line-mode KPT file
            kpt_file = os.path.join(work_path, input_params.get('kpt_file', 'KPT'))
            shutil.copy(band_kpt_file, kpt_file)

            run_abacus(work_path)

            plot_output = abacus_plot_band_nscf(work_path, energy_min, energy_max)

            return {'band_gap': plot_output['band_gap'],
                    'band_calc_dir': Path(work_path).absolute(),
                    'band_picture': Path(plot_output['band_picture']).absolute(),
                    "message": "The band structure is computed via a non-self-consistent field (NSCF) calculation using ABACUS, following a converged self-consistent field (SCF) calculation."}
        else:
            raise ValueError(f"Calculation mode {mode} not in ('pyatb', 'nscf', 'auto')")
    except Exception as e:
        return {'message': f"Calculating band failed: {e}"}


# ============================================================================
# Effective Mass Calculation Functions
# ============================================================================

def load_band_data_for_effective_mass(band_calc_dir: Path) -> Dict[str, Any]:
    """
    Load band data from BANDS_*.dat files and extract k-point information.

    Args:
        band_calc_dir: Path to directory containing band calculation results

    Returns:
        Dict containing:
            - bands: List[List[float]] - bands[i][j] = energy of band i at k-point j
            - kline: List[float] - cumulative k-distances
            - kpoints: List[List[float]] - actual k-point coordinates in reciprocal space
            - efermi: float - Fermi energy
            - nspin: int - spin polarization
            - bands_dw: Optional[List[List[float]]] - spin-down bands if nspin=2
    """
    import numpy as np

    input_params = ReadInput(os.path.join(band_calc_dir, "INPUT"))
    suffix = input_params.get('suffix', 'ABACUS')
    nspin = input_params.get('nspin', 1)

    # Get Fermi energy
    metrics = collect_metrics(band_calc_dir, ['efermi'])
    efermi = metrics['efermi']

    # Read band data
    band_file = os.path.join(band_calc_dir, f"OUT.{suffix}/BANDS_1.dat")
    bands, kline, nbands = read_band_data(band_file, efermi)

    bands_dw = None
    if nspin == 2:
        band_file_dw = os.path.join(band_calc_dir, f"OUT.{suffix}/BANDS_2.dat")
        bands_dw, _, _ = read_band_data(band_file_dw, efermi)

    # Reconstruct k-point coordinates
    kpoints = reconstruct_kpoint_coords(band_calc_dir, kline)

    return {
        'bands': bands,
        'kline': kline,
        'kpoints': kpoints,
        'efermi': efermi,
        'nspin': nspin,
        'bands_dw': bands_dw,
        'nbands': nbands
    }


def reconstruct_kpoint_coords(band_calc_dir: Path, kline: List[float]) -> List[List[float]]:
    """
    Reconstruct actual k-point coordinates from KPT file and k-line distances.

    Args:
        band_calc_dir: Path to band calculation directory
        kline: List of cumulative k-distances

    Returns:
        List of k-point coordinates [[kx, ky, kz], ...]
    """
    import numpy as np

    # Read KPT_band file to get high symmetry points
    kpt_file = os.path.join(band_calc_dir, "KPT_band")
    if not os.path.exists(kpt_file):
        kpt_file = os.path.join(band_calc_dir, "KPT")

    high_symm_kpoints = []
    insert_nums = []

    with open(kpt_file) as fin:
        lines = fin.readlines()
        for line in lines:
            words = line.split()
            if len(words) >= 4:
                try:
                    kx, ky, kz = float(words[0]), float(words[1]), float(words[2])
                    num = int(words[3])
                    high_symm_kpoints.append([kx, ky, kz])
                    insert_nums.append(num)
                except ValueError:
                    continue

    # Interpolate k-points along the path
    kpoints = []
    for i in range(len(high_symm_kpoints) - 1):
        k_start = np.array(high_symm_kpoints[i])
        k_end = np.array(high_symm_kpoints[i + 1])
        num_points = insert_nums[i]

        for j in range(num_points):
            t = j / num_points if num_points > 1 else 0
            k_interp = k_start + t * (k_end - k_start)
            kpoints.append(k_interp.tolist())

    # Add the last point
    kpoints.append(high_symm_kpoints[-1])

    return kpoints


def find_band_extrema(
    bands: List[List[float]],
    kpoints: List[List[float]],
    kline: List[float],
    efermi: float,
    energy_range: List[float],
    band_indices: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Automatically detect band extrema (VBM, CBM, and other local extrema).

    Args:
        bands: Band energies
        kpoints: K-point coordinates
        kline: K-line distances
        efermi: Fermi energy (already subtracted from bands)
        energy_range: [E_min, E_max] relative to Fermi level for extrema detection
        band_indices: Specific band indices to analyze

    Returns:
        List of dicts, each containing extrema information
    """
    import numpy as np

    extrema = []

    # Find VBM and CBM
    vbm_info = None
    cbm_info = None
    vbm_energy = -float('inf')
    cbm_energy = float('inf')

    for band_idx, band in enumerate(bands):
        if band_indices is not None and band_idx not in band_indices:
            continue

        for k_idx, energy in enumerate(band):
            # Find VBM (highest occupied state below Fermi level)
            if energy < 0 and energy > vbm_energy:
                vbm_energy = energy
                vbm_info = {
                    'band_index': band_idx,
                    'kpoint_index': k_idx,
                    'kpoint_coords': kpoints[k_idx],
                    'energy': energy,
                    'extrema_type': 'VBM',
                    'is_degenerate': False
                }

            # Find CBM (lowest unoccupied state above Fermi level)
            if energy > 0 and energy < cbm_energy:
                cbm_energy = energy
                cbm_info = {
                    'band_index': band_idx,
                    'kpoint_index': k_idx,
                    'kpoint_coords': kpoints[k_idx],
                    'energy': energy,
                    'extrema_type': 'CBM',
                    'is_degenerate': False
                }

    if vbm_info:
        extrema.append(vbm_info)
    if cbm_info:
        extrema.append(cbm_info)

    # Find other local extrema within energy range
    for band_idx, band in enumerate(bands):
        if band_indices is not None and band_idx not in band_indices:
            continue

        for k_idx in range(1, len(band) - 1):
            energy = band[k_idx]

            # Check if within energy range
            if energy < energy_range[0] or energy > energy_range[1]:
                continue

            # Check for local maximum
            if band[k_idx] > band[k_idx - 1] and band[k_idx] > band[k_idx + 1]:
                extrema_type = 'local_max'
                if abs(energy - vbm_energy) < 0.01:  # Same as VBM
                    continue

                extrema.append({
                    'band_index': band_idx,
                    'kpoint_index': k_idx,
                    'kpoint_coords': kpoints[k_idx],
                    'energy': energy,
                    'extrema_type': extrema_type,
                    'is_degenerate': False
                })

            # Check for local minimum
            elif band[k_idx] < band[k_idx - 1] and band[k_idx] < band[k_idx + 1]:
                extrema_type = 'local_min'
                if abs(energy - cbm_energy) < 0.01:  # Same as CBM
                    continue

                extrema.append({
                    'band_index': band_idx,
                    'kpoint_index': k_idx,
                    'kpoint_coords': kpoints[k_idx],
                    'energy': energy,
                    'extrema_type': extrema_type,
                    'is_degenerate': False
                })

    return extrema


def fit_parabola_1d(
    k_distances: List[float],
    energies: List[float],
    k0: float = 0.0
) -> Dict[str, Any]:
    """
    Fit parabola E(k) = E0 + a*(k-k0)^2 to band data.

    Args:
        k_distances: K-point distances along direction
        energies: Band energies at those k-points
        k0: Center k-point (default: 0.0)

    Returns:
        Dict containing fit parameters and quality metrics
    """
    import numpy as np

    if len(k_distances) < 3:
        return {
            'E0': None,
            'a': None,
            'curvature': None,
            'r_squared': 0.0,
            'fit_energies': [],
            'residuals': [],
            'error': 'Insufficient data points for fitting'
        }

    # Shift k-points to center at k0
    k_shifted = np.array(k_distances) - k0
    e_array = np.array(energies)

    # Fit parabola: E = E0 + a*k^2
    # Using polyfit with degree 2
    try:
        coeffs = np.polyfit(k_shifted, e_array, 2)
        a, b, E0 = coeffs[0], coeffs[1], coeffs[2]

        # Calculate fitted energies
        fit_energies = np.polyval(coeffs, k_shifted)

        # Calculate R²
        ss_res = np.sum((e_array - fit_energies) ** 2)
        ss_tot = np.sum((e_array - np.mean(e_array)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Curvature is d²E/dk² = 2*a
        curvature = 2 * a

        return {
            'E0': E0,
            'a': a,
            'b': b,
            'curvature': curvature,
            'r_squared': r_squared,
            'fit_energies': fit_energies.tolist(),
            'residuals': (e_array - fit_energies).tolist(),
            'num_points': len(k_distances)
        }
    except Exception as e:
        return {
            'E0': None,
            'a': None,
            'curvature': None,
            'r_squared': 0.0,
            'fit_energies': [],
            'residuals': [],
            'error': f'Fitting failed: {str(e)}'
        }


def calculate_effective_mass_from_curvature(
    curvature: float,
    direction: str
) -> Dict[str, float]:
    """
    Calculate effective mass from band curvature using m* = ℏ²/(d²E/dk²).

    Args:
        curvature: d²E/dk² in eV/Å^-2
        direction: "kx", "ky", or "kz"

    Returns:
        Dict containing effective mass values
    """
    from abacusagent.constant import EFFECTIVE_MASS_FACTOR

    if curvature is None or abs(curvature) < 1e-10:
        return {
            'm_star': float('inf'),
            'm_star_kg': float('inf'),
            'curvature': curvature,
            'is_flat_band': True
        }

    # m*/m_e = EFFECTIVE_MASS_FACTOR / curvature
    m_star = EFFECTIVE_MASS_FACTOR / curvature

    # Convert to kg
    from abacusagent.constant import ELECTRON_MASS_KG
    m_star_kg = m_star * ELECTRON_MASS_KG

    return {
        'm_star': m_star,
        'm_star_kg': m_star_kg,
        'curvature': curvature,
        'is_flat_band': False
    }


def calculate_effective_mass_tensor(
    band_index: int,
    kpoint_index: int,
    kpoints: List[List[float]],
    bands: List[List[float]],
    kline: List[float],
    fitting_window: int,
    directions: List[str]
) -> Dict[str, Any]:
    """
    Calculate effective mass in multiple directions around a k-point.

    Args:
        band_index: Band index
        kpoint_index: K-point index
        kpoints: All k-point coordinates
        bands: All band energies
        kline: K-line distances
        fitting_window: Number of points on each side for fitting
        directions: List of directions to calculate

    Returns:
        Dict containing effective mass tensor components
    """
    import numpy as np

    result = {
        'effective_masses': {},
        'fitting_data': {},
        'anisotropy_ratio': None,
        'm_star_avg': None
    }

    k_center = np.array(kpoints[kpoint_index])
    band = bands[band_index]

    # Determine available k-points within window
    start_idx = max(0, kpoint_index - fitting_window)
    end_idx = min(len(kpoints), kpoint_index + fitting_window + 1)

    # Extract k-points and energies in window
    k_window = np.array([kpoints[i] for i in range(start_idx, end_idx)])
    e_window = [band[i] for i in range(start_idx, end_idx)]
    kline_window = [kline[i] for i in range(start_idx, end_idx)]

    # Calculate effective mass along k-path direction
    # Use kline distances directly
    k_distances = np.array(kline_window) - kline[kpoint_index]

    fit_result = fit_parabola_1d(k_distances.tolist(), e_window, 0.0)

    if fit_result['curvature'] is not None:
        mass_result = calculate_effective_mass_from_curvature(
            fit_result['curvature'],
            'kpath'
        )

        result['effective_masses']['kpath'] = {
            'm_star': mass_result['m_star'],
            'curvature': fit_result['curvature'],
            'r_squared': fit_result['r_squared'],
            'num_points': fit_result['num_points']
        }

        result['fitting_data']['kpath'] = {
            'k_distances': k_distances.tolist(),
            'energies': e_window,
            'fit_energies': fit_result['fit_energies']
        }

        result['m_star_avg'] = mass_result['m_star']

    return result


def find_high_symmetry_kpoint(
    label: str,
    band_calc_dir: Path,
    kpoints: List[List[float]],
    kline: List[float]
) -> Optional[int]:
    """
    Find k-point index corresponding to a high symmetry point label.

    Args:
        label: High symmetry point label (e.g., "G", "M", "K")
        band_calc_dir: Path to band calculation directory
        kpoints: All k-point coordinates
        kline: K-line distances

    Returns:
        K-point index or None if not found
    """
    try:
        high_symm_labels, band_point_nums = read_high_symmetry_labels(band_calc_dir)

        # Normalize label (G -> Γ)
        if label == 'G' or label == 'Gamma':
            label = r'$\Gamma$'

        for i, symm_label in enumerate(high_symm_labels):
            if symm_label == label or symm_label.replace('$', '').replace('\\', '') == label:
                return band_point_nums[i]

        return None
    except Exception as e:
        print(f"Warning: Could not find high symmetry point {label}: {e}")
        return None


def plot_effective_mass_fit(
    result: Dict[str, Any],
    output_path: Path
) -> Path:
    """
    Create a plot showing parabolic fit for effective mass calculation.

    Args:
        result: Effective mass result dictionary
        output_path: Path to save the plot

    Returns:
        Path to saved plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    point_info = result['point_info']
    eff_masses = result['effective_masses']
    fitting_data = result['fitting_data']

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot for kpath direction
    if 'kpath' in fitting_data:
        data = fitting_data['kpath']
        mass_data = eff_masses['kpath']

        # Scatter plot of actual data
        ax.scatter(data['k_distances'], data['energies'],
                  color='blue', s=50, label='Band data', zorder=3)

        # Plot fitted parabola
        ax.plot(data['k_distances'], data['fit_energies'],
               'r-', linewidth=2, label='Parabolic fit', zorder=2)

        # Add text with effective mass info
        textstr = f"m* = {mass_data['m_star']:.3f} $m_e$\n"
        textstr += f"R² = {mass_data['r_squared']:.4f}\n"
        textstr += f"Curvature = {mass_data['curvature']:.4f} eV/Å²"

        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round',
               facecolor='wheat', alpha=0.5))

    ax.set_xlabel('k distance (Å⁻¹)')
    ax.set_ylabel('E - E$_F$ (eV)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    title = f"Effective Mass at {point_info['extrema_type']}"
    if point_info.get('high_symmetry_label'):
        title += f" ({point_info['high_symmetry_label']})"
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return Path(output_path).absolute()


def plot_effective_mass_summary(
    results: List[Dict[str, Any]],
    output_path: Path
) -> Path:
    """
    Create summary plot showing effective masses at different points.

    Args:
        results: List of effective mass results
        output_path: Path to save the plot

    Returns:
        Path to saved plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not results:
        return None

    # Extract data
    labels = []
    masses = []
    colors = []

    for res in results:
        point_info = res['point_info']
        eff_mass = res['effective_masses'].get('kpath', {}).get('m_star')

        if eff_mass is not None and not np.isinf(eff_mass):
            label = point_info['extrema_type']
            if point_info.get('high_symmetry_label'):
                label += f"\n{point_info['high_symmetry_label']}"

            labels.append(label)
            masses.append(abs(eff_mass))

            # Color code by type
            if 'VBM' in point_info['extrema_type']:
                colors.append('blue')
            elif 'CBM' in point_info['extrema_type']:
                colors.append('red')
            else:
                colors.append('gray')

    if not masses:
        return None

    # Create bar plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, masses, color=colors, alpha=0.7, edgecolor='black')

    ax.set_xlabel('Band Extrema')
    ax.set_ylabel('|m*| / $m_e$')
    ax.set_title('Effective Mass Summary')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, mass) in enumerate(zip(bars, masses)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mass:.3f}',
               ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return Path(output_path).absolute()


def abacus_cal_effective_mass(
    band_calc_dir: Path,
    calculation_points: Union[str, List[Dict]],
    fitting_window: int,
    directions: List[str],
    band_indices: Optional[List[int]],
    energy_range: Optional[List[float]],
    output_dir: Optional[Path]
) -> Dict[str, Any]:
    """
    Calculate effective mass from band structure using parabolic fitting.

    Args:
        band_calc_dir: Path to directory containing band calculation results
        calculation_points: Where to calculate effective mass
        fitting_window: Number of k-points on each side for fitting
        directions: Directions for effective mass calculation
        band_indices: Specific band indices to analyze
        energy_range: Energy range for extrema detection
        output_dir: Directory for output files

    Returns:
        Dict containing effective mass results, plots, and summary
    """
    import json
    from datetime import datetime
    import numpy as np

    try:
        # Set default values
        if energy_range is None:
            energy_range = [-2.0, 2.0]

        if output_dir is None:
            output_dir = band_calc_dir

        # Load band data
        print("Loading band data...")
        band_data = load_band_data_for_effective_mass(band_calc_dir)
        bands = band_data['bands']
        kpoints = band_data['kpoints']
        kline = band_data['kline']
        efermi = band_data['efermi']
        nspin = band_data['nspin']
        bands_dw = band_data['bands_dw']

        # Determine calculation points
        calc_points = []

        if calculation_points == "auto":
            # Find only VBM and CBM
            extrema = find_band_extrema(bands, kpoints, kline, efermi,
                                       energy_range, band_indices)
            calc_points = [e for e in extrema if e['extrema_type'] in ['VBM', 'CBM']]

        elif calculation_points == "extrema":
            # Find all extrema
            calc_points = find_band_extrema(bands, kpoints, kline, efermi,
                                           energy_range, band_indices)

        elif isinstance(calculation_points, list):
            # User-specified points
            for point_spec in calculation_points:
                if point_spec['type'] == 'kpoint':
                    # Find nearest k-point to specified coordinates
                    target_k = np.array(point_spec['coords'])
                    k_array = np.array(kpoints)
                    distances = np.linalg.norm(k_array - target_k, axis=1)
                    k_idx = np.argmin(distances)

                    band_idx = point_spec.get('band_index')
                    if band_idx is None:
                        # Find band closest to Fermi level at this k-point
                        energies = [bands[i][k_idx] for i in range(len(bands))]
                        band_idx = np.argmin(np.abs(energies))

                    calc_points.append({
                        'band_index': band_idx,
                        'kpoint_index': k_idx,
                        'kpoint_coords': kpoints[k_idx],
                        'energy': bands[band_idx][k_idx],
                        'extrema_type': 'user_specified',
                        'is_degenerate': False
                    })

                elif point_spec['type'] == 'high_symmetry':
                    label = point_spec['label']
                    k_idx = find_high_symmetry_kpoint(label, band_calc_dir,
                                                     kpoints, kline)
                    if k_idx is not None:
                        band_idx = point_spec.get('band_index')
                        if band_idx is None:
                            energies = [bands[i][k_idx] for i in range(len(bands))]
                            band_idx = np.argmin(np.abs(energies))

                        calc_points.append({
                            'band_index': band_idx,
                            'kpoint_index': k_idx,
                            'kpoint_coords': kpoints[k_idx],
                            'energy': bands[band_idx][k_idx],
                            'extrema_type': 'high_symmetry',
                            'high_symmetry_label': label,
                            'is_degenerate': False
                        })

        if not calc_points:
            return {'message': 'No calculation points found'}

        print(f"Calculating effective mass at {len(calc_points)} points...")

        # Calculate effective mass at each point
        results = []
        plot_paths = []

        for i, point in enumerate(calc_points):
            print(f"Processing point {i+1}/{len(calc_points)}: {point['extrema_type']}")

            # Calculate effective mass tensor
            eff_mass_result = calculate_effective_mass_tensor(
                point['band_index'],
                point['kpoint_index'],
                kpoints,
                bands,
                kline,
                fitting_window,
                directions
            )

            # Compile result
            result = {
                'point_info': point,
                'effective_masses': eff_mass_result['effective_masses'],
                'fitting_data': eff_mass_result['fitting_data']
            }

            results.append(result)

            # Generate plot for this point
            plot_filename = f"effective_mass_{point['extrema_type']}_{i}.png"
            plot_path = os.path.join(output_dir, plot_filename)
            plot_effective_mass_fit(result, plot_path)
            plot_paths.append(Path(plot_path).absolute())

        # Generate summary statistics
        electron_masses = []
        hole_masses = []

        for res in results:
            point_type = res['point_info']['extrema_type']
            m_star = res['effective_masses'].get('kpath', {}).get('m_star')

            if m_star is not None and not np.isinf(m_star):
                if 'CBM' in point_type or 'local_min' in point_type:
                    electron_masses.append(abs(m_star))
                elif 'VBM' in point_type or 'local_max' in point_type:
                    hole_masses.append(abs(m_star))

        summary = {}
        if electron_masses:
            summary['electron_effective_mass'] = {
                'average': float(np.mean(electron_masses)),
                'min': float(np.min(electron_masses)),
                'max': float(np.max(electron_masses)),
                'std': float(np.std(electron_masses))
            }

        if hole_masses:
            summary['hole_effective_mass'] = {
                'average': float(np.mean(hole_masses)),
                'min': float(np.min(hole_masses)),
                'max': float(np.max(hole_masses)),
                'std': float(np.std(hole_masses))
            }

        # Generate summary plot
        summary_plot_path = os.path.join(output_dir, "effective_mass_summary.png")
        plot_effective_mass_summary(results, summary_plot_path)
        if os.path.exists(summary_plot_path):
            plot_paths.append(Path(summary_plot_path).absolute())

        # Write JSON output
        json_output = {
            'metadata': {
                'band_calc_dir': str(band_calc_dir),
                'calculation_date': datetime.now().isoformat(),
                'nspin': nspin,
                'efermi': efermi,
                'fitting_window': fitting_window,
                'directions': directions
            },
            'results': results,
            'summary': summary
        }

        json_path = os.path.join(output_dir, "effective_mass_results.json")
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2, default=str)

        return {
            'effective_mass_results': results,
            'effective_mass_json': Path(json_path).absolute(),
            'effective_mass_plots': plot_paths,
            'summary': summary,
            'message': 'Effective mass calculation completed successfully'
        }

    except Exception as e:
        import traceback
        return {'message': f"Effective mass calculation failed: {e}\n{traceback.format_exc()}"}
