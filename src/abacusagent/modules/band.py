import os
import shutil
from pathlib import Path
from typing import Literal, Optional, TypedDict, Dict, Any, List
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteInput

from abacusagent.init_mcp import mcp
from abacusagent.modules.abacus import abacus_modify_input, abacus_collect_data
from abacusagent.modules.util.comm import run_abacus, run_command, get_physical_cores
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
    for i in range(len(splits)):
        if i == 0:
            start = 0
        else:
            start = splits[i-1]
        
        if i == len(splits) - 1:
            end = splits[-1]
        else:
            end = splits[i]
        
        splited_array.append(array[start:end])
    
    splited_array.append(array[splits[-1]:])
    return splited_array

def read_high_symmetry_labels(abacusjob_dir: Path):
    """
    Read high symmetry labels from INPUT file
    """
    high_symm_labels = []
    band_point_nums = []
    band_point_num = 0
    with open(os.path.join(abacusjob_dir, "KPT")) as fin:
        for lines in fin:
            words = lines.split()
            if len(words) > 2:
                if words[-2] == '#':
                    if words[-1] == 'G':
                        high_symm_labels.append(r'$\Gamma$')
                    else:
                        high_symm_labels.append(words[-1])
                    band_point_nums.append(band_point_num)
                    band_point_num += int(words[-3])
    
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

#@mcp.tool()
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
    
    metrics = abacus_collect_data(abacusjob_dir, ['efermi', 'nelec', 'band_gap'])['collected_metrics']
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

def write_pyatb_input(band_calc_path: Path, connect_line_points=30):
    """
    Write Input file for PYATB
    """
    input_args = ReadInput(os.path.join(band_calc_path, "INPUT"))
    suffix = input_args.get('suffix', 'ABACUS')
    nspin = input_args.get('nspin', 1)
    metrics = abacus_collect_data(band_calc_path, ['efermi', 'cell', 'band_gap'])['collected_metrics']
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
    stru_file = AbacusStru.ReadStru(os.path.join(band_calc_path, "STRU"))
    kpt_file = os.path.join(band_calc_path, "KPT")
    stru_file.get_kline_ase(point_number=connect_line_points,kpt_file=kpt_file)

    kpt_file_content = []
    with open(kpt_file) as fin:
        for lines in fin:
            words = lines.split()
            kpt_file_content.append(words)

    high_symm_nums = int(kpt_file_content[1][0])
    kpoint_label = ''
    for linenum in range(3, 3+high_symm_nums):
        kpoint_label += kpt_file_content[linenum][-1]
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

#@mcp.tool()
def abacus_plot_band_pyatb(band_calc_path: Path,
                           energy_min: float = -10,
                           energy_max: float = 10,
                           connect_line_points=30
) -> Dict[str, Any]:
    """
    Read result from self-consistent (scf) calculation of hybrid functional using uniform grid,
    and calculate and plot band using PYATB.  

    Currently supports only non-spin-polarized and collinear spin-polarized calculations.

    Args:
        band_calc_path (str): Absolute path to the band calculation directory.
        energy_min (float): Lower bound of $E - E_F$ in the plotted band.
        energy_max (float): Upper bound of $E - E_F$ in the plotted band.
        connect_line_points (int): Number of inserted points between consecutive high-symmetry points in k-point path.

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
    band_gap = float(abacus_collect_data(band_calc_path, ['band_gap'])['collected_metrics']['band_gap'])
    if nspin not in (1, 2):
        raise NotImplementedError("Band plot for nspin=4 is not supported yet")
    
    if write_pyatb_input(band_calc_path, connect_line_points=connect_line_points) is not True:
        raise RuntimeError("Failed to write pyatb input file")
    
    # Use pyatb to plot band
    physical_cores = get_physical_cores()
    pyatb_command = f"cd {band_calc_path}; OMP_NUM_THREADS=1 mpirun -np {physical_cores} pyatb"
    return_code, out, err = run_command(pyatb_command)
    if return_code != 0:
        raise RuntimeError(f"pyatb failed with return code {return_code}, out: {out}, err: {err}")

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

@mcp.tool()
def abacus_cal_band(abacus_inputs_path: Path,
                    energy_min: float = -10,
                    energy_max: float = 10
) -> Dict[str, float|str]:
    """
    Calculate band using ABACUS based on prepared directory containing the INPUT, STRU, KPT, and pseudopotential or orbital files.
    PYATB or ABACUS NSCF calculation will be used according to parameters in INPUT.
    Args:
        abacusjob_dir (str): Absolute path to a directory containing the INPUT, STRU, KPT, and pseudopotential or orbital files.
        energy_min (float): Lower bound of $E - E_F$ in the plotted band.
        energy_max (float): Upper bound of $E - E_F$ in the plotted band.
    Returns:
        A dictionary containing band gap, path to the work directory for calculating band and path to the plotted band.
    Raises:
    """
    scf_output = property_calculation_scf(abacus_inputs_path)
    work_path, mode = scf_output["work_path"], scf_output["mode"]
    if mode == 'pyatb':
        # Obtain band using PYATB
        postprocess_output = abacus_plot_band_pyatb(work_path,
                                                    energy_min,
                                                    energy_max)

        return {'band_gap': postprocess_output['band_gap'],
                'band_calc_dir': abacus_inputs_path,
                'band_picture': postprocess_output['band_picture'],
                "message": "The band is calculated using PYATB after SCF calculation using ABACUS"}

    elif mode == 'nscf':
        modified_params = {'calculation': 'nscf',
                           'init_chg': 'file',
                           'out_band': 1,
                           'symmetry': 0}
        remove_params = ['kspacing']
        modified_input = abacus_modify_input(work_path,
                                             extra_input = modified_params,
                                             remove_input = remove_params)

        # Prepare line-mode KPT file
        nscf_stru = AbacusStru.ReadStru(os.path.join(work_path, "STRU"))
        kpt_file = os.path.join(work_path, 'KPT')
        nscf_stru.get_kline_ase(point_number=30,kpt_file=kpt_file)

        run_abacus(work_path)

        plot_output = abacus_plot_band_nscf(work_path, energy_min, energy_max)

        return {'band_gap': plot_output['band_gap'],
                'band_calc_dir': Path(work_path).absolute(),
                'band_picture': Path(plot_output['band_picture']).absolute(),
                "message": "The band structure is computed via a non-self-consistent field (NSCF) calculation using ABACUS, \
                            following a converged self-consistent field (SCF) calculation."}
    else:
        raise ValueError(f"Calculation mode {mode} not in ('pyatb', 'nscf')")
