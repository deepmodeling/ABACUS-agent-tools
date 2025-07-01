import os
import shutil
import time
from pathlib import Path
from typing import Literal, Optional, TypedDict, Dict, Any, List
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteInput
from abacustest.lib_model.model_013_inputs import PrepInput

from abacusagent.init_mcp import mcp
from abacusagent.modules.abacus import abacus_modify_input, abacus_collect_data
from abacusagent.modules.util.comm import run_abacus, link_abacusjob, generate_work_path

@mcp.tool()
def abacus_plot_band(abacusjob_dir: Path,
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
    input_args = ReadInput(abacusjob_dir + "/INPUT")
    suffix = input_args.get('suffix', 'ABACUS')
    nspin = input_args.get('nspin', 1)
    if nspin != 1 and nspin != 2:
        raise NotImplementedError("Band plot for nspin=4 is not supported yet")
    
    metrics = abacus_collect_data(abacusjob_dir, ['efermi', 'nelec', 'band_gap'])['collected_metrics']
    efermi, band_gap = metrics['efermi'], float(metrics['band_gap'])
    band_file = abacusjob_dir + f"/OUT.{suffix}/BANDS_1.dat"
    if nspin == 2:
        band_file_dw = abacusjob_dir + f"/OUT.{suffix}/BANDS_2.dat"
    
    # Read band data
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
        raise RuntimeError("Read data from BANDS_1.dat failed")

    if nspin == 2:
        bands_dw = []
        try:
            with open(band_file_dw) as fin:
                for lines in fin:
                    words = lines.split()
                    if len(bands_dw) == 0:
                        for _ in range(nbands):
                            bands_dw.append([])

                    for i in range(nbands):
                        bands_dw[i].append(float(words[i+2]) - efermi)
        except Exception as e:
            raise RuntimeError("Read data from BANDS_2.dat failed")

    # Read high symmetry labels from KPT file
    high_symm_labels = []
    band_point_nums = []
    band_point_num = 0
    with open(abacusjob_dir + "/KPT") as fin:
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
    
    # Split band and kline by incontinuous points
    def split_array(array, splits):
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
    
    import matplotlib.pyplot as plt
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
    plt.savefig(abacusjob_dir + '/band.png', dpi=300)
    plt.show()

    return {'band_gap': band_gap,
            'band_picture': Path(abacusjob_dir + '/band.png')}

@mcp.tool()
def abacus_postprocess_band_pyatb(band_calc_path: Path,
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
    input_args = ReadInput(band_calc_path + "/INPUT")
    suffix = input_args.get('suffix', 'ABACUS')
    nspin = input_args.get('nspin', 1)
    if nspin != 1 and nspin != 2:
        raise NotImplementedError("Band plot for nspin=4 is not supported yet")
    
    metrics = abacus_collect_data(band_calc_path, ['efermi', 'cell', 'band_gap'])['collected_metrics']
    efermi, cell, band_gap = metrics['efermi'], metrics['cell'], float(metrics['band_gap'])

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
    
    shutil.move(band_calc_path + "/INPUT", band_calc_path + "/INPUT_scf")
    shutil.move(band_calc_path + "/KPT", band_calc_path + "/KPT_scf")
    pyatb_input_file = open(band_calc_path + "/Input", "w")
    
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
    stru_file = AbacusStru.ReadStru(band_calc_path + "/STRU")
    kpt_file = band_calc_path + "/KPT"
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

    # Use pyatb to plot band
    os.system(f"cd {band_calc_path}; OMP_NUM_THREADS=1 pyatb")

    # read band gap
    band_gaps = []
    try:
        with open(band_calc_path + "/Out/Band_Structure/band_info.dat") as fin:
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
    band_picture = band_calc_path + "/band.png"
    os.system(f"cp {band_calc_path + '/Out/Band_Structure/band.png'} {band_picture}")

    return {'band_gap': band_gap,
            'band_picture': Path(band_calc_path + '/band.png')}    

@mcp.tool()
def abacus_cal_band(abacus_inputs_path: Path,
                    dft_functional: str = 'pbe',
                    energy_min: float = -10,
                    energy_max: float = 10
) -> Dict[str, float|str]:
    """
    Calculate band using ABACUS based on prepared directory containing the INPUT, STRU, KPT, and pseudopotential or orbital files.
    Currently calculating hybrid DFT band using this function is broken, and should not be used when a hybrid DFT band calculation is requested.
    Args:
        abacusjob_dir (str): Absolute path to a directory containing the INPUT, STRU, KPT, and pseudopotential or orbital files.
        dft_functional (str): Density functional used to calculate the band
        energy_min (float): Lower bound of $E - E_F$ in the plotted band.
        energy_max (float): Upper bound of $E - E_F$ in the plotted band.
    Returns:
        Band gap, work directory and plotted band.
    Raises:
    """
    work_path = generate_work_path()
    link_abacusjob(src=abacus_inputs_path,
                    dst=work_path,
                    copy_files=["INPUT", "STRU", "KPT"])
    
    # SCF calculation
    modified_params = {'calculation': 'scf',
                       'dft_functional': dft_functional,
                       'out_chg': 1}
    modified_input = abacus_modify_input(work_path + "/INPUT",
                                         extra_input = modified_params)

    run_abacus(work_path)
    
    # NSCF calculation
    modified_params = {'calculation': 'nscf',
                       'init_chg': 'file',
                       'out_band': 1,
                       'symmetry': 0}
    remove_params = ['kspacing']
    modified_input = abacus_modify_input(work_path + "/INPUT",
                                         extra_input = modified_params,
                                         remove_input = remove_params)
    
    # Prepare line-mode KPT file
    nscf_stru = AbacusStru.ReadStru(work_path + "/STRU")
    kpt_file = work_path + '/KPT'
    nscf_stru.get_kline_ase(point_number=30,kpt_file=kpt_file)

    run_abacus(work_path)

    plot_output = abacus_plot_band(work_path, energy_min, energy_max)

    return {'band_gap': plot_output['band_gap'],
            'band_calc_dir': work_path,
            'band_picture': Path(plot_output['band_picture'])}

@mcp.tool()
def abacus_cal_band_pyatb(abacus_inputs_path: Path,
                          dft_functional: str = 'pbe',
                          energy_min: float = -10.0,
                          energy_max: float =  10.0
) -> Dict[str, float|str]:
    """
    Calculate GGA, meta-GGA and hybrid DFT band using PYATB based on results from SCF calculation using ABACUS.
    For hybrid DFT band calculation, this function should be used. For GGA and meta-GGA DFT band calculations, 
    This function can be used but not prior than other functions.
    Args:
        abacusjob_dir (str): Absolute path to a directory containing the INPUT, STRU, KPT, and pseudopotential or orbital files.
        dft_functional (str): Density functional used to calculate the band. For example, 'hse' means HSE06 hybrid density functional.
        energy_min (float): Lower bound of $E - E_F$ in the plotted band.
        energy_max (float): Upper bound of $E - E_F$ in the plotted band.
    Returns:
        A dictionary containing band gap, work directory and plotted band.
    Raises:
    """
    work_path = generate_work_path()
    link_abacusjob(src=abacus_inputs_path,
                    dst=work_path,
                    copy_files=["INPUT", "STRU", "KPT"])
    
    extra_input = {'calculation': 'scf',
                   'dft_functional': dft_functional,
                   'out_mat_hs2': True,
                   'out_mat_r': True}
    
    modified_input = abacus_modify_input(work_path + "/INPUT",
                                         extra_input = extra_input)
    run_abacus(work_path)

    postprocess_output = abacus_postprocess_band_pyatb(work_path,
                                                       energy_min,
                                                       energy_max)
    
    return {'band_gap': postprocess_output['band_gap'],
            'band_calc_dir': abacus_inputs_path,
            'band_picture': Path(postprocess_output['band_picture'])}

if __name__ == '__main__':
    from abacusagent.env import set_envs, create_workpath
    set_envs()
    create_workpath()
    
    abacusjob_dir = '/mnt/e/temp/abacus-agent-develop-7/abacus-agent/tests/band'
    abacus_plot_band(abacusjob_dir,
                          energy_min = -6.0,
                          energy_max = 6.0)
