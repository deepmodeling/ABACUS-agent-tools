import os
import shutil
import time
from pathlib import Path
from typing import Literal, Optional, TypedDict, Dict, Any, List
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteInput
from abacustest.lib_model.model_013_inputs import PrepInput

from abacusagent.init_mcp import mcp
from abacusagent.modules.abacus import abacus_modify_input, abacus_collect_data, run_abacus_onejob

def get_suffix(path: Path, default='ABACUS') -> str:
    """
    Get `suffix` keywork in INPUT file
    """
    inputs = ReadInput(path / "INPUT")
    return inputs.get('suffix', default)

def abacus_cal_band_scf(stru_file,
                        stru_type,
                        lcao,
                        extra_input,
                        band_work_path):
    """
    Do scf calculation step in the band calculation.
    Return True if calculation succeed, and return False if calculation failed.
    """
    pp_path = os.environ.get("ABACUS_PP_PATH")
    orb_path = os.environ.get("ABACUS_ORB_PATH")

    extra_input['out_chg'] = 1

    extra_input_file = None
    if extra_input is not None:
        # write extra input to the input file
        extra_input_file = "INPUT.tmp"
        WriteInput(extra_input, extra_input_file)
    
    try:
        _, job_path = PrepInput(files=str(stru_file),
                                filetype=stru_type,
                                jobtype='scf',
                                pp_path=pp_path,
                                orb_path=orb_path,
                                input_file=extra_input_file,
                                lcao=lcao
                                ).run()
    except Exception as e:
        raise RuntimeError(f"Error preparing input files: {e}")
    
    old_job_path = job_path[0]
    band_scf_work_path = (band_work_path / "scf").absolute()
    shutil.move(old_job_path, band_scf_work_path)

    run_abacus_onejob(band_scf_work_path)
    scf_metrics = abacus_collect_data(band_scf_work_path, ['normal_end'])['collected_metrics']

    if scf_metrics['normal_end'] is True:
        return True
    else:
        return False

def abacus_cal_band_nscf(stru_file,
                         stru_type,
                         lcao,
                         extra_input,
                         band_work_path,
                         connect_line_points: int = 30):
    """
    Do nscf calculation step in the band calculation.
    Return True if calculation succeed, and return False if calculation failed.
    """
    pp_path = os.environ.get("ABACUS_PP_PATH")
    orb_path = os.environ.get("ABACUS_ORB_PATH")
    
    extra_input_file = None
    if extra_input is not None:
        # write extra input to the input file
        extra_input_file = "INPUT.tmp"
        WriteInput(extra_input, extra_input_file)
    
    try:
        _, job_path = PrepInput(files=str(stru_file),
                                filetype=stru_type,
                                jobtype='scf',
                                pp_path=pp_path,
                                orb_path=orb_path,
                                input_file=extra_input_file,
                                lcao=lcao
                                ).run()
    except Exception as e:
        raise RuntimeError(f"Error preparing input files: {e}")

    old_job_path = job_path[0]
    band_nscf_work_path = (band_work_path / "nscf").absolute()
    shutil.move(old_job_path, band_nscf_work_path)

    # Modify INPUT parameters to adapt nscf calculation 
    modified_params = {'calculation': 'nscf',
                           'init_chg': 'file',
                           'out_band': 1,
                           'symmetry': 0}
    remove_params = ['kspacing']
    modified_input = abacus_modify_input(band_nscf_work_path / "INPUT",
                                         extra_input = modified_params,
                                         remove_input = remove_params)
    
    # Copy scf charge to nscf dir
    band_scf_work_path = (band_work_path / "scf").absolute()
    scf_suffix = get_suffix(band_scf_work_path)
    scf_chg_file = band_scf_work_path / f"OUT.{scf_suffix}" / f"{scf_suffix}-CHARGE-DENSITY.restart"
    nscf_suffix = get_suffix(band_nscf_work_path)
    if os.path.exists(band_nscf_work_path / f"OUT.{nscf_suffix}"):
        shutil.rmtree(band_nscf_work_path / f"OUT.{nscf_suffix}")
    os.mkdir(band_nscf_work_path / f"OUT.{nscf_suffix}")
    nscf_chg_file = band_nscf_work_path / f"OUT.{nscf_suffix}" / f"{nscf_suffix}-CHARGE-DENSITY.restart"
    shutil.copy(scf_chg_file, nscf_chg_file)

    # Copy onsite.dm in SCF calculation to nscf dir
    modified_input_params = ReadInput(modified_input['input_path'])
    if 'dft_plus_u' in modified_input_params.keys():
        scf_onsite_dm_file = band_scf_work_path / f"OUT.{scf_suffix}" / "onsite.dm"
        nscf_onsite_dm_file = band_nscf_work_path / f"OUT.{nscf_suffix}" / "onsite.dm"
        shutil.copy(scf_onsite_dm_file, nscf_onsite_dm_file)
    
    # Prepare line-mode KPT file
    nscf_stru = AbacusStru.ReadStru(band_nscf_work_path / "STRU")
    kpt_file = band_nscf_work_path / 'KPT'
    nscf_stru.get_kline_ase(point_number=connect_line_points,kpt_file=kpt_file)

    run_abacus_onejob(band_nscf_work_path)
    nscf_metrics = abacus_collect_data(band_nscf_work_path, ['normal_end'])['collected_metrics']

    if nscf_metrics['normal_end'] is True:
        return True
    else:
        return False

def abacus_plot_band(band_work_path,
                     material_name: str,
                     nspin: int = 1,
                     energy_range: float = 10.0,
                     band_pic_path: str = "BAND.png"):
    """
    Plot band after band calculation finishes.
    Currently support non-spin-polarized and collinear spin polarized band plot only.
    """
    if nspin != 1 and nspin != 2:
        raise NotImplementedError("Band plot for nspin=4 is not supported yet")
    
    nscf_dir = band_work_path / "nscf"
    nscf_suffix = get_suffix(nscf_dir)
    metrics = abacus_collect_data(band_work_path / "scf", ['efermi', 'nelec'])['collected_metrics']
    efermi, nelec = metrics['efermi'], int(metrics['nelec'])

    band_file = band_work_path / "nscf" / f"OUT.{nscf_suffix}" / "BANDS_1.dat"
    if nspin == 2:
        band_file_dw = band_work_path / "nscf" / f"OUT.{nscf_suffix}" / "BANDS_2.dat"
    
    # Read band data
    bands, kline = [], []
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
    if nspin == 2:
        bands_dw = []
        with open(band_file_dw) as fin:
            for lines in fin:
                words = lines.split()
                if len(bands_dw) == 0:
                    for _ in range(nbands):
                        bands_dw.append([])

                for i in range(nbands):
                    bands_dw[i].append(float(words[i+2]) - efermi)

    # Calculate the band gap
    band_maxes, band_mins = [], []
    for i in range(nbands):
        band_max, band_min = max(bands[i]), min(bands[i])
        if band_max * band_min <= 0: # If band crosses E_fermi
            band_gap = 0
            break
        else:
            band_maxes.append(band_max)
            band_mins.append(band_min)
    
    if nspin == 1:
        if 'band_gap' not in locals():
            # If not metallic for nspin=1, use nelec to calculate number of valence band and find CBM and VBM
            valence_bands = int(nelec / 2)
            band_gap = min(band_mins[valence_bands:]) - max(band_maxes[:valence_bands])
    
    elif nspin == 2:
        # Calculate band gap for down spin in nspin=2 case
        band_dw_maxes, band_dw_mins = [], []
        for i in range((len(bands_dw))):
            band_dw_max, band_dw_min = max(bands_dw[i]), min(bands_dw[i])
            if band_dw_max * band_dw_min <= 0: # If band crosses E_fermi
                band_gap_dw = 0
                break
            else:
                band_dw_maxes.append(band_dw_max)
                band_dw_mins.append(band_dw_min)
        
        # If not metallic, find minimun in unoccupied band and maximun in occupied band to calculate band gap
        if 'band_gap_dw' not in locals():
            occupied_band_dw_max = min(band_dw_maxes)
            for band_dw_max in band_dw_maxes:
                if band_dw_max > occupied_band_dw_max and band_dw_max <= 0:
                    occupied_band_dw_max = band_dw_max
            
            unoccupied_band_dw_min = max(band_dw_mins)
            for band_dw_min in band_dw_mins:
                if band_dw_min < unoccupied_band_dw_min and band_dw_min >= 0:
                    unoccupied_band_dw_min = band_dw_min
            
            band_gap_dw = unoccupied_band_dw_min - occupied_band_dw_max
        
        # The band gap of upper spin should be calculated using the same method with down spin 
        if 'band_gap' not in locals():
            occupied_band_max = min(band_maxes)
            for band_max in band_maxes:
                if band_max > occupied_band_max and band_max <= 0:
                    occupied_band_max = band_max
            
            unoccupied_band_min = max(band_mins)
            for band_min in band_mins:
                if band_min < unoccupied_band_min and band_min >= 0:
                    unoccupied_band_min = band_min
            
            band_gap = unoccupied_band_min - occupied_band_max

    else:
        raise NotImplementedError("Band plot for nspin=4 is not supported yet")

    # Read high symmetry labels from KPT file
    high_symm_labels = []
    band_point_nums = []
    band_point_num = 0
    with open(nscf_dir / "KPT") as fin:
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
    plt.ylim(-energy_range, energy_range)
    plt.ylabel(r"$E-E_\text{F}$/eV")
    plt.xticks(high_symm_poses, high_symm_labels)
    plt.grid()
    if nspin == 1:
        plt.title(f"Band structure of {material_name} (Gap = {band_gap:.2f} eV)")
    elif nspin == 2:
        plt.title(f"Band structure of {material_name}\n(Up spin gap = {band_gap:.2f} eV, down spin gap = {band_gap_dw:.2f} eV)")
    plt.legend()
    plt.savefig(band_work_path / band_pic_path, dpi=300)
    plt.show()

    if nspin == 1:
        return band_gap
    else:
        return [band_gap, band_gap_dw]

def abacus_cal_band_scf_pyatb(stru_file,
                               stru_type,
                               lcao,
                               band_work_path,
                               dft_functional = 'hse',
                               kspacing = 0.16,
                               scf_thr = 1e-6,
                               extra_input = None):
    """
    Do scf calculation using hybrid functionals on a uniform k-mesh.
    Used by calculating hybrid density functional band by pyatb.
    """
    pp_path = os.environ.get("ABACUS_PP_PATH")
    orb_path = os.environ.get("ABACUS_ORB_PATH")

    extra_input['kspacing'] = kspacing
    extra_input['dft_functional'] = dft_functional
    extra_input['scf_thr'] = scf_thr
    extra_input['out_mat_hs2'] = True
    extra_input['out_mat_r']   = True

    extra_input_file = "INPUT.tmp"
    WriteInput(extra_input, extra_input_file)

    try:
        _, job_path = PrepInput(files=str(stru_file),
                                filetype=stru_type,
                                jobtype='scf',
                                pp_path=pp_path,
                                orb_path=orb_path,
                                input_file=extra_input_file,
                                lcao=lcao
                                ).run()
    except Exception as e:
        raise RuntimeError(f"Error preparing input files: {e}")
    
    old_job_path = job_path[0]
    band_scf_work_path = (band_work_path / "scf").absolute()
    shutil.move(old_job_path, band_scf_work_path)

    run_abacus_onejob(band_scf_work_path)
    scf_metrics = abacus_collect_data(band_scf_work_path, ['normal_end'])['collected_metrics']

    if scf_metrics['normal_end'] is True:
        return True
    else:
        return False

def abacus_postprocess_band_pyatb(nspin,
                                  band_work_path,
                                  band_pdf_path,
                                  energy_range = 10,
                                  connect_line_points=30):
    """
    Post process SCF data and plot band using pyatb. 
    This function is used for hybrid density functional band calculation as a workaround.
    """
    pyatb_process_dir = Path(band_work_path) / 'pyatb'
    os.system(f"mkdir {pyatb_process_dir}")

    scf_dir = (band_work_path / "scf").absolute()
    scf_metrics = abacus_collect_data(scf_dir, ['efermi', 'cell'])['collected_metrics']

    input_parameters = {
        'nspin': nspin,
        'package': "ABACUS",
        'fermi_energy': scf_metrics['efermi'],
        'HR_route': "../scf/OUT.ABACUS/data-HR-sparse_SPIN0.csr",
        'SR_route': "../scf/OUT.ABACUS/data-SR-sparse_SPIN0.csr",
        'rR_route': "../scf/OUT.ABACUS/data-rR-sparse.csr",
        "HR_unit":  "Ry",
        "rR_unit": "Bohr"
    }
    if nspin == 2:
        input_parameters['HR_route'] += ' ../scf/OUT.ABACUS/data-HR-sparse_SPIN1.csr'
        input_parameters['SR_route'] += ' ../scf/OUT.ABACUS/data-SR-sparse_SPIN1.csr'

    pyatb_input_file = open((pyatb_process_dir / "Input").absolute(), "w")
    
    pyatb_input_file.write("INPUT_PARAMETERS\n{\n")
    for key, value in input_parameters.items():
        pyatb_input_file.write(f"    {key}  {value}\n")
    pyatb_input_file.write("}\n\nLATTICE\n{\n")

    pyatb_input_file.write(f"    {'lattice_constant'}  {1.8897162}\n")
    pyatb_input_file.write(f"    {'lattice_constant_unit'}  {'Bohr'}\n    lattice_vector\n")
    for cell_vec in scf_metrics['cell']:
        pyatb_input_file.write(f"    {cell_vec[0]:.8f}  {cell_vec[1]:.8f}  {cell_vec[2]:.8f}\n")
    pyatb_input_file.write("}\n\nBAND_STRUCTURE\n{\n    kpoint_mode   line\n")

    # Get kline and write to pyatb Input file
    shutil.copy(band_work_path / "scf" / "STRU", pyatb_process_dir / "STRU")
    scf_stru = AbacusStru.ReadStru(band_work_path / "scf" / "STRU")
    kpt_file = pyatb_process_dir / "KPT"
    scf_stru.get_kline_ase(point_number=connect_line_points,kpt_file=kpt_file)

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
    os.chdir(pyatb_process_dir)
    os.system("OMP_NUM_THREADS=1 pyatb")

    # read band gap
    band_gaps = []
    with open(pyatb_process_dir / "Out/Band_Structure/band_info.dat") as fin:
        for lines in fin:
            if "Band gap" in lines:
                band_gaps.append(float(lines.split()[-1]))
    
    # Modify auto generated plot_band.py and replot the band
    os.chdir(pyatb_process_dir / "Out/Band_structure")
    os.system(f'sed -i "16c y_min = {-energy_range} # eV" plot_band.py')
    os.system(f'sed -i "17c y_max =  {energy_range} # eV" plot_band.py')
    os.system("python plot_band.py")
    os.chdir(pyatb_process_dir)
    
    # Copy plotted band.pdf to given directory
    os.system(f"cp {pyatb_process_dir / 'Out' / 'Band_Structure' / 'band.pdf'} {band_work_path / band_pdf_path}")

    if nspin == 1:
        return band_gaps[0]
    else:
        return [band_gaps[0], band_gaps[1]]

@mcp.tool()
def abacus_cal_band(stru_file: str,
                    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
                    nspin: Optional[Literal[1, 2, 4]] = 1,
                    material_name: str = 'the material',
                    band_pic_path: str = "band.png",
                    energy_range: float = 10.0,
                    lcao: bool = True,
                    extra_input: Optional[Dict[str, Any]] = None
) -> TypedDict("results", {"band_gap": float, "band_picture": str}):
    """
    Calculate the electronic band of given material.
    If a band calculation is requested, this function should be used directly.
    For GGA (and rarely used LDA) DFT bands, the calculation will calculate band using converged charge of the
    same given density functional (default is PBE, specified by `dft_functional` in extra_input).
    For hybrid DFT bands, this function should not be used.
    Args:
        stru_file: Structure file in cif, poscar, or abacus/stru format.
        stru_type: Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        material_name: Name of the material, used in the title of plotted band
        nspin: Type of spin polarization. 1 means no spin polarization, 2 means collinear spin polarization, and 4 means noncollinear spin polarization. Currently only nspin=1 and 2 are supported.
        band_pic_path: Path to save picture of plotted band.
        energy_range: Set [-energy_range, energy_range] for $E-E_\text{fermi}$ in the plotted band. The default 10 eV is OK.
        lcao: Whether to use LCAO basis set, default is True.
        extra_input: Extra input parameters for ABACUS. 
    Returns:
        The bandgap (in eV) of the structure and path to the plotted band.
    
    Raises:
        NotImplementedError: If nspin=4 is requested
    """
    if extra_input is None:
        extra_input = {}
    if nspin not in extra_input.keys():
        extra_input['nspin'] = nspin    
    if nspin == 2 or nspin == 4:
        extra_input['mixing_beta'] = 0.4
    
    # Create work dir for band calculation
    cwd = Path(os.getcwd()).absolute()
    band_work_path = cwd / f"./band-{time.strftime('%Y%m%d%H%M%S')}"
    if not os.path.exists(band_work_path):
        os.mkdir(band_work_path)
    
    scfrun_state = abacus_cal_band_scf(stru_file,
                                       stru_type,
                                       lcao,
                                       extra_input,
                                       band_work_path)
    if scfrun_state is False:
        raise RuntimeError("SCF step in band calculation failed")
    
    nscfrun_state = abacus_cal_band_nscf(stru_file,
                                         stru_type,
                                         lcao,
                                         extra_input,
                                         band_work_path)
    if nscfrun_state is False:
        raise RuntimeError("NSCF step in band calculation failed")
    
    band_gap = abacus_plot_band(band_work_path,
                               material_name,
                               nspin,
                               energy_range,
                               band_pic_path)
    if nspin == 1:
        return {'band_gap': band_gap, 'band_picture': band_pic_path}
    elif nspin == 2:
        return {'band_gap_up': band_gap[0], 'band_gap_down': band_gap[1], 'band_picture': band_pic_path}
    else:
        raise NotImplementedError("Band plot for nspin=4 is not supported yet")

@mcp.tool()
def abacus_cal_band_pyatb(stru_file: str,
                          stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
                          nspin: Optional[Literal[1, 2, 4]] = 1,
                          hybrid_dft: Optional[str] = 'hse',
                          kspacing: Optional[float] = 0.16,
                          scf_thr: Optional[float] = 1e-6,
                          band_pdf_path: str = "band.pdf",
                          energy_range: float = 10.0,
                          lcao: bool = True,
                          extra_input: Optional[Dict[str, Any]] = None
) -> TypedDict("results", {"band_gap": List[float], "band_picture": str}):
    """
    Calculate band using hybrid functional using ABACUS to do SCF calculation and using pyatb to postprocess. 
    Implemented for calculate band using hybrid functionals as a workaround. Should be used only if a hybrid functional is 
    requested to calculate the band.
    Args:
        stru_file: Structure file in cif, poscar, or abacus/stru format.
        stru_type: Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        nspin: Type of spin polarization. 1 means no spin polarization, 2 means collinear spin polarization, and 4 means noncollinear spin polarization. Currently only nspin=1 and 2 are supported.
        hybrid_dft: Hybrid dft functional to be used to calculate the band. For example, `hse` means using HSE06.
        kspacing: The `kspacing` in INPUT file used in SCF calculation. Can be changed to 0.18 or 0.20 if the system is too large.
        band_pdf_path: Path to save plotted band. Currently only pdf format are supported.
        energy_range: Set [-energy_range, energy_range] for $E-E_\text{F}$ in the plotted band. The default 10 eV is generally OK.
        lcao: Whether to use LCAO basis set, default is True.
        extra_input: Extra input parameters for ABACUS. 
    Returns:
        The bandgap (in eV) of the structure and path to the plotted band. Contains only 1 value for nspin=1, and 2 values (for up and down spin respectively) for nspin=2.
    
    Raises:
        NotImplementedError: If nspin=4 is requested
    """
    if extra_input is None:
        extra_input = {}
    if nspin not in extra_input.keys():
        extra_input['nspin'] = nspin
    
    if nspin == 2 or nspin == 4:
        extra_input['mixing_beta'] = 0.4
    
    # Create work dir for band calculation
    cwd = Path(os.getcwd()).absolute()
    band_work_path = cwd / f"./band-{time.strftime('%Y%m%d%H%M%S')}"
    if not os.path.exists(band_work_path):
        os.mkdir(band_work_path)
    
    scfrun_state = abacus_cal_band_scf_pyatb(stru_file,
                                             stru_type,
                                             lcao,
                                             band_work_path,
                                             dft_functional=hybrid_dft,
                                             kspacing=kspacing,
                                             scf_thr=scf_thr,
                                             extra_input=extra_input)
    if scfrun_state is False:
        raise RuntimeError("SCF step in band calculation failed")
    
    band_gap = abacus_postprocess_band_pyatb(nspin,
                                             band_work_path,
                                             band_pdf_path,
                                             energy_range)
    
    return {'band_gap': band_gap,
            'band_picture': band_pdf_path}
