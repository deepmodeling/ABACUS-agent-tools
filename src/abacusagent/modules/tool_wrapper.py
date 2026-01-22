from pathlib import Path
import os
from typing import Literal, Optional, Dict, Any, List, Tuple, Union
from typing_extensions import TypedDict

from abacusagent.init_mcp import mcp
from abacusagent.modules.util.comm import get_relax_precision
from abacusagent.modules.submodules.abacus import abacus_prepare
from abacusagent.modules.submodules.scf import abacus_calculation_scf as _abacus_calculation_scf
from abacusagent.modules.submodules.cube import abacus_cal_elf as _abacus_cal_elf
from abacusagent.modules.submodules.band import abacus_cal_band as _abacus_cal_band
from abacusagent.modules.submodules.bader import abacus_badercharge_run as _abacus_badercharge_run
from abacusagent.modules.submodules.dos import abacus_dos_run as _abacus_dos_run
from abacusagent.modules.submodules.phonon import abacus_phonon_dispersion as _abacus_phonon_dispersion
from abacusagent.modules.submodules.elastic import abacus_cal_elastic as _abacus_cal_elastic
from abacusagent.modules.submodules.eos import abacus_eos as _abacus_eos
from abacusagent.modules.submodules.relax import abacus_do_relax as _abacus_do_relax
from abacusagent.modules.submodules.md import abacus_run_md as _abacus_run_md
from abacusagent.modules.submodules.work_function import abacus_cal_work_function as _abacus_cal_work_function
from abacusagent.modules.submodules.vacancy import abacus_cal_vacancy_formation_energy as _abacus_cal_vacancy_formation_energy

class DFTUParam(TypedDict):
    """Definition of DFT+U params"""
    element: List[str]
    orbital: List[Literal['p', 'd', 'f']]
    U_value: List[float]

class InitMagParam(TypedDict):
    """Definition of initial magnetic params"""
    element: List[str]
    mag: List[float]

def transform_dftu_param(dftu_param):
    """
    Transform DFT+U param definition used in tools in this file to the format used by abacus_prepare.
    """
    assert len(dftu_param['element']) == len(dftu_param['orbital'])
    assert len(dftu_param['element']) == len(dftu_param['U_value'])
    dftu_param_new = {}
    for i in range(len(dftu_param['element'])):
        dftu_param_new[dftu_param['element'][i]] = (dftu_param['orbital'][i], dftu_param['U_value'][i])

    return dftu_param_new

def transform_initmag_param(initmag_param):
    """
    Transform initial magnetic param definition used in tools in this file to the format used by abacus_prepare.
    """
    assert len(initmag_param['element']) == len(initmag_param['mag'])
    initmag_param_new = {}
    for i in range(len(initmag_param['element'])):
        initmag_param_new[initmag_param['element'][i]] = initmag_param['mag'][i]

    return initmag_param_new

def prepare_abacus_inputs(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2, 4] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: Optional[Union[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]],
                         Literal['auto']]] = None,
    init_mag: Optional[Dict[str, float]] = None,
    #afm: bool = False,
) -> Dict[str, Any]:
    """
    Commom prepare ABACUS inputs for ABACUS calculation in this file.
    """
    extra_input = {}
    if dft_functional in ['PBE', 'PBEsol', 'LDA', 'SCAN']:
        extra_input['dft_functional'] = dft_functional
    elif dft_functional in ['R2SCAN']:
        extra_input['dft_functional'] = 'MGGA_X_R2SCAN+MGGA_C_R2SCAN'
    elif dft_functional in [ 'HSE', 'PBE0']:
        print("Calculating with hybird functionals like HSE and PBE0 needs much longer time and much more meory than GGA functionals such as PBE.")
        extra_input['dft_functional'] = dft_functional
        os.environ['ABACUS_COMMAND'] = "OMP_NUM_THREADS=16 abacus" # Set to use OpenMP for hybrid functionals like HSE and PBE0
    else:
        print("DFT functional not supported now. Use dafault PBE functional.")
    
    abacus_prepare_outputs = abacus_prepare(stru_file=stru_file,
                                            stru_type=stru_type,
                                            lcao=lcao,
                                            nspin=nspin,
                                            #soc=soc,
                                            dftu=dftu,
                                            dftu_param=dftu_param,
                                            init_mag=init_mag,
                                            #afm=afm,
                                            extra_input=extra_input)
    
    abacus_inputs_dir = abacus_prepare_outputs['abacus_inputs_dir']
    
    return abacus_inputs_dir

def do_relax(
    abacus_inputs_dir: Path = None,
    max_steps: int = 100,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    fixed_axes: Optional[Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"]] = None,
    relax_method: Optional[Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"]] = None,
) -> Dict[str, Any]:
    """
    Do relax calculation using ABACUS.
    """
    relax_thresholds = get_relax_precision(relax_precision)
    
    if relax_cell is False: # For ABACUS LTSv3.10.0
        relax_method = 'bfgs_trad'
    else:
        relax_method = 'cg'
    
    relax_outputs = _abacus_do_relax(abacus_inputs_dir,
                                     force_thr_ev=relax_thresholds['force_thr_ev'],
                                     stress_thr_kbar=relax_thresholds['stress_thr'],
                                     max_steps=max_steps,
                                     relax_cell=relax_cell,
                                     relax_method=relax_method,
                                     fixed_axes=fixed_axes)
    
    if relax_outputs['normal_end'] is False:
        raise ValueError('Relaxation calculation failed')
    elif relax_outputs['relax_converge'] is False:
        return {"msg":f'Relaxation calculation did not converge in {max_steps} steps',
                "final_stru": Path(relax_outputs['new_abacus_inputs_dir']) / "STRU",
                **relax_outputs["result"]}
    else:
        print("Relax calculation completed successfully.")
        abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']
    
    return {'final_stru': Path(relax_outputs['new_abacus_inputs_dir']) / "STRU",
            **relax_outputs}

@mcp.tool()
def abacus_calculation_scf(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: DFTUParam = None,
    init_mag: InitMagParam = None,
    #afm: bool = False,
) -> Dict[str, Any]:
    """
    Run ABACUS SCF calculation.

    Args:
        The following parameters are same with other tools:
        stru_file (Path): Structure file in cif, poscar, or abacus/stru format.
        stru_type (Literal["cif", "poscar", "abacus/stru"] = "cif"): Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        lcao (bool): Whether to use LCAO basis set, default is True.
        nspin (int): The number of spins, can be 1 (no spin), 2 (spin polarized). Default is 1.
        dft_functional (Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN']): The DFT functional to use, can be 'PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', 'PBE0', or 'R2SCAN'. Default is 'PBE'.
            If hybrid functionals like HSE and PBE0 are used, the calculation may be much slower than GGA functionals like PBE.
        dftu (bool): Whether to use DFT+U, default is False.
        dftu_param (dict or None): The DFT+U parameters, should be a dict containing the following keys:
            - 'element' (List[str]): List of elements to apply DFT+U to.
            - 'orbital' (List[str]): List of orbitals to apply DFT+U to for each element. Should be 'p', 'd' or 'f'.
            - 'U_value' (List[float]): List of Hubbard U values for each element.
            The length of list for each key in dftu_param should be the same.
        init_mag (dict or None): The initial magnetic moment for magnetic elements, should be a dict containing the following keys:
            - 'element': List of elements to apply initial magnetic moment to.
            - 'mag': List of initial magnetic moments for each element.
            The length of list for each key in init_mag should be the same.

    Returns:
        A dictionary containing whether the SCF calculation finished normally, the SCF is converged or not and the converged SCF energy.
    """
    dftu_param = transform_dftu_param(dftu_param) if dftu_param is not None else None
    init_mag = transform_initmag_param(init_mag) if init_mag is not None else None
    abacus_inputs_dir = prepare_abacus_inputs(stru_file=stru_file,
                                              stru_type=stru_type,
                                              lcao=lcao,
                                              nspin=nspin,
                                              dft_functional=dft_functional,
                                              dftu=dftu,
                                              dftu_param=dftu_param,
                                              init_mag=init_mag)
    
    results =  _abacus_calculation_scf(abacus_inputs_dir)

    return {'energy': results.get('energy', None),
            'converge': results.get('converge', None),
            'normal_end': results.get('normal_end', None)}

@mcp.tool()
def abacus_do_relax(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: DFTUParam = None,
    init_mag: InitMagParam = None,
    #afm: bool = False,
    max_steps: int = 100,
    relax_cell: bool = False,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    relax_method: Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"] = "cg",
    fixed_axes: Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"] = None,
) -> Dict[str, Any]:
    """
    Perform relaxation calculations using ABACUS based on the provided input files. The results of the relaxation and 
    the new ABACUS input files containing final relaxed structure will be returned.
    Args:
        The following parameters are same with other tools:
        stru_file (Path): Structure file in cif, poscar, or abacus/stru format.
        stru_type (Literal["cif", "poscar", "abacus/stru"] = "cif"): Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        lcao (bool): Whether to use LCAO basis set, default is True.
        nspin (int): The number of spins, can be 1 (no spin), 2 (spin polarized). Default is 1.
        dft_functional (Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN']): The DFT functional to use, can be 'PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', 'PBE0', or 'R2SCAN'. Default is 'PBE'.
            If hybrid functionals like HSE and PBE0 are used, the calculation may be much slower than GGA functionals like PBE.
        dftu (bool): Whether to use DFT+U, default is False.
        dftu_param (dict or None): The DFT+U parameters, should be a dict containing the following keys:
            - 'element' (List[str]): List of elements to apply DFT+U to.
            - 'orbital' (List[str]): List of orbitals to apply DFT+U to for each element. Should be 'p', 'd' or 'f'.
            - 'U_value' (List[float]): List of Hubbard U values for each element.
            The length of list for each key in dftu_param should be the same.
        init_mag (dict or None): The initial magnetic moment for magnetic elements, should be a dict containing the following keys:
            - 'element': List of elements to apply initial magnetic moment to.
            - 'mag': List of initial magnetic moments for each element.
            The length of list for each key in init_mag should be the same.
        max_steps: Maximum number of relaxation steps, default is 100.
        relax_cell: Whether to relax the cell parameters, default is False.
        relax_precision (Literal['low', 'medium', 'high']): The precision of the relax calculation, can be 'low', 'medium', or 'high'. Default is 'medium'.
            'low' means the relax calculation will be done with force_thr_ev=0.05 and stress_thr_kbar=5.
            'medium' means the relax calculation will be done with force_thr_ev=0.01 and stress_thr_kbar=1.0.
            'high' means the relax calculation will be done with force_thr_ev=0.005 and stress_thr_kbar=0.5.
        relax_method: The relaxation method to use, can be 'cg', 'bfgs', 'bfgs_trad', 'cg_bfgs', 'sd', or 'fire'. Default is 'cg'.
        fixed_axes: Specifies which axes to fix during relaxation. Only effective when `relax_cell` is True. Options are:
            - None: relax all axes (default)
            - volume: relax with fixed volume
            - shape: relax with fixed shape but changing volume (i.e. only lattice constant changes)
            - a: fix a axis
            - b: fix b axis
            - c: fix c axis
            - ab: fix both a and b axes
            - ac: fix both a and c axes
            - bc: fix both b and c axes
    Returns:
        A dictionary containing result of the relaxation calculation:
            - final_stru: The final STRU file path after relaxation.
            - relax_steps: The number of relaxation steps taken.
            - relax_converge: Whether the relaxation converged.
    """
    dftu_param = transform_dftu_param(dftu_param) if dftu_param is not None else None
    init_mag = transform_initmag_param(init_mag) if init_mag is not None else None
    abacus_inputs_dir = prepare_abacus_inputs(stru_file=stru_file,
                                              stru_type=stru_type,
                                              lcao=lcao,
                                              nspin=nspin,
                                              dft_functional=dft_functional,
                                              dftu=dftu,
                                              dftu_param=dftu_param,
                                              init_mag=init_mag)
    
    relax_outputs = do_relax(abacus_inputs_dir=abacus_inputs_dir,
                             max_steps=max_steps,
                             relax_cell=relax_cell,
                             relax_precision=relax_precision,
                             fixed_axes=fixed_axes,
                             relax_method=relax_method)

    return {'final_stru': relax_outputs.get('final_stru', None),
            'relax_converge': relax_outputs.get('relax_converge', None),
            'relax_steps': relax_outputs.get('relax_steps', None),}

@mcp.tool()
def abacus_badercharge_run(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: DFTUParam = None,
    init_mag: InitMagParam = None,
    #afm: bool = False,
    max_steps: int = 100,
    relax: bool = False,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    relax_method: Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"] = "cg",
    fixed_axes: Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"] = None,
) -> List[float]:
    """
    Calculate Bader charges for a given structure file, with ABACUS as
    the dft software to calculate the charge density, and then postprocess
    the charge density with the cube manipulator and Bader analysis.

    Args:
        The following parameters are commom for all properties:
        stru_file (Path): Structure file in cif, poscar, or abacus/stru format.
        stru_type (Literal["cif", "poscar", "abacus/stru"] = "cif"): Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        lcao (bool): Whether to use LCAO basis set, default is True.
        nspin (int): The number of spins, can be 1 (no spin), 2 (spin polarized). Default is 1.
        dft_functional (Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN']): The DFT functional to use, can be 'PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', 'PBE0', or 'R2SCAN'. Default is 'PBE'.
            If hybrid functionals like HSE and PBE0 are used, the calculation may be much slower than GGA functionals like PBE.
        dftu (bool): Whether to use DFT+U, default is False.
        dftu_param (dict or None): The DFT+U parameters, should be a dict containing the following keys:
            - 'element' (List[str]): List of elements to apply DFT+U to.
            - 'orbital' (List[str]): List of orbitals to apply DFT+U to for each element. Should be 'p', 'd' or 'f'.
            - 'U_value' (List[float]): List of Hubbard U values for each element.
            The length of list for each key in dftu_param should be the same.
        init_mag (dict or None): The initial magnetic moment for magnetic elements, should be a dict containing the following keys:
            - 'element': List of elements to apply initial magnetic moment to.
            - 'mag': List of initial magnetic moments for each element.
            The length of list for each key in init_mag should be the same.
        max_steps (int): The maximum number of steps for the relax calculation. Default is 100.
        relax: Whether to do a relax calculation before doing the property calculation. Default is False.
            If the calculated property is phonon dispersion or elastic properties, the crystal should be relaxed first with relax_cell set to True and `relax_precision` is strongly recommended be set to `high`.
        relax_cell (bool): Whether to relax the cell size during the relax calculation. Default is True.
        relax_precision (Literal['low', 'medium', 'high']): The precision of the relax calculation, can be 'low', 'medium', or 'high'. Default is 'medium'.
            'low' means the relax calculation will be done with force_thr_ev=0.05 and stress_thr_kbar=5.
            'medium' means the relax calculation will be done with force_thr_ev=0.01 and stress_thr_kbar=1.0.
            'high' means the relax calculation will be done with force_thr_ev=0.005 and stress_thr_kbar=0.5.
        fixed_axes: Specifies which axes to fix during relaxation. Only effective when `relax_cell` is True. Options are:
            - None: relax all axes (default)
            - volume: relax with fixed volume
            - shape: relax with fixed shape but changing volume (i.e. only lattice constant changes)
            - a: fix a axis
            - b: fix b axis
            - c: fix c axis
            - ab: fix both a and b axes
            - ac: fix both a and c axes
            - bc: fix both b and c axes

    Returns:
    dict: A dictionary containing: 
        - bader_results_csv: Path to the Bader results csv file.
    """
    dftu_param = transform_dftu_param(dftu_param) if dftu_param is not None else None
    init_mag = transform_initmag_param(init_mag) if init_mag is not None else None
    abacus_inputs_dir = prepare_abacus_inputs(stru_file=stru_file,
                                              stru_type=stru_type,
                                              lcao=lcao,
                                              nspin=nspin,
                                              dft_functional=dft_functional,
                                              dftu=dftu,
                                              dftu_param=dftu_param,
                                              init_mag=init_mag)
    
    if relax:
        relax_outputs = do_relax(abacus_inputs_dir=abacus_inputs_dir,
                                 max_steps=max_steps,
                                 relax_cell=relax_cell,
                                 relax_precision=relax_precision,
                                 fixed_axes=fixed_axes,
                                 relax_method=relax_method)

        abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']

    badercharge_results = _abacus_badercharge_run(abacus_inputs_dir)

    return {"bader_result_csv": Path(badercharge_results['bader_result_csv']).absolute()}

@mcp.tool()
def abacus_dos_run(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: DFTUParam = None,
    init_mag: InitMagParam = None,
    #afm: bool = False,
    max_steps: int = 100,
    relax: bool = False,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    relax_method: Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"] = "cg",
    fixed_axes: Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"] = None,
    pdos_mode: Literal['species', 'species+shell', 'species+orbital'] = 'species+shell',
    dos_edelta_ev: float = 0.01,
    dos_sigma: float = 0.07,
    dos_scale: float = 0.01,
    dos_emin_ev: float = None,
    dos_emax_ev: float = None,
    dos_nche: int = None,
) -> Dict[str, Any]:
    """
    Run the DOS and PDOS calculation.
    If the INPUT parameter "basis_type" is "PW", then out_dos will be set to 1, and only DOS will be calculated and plotted.
    If the INPUT parameter "basis_type" is "LCAO", then out_dos will be set to 2, and both DOS and PDOS will be calculated and plotted.

    Args:
        The following parameters are commom for all properties:
        stru_file (Path): Structure file in cif, poscar, or abacus/stru format.
        stru_type (Literal["cif", "poscar", "abacus/stru"] = "cif"): Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        lcao (bool): Whether to use LCAO basis set, default is True.
        nspin (int): The number of spins, can be 1 (no spin), 2 (spin polarized). Default is 1.
        dft_functional (Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN']): The DFT functional to use, can be 'PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', 'PBE0', or 'R2SCAN'. Default is 'PBE'.
            If hybrid functionals like HSE and PBE0 are used, the calculation may be much slower than GGA functionals like PBE.
        dftu (bool): Whether to use DFT+U, default is False.
        dftu_param (dict or None): The DFT+U parameters, should be a dict containing the following keys:
            - 'element' (List[str]): List of elements to apply DFT+U to.
            - 'orbital' (List[str]): List of orbitals to apply DFT+U to for each element. Should be 'p', 'd' or 'f'.
            - 'U_value' (List[float]): List of Hubbard U values for each element.
            The length of list for each key in dftu_param should be the same.
        init_mag (dict or None): The initial magnetic moment for magnetic elements, should be a dict containing the following keys:
            - 'element': List of elements to apply initial magnetic moment to.
            - 'mag': List of initial magnetic moments for each element.
            The length of list for each key in init_mag should be the same.
        max_steps (int): The maximum number of steps for the relax calculation. Default is 100.
        relax: Whether to do a relax calculation before doing the property calculation. Default is False.
            If the calculated property is phonon dispersion or elastic properties, the crystal should be relaxed first with relax_cell set to True and `relax_precision` is strongly recommended be set to `high`.
        relax_cell (bool): Whether to relax the cell size during the relax calculation. Default is True.
        relax_precision (Literal['low', 'medium', 'high']): The precision of the relax calculation, can be 'low', 'medium', or 'high'. Default is 'medium'.
            'low' means the relax calculation will be done with force_thr_ev=0.05 and stress_thr_kbar=5.
            'medium' means the relax calculation will be done with force_thr_ev=0.01 and stress_thr_kbar=1.0.
            'high' means the relax calculation will be done with force_thr_ev=0.005 and stress_thr_kbar=0.5.
        fixed_axes: Specifies which axes to fix during relaxation. Only effective when `relax_cell` is True. Options are:
            - None: relax all axes (default)
            - volume: relax with fixed volume
            - shape: relax with fixed shape but changing volume (i.e. only lattice constant changes)
            - a: fix a axis
            - b: fix b axis
            - c: fix c axis
            - ab: fix both a and b axes
            - ac: fix both a and c axes
            - bc: fix both b and c axes
        pdos_mode: Mode of plotted PDOS file.
            - "species": Total PDOS of any species will be plotted in a picture.
            - "species+shell": PDOS for any shell (s, p, d, f, g,...) of any species will be plotted. PDOS of a shell of a species willbe plotted in a subplot.
            - “species+orbital": Orbital-resolved PDOS will be plotted. PDOS of orbitals in the same shell of a species will be plotted in a subplot.
        dos_edelta_ev: Step size in writing Density of States (DOS) in eV.
        dos_sigma: Width of the Gaussian factor when obtaining smeared Density of States (DOS) in eV. 
        dos_scale: Defines the energy range of DOS output as (emax-emin)*(1+dos_scale), centered at (emax+emin)/2. 
                   This parameter will be used when dos_emin_ev and dos_emax_ev are not set.
        dos_emin_ev: Minimal range for Density of States (DOS) in eV.
        dos_emax_ev: Maximal range for Density of States (DOS) in eV.
        dos_nche: The order of Chebyshev expansions when using Stochastic Density Functional Theory (SDFT) to calculate DOS.
    
    Returns:
        Dict[str, Any]: A dictionary containing:
            - dos_fig_path: Path to the plotted DOS.
            - pdos_fig_path: Path to the plotted PDOS. Only for LCAO basis.
            - scf_normal_end: If the SCF calculation ended normally.
            - scf_converge: If the SCF calculation converged.
            - scf_energy: The calculated energy of SCF calculation.
            - nscf_normal_end: If the SCF calculation ended normally.
    """
    dftu_param = transform_dftu_param(dftu_param) if dftu_param is not None else None
    init_mag = transform_initmag_param(init_mag) if init_mag is not None else None
    abacus_inputs_dir = prepare_abacus_inputs(stru_file=stru_file,
                                              stru_type=stru_type,
                                              lcao=lcao,
                                              nspin=nspin,
                                              dft_functional=dft_functional,
                                              dftu=dftu,
                                              dftu_param=dftu_param,
                                              init_mag=init_mag)
    
    if relax:
        relax_outputs = do_relax(abacus_inputs_dir=abacus_inputs_dir,
                                 max_steps=max_steps,
                                 relax_cell=relax_cell,
                                 relax_precision=relax_precision,
                                 fixed_axes=fixed_axes,
                                 relax_method=relax_method)
        abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']

    dos_results = _abacus_dos_run(abacus_inputs_dir,
                                  pdos_mode,
                                  dos_edelta_ev,
                                  dos_sigma,
                                  dos_scale,
                                  dos_emin_ev,
                                  dos_emax_ev,
                                  dos_nche)
    
    return {'dos_fig_path': dos_results.get('dos_fig_path', None),
            'pdos_fig_path': dos_results.get('pdos_fig_path', None),
            'scf_normal_end': dos_results.get('scf_normal_end', None),
            'scf_converge': dos_results.get('scf_converge', None),
            'scf_energy': dos_results.get('scf_energy', None),
            'nscf_normal_end': dos_results.get('nscf_normal_end', None)}

@mcp.tool()
def abacus_cal_band(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: DFTUParam = None,
    init_mag: InitMagParam = None,
    #afm: bool = False,
    max_steps: int = 100,
    relax: bool = False,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    relax_method: Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"] = "cg",
    fixed_axes: Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"] = None,
    mode: Literal["nscf", "pyatb", "auto"] = "auto",
    energy_min: float = -10,
    energy_max: float = 10,
    insert_point_nums: int = 30
) -> Dict[str, Any]:
    """
    Calculate band using ABACUS for the given structure.
    PYATB or ABACUS NSCF calculation will be used according to parameters in INPUT.

    Args:
        The following parameters are commom for all properties:
        stru_file (Path): Structure file in cif, poscar, or abacus/stru format.
        stru_type (Literal["cif", "poscar", "abacus/stru"] = "cif"): Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        lcao (bool): Whether to use LCAO basis set, default is True.
        nspin (int): The number of spins, can be 1 (no spin), 2 (spin polarized). Default is 1.
        dft_functional (Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN']): The DFT functional to use, can be 'PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', 'PBE0', or 'R2SCAN'. Default is 'PBE'.
            If hybrid functionals like HSE and PBE0 are used, the calculation may be much slower than GGA functionals like PBE.
        dftu (bool): Whether to use DFT+U, default is False.
        dftu_param (dict or None): The DFT+U parameters, should be a dict containing the following keys:
            - 'element' (List[str]): List of elements to apply DFT+U to.
            - 'orbital' (List[str]): List of orbitals to apply DFT+U to for each element. Should be 'p', 'd' or 'f'.
            - 'U_value' (List[float]): List of Hubbard U values for each element.
            The length of list for each key in dftu_param should be the same.
        init_mag (dict or None): The initial magnetic moment for magnetic elements, should be a dict containing the following keys:
            - 'element': List of elements to apply initial magnetic moment to.
            - 'mag': List of initial magnetic moments for each element.
            The length of list for each key in init_mag should be the same.
        max_steps (int): The maximum number of steps for the relax calculation. Default is 100.
        relax: Whether to do a relax calculation before doing the property calculation. Default is False.
            If the calculated property is phonon dispersion or elastic properties, the crystal should be relaxed first with relax_cell set to True and `relax_precision` is strongly recommended be set to `high`.
        relax_cell (bool): Whether to relax the cell size during the relax calculation. Default is True.
        relax_precision (Literal['low', 'medium', 'high']): The precision of the relax calculation, can be 'low', 'medium', or 'high'. Default is 'medium'.
            'low' means the relax calculation will be done with force_thr_ev=0.05 and stress_thr_kbar=5.
            'medium' means the relax calculation will be done with force_thr_ev=0.01 and stress_thr_kbar=1.0.
            'high' means the relax calculation will be done with force_thr_ev=0.005 and stress_thr_kbar=0.5.
        fixed_axes: Specifies which axes to fix during relaxation. Only effective when `relax_cell` is True. Options are:
            - None: relax all axes (default)
            - volume: relax with fixed volume
            - shape: relax with fixed shape but changing volume (i.e. only lattice constant changes)
            - a: fix a axis
            - b: fix b axis
            - c: fix c axis
            - ab: fix both a and b axes
            - ac: fix both a and c axes
            - bc: fix both b and c axes
        mode: Method used to plot band. Should be `auto`, `pyatb` or `nscf`. 
            - `nscf` means using `nscf` calculation in ABACUS to calculate and plot the band
            - `pyatb` means using PYATB to plot the band
            - `auto` means deciding use `nscf` or `pyatb` mode according to the `basis_type` in INPUT file and files included in `abacus_inputs_dir`.
                -- If charge files are in `abacus_input_dir`, `nscf` mode will be used.
                -- If matrix files are in `abacus_input_dir`, `pyatb` mode will be used.
                -- If no matrix file or charge file are in `abacus_input_dir`, will determine mode by `basis_type`. If `basis_type` is lcao, will use `pyatb` mode.
                    If `basis_type` is pw, will use `nscf` mode.
        energy_min (float): Lower bound of $E - E_F$ in the plotted band.
        energy_max (float): Upper bound of $E - E_F$ in the plotted band.
        insert_point_nums (int): Number of points to insert between two high symmetry points. Default is 30.
    
    Returns:
        A dictionary containing band gap and path to the plotted band.
    """
    dftu_param = transform_dftu_param(dftu_param) if dftu_param is not None else None
    init_mag = transform_initmag_param(init_mag) if init_mag is not None else None
    abacus_inputs_dir = prepare_abacus_inputs(stru_file=stru_file,
                                              stru_type=stru_type,
                                              lcao=lcao,
                                              nspin=nspin,
                                              dft_functional=dft_functional,
                                              dftu=dftu,
                                              dftu_param=dftu_param,
                                              init_mag=init_mag)
    
    if relax:
        relax_outputs = do_relax(abacus_inputs_dir=abacus_inputs_dir,
                                 max_steps=max_steps,
                                 relax_cell=relax_cell,
                                 relax_precision=relax_precision,
                                 fixed_axes=fixed_axes,
                                 relax_method=relax_method)
        abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']
    
    band_calculation_outputs = _abacus_cal_band(abacus_inputs_dir,
                                                mode,
                                                kpath=None,
                                                high_symm_points=None,
                                                energy_min=energy_min,
                                                energy_max=energy_max,
                                                insert_point_nums=insert_point_nums)
    
    return {'band_gap': band_calculation_outputs.get('band_gap', None),
            'band_picture': band_calculation_outputs.get('band_picture', None),
            'message': band_calculation_outputs.get('message', None),}

@mcp.tool()
def abacus_phonon_dispersion(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: DFTUParam = None,
    init_mag: InitMagParam = None,
    #afm: bool = False,
    max_steps: int = 100,
    relax: bool = True,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    relax_method: Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"] = "cg",
    fixed_axes: Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"] = None,
    supercell: Optional[List[int]] = None,
    displacement_stepsize: float = 0.01,
    temperature: Optional[float] = 298.15,
    min_supercell_length: float = 10.0,
    #qpath: Optional[Union[List[str], List[List[str]]]] = None,
    #high_symm_points: Optional[Dict[str, List[float]]] = None
) -> Dict[str, Any]:
    """
    Calculate phonon dispersion with finite-difference method using Phonopy with ABACUS as the calculator. 

    Args:
        The following parameters are commom for all properties:
        stru_file (Path): Structure file in cif, poscar, or abacus/stru format.
        stru_type (Literal["cif", "poscar", "abacus/stru"] = "cif"): Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        lcao (bool): Whether to use LCAO basis set, default is True.
        nspin (int): The number of spins, can be 1 (no spin), 2 (spin polarized). Default is 1.
        dft_functional (Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN']): The DFT functional to use, can be 'PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', 'PBE0', or 'R2SCAN'. Default is 'PBE'.
            If hybrid functionals like HSE and PBE0 are used, the calculation may be much slower than GGA functionals like PBE.
        dftu (bool): Whether to use DFT+U, default is False.
        dftu_param (dict or None): The DFT+U parameters, should be a dict containing the following keys:
            - 'element' (List[str]): List of elements to apply DFT+U to.
            - 'orbital' (List[str]): List of orbitals to apply DFT+U to for each element. Should be 'p', 'd' or 'f'.
            - 'U_value' (List[float]): List of Hubbard U values for each element.
            The length of list for each key in dftu_param should be the same.
        init_mag (dict or None): The initial magnetic moment for magnetic elements, should be a dict containing the following keys:
            - 'element': List of elements to apply initial magnetic moment to.
            - 'mag': List of initial magnetic moments for each element.
            The length of list for each key in init_mag should be the same.
        max_steps (int): The maximum number of steps for the relax calculation. Default is 100.
        relax: Whether to do a relax calculation before doing the property calculation. Default is False.
            If the calculated property is phonon dispersion or elastic properties, the crystal should be relaxed first with relax_cell set to True and `relax_precision` is strongly recommended be set to `high`.
        relax_cell (bool): Whether to relax the cell size during the relax calculation. Default is True.
        relax_precision (Literal['low', 'medium', 'high']): The precision of the relax calculation, can be 'low', 'medium', or 'high'. Default is 'medium'.
            'low' means the relax calculation will be done with force_thr_ev=0.05 and stress_thr_kbar=5.
            'medium' means the relax calculation will be done with force_thr_ev=0.01 and stress_thr_kbar=1.0.
            'high' means the relax calculation will be done with force_thr_ev=0.005 and stress_thr_kbar=0.5.
        fixed_axes: Specifies which axes to fix during relaxation. Only effective when `relax_cell` is True. Options are:
            - None: relax all axes (default)
            - volume: relax with fixed volume
            - shape: relax with fixed shape but changing volume (i.e. only lattice constant changes)
            - a: fix a axis
            - b: fix b axis
            - c: fix c axis
            - ab: fix both a and b axes
            - ac: fix both a and c axes
            - bc: fix both b and c axes
        supercell (List[int], optional): Supercell matrix for phonon calculations. If default value None are used,
            the supercell matrix will be determined by how large a supercell can have a length of lattice vector
            along all 3 directions larger than 10.0 Angstrom.
        displacement_stepsize (float, optional): Displacement step size for finite difference. Defaults to 0.01 Angstrom.
        temperature (float, optional): Temperature in Kelvin for thermal properties. Defaults to 298.15. Units in Kelvin.
        min_supercell_length (float): If supercell is not provided, the generated supercell will have a length of lattice vector
            along all 3 directions larger than min_supercell_length. Defaults to 10.0 Angstrom. Units in Angstrom.
    
    Returns:
        A dictionary containing:
            - band_dos_plot: Path to the phonon dispersion plot.
            - entropy: Entropy at the specified temperature.
            - free_energy: Free energy at the specified temperature.
            - heat_capacity: Heat capacity at the specified temperature.
            - max_frequency_THz: Maximum phonon frequency in THz.
            - max_frequency_K: Maximum phonon frequency in Kelvin.
    """
    dftu_param = transform_dftu_param(dftu_param) if dftu_param is not None else None
    init_mag = transform_initmag_param(init_mag) if init_mag is not None else None
    abacus_inputs_dir = prepare_abacus_inputs(stru_file=stru_file,
                                              stru_type=stru_type,
                                              lcao=lcao,
                                              nspin=nspin,
                                              dft_functional=dft_functional,
                                              dftu=dftu,
                                              dftu_param=dftu_param,
                                              init_mag=init_mag)
    
    if relax:
        relax_outputs = do_relax(abacus_inputs_dir=abacus_inputs_dir,
                                 max_steps=max_steps,
                                 relax_cell=relax_cell,
                                 relax_precision=relax_precision,
                                 fixed_axes=fixed_axes,
                                 relax_method=relax_method)
        abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']
    
    phonon_outputs = _abacus_phonon_dispersion(abacus_inputs_dir,
                                               supercell,
                                               displacement_stepsize,
                                               temperature,
                                               min_supercell_length,
                                               qpath=None,
                                               high_symm_points=None)
    
    return {'band_dos_plot': phonon_outputs.get('band_dos_plot', None),
            'entropy': phonon_outputs.get('entropy', None),
            'free_energy': phonon_outputs.get('free_energy', None),
            'heat_capacity': phonon_outputs.get('heat_capacity', None),
            'max_frequency_THz': phonon_outputs.get('max_frequency_THz', None),
            'max_frequency_K': phonon_outputs.get('max_frequency_K', None)}

@mcp.tool()
def abacus_cal_elastic(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: DFTUParam = None,
    init_mag: InitMagParam = None,
    #afm: bool = False,
    max_steps: int = 100,
    relax: bool = True,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    relax_method: Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"] = "cg",
    fixed_axes: Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"] = None,
    norm_strain: float = 0.01,
    shear_strain: float = 0.01,
    kspacing: float = 0.08,
    relax_force_thr_ev: float = 0.01
) -> Dict[str, Any]:
    """
    Calculate various elastic constants for a given structure using ABACUS. 

    Args:
        The following parameters are commom for all properties:
        stru_file (Path): Structure file in cif, poscar, or abacus/stru format.
        stru_type (Literal["cif", "poscar", "abacus/stru"] = "cif"): Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        lcao (bool): Whether to use LCAO basis set, default is True.
        nspin (int): The number of spins, can be 1 (no spin), 2 (spin polarized), or 4 (non-collinear spin). Default is 1.
        dft_functional (Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN']): The DFT functional to use, can be 'PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', 'PBE0', or 'R2SCAN'. Default is 'PBE'.
            If hybrid functionals like HSE and PBE0 are used, the calculation may be much slower than GGA functionals like PBE.
        dftu (bool): Whether to use DFT+U, default is False.
        dftu_param (dict or None): The DFT+U parameters, should be a dict containing the following keys:
            - 'element' (List[str]): List of elements to apply DFT+U to.
            - 'orbital' (List[str]): List of orbitals to apply DFT+U to for each element. Should be 'p', 'd' or 'f'.
            - 'U_value' (List[float]): List of Hubbard U values for each element.
            The length of list for each key in dftu_param should be the same.
        init_mag (dict or None): The initial magnetic moment for magnetic elements, should be a dict containing the following keys:
            - 'element': List of elements to apply initial magnetic moment to.
            - 'mag': List of initial magnetic moments for each element.
            The length of list for each key in init_mag should be the same.
        max_steps (int): The maximum number of steps for the relax calculation. Default is 100.
        relax: Whether to do a relax calculation before doing the property calculation. Default is False.
            If the calculated property is phonon dispersion or elastic properties, the crystal should be relaxed first with relax_cell set to True and `relax_precision` is strongly recommended be set to `high`.
        relax_cell (bool): Whether to relax the cell size during the relax calculation. Default is True.
        relax_precision (Literal['low', 'medium', 'high']): The precision of the relax calculation, can be 'low', 'medium', or 'high'. Default is 'medium'.
            'low' means the relax calculation will be done with force_thr_ev=0.05 and stress_thr_kbar=5.
            'medium' means the relax calculation will be done with force_thr_ev=0.01 and stress_thr_kbar=1.0.
            'high' means the relax calculation will be done with force_thr_ev=0.005 and stress_thr_kbar=0.5.
        fixed_axes: Specifies which axes to fix during relaxation. Only effective when `relax_cell` is True. Options are:
            - None: relax all axes (default)
            - volume: relax with fixed volume
            - shape: relax with fixed shape but changing volume (i.e. only lattice constant changes)
            - a: fix a axis
            - b: fix b axis
            - c: fix c axis
            - ab: fix both a and b axes
            - ac: fix both a and c axes
            - bc: fix both b and c axes
        norm_strain (float): Normal strain to calculate elastic constants, default is 0.01.
        shear_strain (float): Shear strain to calculate elastic constants, default is 0.01.
        kspacing (float): K-point spacing for ABACUS calculation, default is 0.08. Units in Bohr^{-1}.
        relax_force_thr_ev (float): Threshold for force convergence of the relax calculation for each deformed structure, default is 0.02. Units in eV/Angstrom.
    
    Returns:
        A dictionary containing the following keys:
        - elastic_tensor (np.array in (6,6) dimension): Calculated elastic constants in Voigt notation. Units in GPa.
        - bulk_modulus (float): Calculated bulk modulus in GPa.
        - shear_modulus (float): Calculated shear modulus in GPa.
        - young_modulus (float): Calculated Young's modulus in GPa.
        - poisson_ratio (float): Calculated Poisson's ratio.
    """
    dftu_param = transform_dftu_param(dftu_param) if dftu_param is not None else None
    init_mag = transform_initmag_param(init_mag) if init_mag is not None else None
    abacus_inputs_dir = prepare_abacus_inputs(stru_file=stru_file,
                                              stru_type=stru_type,
                                              lcao=lcao,
                                              nspin=nspin,
                                              dft_functional=dft_functional,
                                              dftu=dftu,
                                              dftu_param=dftu_param,
                                              init_mag=init_mag)
    
    if relax:
        relax_outputs = do_relax(abacus_inputs_dir=abacus_inputs_dir,
                                 max_steps=max_steps,
                                 relax_cell=relax_cell,
                                 relax_precision=relax_precision,
                                 fixed_axes=fixed_axes,
                                 relax_method=relax_method)
        abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']
    
    elactic_outputs = _abacus_cal_elastic(abacus_inputs_dir,
                                          norm_strain,
                                          shear_strain,
                                          kspacing,
                                          relax_force_thr_ev)
    
    return {'elastic_tensor': elactic_outputs.get('elastic_tensor', None),
            'bulk_modulus': elactic_outputs.get('bulk_modulus', None),
            'shear_modulus': elactic_outputs.get('shear_modulus', None),
            'young_modulus': elactic_outputs.get('young_modulus', None),
            'poisson_ratio': elactic_outputs.get('poisson_ratio', None),}

@mcp.tool()
def abacus_vacancy_formation_energy(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: DFTUParam = None,
    init_mag: InitMagParam = None,
    #afm: bool = False,
    max_steps: int = 100,
    relax: bool = True,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    relax_method: Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"] = "cg",
    fixed_axes: Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"] = None,
    supercell: List[int] = [1, 1, 1],
    vacancy_index: int = 1,
    vacancy_relax_precision: Literal['low', 'medium', 'high'] = 'low',
) -> Dict[str, Any]:
    """
    Calculate vacancy formation energy. Currenly only non-charged vacancy of limited elements are suppoted. 
    Supported elements include: Li, Be, Na, Mg, Al, Si, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, 
    Ge, Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Cs, Ba, La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, 
    Dy, Ho, Er, Tm, Yb, Lu, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb.
    The most stable crystal structure are used.

    Args:
        The following parameters are commom for all properties:
        stru_file (Path): Structure file in cif, poscar, or abacus/stru format.
        stru_type (Literal["cif", "poscar", "abacus/stru"] = "cif"): Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        lcao (bool): Whether to use LCAO basis set, default is True.
        nspin (int): The number of spins, can be 1 (no spin), 2 (spin polarized). Default is 1.
        dft_functional (Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN']): The DFT functional to use, can be 'PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', 'PBE0', or 'R2SCAN'. Default is 'PBE'.
            If hybrid functionals like HSE and PBE0 are used, the calculation may be much slower than GGA functionals like PBE.
        dftu (bool): Whether to use DFT+U, default is False.
        dftu_param (dict or None): The DFT+U parameters, should be a dict containing the following keys:
            - 'element' (List[str]): List of elements to apply DFT+U to.
            - 'orbital' (List[str]): List of orbitals to apply DFT+U to for each element. Should be 'p', 'd' or 'f'.
            - 'U_value' (List[float]): List of Hubbard U values for each element.
            The length of list for each key in dftu_param should be the same.
        init_mag (dict or None): The initial magnetic moment for magnetic elements, should be a dict containing the following keys:
            - 'element': List of elements to apply initial magnetic moment to.
            - 'mag': List of initial magnetic moments for each element.
            The length of list for each key in init_mag should be the same.
        max_steps (int): The maximum number of steps for the relax calculation. Default is 100.
        relax: Whether to do a relax calculation before doing the property calculation. Default is False.
            If the calculated property is phonon dispersion or elastic properties, the crystal should be relaxed first with relax_cell set to True and `relax_precision` is strongly recommended be set to `high`.
        relax_cell (bool): Whether to relax the cell size during the relax calculation. Default is True.
        relax_precision (Literal['low', 'medium', 'high']): The precision of the relax calculation, can be 'low', 'medium', or 'high'. Default is 'medium'.
            'low' means the relax calculation will be done with force_thr_ev=0.05 and stress_thr_kbar=5.
            'medium' means the relax calculation will be done with force_thr_ev=0.01 and stress_thr_kbar=1.0.
            'high' means the relax calculation will be done with force_thr_ev=0.005 and stress_thr_kbar=0.5.
        fixed_axes: Specifies which axes to fix during relaxation. Only effective when `relax_cell` is True. Options are:
            - None: relax all axes (default)
            - volume: relax with fixed volume
            - shape: relax with fixed shape but changing volume (i.e. only lattice constant changes)
            - a: fix a axis
            - b: fix b axis
            - c: fix c axis
            - ab: fix both a and b axes
            - ac: fix both a and c axes
            - bc: fix both b and c axes
        abacus_inputs_dir (Path): Path to the directory containing the ABACUS inputs.
        supercell (List[int]): Supercell matrix. Defaults to [1, 1, 1], which means no supercell.
        vacancy_index (int): Index of the vacancy element. The index starts from 1 and is in the original structure. Defaults to 1.
        relax_precision (Literal['low', 'medium', 'high']): Precision of the relax calculation. The unit of `force_thr_ev` is eV/Angstrom, and the unit of `stress_thr` is kbar.
        - 'low' means the relax calculation will be done with force_thr_ev=0.05 and stress_thr=5.0.
        - 'medium' means the relax calculation will be done with force_thr_ev=0.01 and stress_thr=1.0.
        - 'high' means the relax calculation will be done with force_thr_ev=0.005 and stress_thr=0.5.
    Returns:
        A dictionary containing:
        - "vacancy_formation_energy": Calculated vacancy formation energy.
        - "original_stru_job_relax_converge": If the relax calculation of original structure is converged.
        - "defect_supercell_job_relax_converge": If the defect supercell relax calculation is converged.
    """
    dftu_param = transform_dftu_param(dftu_param) if dftu_param is not None else None
    init_mag = transform_initmag_param(init_mag) if init_mag is not None else None
    abacus_inputs_dir = prepare_abacus_inputs(stru_file=stru_file,
                                              stru_type=stru_type,
                                              lcao=lcao,
                                              nspin=nspin,
                                              dft_functional=dft_functional,
                                              dftu=dftu,
                                              dftu_param=dftu_param,
                                              init_mag=init_mag)
    
    if relax:
        relax_outputs = do_relax(abacus_inputs_dir=abacus_inputs_dir,
                                 max_steps=max_steps,
                                 relax_cell=relax_cell,
                                 relax_precision=relax_precision,
                                 fixed_axes=fixed_axes,
                                 relax_method=relax_method)
        abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']
    
    if stru_type in ['cif', 'poscar']:
        # Get actual atom index in transformed STRU file
        from abacustest.lib_model.model_017_vacancy import get_categorized_idx
        categorized_idx = get_categorized_idx(stru_file, stru_type)
        vacancy_index = categorized_idx.index(vacancy_index-1) + 1

    vacancy_outputs = _abacus_cal_vacancy_formation_energy(abacus_inputs_dir,
                                                           supercell,
                                                           vacancy_index,
                                                           vacancy_relax_precision)
    
    return {'vacancy_formation_energy': vacancy_outputs.get('vac_formation_energy', None),
            'original_stru_job_relax_converge': vacancy_outputs.get('original_stru_job_relax_converge', None),
            'defect_supercell_job_relax_converge': vacancy_outputs.get('defect_supercell_job_relax_converge', None),}

@mcp.tool()
def abacus_cal_work_function(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: DFTUParam = None,
    init_mag: InitMagParam = None,
    #afm: bool = False,
    max_steps: int = 100,
    relax: bool = False,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    relax_method: Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"] = "cg",
    fixed_axes: Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"] = None,
    vacuum_direction: Literal['x', 'y', 'z'] = 'z',
    dipole_correction: bool = False,
) -> Dict[str, Any]:
    """
    Calculate the electrostatic potential and work function using ABACUS.
    
    Args:
        The following parameters are commom for all properties:
        stru_file (Path): Structure file in cif, poscar, or abacus/stru format.
        stru_type (Literal["cif", "poscar", "abacus/stru"] = "cif"): Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        lcao (bool): Whether to use LCAO basis set, default is True.
        nspin (int): The number of spins, can be 1 (no spin), 2 (spin polarized). Default is 1.
        dft_functional (Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN']): The DFT functional to use, can be 'PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', 'PBE0', or 'R2SCAN'. Default is 'PBE'.
            If hybrid functionals like HSE and PBE0 are used, the calculation may be much slower than GGA functionals like PBE.
        dftu (bool): Whether to use DFT+U, default is False.
        dftu_param (dict or None): The DFT+U parameters, should be a dict containing the following keys:
            - 'element' (List[str]): List of elements to apply DFT+U to.
            - 'orbital' (List[str]): List of orbitals to apply DFT+U to for each element. Should be 'p', 'd' or 'f'.
            - 'U_value' (List[float]): List of Hubbard U values for each element.
            The length of list for each key in dftu_param should be the same.
        init_mag (dict or None): The initial magnetic moment for magnetic elements, should be a dict containing the following keys:
            - 'element': List of elements to apply initial magnetic moment to.
            - 'mag': List of initial magnetic moments for each element.
            The length of list for each key in init_mag should be the same.
        max_steps (int): The maximum number of steps for the relax calculation. Default is 100.
        relax: Whether to do a relax calculation before doing the property calculation. Default is False.
            If the calculated property is phonon dispersion or elastic properties, the crystal should be relaxed first with relax_cell set to True and `relax_precision` is strongly recommended be set to `high`.
        relax_cell (bool): Whether to relax the cell size during the relax calculation. Default is True.
        relax_precision (Literal['low', 'medium', 'high']): The precision of the relax calculation, can be 'low', 'medium', or 'high'. Default is 'medium'.
            'low' means the relax calculation will be done with force_thr_ev=0.05 and stress_thr_kbar=5.
            'medium' means the relax calculation will be done with force_thr_ev=0.01 and stress_thr_kbar=1.0.
            'high' means the relax calculation will be done with force_thr_ev=0.005 and stress_thr_kbar=0.5.
        fixed_axes: Specifies which axes to fix during relaxation. Only effective when `relax_cell` is True. Options are:
            - None: relax all axes (default)
            - volume: relax with fixed volume
            - shape: relax with fixed shape but changing volume (i.e. only lattice constant changes)
            - a: fix a axis
            - b: fix b axis
            - c: fix c axis
            - ab: fix both a and b axes
            - ac: fix both a and c axes
            - bc: fix both b and c axes
        vacuum_direction (Literal['x', 'y', 'z']): The direction of the vacuum.
        dipole_correction (bool): Whether to apply dipole correction along the vacuum direction. For polar slabs, it is recommended to enable dipole correction.
    
    Returns:
        A dictionary containing:
        - elecstat_pot_file (Path): Path to the cube file containing the electrostatic potential.
        - averaged_elecstat_pot_plot (Path): Path to the plot of the averaged electrostatic potential.
        - work_function_results (list): A list of 1 or 2 dictionary. If dipole correction is not used, only 1 dictionaray will be returned. 
          If dipole correction is used, there will be 2 dictionarys for calculated work function of 2 surfaces of the slab. Each dictionary contains 3 keys:
            - 'work_function': calculated work function
            - 'plateau_start_fractional': Fractional coordinate of start of the identified plateau in the given vacuum direction
            - 'plateau_end_fractional': Fractional coordinate of end of the identified plateau in the given vacuum direction
    """
    dftu_param = transform_dftu_param(dftu_param) if dftu_param is not None else None
    init_mag = transform_initmag_param(init_mag) if init_mag is not None else None
    abacus_inputs_dir = prepare_abacus_inputs(stru_file=stru_file,
                                              stru_type=stru_type,
                                              lcao=lcao,
                                              nspin=nspin,
                                              dft_functional=dft_functional,
                                              dftu=dftu,
                                              dftu_param=dftu_param,
                                              init_mag=init_mag)
    
    if relax:
        relax_outputs = do_relax(abacus_inputs_dir=abacus_inputs_dir,
                                 max_steps=max_steps,
                                 relax_cell=relax_cell,
                                 relax_precision=relax_precision,
                                 fixed_axes=fixed_axes,
                                 relax_method=relax_method)
        abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']
    
    work_function_outputs = _abacus_cal_work_function(abacus_inputs_dir,
                                                      vacuum_direction,
                                                      dipole_correction)
    
    return {'elecstat_pot_file': work_function_outputs.get('elecstat_pot_file', None),
            'averaged_elecstat_pot_plot': work_function_outputs.get('averaged_elecstat_pot_plot', None),
            'work_function_results': work_function_outputs.get('work_function_results', None)}

@mcp.tool()
def abacus_cal_elf(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: DFTUParam = None,
    init_mag: InitMagParam = None,
    #afm: bool = False,
    max_steps: int = 100,
    relax: bool = False,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    relax_method: Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"] = "cg",
    fixed_axes: Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"] = None,
) -> Dict[str, Any]:
    """
    Calculate electron localization function (ELF) using ABACUS.
    Args:
        The following parameters are commom for all properties:
        stru_file (Path): Structure file in cif, poscar, or abacus/stru format.
        stru_type (Literal["cif", "poscar", "abacus/stru"] = "cif"): Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        lcao (bool): Whether to use LCAO basis set, default is True.
        nspin (int): The number of spins, can be 1 (no spin), 2 (spin polarized). Default is 1.
        dft_functional (Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN']): The DFT functional to use, can be 'PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', 'PBE0', or 'R2SCAN'. Default is 'PBE'.
            If hybrid functionals like HSE and PBE0 are used, the calculation may be much slower than GGA functionals like PBE.
        dftu (bool): Whether to use DFT+U, default is False.
        dftu_param (dict or None): The DFT+U parameters, should be a dict containing the following keys:
            - 'element' (List[str]): List of elements to apply DFT+U to.
            - 'orbital' (List[str]): List of orbitals to apply DFT+U to for each element. Should be 'p', 'd' or 'f'.
            - 'U_value' (List[float]): List of Hubbard U values for each element.
            The length of list for each key in dftu_param should be the same.
        init_mag (dict or None): The initial magnetic moment for magnetic elements, should be a dict containing the following keys:
            - 'element': List of elements to apply initial magnetic moment to.
            - 'mag': List of initial magnetic moments for each element.
            The length of list for each key in init_mag should be the same.
        max_steps (int): The maximum number of steps for the relax calculation. Default is 100.
        relax: Whether to do a relax calculation before doing the property calculation. Default is False.
            If the calculated property is phonon dispersion or elastic properties, the crystal should be relaxed first with relax_cell set to True and `relax_precision` is strongly recommended be set to `high`.
        relax_cell (bool): Whether to relax the cell size during the relax calculation. Default is True.
        relax_precision (Literal['low', 'medium', 'high']): The precision of the relax calculation, can be 'low', 'medium', or 'high'. Default is 'medium'.
            'low' means the relax calculation will be done with force_thr_ev=0.05 and stress_thr_kbar=5.
            'medium' means the relax calculation will be done with force_thr_ev=0.01 and stress_thr_kbar=1.0.
            'high' means the relax calculation will be done with force_thr_ev=0.005 and stress_thr_kbar=0.5.
        fixed_axes: Specifies which axes to fix during relaxation. Only effective when `relax_cell` is True. Options are:
            - None: relax all axes (default)
            - volume: relax with fixed volume
            - shape: relax with fixed shape but changing volume (i.e. only lattice constant changes)
            - a: fix a axis
            - b: fix b axis
            - c: fix c axis
            - ab: fix both a and b axes
            - ac: fix both a and c axes
            - bc: fix both b and c axes
    Returns:
        Dict[str, Any]: A dictionary containing:
         - elf_file: ELF file path (in .cube file format).
    """
    dftu_param = transform_dftu_param(dftu_param) if dftu_param is not None else None
    init_mag = transform_initmag_param(init_mag) if init_mag is not None else None
    abacus_inputs_dir = prepare_abacus_inputs(stru_file=stru_file,
                                              stru_type=stru_type,
                                              lcao=lcao,
                                              nspin=nspin,
                                              dft_functional=dft_functional,
                                              dftu=dftu,
                                              dftu_param=dftu_param,
                                              init_mag=init_mag)
    
    if relax:
        relax_outputs = do_relax(abacus_inputs_dir=abacus_inputs_dir,
                                 max_steps=max_steps,
                                 relax_cell=relax_cell,
                                 relax_precision=relax_precision,
                                 fixed_axes=fixed_axes,
                                 relax_method=relax_method)
        abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']
    
    elf_outputs = _abacus_cal_elf(abacus_inputs_dir)

    return {'elf_file': elf_outputs.get('elf_file', None)}

@mcp.tool()
def abacus_eos(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: DFTUParam = None,
    init_mag: InitMagParam = None,
    #afm: bool = False,
    max_steps: int = 100,
    relax: bool = False,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    relax_method: Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"] = "cg",
    fixed_axes: Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"] = None,
    stru_scale_number: int = 3,
    scale_stepsize: float = 0.02
) -> Dict[str, Any]:
    """
    Use Birch-Murnaghan equation of state (EOS) to calculate the EOS data. The shape of fitted crystal is limited to cubic now.

    Args:
        The following parameters are commom for all properties:
        stru_file (Path): Structure file in cif, poscar, or abacus/stru format.
        stru_type (Literal["cif", "poscar", "abacus/stru"] = "cif"): Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        lcao (bool): Whether to use LCAO basis set, default is True.
        nspin (int): The number of spins, can be 1 (no spin), 2 (spin polarized). Default is 1.
        dft_functional (Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN']): The DFT functional to use, can be 'PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', 'PBE0', or 'R2SCAN'. Default is 'PBE'.
            If hybrid functionals like HSE and PBE0 are used, the calculation may be much slower than GGA functionals like PBE.
        dftu (bool): Whether to use DFT+U, default is False.
        dftu_param (dict or None): The DFT+U parameters, should be a dict containing the following keys:
            - 'element' (List[str]): List of elements to apply DFT+U to.
            - 'orbital' (List[str]): List of orbitals to apply DFT+U to for each element. Should be 'p', 'd' or 'f'.
            - 'U_value' (List[float]): List of Hubbard U values for each element.
            The length of list for each key in dftu_param should be the same.
        init_mag (dict or None): The initial magnetic moment for magnetic elements, should be a dict containing the following keys:
            - 'element': List of elements to apply initial magnetic moment to.
            - 'mag': List of initial magnetic moments for each element.
            The length of list for each key in init_mag should be the same.
        max_steps (int): The maximum number of steps for the relax calculation. Default is 100.
        relax: Whether to do a relax calculation before doing the property calculation. Default is False.
            If the calculated property is phonon dispersion or elastic properties, the crystal should be relaxed first with relax_cell set to True and `relax_precision` is strongly recommended be set to `high`.
        relax_cell (bool): Whether to relax the cell size during the relax calculation. Default is True.
        relax_precision (Literal['low', 'medium', 'high']): The precision of the relax calculation, can be 'low', 'medium', or 'high'. Default is 'medium'.
            'low' means the relax calculation will be done with force_thr_ev=0.05 and stress_thr_kbar=5.
            'medium' means the relax calculation will be done with force_thr_ev=0.01 and stress_thr_kbar=1.0.
            'high' means the relax calculation will be done with force_thr_ev=0.005 and stress_thr_kbar=0.5.
        fixed_axes: Specifies which axes to fix during relaxation. Only effective when `relax_cell` is True. Options are:
            - None: relax all axes (default)
            - volume: relax with fixed volume
            - shape: relax with fixed shape but changing volume (i.e. only lattice constant changes)
            - a: fix a axis
            - b: fix b axis
            - c: fix c axis
            - ab: fix both a and b axes
            - ac: fix both a and c axes
            - bc: fix both b and c axes
        stru_scale_number (int): Number of structures to generate for EOS calculation.
        scale_stepsize (float): Step size for scaling. Default is 0.02, which means 2% of the original cell size.

    Returns:
        Dict[str, Any]: A dictionary containing EOS calculation results:
            - "eos_fig_path" (Path): Path to the EOS fitting plot (energy vs. volume).
            - "E0" (float): Minimum energy (in eV) from the EOS fit.
            - "V0" (float): Equilibrium volume (in Å³) corresponding to E0.
            - "B0" (float): Bulk modulus (in GPa) at equilibrium volume.
            - "B0_deriv" (float): Pressure derivative of the bulk modulus.
    """
    dftu_param = transform_dftu_param(dftu_param) if dftu_param is not None else None
    init_mag = transform_initmag_param(init_mag) if init_mag is not None else None
    abacus_inputs_dir = prepare_abacus_inputs(stru_file=stru_file,
                                              stru_type=stru_type,
                                              lcao=lcao,
                                              nspin=nspin,
                                              dft_functional=dft_functional,
                                              dftu=dftu,
                                              dftu_param=dftu_param,
                                              init_mag=init_mag)
    
    if relax:
        relax_outputs = do_relax(abacus_inputs_dir=abacus_inputs_dir,
                                 max_steps=max_steps,
                                 relax_cell=relax_cell,
                                 relax_precision=relax_precision,
                                 fixed_axes=fixed_axes,
                                 relax_method=relax_method)
        abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']
    
    eos_outputs = _abacus_eos(abacus_inputs_dir,
                              stru_scale_number,
                              scale_stepsize)
    
    return {'eos_fig_path': eos_outputs.get('eos_fig_path', None),
            'E0': eos_outputs.get('E0', None),
            'V0': eos_outputs.get('V0', None),
            'B0': eos_outputs.get('B0', None),
            'B0_deriv': eos_outputs.get('B0_deriv', None)}

@mcp.tool()
def abacus_run_md(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    lcao: bool = True,
    nspin: Literal[1, 2] = 1,
    dft_functional: Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN'] = 'PBE',
    #soc: bool = False,
    dftu: bool = False,
    dftu_param: DFTUParam = None,
    init_mag: InitMagParam = None,
    #afm: bool = False,
    max_steps: int = 100,
    relax: bool = False,
    relax_cell: bool = True,
    relax_precision: Literal['low', 'medium', 'high'] = 'medium',
    relax_method: Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"] = "cg",
    fixed_axes: Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"] = None,
    md_type: Literal['nve', 'nvt', 'npt', 'langevin'] = 'nve',
    md_nstep: int = 10,
    md_dt: float = 1.0,
    md_tfirst: float = 300.0,
    md_tlast: float = 300.0,
    md_thermostat: Literal['nhc', 'anderson', 'berendsen', 'rescaling', 'rescale_v'] = 'nhc',
    md_pmode: Literal['iso', 'aniso', 'tri'] = 'iso',
    md_pcouple: Literal['none', 'xy', 'xz', 'yz', 'xyz'] = 'none',
    md_dumpfreq: int = 1,
    md_seed: int = -1
) -> Dict[str, Any]:
    """
    Use ABACUS to do ab-initio molecular dynamics calculation.

    Args:
        The following parameters are commom for all properties:
        stru_file (Path): Structure file in cif, poscar, or abacus/stru format.
        stru_type (Literal["cif", "poscar", "abacus/stru"] = "cif"): Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        lcao (bool): Whether to use LCAO basis set, default is True.
        nspin (int): The number of spins, can be 1 (no spin), 2 (spin polarized). Default is 1.
        dft_functional (Literal['PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', "PBE0", 'R2SCAN']): The DFT functional to use, can be 'PBE', 'PBEsol', 'LDA', 'SCAN', 'HSE', 'PBE0', or 'R2SCAN'. Default is 'PBE'.
            If hybrid functionals like HSE and PBE0 are used, the calculation may be much slower than GGA functionals like PBE.
        dftu (bool): Whether to use DFT+U, default is False.
        dftu_param (dict or None): The DFT+U parameters, should be a dict containing the following keys:
            - 'element' (List[str]): List of elements to apply DFT+U to.
            - 'orbital' (List[str]): List of orbitals to apply DFT+U to for each element. Should be 'p', 'd' or 'f'.
            - 'U_value' (List[float]): List of Hubbard U values for each element.
            The length of list for each key in dftu_param should be the same.
        init_mag (dict or None): The initial magnetic moment for magnetic elements, should be a dict containing the following keys:
            - 'element': List of elements to apply initial magnetic moment to.
            - 'mag': List of initial magnetic moments for each element.
            The length of list for each key in init_mag should be the same.
        max_steps (int): The maximum number of steps for the relax calculation. Default is 100.
        relax: Whether to do a relax calculation before doing the property calculation. Default is False.
            If the calculated property is phonon dispersion or elastic properties, the crystal should be relaxed first with relax_cell set to True and `relax_precision` is strongly recommended be set to `high`.
        relax_cell (bool): Whether to relax the cell size during the relax calculation. Default is True.
        relax_precision (Literal['low', 'medium', 'high']): The precision of the relax calculation, can be 'low', 'medium', or 'high'. Default is 'medium'.
            'low' means the relax calculation will be done with force_thr_ev=0.05 and stress_thr_kbar=5.
            'medium' means the relax calculation will be done with force_thr_ev=0.01 and stress_thr_kbar=1.0.
            'high' means the relax calculation will be done with force_thr_ev=0.005 and stress_thr_kbar=0.5.
        fixed_axes: Specifies which axes to fix during relaxation. Only effective when `relax_cell` is True. Options are:
            - None: relax all axes (default)
            - volume: relax with fixed volume
            - shape: relax with fixed shape but changing volume (i.e. only lattice constant changes)
            - a: fix a axis
            - b: fix b axis
            - c: fix c axis
            - ab: fix both a and b axes
            - ac: fix both a and c axes
            - bc: fix both b and c axes
        md_type (Literal['nve', 'nvt', 'npt', 'langevin']): The algorithm to integrate the equation of motion for molecular dynamics (MD).
            - nve: NVE ensemble with velocity Verlet algorithm.
            - nvt: NVT ensemble.
            - npt: Nose-Hoover style NPT ensemble.
            - langevin: NVT ensemble with Langevin thermostat.
        md_nstep (int): The total number of molecular dynamics steps.
        md_dt (float): The time step used in molecular dynamics calculations. THe unit in fs.
        md_tfirst (float): If set to larger than 0, initial velocity will be generated according to its value.
            If unset or smaller than 0, initial velocity will try to be read from STRU file.
        md_tlast (float): Only used in NVT/NPT simulations. If md_tlast is unset or less than zero, 
            md_tlast is set to md_tfirst. If md_tlast is set to be different from md_tfirst, 
            ABACUS will automatically change the temperature from md_tfirst to md_tlast
        md_thermostat (str): Specify the temperature control method used in NVT ensemble.
            - nhc: Nose-Hoover chain, see md_tfreq and md_tchain in detail.
            - anderson: Anderson thermostat, see md_nraise in detail.
            - berendsen: Berendsen thermostat, see md_nraise in detail.
            - rescaling: velocity Rescaling method 1, see md_tolerance in detail.
            - rescale_v: velocity Rescaling method 2, see md_nraise in detail.
        md_pmode (str): Specify the cell fluctuation mode in NPT ensemble based on the Nose-Hoover style non-Hamiltonian equations of motion.
            - iso: The three diagonal elements of the lattice are fluctuated isotropically.
            - aniso: The three diagonal elements of the lattice are fluctuated anisotropically.
            - tri: The lattice must be a lower-triangular matrix, and all six freedoms are fluctuated.
        md_pcouple (str): The coupled lattice vectors will scale proportionally in NPT ensemble based on the Nose-Hoover style non-Hamiltonian equations of motion.
            - none: Three lattice vectors scale independently.
            - xyz: Lattice vectors x, y, and z scale proportionally.
            - xy: Lattice vectors x and y scale proportionally.
            - xz: Lattice vectors x and z scale proportionally.
            - yz: Lattice vectors y and z scale proportionally.
        md_dumpfreq (int): The output frequency of OUT.$[suffix]/MD_dump in molecular dynamics calculations. Generally the default value 1
            is OK. For very long ab-initio MD calculations, increasing md_dumpfreq can help reducing the size of MD_dump.
        md_seed (int): The random seed to initialize random numbers used in molecular dynamics calculations.
            - < 0: No srand() function is called.
            - >= 0: The function srand(md_seed) is called.
    Returns:
        A dictionary containing:
            - md_work_path (Path): The working directory of the molecular dynamics calculation.
            - md_traj_file (Path): The path to the ASE trajectory file containing the MD steps.
            - traj_frame_nums (int): Number of frames in returned trajectory file.
            - normal_end (bool): Whether the ab-initio molecular dynamics calculation ended normally.
    """
    dftu_param = transform_dftu_param(dftu_param) if dftu_param is not None else None
    init_mag = transform_initmag_param(init_mag) if init_mag is not None else None
    abacus_inputs_dir = prepare_abacus_inputs(stru_file=stru_file,
                                              stru_type=stru_type,
                                              lcao=lcao,
                                              nspin=nspin,
                                              dft_functional=dft_functional,
                                              dftu=dftu,
                                              dftu_param=dftu_param,
                                              init_mag=init_mag)
    
    if relax:
        relax_outputs = do_relax(abacus_inputs_dir=abacus_inputs_dir,
                                 max_steps=max_steps,
                                 relax_cell=relax_cell,
                                 relax_precision=relax_precision,
                                 fixed_axes=fixed_axes,
                                 relax_method=relax_method)
        abacus_inputs_dir = relax_outputs['new_abacus_inputs_dir']
    
    md_outputs = _abacus_run_md(abacus_inputs_dir,
                               md_type,
                               md_nstep,
                               md_dt,
                               md_tfirst,
                               md_tlast,
                               md_thermostat,
                               md_pmode,
                               md_pcouple,
                               md_dumpfreq,
                               md_seed)
    
    return {'md_traj_file': md_outputs.get('md_traj_file', None),
            'traj_frame_nums': md_outputs.get('traj_frame_nums', None),
            'normal_end': md_outputs.get('normal_end', None)}
