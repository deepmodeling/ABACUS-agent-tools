import os
import json
from pathlib import Path
from typing import Literal, Optional, TypedDict, Dict, Any, List, Tuple, Union
from abacustest.lib_model.model_013_inputs import PrepInput
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteInput
from abacustest.lib_collectdata.collectdata import RESULT

from abacusagent.init_mcp import mcp
from abacusagent.modules.util.comm import run_abacus, link_abacusjob, generate_work_path


@mcp.tool()
def abacus_do_relax(
    abacus_inputs_path: str,
    force_thr_ev: Optional[float] = None,
    stress_thr_kbar: Optional[float] = None,
    max_steps: Optional[int] = None,
    relax_cell: Optional[bool] = None,
    fixed_axes: Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"] = None,
    relax_method: Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"] = None,
    relax_new: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Specially modify the ABACUS input for relaxation calculations.
    
    Args:
        abacus_inputs_path: Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        force_thr_ev: Force convergence threshold in eV/Å, default is 0.01.
        stress_thr_kbar: Stress convergence threshold in kbar, default is 1.0, this is only used when relax_cell is True.
        max_steps: Maximum number of relaxation steps, default is 100.
        relax_cell: Whether to relax the cell parameters, default is False.
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
        relax_method: The relaxation method to use, can be 'cg', 'bfgs', 'bfgs_trad', 'cg_bfgs', 'sd', or 'fire'. Default is 'cg'.
        relax_new: If use new implemented CG method, default is True.

    Returns:
        A dictionary containing:
        - job_path: The absolute path to the job directory.
        - result: The result of the relaxation calculation as a RESULT object.
    Raises:
        FileNotFoundError: If the job directory does not exist or does not contain necessary files.
        RuntimeError: If the ABACUS calculation fails or returns an error.
    """
    work_path = generate_work_path()
    link_abacusjob(    src=abacus_inputs_path,
                        dst=work_path,
                        copy_files=["INPUT", "STRU", "KPT"])
    
    prepare_relax_inputs(
        work_path=work_path,
        force_thr_ev=force_thr_ev,
        stress_thr_kbar=stress_thr_kbar,
        max_steps=max_steps,
        relax_cell=relax_cell,
        fixed_axes=fixed_axes,
        relax_method=relax_method,
        relax_new=relax_new,
    )
    
    run_abacus(work_path)
    
    results = relax_postprocess(work_path)

    return results


def prepare_relax_inputs(
    work_path: str,
    force_thr_ev: Optional[float] = None,
    stress_thr_kbar: Optional[float] = None,
    max_steps: Optional[int] = None,
    relax_cell: Optional[bool] = None,
    fixed_axes: Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"] = None,
    relax_method: Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"] = None,
    relax_new: Optional[bool] = None,):
    """
    Prepare the ABACUS input files for relaxation calculations.
    """
    
    input_param = ReadInput(work_path+"/INPUT")
    
    # check calculation type
    if relax_cell is None and "calculation" not in input_param:
        input_param["calculation"] = "relax"
    elif relax_cell:
        input_param["calculation"] = "cell-relax"
    else:
        input_param["calculation"] = "relax"
        
    # check force threshold
    if force_thr_ev is not None:
        input_param["force_thr_ev"] = force_thr_ev
        if "force_thr" in input_param:
            del input_param["force_thr"]
    
    if stress_thr_kbar is not None:
        input_param["stress_thr"] = stress_thr_kbar
    
    if max_steps is not None:
        input_param["max_steps"] = max_steps
        
    if fixed_axes is not None:
        input_param["fixed_axes"] = fixed_axes
        
    if relax_method is not None:
        input_param["relax_method"] = relax_method
        if relax_method == "fire":
            print("Using FIRE method for relaxation. Setting calculation type to 'md'.")
            input_param["calculation"] = "md"
            input_param["md_type"] = "fire"
            input_param.pop("relax_method", None)
    
    if relax_new is not None:
        input_param["relax_new"] = relax_new
    
    WriteInput(input_param, work_path+"/INPUT")
    

def relax_postprocess(work_path):
    pass