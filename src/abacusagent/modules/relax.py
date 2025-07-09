import os
import json
from pathlib import Path
from typing import Literal, Optional, TypedDict, Dict, Any, List, Tuple, Union
from abacustest.lib_model.model_013_inputs import PrepInput
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteInput
from abacustest.lib_collectdata.collectdata import RESULT
from abacustest.collectdata import parse_value

from abacusagent.init_mcp import mcp
from abacusagent.modules.util.comm import run_abacus, link_abacusjob, generate_work_path


@mcp.tool()
def abacus_do_relax(
    abacus_inputs_path: Path,
    force_thr_ev: Optional[float] = None,
    stress_thr_kbar: Optional[float] = None,
    max_steps: Optional[int] = None,
    relax_cell: Optional[bool] = None,
    fixed_axes: Optional[Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"]] = None,
    relax_method: Optional[Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"]] = None,
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
        - job_path: The job path of the relaxation calculation.
        - result: The result of the relaxation calculation with a dictionary containing:
            - normal_end: Whether the relaxation calculation ended normally.
            - relax_steps: The number of relaxation steps taken.
            - largest_gradient: The largest force gradient during the relaxation.
            - relax_converge: Whether the relaxation converged.
            - energies: The energies at each step of the relaxation.
    Raises:
        FileNotFoundError: If the job directory does not exist or does not contain necessary files.
        RuntimeError: If the ABACUS calculation fails or returns an error.
    """
    abacus_inputs_path = Path(abacus_inputs_path).absolute()
    work_path = Path(generate_work_path()).absolute()
    link_abacusjob(src=abacus_inputs_path,
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

    return {
        "job_path": Path(work_path).absolute(),
        "result": results
    }

@mcp.tool()
def abacus_prepare_inputs_from_relax_results(
    relax_jobpath: Path
)-> Dict[str, Any]:
    """
    Prepare ABACUS input files based on the structure of the last relaxation step.
    The INPUT/KPT and pseudopotential/orbital files will be copied from the relaxation job directory.
    
    Args:
        relax_jobpath: Path to the relaxation results.
    
    Returns:
        A dictionary containing the job path.
        - 'job_path': The absolute path to the job directory.
        - 'input_content': The content of the generated INPUT file.
        - 'input_files': A list of files in the job directory.
    """
    relax_jobpath = Path(relax_jobpath).absolute()
    rs = RESULT(path=relax_jobpath, fmt="abacus")
    final_stru = Path(os.path.join(relax_jobpath, f"OUT.{rs.SUFFIX}", "STRU_ION_D")).absolute() # the structure file of the last relax step
    
    if not os.path.isfile(final_stru):
        raise FileNotFoundError(f"We can not find the structure file of last relax step {final_stru}. \
            Please check the path and ensure the relaxation calculation has completed successfully.")
    
    work_path = Path(generate_work_path()).absolute()
    
    link_abacusjob(
        src=relax_jobpath,
        dst=work_path,
        copy_files=["INPUT", "STRU", "KPT"],
        exclude=["OUT.*", "*.log", "*.out", "*.json", "log"],
        exclude_directories=True
    )
    if os.path.isfile(os.path.join(work_path, "STRU")):
        os.unlink(os.path.join(work_path, "STRU"))
    os.symlink(final_stru, os.path.join(work_path, "STRU"))

    return {
        "job_path": Path(work_path).absolute(),
        "input_content": ReadInput(os.path.join(work_path, "INPUT")),
        "input_files": [f for f in Path(work_path).iterdir()]
    }


def prepare_relax_inputs(
    work_path: Path,
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
    work_path = Path(work_path).absolute()
    
    input_param = ReadInput(os.path.join(work_path, "INPUT"))
    
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
        input_param["relax_nmax"] = max_steps
        
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
    
    WriteInput(input_param, os.path.join(work_path, "INPUT"))
    

def relax_postprocess(work_path: Path) -> Dict[str, Any]:
    work_path = Path(work_path).absolute()
    rs = RESULT(path=work_path, fmt="abacus")
    
    metrics = ["normal_end", "relax_steps", "largest_gradient", "largest_gradient_stress", "relax_converge", "energies"]
    
    results = parse_value(rs, metrics)
    
    return results
