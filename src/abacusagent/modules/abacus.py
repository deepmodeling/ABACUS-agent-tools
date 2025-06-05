import os
from pathlib import Path
from typing import Literal, Optional, TypedDict, Dict, Any
from abacustest.lib_model.model_013_inputs import PrepInput
from abacustest.lib_prepare.abacus import WriteInput

from abacusagent.init_mcp import mcp

@mcp.tool()
def abacus_prepare(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    pp_path: Optional[Path] = None,
    orb_path: Optional[Path] = None,
    job_type: Literal["scf", "relax", "cell-relax", "md"] = "scf",
    lcao: bool = True,
    extra_input: Optional[Dict[str, Any]] = None,
) -> TypedDict("results",{"job_path": Path}):
    """
    Prepare input files for ABACUS calculation.
    Args:
        stru_file: Structure file in cif, poscar, or abacus/stru format.
        stru_type: Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        pp_path: The pseudopotential library path, if is None, will use the value of environment variable ABACUS_PP_PATH.
        orb_path: The orbital library path, if is None, will use the value of environment variable ABACUS_ORB_PATH.
        job_type: The type of job to be performed, can be 'scf', 'relax', 'cell-relax', or 'md'. 'scf' is the default.
        lcao: Whether to use LCAO basis set, default is True. If True, the orbital library path must be provided.
        extra_input: Extra input parameters for ABACUS. 
    
    Returns:
        A dictionary containing the job path.
    Raises:
        FileNotFoundError: If the structure file or pseudopotential path does not exist.
        ValueError: If LCAO basis set is selected but no orbital library path is provided.
        RuntimeError: If there is an error preparing input files.
    """
    
    if not os.path.isfile(stru_file):
        raise FileNotFoundError(f"Structure file {stru_file} does not exist.")
    
    # Check if the pseudopotential path exists
    pp_path = pp_path if pp_path is not None else os.getenv("ABACUS_PP_PATH")
    if pp_path is None or not os.path.exists(pp_path):
        raise FileNotFoundError(f"Pseudopotential path {pp_path} does not exist.")
    
    if orb_path is None and os.getenv("ABACUS_ORB_PATH") is not None:
        orb_path = os.getenv("ABACUS_ORB_PATH")
    
    if lcao and orb_path is None:
        raise ValueError("LCAO basis set is selected but no orbital library path is provided.")
    
    extra_input_file = None
    if extra_input is not None:
        # write extra input to the input file
        extra_input_file = "INPUT.tmp"
        WriteInput(extra_input, extra_input_file)
    
    try:
        _, job_path = PrepInput(
            files=str(stru_file),
            filetype=stru_type,
            jobtype=job_type,
            pp_path=pp_path,
            orb_path=orb_path,
            input_file=extra_input_file,
            lcao=lcao,
        ).run()
    except Exception as e:
        raise RuntimeError(f"Error preparing input files: {e}")

    if len(job_path) == 0:
        raise RuntimeError("No job path returned from PrepInput.")
    
    return {"job_path": Path(job_path[0]).absolute()}