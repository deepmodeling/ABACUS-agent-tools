"""
Use Pyatb to do property calculation.
"""
import os
from pathlib import Path
from typing import Dict, Any, Literal
from abacusagent.modules.util.comm import (
    generate_work_path, 
    link_abacusjob, 
    run_abacus, 
    has_chgfile, 
    has_pyatb_matrix_files
)

from abacustest.lib_prepare.abacus import ReadInput, WriteInput, AbacusStru
from abacustest.lib_collectdata.collectdata import RESULT


def property_calculation_scf(
    abacus_inputs_path: Path,
    mode: Literal["nscf", "pyatb", "auto"] = "auto",
                    ):
    """Perform the SCF calculation for property calculations like DOS or band structure.

    Args:
        abacus_inputs_path (Path): Path to the ABACUS input files.
        mode (Literal["nscf", "pyatb", "auto"]): Mode of operation, default is "auto".
            nscf: first run SCF with out_chg=1, then run nscf with init_chg=file.
            pyatb: run SCF with out_mat_r and out_mat_hs2 = 1, then calculate properties using Pyatb.
            auto: automatically determine the mode based on the input parameters. If basis is LCAO, use "pyatb", otherwise use "nscf".

    Returns:
        Dict[str, Any]: A dictionary containing the work path, normal end status, SCF steps, convergence status, and energies.
    """

    input_param = ReadInput(os.path.join(abacus_inputs_path, 'INPUT'))
    basis_type = input_param.get("basis_type", "pw")
    if (mode in ["pyatb", "auto"] and has_pyatb_matrix_files(abacus_inputs_path)):
        print("Matrix files already exist, skipping SCF calculation.")
        work_path = abacus_inputs_path
    elif (mode in ["nscf", "auto"] and has_chgfile(abacus_inputs_path)):
        print("Charge files already exist, skipping SCF calculation.")
        work_path = abacus_inputs_path
    else:
        if mode == "auto":
            if basis_type.lower() == "lcao":
                mode = "pyatb"
            else:
                mode = "nscf"

        if basis_type == "pw" and mode == "pyatb":
            raise ValueError("Pyatb mode is not supported for PW basis. Please use 'nscf' mode instead.")

        work_path = Path(generate_work_path()).absolute()
        link_abacusjob(src=abacus_inputs_path,
                       dst=work_path,
                       copy_files=["INPUT", "STRU", "KPT"])
        if mode == "nscf":
            input_param["calculation"] = "scf"
            input_param["out_chg"] = 1
        elif mode == "pyatb":
            input_param["calculation"] = "scf"
            input_param["out_mat_r"] = 1
            input_param["out_mat_hs2"] = 1
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'nscf', 'pyatb', or 'auto'.")
        
        WriteInput(input_param, os.path.join(work_path, 'INPUT'))
        run_abacus(work_path, "abacus.log")
        
    rs = RESULT(path=work_path, fmt="abacus")

    return {
        "work_path": Path(work_path).absolute(),
        "normal_end": rs["normal_end"],
        "scf_steps": rs["scf_steps"],
        "converge": rs["converge"],
        "energies": rs["energies"],
        "mode": mode
    }
        
    