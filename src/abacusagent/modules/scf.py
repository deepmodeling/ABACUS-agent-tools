import os
from pathlib import Path
from typing import Literal, Optional, TypedDict, Dict, Any, List, Tuple, Union

from abacustest.lib_model.model_013_inputs import PrepInput
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteInput
from abacustest.lib_collectdata.collectdata import RESULT
from abacustest.lib_model.comm import check_abacus_inputs

from abacusagent.init_mcp import mcp
from abacusagent.modules.util.comm import generate_work_path, link_abacusjob, run_abacus, collect_metrics


@mcp.tool()
def abacus_calculation_scf(
    abacusjob_path: Path,
) -> Dict[str, Any]:
    """
    Run ABACUS SCF calculation.

    Args:
        abacusjob (str): Path to the directory containing the ABACUS input files.
    Returns:
        A dictionary containing the path to output file of ABACUS calculation, and a dictionary containing whether the SCF calculation
        finished normally, the SCF is converged or not, the converged SCF energy and total time used.
    """
    try:
        is_valid, msg = check_abacus_inputs(abacusjob_path)
        if not is_valid:
            raise RuntimeError(f"Invalid ABACUS input files: {msg}")
        
        work_path = Path(generate_work_path()).absolute()
        link_abacusjob(src=abacusjob_path, dst=work_path, copy_files=['INPUT', 'STRU'])
        input_params = ReadInput(os.path.join(work_path, "INPUT"))

        input_params['calculation'] = 'scf'
        WriteInput(input_params, os.path.join(work_path, "INPUT"))

        run_abacus(work_path)

        return_dict = {'abacusjob_dir': Path(work_path).absolute()}
        return_dict.update(collect_metrics(work_path, metrics_names=['normal_end', 'converge', 'energy', 'total_time']))

        return return_dict
    except Exception as e:
        return {"abacusjob_dir": None,
                "normal_end": None,
                "converge": None,
                "energy": None,
                "total_time": None}