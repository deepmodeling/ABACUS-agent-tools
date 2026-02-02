import os
from pathlib import Path
from typing import Literal, Optional, Dict, Any, List

from abacustest.lib_model.comm import check_abacus_inputs
from abacustest.lib_model.model_020_workfunc import prep_abacus_workfunc_calc, post_workfunc_calc

from abacusagent.modules.util.comm import run_abacus, generate_work_path, link_abacusjob

def abacus_cal_work_function(
    abacus_inputs_dir: Path,
    vacuum_direction: Literal['a', 'b', 'c', 'auto'] = 'c',
    dipole_correction: bool = False,
) -> Dict[str, Any]:
    """
    Calculate the electrostatic potential and work function using ABACUS.
    
    Args:
        abacus_inputs_dir (Path): Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        vacuum_direction (Literal['a', 'b', 'c', 'auto']): The direction of the vacuum. If set to auto, the direction will try to be determined automatically.
        dipole_correction (bool): Whether to apply dipole correction along the vacuum direction. For polar slabs, it is recommended to enable dipole correction.

    Returns:
        A dictionary containing:
        - elecstat_pot_work_function_work_path (Path): Path to the ABACUS job directory calculating electrostatic potential and work function.
        - elecstat_pot_file (Path): Path to the cube file containing the electrostatic potential.
        - averaged_elecstat_pot_plot (Path): Path to the plot of the averaged electrostatic potential.
        - averaged_elecstat_pot_dat_file (Path): Path to the data used in the plot of averaged electrostatic potential.
        - work_function_results (list): A list of 1 or 2 dictionary. If dipole correction is not used, only 1 dictionaray will be returned. 
          If dipole correction is used, there will be 2 dictionarys for calculated work function of 2 surfaces of the slab. Each dictionary contains 3 keys:
            - 'work_function': calculated work function
            - 'plateau_start_fractional': Fractional coordinate of start of the identified plateau in the given vacuum direction
            - 'plateau_end_fractional': Fractional coordinate of end of the identified plateau in the given vacuum direction
    """
    try:
        is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
        if not is_valid:
            raise RuntimeError(f"Invalid ABACUS input files: {msg}")
        
        work_path = Path(generate_work_path()).absolute()
        link_abacusjob(src=abacus_inputs_dir,dst=work_path,copy_files=["INPUT", "STRU"], exclude_directories=True)
        workfunc_work_dir = prep_abacus_workfunc_calc(work_path, vacuum_direction, dipole_correction, os.path.join(work_path, "workfunc_job"))
        
        run_abacus(workfunc_work_dir)

        work_function_results, plot_path, pot_file, plot_data_file = post_workfunc_calc(work_path, jobtype="abacus")

        return {'elecstat_pot_work_function_work_path': Path(work_path).absolute(),
                'elecstat_pot_file': Path(pot_file).absolute(),
                'averaged_elecstat_pot_plot': Path(plot_path).absolute(),
                'averaged_elecstat_pot_dat_file': Path(plot_data_file).absolute(),
                'work_function_results': work_function_results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'message': f"Calculating electrostatic potential and work function failed: {e}"}
