from pathlib import Path
from typing import Literal, Dict, List, Union, Any

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.band import abacus_cal_band as _abacus_cal_band

@mcp.tool()
def abacus_cal_band(abacus_inputs_dir: Path,
                    mode: Literal["nscf", "pyatb", "auto"] = "auto",
                    kpath: Union[List[str], List[List[str]]] = None,
                    high_symm_points: Dict[str, List[float]] = None,
                    energy_min: float = -10,
                    energy_max: float = 10,
                    insert_point_nums: int = 30
) -> Dict[str, Any]:
    """
    Calculate band using ABACUS based on prepared directory containing the INPUT, STRU, KPT, and pseudopotential or orbital files.
    PYATB or ABACUS NSCF calculation will be used according to parameters in INPUT.
    Args:
        abacus_inputs_dir (str): Absolute path to a directory containing the INPUT, STRU, KPT, and pseudopotential or orbital files.
        mode: Method used to plot band. Should be `auto`, `pyatb` or `nscf`. 
            - `nscf` means using `nscf` calculation in ABACUS to calculate and plot the band
            - `pyatb` means using PYATB to plot the band
            - `auto` means deciding use `nscf` or `pyatb` mode according to the `basis_type` in INPUT file and files included in `abacus_inputs_dir`.
                -- If charge files are in `abacus_input_dir`, `nscf` mode will be used.
                -- If matrix files are in `abacus_input_dir`, `pyatb` mode will be used.
                -- If no matrix file or charge file are in `abacus_input_dir`, will determine mode by `basis_type`. If `basis_type` is lcao, will use `pyatb` mode.
                    If `basis_type` is pw, will use `nscf` mode.
        kpath (Tuple[List[str], List[List[str]]]): 
                A list of name of high symmetry points in the band path. Non-continuous line of high symmetry points are stored as seperate lists.
                For example, ['G', 'M', 'K', 'G'] and [['G', 'X', 'P', 'N', 'M', 'S'], ['S_0', 'G', R']] are both acceptable inputs.
                Default is None. If None, will use automatically generated kpath.
                `kpath` must be used with `high_symm_points` to take effect.
        high_symm_points: A dictionary containing high symmetry points and their coordinates in the band path. All points in `kpath` should be included.
                For example, {'G': [0, 0, 0], 'M': [0.5, 0.0, 0.0], 'K': [0.33333333, 0.33333333, 0.0], 'G': [0, 0, 0]}.
                Default is None. If None, will use automatically generated high symmetry points.
        energy_min (float): Lower bound of $E - E_F$ in the plotted band.
        energy_max (float): Upper bound of $E - E_F$ in the plotted band.
        insert_point_nums (int): Number of points to insert between two high symmetry points. Default is 30.
    Returns:
        A dictionary containing band gap, path to the work directory for calculating band and path to the plotted band.
    Raises:
    """
    return _abacus_cal_band(abacus_inputs_dir, mode, kpath, high_symm_points, energy_min, energy_max, insert_point_nums)

