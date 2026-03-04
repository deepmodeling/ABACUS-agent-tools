from pathlib import Path
from typing import Dict, Any, Literal

from abacusagent.init_mcp import mcp

@mcp.tool()
def get_high_symm_points_from_stru(stru_file: Path,
                                   stru_type: Literal['cif', 'poscar', 'abacus/stru'] = 'cif'
) -> Dict[str, Any]:
    """
    Get high symmetry points and kpath from structure file.
    Args:
        stru_file (Path): Absolute path to the structure file.
        stru_type (Literal): Type of the structure file, can be 'cif', 'poscar', or 'abacus/stru'.
    Returns:
        A Dictionary with keys:
        - high_symm_points: A list of dictionary, where the key is the label of the point, and the value is the fractional coordinates of the point.
        - kpath: A list of tuples, where each tuple is a pair of labels of the two points in the kpath.
    """
    from abacusagent.modules.util.symmetry import get_high_symm_points_from_stru as _get_high_symm_points_from_stru

    return _get_high_symm_points_from_stru(stru_file, stru_type)

@mcp.tool()
def get_high_symm_points_from_abacus_inputs_dir(abacusjob_dir: Path) -> Dict[str, Any]:
    """
    Get high symmetry points and kpath for STRU file in ABACUS inputs directory.
    Args:
        abacusjob_dir (str): Absolute path to a directory containing the INPUT, STRU, KPT, and pseudopotential or orbital files.
    Returns:
        A dictionary containing high symmetry points and suggested kpath for STRU file in ABACUS inputs directory. The most important keys are:
        - path (List[List[str]]): Suggested path for the given structure.
        - point_coords: Coordinates of high symmetry points in reciprocal space.
    """
    from abacusagent.modules.util.symmetry import get_high_symm_points_from_abacus_inputs_dir as _get_high_symm_points_from_abacus_inputs_dir

    return _get_high_symm_points_from_abacus_inputs_dir(abacusjob_dir)
