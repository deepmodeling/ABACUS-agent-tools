import os

from pathlib import Path
from typing import Literal, Dict, Any
from abacustest import AbacusSTRU, ReadInput

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
    if stru_type in ['cif', 'poscar', 'abacus/stru']:
        stru = AbacusSTRU.read(stru_file, stru_type)
    else:
        raise ValueError("stru_type should be 'cif', 'poscar', or 'abacus/stru'")
    
    high_symm_points, kpath = stru.get_kline()
    return {"high_symm_points": high_symm_points, "kpath": kpath}

def get_high_symm_points_from_abacus_inputs_dir(abacus_inputs_dir: Path) -> Dict[str, Any]:
    """
    Get high symmetry points and kpath for STRU file in ABACUS inputs directory.
    """
    input_params = ReadInput(os.path.join(Path(abacus_inputs_dir), "INPUT"))
    stru_file = os.path.join(abacus_inputs_dir, input_params.get('stru_file', "STRU"))
    return get_high_symm_points_from_stru(stru_file, stru_type='abacus/stru')
