import os
from pathlib import Path
from typing import Literal, Tuple, Dict, Any, Optional

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.structure_editor import build_slab as _build_slab
from abacusagent.modules.submodules.structure_editor import convert_to_primitive as _convert_to_primitive
from abacusagent.modules.submodules.structure_editor import convert_to_conventional as _convert_to_conventional

@mcp.tool()
def build_slab(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    miller_indices: Tuple[int, int, int] = (1, 0, 0),
    layers: int = 3,
    surface_supercell: Tuple[int, int] = (1, 1),
    vacuum: float = 15.0,
    vacuum_direction: Literal["a", "b", "c"] = "b",
):
    """
    Build slab from given structure file.

    Args:
        stru_file (Path): Path to structure file.
        stru_type (Literal["cif", "poscar", "abacus/stru"]): Type of structure file. Defaults to "cif".
        miller_indices (Tuple[int, int, int]): Miller indices of the surface. Defaults to (1, 0, 0), which means (100) surface of the structure.
        layers (int, optional): Number of layers of the surface. Note that the layers is number of equivalent layers, not number of layers of atoms. Defaults to 3.
        surface_supercell (Tuple[int, int], optional): Supercell size of the surface. Default is (1, 1), which means no supercell.
        vacuum (float, optional): Vacuum space between the cleaved surface and its periodic image. The total vacuum size will be twice this value. Units in Angstrom. Defaults to 15.0.
        vacuum_direction (Literal['a', 'b', 'c']): The direction of the vacuum space. Defaults to 'b'.
    Returns:
        A dictionary containing the path to the surface structure file.
        Keys:
            - surface_stru_file: Path to the surface structure file. The format of the generated structure file depends on the input structure file.
    Raises:
        ValueError: If stru_type is not supported.
    """
    return _build_slab(stru_file, stru_type, miller_indices, layers, surface_supercell, vacuum, vacuum_direction)

@mcp.tool()
def convert_to_primitive(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    output_format: Optional[Literal["cif", "poscar", "abacus/stru"]] = None,
    tolerance: float = 0.25,
) -> Dict[str, Any]:
    """
    Convert a crystal structure to its primitive cell.

    This function takes a crystal structure in CIF, POSCAR, or ABACUS STRU format
    and converts it to its primitive cell using pymatgen's symmetry analysis.

    Args:
        stru_file: Path to the input structure file.
        stru_type: Type of the input structure file. Options are:
            - 'cif': Crystallographic Information File format
            - 'poscar': VASP POSCAR format
            - 'abacus/stru': ABACUS STRU format
        output_format: Format of the output file. If not specified, uses the same
            format as the input. Options are: 'cif', 'poscar', 'abacus/stru'.
        tolerance: Tolerance for symmetry detection in Angstroms. Default is 0.25.
            Structures with atoms closer than this distance are considered symmetric.

    Returns:
        A dictionary containing:
        - 'output_file': Path to the generated primitive structure file.
        - 'num_atoms': Number of atoms in the primitive cell.
        - 'cell': Cell parameters of the primitive cell as a 3x3 list of lists.
        - 'spacegroup': Space group symbol of the structure.

    Raises:
        FileNotFoundError: If the input structure file does not exist.
        ValueError: If the structure file type or output format is not supported.

    Examples:
        >>> # Convert a CIF file to primitive cell in POSCAR format
        >>> convert_structure_to_primitive("Si.cif", stru_type="cif", output_format="poscar")

        >>> # Convert ABACUS STRU to primitive cell in CIF format
        >>> convert_structure_to_primitive("STRU", stru_type="abacus/stru", output_format="cif")
    """
    return _convert_to_primitive(stru_file, stru_type, output_format, tolerance)


@mcp.tool()
def convert_to_conventional(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    output_format: Optional[Literal["cif", "poscar", "abacus/stru"]] = None,
    tolerance: float = 0.01,
) -> Dict[str, Any]:
    """
    Convert a crystal structure to its conventional standard cell.

    This function takes a crystal structure in CIF, POSCAR, or ABACUS STRU format
    and converts it to its conventional standard cell using pymatgen's SpacegroupAnalyzer.
    The conventional cell follows the standard conventions for each space group,
    ensuring proper cell orientation and lattice parameter assignment.

    Args:
        stru_file: Path to the input structure file.
        stru_type: Type of the input structure file. Options are:
            - 'cif': Crystallographic Information File format
            - 'poscar': VASP POSCAR format
            - 'abacus/stru': ABACUS STRU format
        output_format: Format of the output file. If not specified, uses the same
            format as the input. Options are: 'cif', 'poscar', 'abacus/stru'.
        tolerance: Tolerance for symmetry detection in Angstroms. Default is 0.01.
            Lower values are more strict in detecting symmetry.

    Returns:
        A dictionary containing:
        - 'output_file': Path to the generated conventional structure file.
        - 'num_atoms': Number of atoms in the conventional cell.
        - 'cell': Cell parameters of the conventional cell as a 3x3 list of lists.
        - 'spacegroup': Space group symbol of the structure.

    Raises:
        FileNotFoundError: If the input structure file does not exist.
        ValueError: If the structure file type or output format is not supported.

    Examples:
        >>> # Convert a CIF file to conventional cell in POSCAR format
        >>> convert_structure_to_conventional("Si.cif", stru_type="cif", output_format="poscar")

        >>> # Convert ABACUS STRU to conventional cell in CIF format
        >>> convert_structure_to_conventional("STRU", stru_type="abacus/stru", output_format="cif")
    """
    return _convert_to_conventional(stru_file, stru_type, output_format, tolerance)
