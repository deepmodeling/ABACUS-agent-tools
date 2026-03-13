import os
from pathlib import Path
from typing import Literal, Tuple, Dict, Any
from ase.build import surface, make_supercell
from ase.io import read
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from abacustest.constant import A2BOHR
from abacustest.lib_prepare.stru import AbacusSTRU

def build_slab(stru_file: Path,
               stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
               miller_indices: Tuple[int, int, int] = (1, 0, 0),
               layers: int = 3,
               surface_supercell: Tuple[int, int] = (1, 1),
               vacuum: float = 15.0,
               vacuum_direction: Literal['a', 'b', 'c'] = 'b'):
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
    if stru_type == "abacus/stru":
        stru = AbacusSTRU.read(stru_file, fmt="stru")
        stru_ase = stru.to("ase")
    elif stru_type in ["cif", "poscar"]:
        stru_ase = read(stru_file, format=stru_type)
    else:
        raise ValueError(f"Unsupported structure file type: {stru_type}")

    stru_surface = surface(stru_ase, miller_indices, layers, vacuum=vacuum / 2, periodic=True)
    stru_surface = make_supercell(
        stru_surface,
        [[surface_supercell[0], 0, 0], [0, surface_supercell[1], 0], [0, 0, 1]],
    )
    stru_surface_abacusstru = AbacusSTRU.from_ase(
        stru_surface,
        metadata={
            "lattice_constant": A2BOHR,
            "atom_type": "cartesian",
        },
    )
    stru_surface_abacusstru.sort()

    # Permute axis to set vacuum direction along given axis. The vacuum direction create by ase.build.surface is always along z axis.
    if vacuum_direction == "a":
        stru_surface_abacusstru.permute_lat_vec(mode="cab", rotate_cart_coord=True)
    elif vacuum_direction == "b":
        stru_surface_abacusstru.permute_lat_vec(mode="bca", rotate_cart_coord=True)
    elif vacuum_direction == "c":
        pass

    h, k, l = miller_indices
    suffix = "STRU" if stru_type == "abacus/stru" else stru_type
    surface_stru_file = Path(f"./{stru_file.stem}_{h}{k}{l}_{layers}layer.{suffix}").absolute()
    stru_surface_abacusstru.write(surface_stru_file, fmt=stru_type)

    return {"surface_stru_file": surface_stru_file}

def _read_structure(
    stru_file: Path, stru_type: Literal["cif", "poscar", "abacus/stru"]
) -> Structure:
    """
    Read a structure file and convert it to pymatgen Structure object.

    Args:
        stru_file: Path to the structure file.
        stru_type: Type of structure file ('cif', 'poscar', or 'abacus/stru').

    Returns:
        pymatgen Structure object.

    Raises:
        FileNotFoundError: If the structure file does not exist.
        ValueError: If the structure file type is not supported.
    """
    if not os.path.isfile(stru_file):
        raise FileNotFoundError(f"Structure file {stru_file} does not exist.")

    if stru_type == "abacus/stru":
        stru = AbacusSTRU.read(stru_file, fmt="stru")
        return Structure(
            lattice=stru.cell,
            species=stru.labels,
            coords=stru.coords_direct,
            coords_are_cartesian=False,
        )
    elif stru_type == "cif":
        return Structure.from_file(stru_file)
    elif stru_type == "poscar":
        from pymatgen.io.vasp import Poscar

        return Poscar.from_file(stru_file).structure
    else:
        raise ValueError(f"Unsupported structure file type: {stru_type}")

def _write_structure(
    structure: Structure,
    output_file: Path,
    output_format: Literal["cif", "poscar", "abacus/stru"],
) -> Path:
    """
    Write a pymatgen Structure to file in the specified format.

    Args:
        structure: pymatgen Structure object.
        output_file: Path to the output file.
        output_format: Format of the output file ('cif', 'poscar', or 'abacus/stru').

    Returns:
        Path to the output file.
    """
    if output_format == "cif":
        structure.to(filename=output_file, fmt="cif")
    elif output_format == "poscar":
        structure.to(filename=output_file, fmt="poscar")
    elif output_format == "abacus/stru":
        from pymatgen.io.ase import AseAtomsAdaptor

        ase_atoms = AseAtomsAdaptor.get_atoms(structure)
        abacus_stru = AbacusSTRU.from_ase(ase_atoms, metadata={"lattice_constant": A2BOHR})
        abacus_stru.write(output_file, fmt="stru")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    return output_file

def convert_to_primitive(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    output_format: Literal["cif", "poscar", "abacus/stru"] = None,
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
        >>> convert_to_primitive("Si.cif", stru_type="cif", output_format="poscar")

        >>> # Convert ABACUS STRU to primitive cell in CIF format
        >>> convert_to_primitive("STRU", stru_type="abacus/stru", output_format="cif")
    """
    try:
        # Set default output format if not specified
        if output_format is None:
            output_format = stru_type

        # Read the structure
        structure = _read_structure(stru_file, stru_type)

        # Get primitive structure
        primitive_structure = structure.get_primitive_structure(tolerance=tolerance)

        # Get space group
        sga = SpacegroupAnalyzer(primitive_structure, symprec=tolerance)
        spacegroup = sga.get_space_group_symbol()

        # Generate output filename
        input_path = Path(stru_file)
        suffix_map = {"cif": ".cif", "poscar": ".vasp", "abacus/stru": ".stru"}
        output_suffix = suffix_map.get(output_format, ".cif")
        output_file = Path(f"{input_path.stem}_primitive{output_suffix}").absolute()

        # Write the primitive structure
        _write_structure(primitive_structure, output_file, output_format)

        return {
            "output_file": output_file,
            "num_atoms": len(primitive_structure),
            "cell": primitive_structure.lattice.matrix.tolist(),
            "spacegroup": spacegroup,
        }
    except Exception as e:
        return {"message": f"Converting to primitive cell failed: {e}"}

def convert_to_conventional(
    stru_file: Path,
    stru_type: Literal["cif", "poscar", "abacus/stru"] = "cif",
    output_format: Literal["cif", "poscar", "abacus/stru"] = None,
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
        >>> convert_to_conventional("Si.cif", stru_type="cif", output_format="poscar")

        >>> # Convert ABACUS STRU to conventional cell in CIF format
        >>> convert_to_conventional("STRU", stru_type="abacus/stru", output_format="cif")
    """
    try:
        # Set default output format if not specified
        if output_format is None:
            output_format = stru_type

        # Read the structure
        structure = _read_structure(stru_file, stru_type)

        # Get conventional standard structure
        sga = SpacegroupAnalyzer(structure, symprec=tolerance)
        conventional_structure = sga.get_conventional_standard_structure()
        spacegroup = sga.get_space_group_symbol()

        # Generate output filename
        input_path = Path(stru_file)
        suffix_map = {"cif": ".cif", "poscar": ".vasp", "abacus/stru": ".stru"}
        output_suffix = suffix_map.get(output_format, ".cif")
        output_file = Path(f"{input_path.stem}_conventional{output_suffix}").absolute()

        # Write the conventional structure
        _write_structure(conventional_structure, output_file, output_format)

        return {
            "output_file": output_file,
            "num_atoms": len(conventional_structure),
            "cell": conventional_structure.lattice.matrix.tolist(),
            "spacegroup": spacegroup,
        }
    except Exception as e:
        return {"message": f"Converting to conventional cell failed: {e}"}
