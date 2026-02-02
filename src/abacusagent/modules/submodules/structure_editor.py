from pathlib import Path
from typing import Literal, Tuple
from ase.build import surface, make_supercell
from ase.io import read

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
    if stru_type == 'abacus/stru':
        stru = AbacusSTRU.read(stru_file, fmt="stru")
        stru_ase = stru.to("ase")
    elif stru_type in ['cif', 'poscar']:
        stru_ase = read(stru_file, format=stru_type)
    else:
        raise ValueError(f"Unsupported structure file type: {stru_type}")
    
    stru_surface = surface(stru_ase, miller_indices, layers, vacuum=vacuum/2, periodic=True)
    stru_surface = make_supercell(stru_surface, [[surface_supercell[0], 0, 0], [0, surface_supercell[1], 0], [0, 0, 1]])
    stru_surface_abacusstru = AbacusSTRU.from_ase(stru_surface, metadata={
                "lattice_constant": A2BOHR,
                "atom_type": "cartesian",
            })
    stru_surface_abacusstru.sort()
    
    # Permute axis to set vacuum direction along given axis. The vacuum direction create by ase.build.surface is always along z axis.
    if vacuum_direction == "a":
        stru_surface_abacusstru.permute_lat_vec(mode="cab", rotate_cart_coord=True)
    elif vacuum_direction == "b":
        stru_surface_abacusstru.permute_lat_vec(mode="bca", rotate_cart_coord=True)
    elif vacuum_direction == "c":
        pass

    h, k, l = miller_indices
    suffix = 'STRU' if stru_type == 'abacus/stru' else stru_type
    surface_stru_file = Path(f"./{stru_file.stem}_{h}{k}{l}_{layers}layer.{suffix}").absolute()
    stru_surface_abacusstru.write(surface_stru_file, fmt=stru_type)

    return {'surface_stru_file': surface_stru_file}
