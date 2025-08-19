from ase import Atoms
from ase.io import read, write
from ase.build import molecule
from ase.data import chemical_symbols
from ase.collections import g2
from pymatgen.core import Structure, Lattice

from pathlib import Path
from typing import Literal, Optional, Dict, Any, List, Tuple, Union

from abacusagent.init_mcp import mcp
from abacusagent.modules.util.comm import generate_work_path 


@mcp.tool()
def generate_bulk_structure(element: str, 
                           crystal_structure:Literal["sc", "fcc", "bcc","hcp","diamond", "zincblende", "rocksalt"]='fcc', 
                           a:float =None, 
                           c: float =None,
                           cubic: bool =False,
                           orthorhombic: bool =False,
                           file_format: Literal["cif", "poscar"] = "cif",
                           ) -> Dict[str, Any]:
    """
    Generate a bulk crystal structure using ASE's `bulk` function.
    
    Args:
        element (str): The chemical symbol of the element (e.g., 'Cu', 'Si', 'NaCl').
        crystal_structure (str): The type of crystal structure to generate. Options include:
            - 'sc' (simple cubic), a is needed
            - 'fcc' (face-centered cubic), a is needed
            - 'bcc' (body-centered cubic), a is needed
            - 'hcp' (hexagonal close-packed), a is needed, if c is None, c will be set to sqrt(8/3) * a.
            - 'diamond' (diamond cubic structure), a is needed
            - 'zincblende' (zinc blende structure), a is needed, two elements are needed, e.g., 'GaAs'
            - 'rocksalt' (rock salt structure), a is needed, two elements are needed, e.g., 'NaCl'
        a (float, optional): Lattice constant in Angstroms. Required for all structures.
        c (float, optional): Lattice constant for the c-axis in Angstroms. Required for 'hcp' structure.
        cubic (bool, optional): If constructing a cubic supercell for fcc, bcc, diamond, zincblende, or rocksalt structures.
        orthorhombic (bool, optional): If constructing orthorhombic cell for 'hcp' structure.
        file_format (str, optional): The format of the output file. Options are 'cif' or 'poscar'. Default is 'cif'.
    
    Notes: all crystal need the lattice constant a, which is the length of the unit cell (or conventional cell).

    Returns:
        structure_file: The path to generated structure file.
        cell: The cell parameters of the generated structure as a list of lists.
        coordinate: The atomic coordinates of the generated structure as a list of lists.
    
    Examples:
    >>> # FCC Cu
    >>> cu_fcc = generate_bulk_structure('Cu', 'fcc', a=3.6)
    >>>
    >>> # HCP Mg with custom c-axis
    >>> mg_hcp = generate_bulk_structure('Mg', 'hcp', a=3.2, c=5.2, orthorhombic=True)
    >>>
    >>> # Diamond Si
    >>> si_diamond = generate_bulk_structure('Si', 'diamond', a=5.43, cubic=True)
    >>> # Zincblende GaAs
    >>> gaas_zincblende = generate_bulk_structure('GaAs', 'zincblende', a=5.65, cubic=True)
    
    """
    try:
        if a is None:
            raise ValueError("Lattice constant 'a' must be provided for all crystal structures.")

        from ase.build import bulk
        special_params = {}

        if crystal_structure == 'hcp':
            if c is not None:
                special_params['c'] = c
            special_params['orthorhombic'] = orthorhombic

        if crystal_structure in ['fcc', 'bcc', 'diamond', 'zincblende']:
            special_params['cubic'] = cubic

        structure = bulk(
            name=element,
            crystalstructure=crystal_structure,
            a=a,
            **special_params
        )
        work_path = generate_work_path(create=True)

        if file_format == "cif":
            structure_file = f"{work_path}/{element}_{crystal_structure}.cif"
            structure.write(structure_file, format="cif")
        elif file_format == "poscar":
            structure_file = f"{work_path}/{element}_{crystal_structure}.vasp"
            structure.write(structure_file, format="vasp")
        else:
            raise ValueError("Unsupported file format. Use 'cif' or 'poscar'.")

        return {
            "structure_file": Path(structure_file).absolute(),
        }
    except Exception as e:
        return {
            "structure_file": None,
            "message": f"Generating bulk structure failed: {e}"
        }

@mcp.tool()
def generate_bulk_structure_from_wyckoff_position(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    spacegroup: str | int,
    wyckoff_positions: List[Tuple[str, List[float], str]],
    crystal_name: str = 'crystal',
    format: Literal["cif", "poscar"] = "cif"
) -> Dict[str, Any]:
    """
    Generate crystal structure from lattice parameters, space group and wyckoff positions.

    Args:
        a, b, c (float): Length of 3 lattice vectors
        alpha, beta, gamma (float): Angles between \vec{b} and \vec{c}, \vec{c} and \vec{a}, \vec{a} and \vec{b} respectively.
        spacegroup (str | int): International space group names or index of space group in standard crystal tables. 
        wyckoff_positions (List[Tuple[str, List[int], str]]): List of Wyckoff positions in the crystal. For each wyckoff position, 
            the first is the symbol of the element, the second is the fractional coordinate, and the third is symbol of the wyckoff position.
    
    Returns:
        Path to the generated crystal structure file.

    Raises:
    """
    try:
        lattice = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

        crys_stru = Structure.from_spacegroup(
            sg=spacegroup,
            lattice=lattice,
            species=[wyckoff_position[0] for wyckoff_position in wyckoff_positions],
            coords=[wyckoff_position[1] for wyckoff_position in wyckoff_positions],
            tol=0.001,
        )

        work_path = generate_work_path(create=True)
        
        crys_file_name = Path(f"{work_path}/{crystal_name}.{format}").absolute()
        write(crys_file_name, crys_stru.to_ase_atoms(), format)

        return {"structure_file": crys_file_name}
    except Exception as e:
        return {"structure_file": None,
                "message": f"Generating bulk structure from Wyckoff position failed: {e}"}

@mcp.tool()
def generate_molecule_structure(
    molecule_name: Literal['PH3', 'P2', 'CH3CHO', 'H2COH', 'CS', 'OCHCHO', 'C3H9C', 'CH3COF',
                           'CH3CH2OCH3', 'HCOOH', 'HCCl3', 'HOCl', 'H2', 'SH2', 'C2H2', 'C4H4NH',
                           'CH3SCH3', 'SiH2_s3B1d', 'CH3SH', 'CH3CO', 'CO', 'ClF3', 'SiH4',
                           'C2H6CHOH', 'CH2NHCH2', 'isobutene', 'HCO', 'bicyclobutane', 'LiF',
                           'Si', 'C2H6', 'CN', 'ClNO', 'S', 'SiF4', 'H3CNH2', 'methylenecyclopropane',
                           'CH3CH2OH', 'F', 'NaCl', 'CH3Cl', 'CH3SiH3', 'AlF3', 'C2H3', 'ClF', 'PF3',
                           'PH2', 'CH3CN', 'cyclobutene', 'CH3ONO', 'SiH3', 'C3H6_D3h', 'CO2', 'NO',
                           'trans-butane', 'H2CCHCl', 'LiH', 'NH2', 'CH', 'CH2OCH2', 'C6H6',
                           'CH3CONH2', 'cyclobutane', 'H2CCHCN', 'butadiene', 'C', 'H2CO', 'CH3COOH',
                           'HCF3', 'CH3S', 'CS2', 'SiH2_s1A1d', 'C4H4S', 'N2H4', 'OH', 'CH3OCH3',
                           'C5H5N', 'H2O', 'HCl', 'CH2_s1A1d', 'CH3CH2SH', 'CH3NO2', 'Cl', 'Be', 'BCl3',
                           'C4H4O', 'Al', 'CH3O', 'CH3OH', 'C3H7Cl', 'isobutane', 'Na', 'CCl4',
                           'CH3CH2O', 'H2CCHF', 'C3H7', 'CH3', 'O3', 'P', 'C2H4', 'NCCN', 'S2', 'AlCl3',
                           'SiCl4', 'SiO', 'C3H4_D2d', 'H', 'COF2', '2-butyne', 'C2H5', 'BF3', 'N2O',
                           'F2O', 'SO2', 'H2CCl2', 'CF3CN', 'HCN', 'C2H6NH', 'OCS', 'B', 'ClO',
                           'C3H8', 'HF', 'O2', 'SO', 'NH', 'C2F4', 'NF3', 'CH2_s3B1d', 'CH3CH2Cl',
                           'CH3COCl', 'NH3', 'C3H9N', 'CF4', 'C3H6_Cs', 'Si2H6', 'HCOOCH3', 'O', 'CCH',
                           'N', 'Si2', 'C2H6SO', 'C5H8', 'H2CF2', 'Li2', 'CH2SCH2', 'C2Cl4', 'C3H4_C3v',
                           'CH3COCH3', 'F2', 'CH4', 'SH', 'H2CCO', 'CH3CH2NH2', 'Li', 'N2', 'Cl2', 'H2O2',
                           'Na2', 'BeH', 'C3H4_C2v', 'NO2', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F',
                           'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
                           'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                           'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
                           'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm',
                           'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re',
                           'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra',
                           'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                           'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl',
                           'Mc', 'Lv', 'Ts', 'Og'] = "H2O",
    cell: Optional[List[List[float]]] = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
    vacuum: Optional[float] = 5.0,
    output_file_format: Literal["cif", "poscar", "abacus"] = "abacus") -> Dict[str, Any]:
    """
    Generate molecule structure from ase's collection of molecules or single atoms.
    Args:
        molecule_name: The name of the molecule or atom to generate. It can be a chemical symbol (e.g., 'H', 'O', 'C') or
                       a molecule name in g2 collection contained in ASE's collections.
        cell: The cell parameters for the generated structure. Default is a 10x10x10 Angstrom cell. Units in angstrom.
        vcuum: The vacuum space to add around the molecule. Default is 7.0 Angstrom.
        output_file_format: The format of the output file. Default is 'abacus'. 'poscar' represents POSCAR format used by VASP.
    Returns:
        A dictionary containing:
        - structure_file: The absolute path to the generated structure file.
        - cell: The cell parameters of the generated structure as a list of lists.
        - coordinate: The atomic coordinates of the generated structure as a list of lists.
    """
    try:
        if output_file_format == "poscar":
            output_file_format = "vasp"  # ASE uses 'vasp' format for POSCAR files
        if molecule_name in g2.names:
            atoms = molecule(molecule_name)
            atoms.set_cell(cell)
            atoms.center(vacuum=vacuum)
        elif molecule_name in chemical_symbols and molecule_name != "X":
            atoms = Atoms(symbol=molecule_name, positions=[[0, 0, 0]], cell=cell)

        work_path = generate_work_path(create=True)
        
        if output_file_format == "abacus":
            stru_file_path = Path(f"{work_path}/{molecule_name}.stru").absolute()
        else:
            stru_file_path = Path(f"{work_path}/{molecule_name}.{output_file_format}").absolute()

        atoms.write(stru_file_path, format=output_file_format)

        return {
            "structure_file": Path(stru_file_path).absolute(),
        }
    except Exception as e:
        return {
            "structure_file": None,
            "message": f"Generating molecule structure failed: {e}"
        }