"""
Functions about cube files. Currently including:
- Electron localization function (ELF)
- Charge density difference
"""
from pathlib import Path
from typing import Literal, Optional, TypedDict, Dict, Any, List, Tuple, Union

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.cube import abacus_cal_elf as _abacus_cal_elf
from abacusagent.modules.submodules.cube import abacus_cal_charge_density_difference as _abacus_cal_charge_density_difference
from abacusagent.modules.submodules.cube import abacus_cal_spin_density as _abacus_cal_spin_density

@mcp.tool()
def abacus_cal_elf(abacus_inputs_dir: Path):
    """
    Calculate electron localization function (ELF) using ABACUS.
    
    Args:
        abacus_inputs_dir (Path): Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
    
    Returns:
        Dict[str, Any]: A dictionary containing:
         - elf_work_path: Path to the directory containing ABACUS input files and output files when calculating ELF.
         - elf_file: ELF file path (in .cube file format).
    
    Raises:
        ValueError: If the nspin in INPUT is not 1 or 2.
        FileNotFoundError: If the ELF file is not found in the output directory.
    """
    return _abacus_cal_elf(abacus_inputs_dir)

@mcp.tool()
def abacus_cal_charge_density_difference(
    abacus_inputs_dir: Path,
    subsys1_atom_index: Optional[List[int]] = [0],
) -> Dict[str, Any]:
    """
    Calculate charge density difference using ABACUS.
    
    Args:
        abacus_inputs_dir (Path): Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        subsys1_atom_index (Optional[List[int]]): Atom indices of the first subsystem. Should not be empty. The atom indices of
            the second subsystem will be determined by the remaining atoms in the full system.
    
    Returns:
        Dict[str, Any]: A dictionary containing:
         - charge_density_diff_work_path: Path to the directory containing ABACUS input files and output files when calculating charge density difference.
         - charge_density_diff_file: Charge density difference file path (in .cube file format).
    
    Raises:
        FileNotFoundError: If the charge density difference file is not found in the output directory.
    """
    return _abacus_cal_charge_density_difference(abacus_inputs_dir, subsys1_atom_index)

@mcp.tool()
def abacus_cal_spin_density(
    abacus_inputs_dir: Path
) -> Dict[str, Any]:
    """
    Calculate the spin density for collinear spin-polarized system (nspin=2).

    Args:
        abacus_inputs_dir (Path): Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
    
    Returns:
        A dictionary containing the following keys:
        - spin_density_work_path (Path): Path to the ABACUS job directory calculating spin density.
        - spin_density_file (Path): Path to the cube file containing the spin density.
    
    Raises:
        ValueError: If nspin in INPUT file is not 2.
    """
    return _abacus_cal_spin_density(abacus_inputs_dir)
