"""
Functions about cube files. Currently including:
- Electron localization function (ELF)
- Charge density difference
"""
import os
from pathlib import Path
from abacustest.lib_prepare.abacus import ReadInput, WriteInput

from abacusagent.init_mcp import mcp
from abacusagent.modules.util.comm import run_abacus, generate_work_path, link_abacusjob

@mcp.tool()
def abacus_cal_elf(abacusjob_dir: Path):
    """
    Calculate electron localization function (ELF) using ABACUS.
    
    Args:
        abacusjob_dir (Path): Path to the ABACUS job directory.
    
    Returns:
        Dict[str, Any]: A dictionary containing:
         - work_path: Path to the directory containing ABACUS input files and output files when calculating ELF.
         - elf_file: ELF file path (in .cube file format).
    
    Raises:
        ValueError: If the nspin in INPUT is not 1 or 2.
        FileNotFoundError: If the ELF file is not found in the output directory.
    """
    work_path = Path(generate_work_path()).absolute()
    link_abacusjob(src=abacusjob_dir, dst=work_path, copy_files=["INPUT"])

    input_params = ReadInput(os.path.join(work_path, "INPUT"))
    if input_params.get('nspin', 1) not in [1, 2]:
        raise ValueError("ELF calculation only supports nspin=1 or nspin=2.")
    
    input_params['calculation'] = 'scf'
    input_params['out_elf'] = 1
    WriteInput(input_params, os.path.join(work_path, "INPUT"))

    run_abacus(work_path)
    
    suffix = input_params.get('suffix', 'ABACUS')
    elf_file = os.path.join(work_path, f'OUT.{suffix}/ELF.cube')
    if not os.path.exists(elf_file):
        raise FileNotFoundError(f"ELF file not found in {work_path}")

    return {
        "work_path": Path(work_path).absolute(),
        "elf_file": Path(elf_file).absolute()
    }
