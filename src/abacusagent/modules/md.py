from pathlib import Path
from typing import Dict, Any, List, Literal

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.md import abacus_run_md as _abacus_run_md

@mcp.tool()
def abacus_run_md(
    abacus_inputs_dir: Path,
    md_type: Literal['nve', 'nvt', 'npt', 'langevin'] = 'nve',
    md_nstep: int = 10,
    md_dt: float = 1.0,
    md_tfirst: float = 300.0,
    md_tlast: float = 300.0,
    md_thermostat: Literal['nhc', 'anderson', 'berendsen', 'rescaling', 'rescale_v'] = 'nhc',
    md_pmode: Literal['iso', 'aniso', 'tri'] = 'iso',
    md_pcouple: Literal['none', 'xy', 'xz', 'yz', 'xyz'] = 'none',
    md_dumpfreq: int = 1,
    md_seed: int = -1
) -> Dict[str, Any]:
    """
    Use ABACUS to do ab-initio molecular dynamics calculation.

    Args:
        abacus_inputs_dir (Path): Path to ABACUS input files.
        md_type (Literal['nve', 'nvt', 'npt', 'langevin']): The algorithm to integrate the equation of motion for molecular dynamics (MD).
            - nve: NVE ensemble with velocity Verlet algorithm.
            - nvt: NVT ensemble.
            - npt: Nose-Hoover style NPT ensemble.
            - langevin: NVT ensemble with Langevin thermostat.
        md_nstep (int): The total number of molecular dynamics steps.
        md_dt (float): The time step used in molecular dynamics calculations. THe unit in fs.
        md_tfirst (float): If set to larger than 0, initial velocity will be generated according to its value.
            If unset or smaller than 0, initial velocity will try to be read from STRU file.
        md_tlast (float): Only used in NVT/NPT simulations. If md_tlast is unset or less than zero, 
            md_tlast is set to md_tfirst. If md_tlast is set to be different from md_tfirst, 
            ABACUS will automatically change the temperature from md_tfirst to md_tlast
        md_thermostat (str): Specify the temperature control method used in NVT ensemble.
            - nhc: Nose-Hoover chain, see md_tfreq and md_tchain in detail.
            - anderson: Anderson thermostat, see md_nraise in detail.
            - berendsen: Berendsen thermostat, see md_nraise in detail.
            - rescaling: velocity Rescaling method 1, see md_tolerance in detail.
            - rescale_v: velocity Rescaling method 2, see md_nraise in detail.
        md_pmode (str): Specify the cell fluctuation mode in NPT ensemble based on the Nose-Hoover style non-Hamiltonian equations of motion.
            - iso: The three diagonal elements of the lattice are fluctuated isotropically.
            - aniso: The three diagonal elements of the lattice are fluctuated anisotropically.
            - tri: The lattice must be a lower-triangular matrix, and all six freedoms are fluctuated.
        md_pcouple (str): The coupled lattice vectors will scale proportionally in NPT ensemble based on the Nose-Hoover style non-Hamiltonian equations of motion.
            - none: Three lattice vectors scale independently.
            - xyz: Lattice vectors x, y, and z scale proportionally.
            - xy: Lattice vectors x and y scale proportionally.
            - xz: Lattice vectors x and z scale proportionally.
            - yz: Lattice vectors y and z scale proportionally.
        md_dumpfreq (int): The output frequency of OUT.$[suffix]/MD_dump in molecular dynamics calculations. Generally the default value 1
            is OK. For very long ab-initio MD calculations, increasing md_dumpfreq can help reducing the size of MD_dump.
        md_seed (int): The random seed to initialize random numbers used in molecular dynamics calculations.
            - < 0: No srand() function is called.
            - >= 0: The function srand(md_seed) is called.
    Returns:
        A dictionary containing:
            - md_work_path (Path): The working directory of the molecular dynamics calculation.
            - md_traj_file (Path): The path to the ASE trajectory file containing the MD steps.
            - traj_frame_nums (int): Number of frames in returned trajectory file.
            - normal_end (bool): Whether the ab-initio molecular dynamics calculation ended normally.
    """
    return _abacus_run_md(
        abacus_inputs_dir,
        md_type,
        md_nstep,
        md_dt,
        md_tfirst,
        md_tlast,
        md_thermostat,
        md_pmode,
        md_pcouple,
        md_dumpfreq,
        md_seed
    )
