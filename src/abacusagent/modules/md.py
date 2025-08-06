import os
from pathlib import Path
from typing import Dict, Any, List, Literal
import re
from pprint import pprint

import numpy as np
from ase import Atoms
from ase.io.trajectory import TrajectoryWriter
from abacustest.lib_prepare.abacus import ReadInput, WriteInput, AbacusStru

from abacusagent.init_mcp import mcp
from abacusagent.modules.util.comm import generate_work_path, link_abacusjob, run_abacus


def get_last_md_stru(md_stru_outputdir: Path) -> Path:
    if md_stru_outputdir.exists() and md_stru_outputdir.is_dir():
        stru_files = [file.name for file in md_stru_outputdir.iterdir() if file.is_file()]
    
    last_frame_idx = 0
    last_stru = Path(os.path.join(md_stru_outputdir, "STRU_MD_0")).absolute()
    for stru_file in stru_files:
        frame_idx = int(stru_file.split('_')[-1])
        if frame_idx > last_frame_idx:
            last_frame_idx = frame_idx
            last_stru = Path(os.path.join(md_stru_outputdir, stru_file)).absolute()
    
    return last_stru

def convert_md_dump_to_ase_traj(md_dump_path: Path, traj_filename: str="md_traj.traj"):
    def list_to_ndarray(lst):
        for i in range(len(lst)):
            lst[i] = float(lst[i])
        return np.array(lst)

    md_step_line_num = []
    with open(md_dump_path) as fin:
        for line_number, lines in enumerate(fin, start=1):
            if 'MDSTEP:' in lines:
                md_step_line_num.append(line_number)
    
    md_steps = []
    md_dump_file = open(md_dump_path)
    for i in range(len(md_step_line_num)):
        if i != len(md_step_line_num) - 1:
            md_step_data = []
            for j in range(md_step_line_num[i+1]-md_step_line_num[i]):
                md_step_data.append(md_dump_file.readline())
        else:
            md_step_data = md_dump_file.readlines()
        
        for idx, line in enumerate(md_step_data):
            if 'LATTICE_CONSTANT: ' in line:
                words = line.split()
                lattice_constant = float(words[1])
            
            if 'LATTICE_VECTORS' in line:
                a, b, c = md_step_data[idx+1].split(), md_step_data[idx+2].split(), md_step_data[idx+3].split()
                lattice_vectors = [a, b, c]
                for j1 in range(len(lattice_vectors)):
                    for j2 in range(len(lattice_vectors[j1])):
                        lattice_vectors[j1][j2] = float(lattice_vectors[j1][j2])
                
                lattice_vectors = np.array(lattice_vectors) * lattice_constant
            
            elements, positions, forces, velocities = [], [], [], []
            if 'INDEX    LABEL' in line:
                for coord_line in md_step_data[idx+1:]:
                    if coord_line.strip() != '':
                        words = coord_line.split()
                        elements.append(words[1])
                        positions.append(list_to_ndarray(words[2:5]))
                        forces.append(list_to_ndarray(words[5:8]))
                        velocities.append(list_to_ndarray(words[8:]))

            if len(elements) != 0 and len(positions) != 0:
                md_step = Atoms(symbols = elements,
                                positions=positions,
                                cell=lattice_vectors,
                                velocities=velocities)
                md_steps.append(md_step)
    
    traj_writer = TrajectoryWriter(Path(os.path.join(md_dump_path.parent, traj_filename)).absolute(), mode="a", atoms=md_step)
    for md_step in md_steps:
        traj_writer.write(md_step)
    traj_writer.close()
    
    return Path(traj_filename).absolute()

@mcp.tool()
def abacus_run_md(
    abacusjob_dir: Path,
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
        abacusjob_dir (Path): Path to ABACUS input files.
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
        md_dumpfreq (int): The output frequency of OUT.${suffix}/MD_dump in molecular dynamics calculations. Generally the default value 1
            is OK. For very long ab-initio MD calculations, increasing md_dumpfreq can help reducing the size of MD_dump.
        md_seed (int): The random seed to initialize random numbers used in molecular dynamics calculations.
            - < 0: No srand() function is called.
            - >= 0: The function srand(md_seed) is called.
    Returns:
    Raises:

    """
    try:
        work_path = Path(generate_work_path()).absolute()
        link_abacusjob(src=abacusjob_dir, dst=work_path, copy_files=['INPUT', 'STRU'], exclude_directories=True)
        input_params = ReadInput(os.path.join(work_path, "INPUT"))

        input_params['calculation'] = 'md'
        for keyword, value in {
            'md_type': md_type,
            'md_nstep': md_nstep,
            'md_dt': md_dt,
            'md_tfirst': md_tfirst,
            'md_tlast': md_tlast,
            'md_thermostat': md_thermostat,
            'md_pmode': md_pmode,
            'md_pcouple': md_pcouple,
            'md_dumpfreq': md_dumpfreq,
            'md_seed': md_seed
        }.items():
            input_params[keyword] = value

        if 'init_vel' in input_params.keys():
            if md_tfirst > 0:
                input_params.pop('init_vel')
            else:
                stru_file = os.path.join(work_path, input_params.get('stru_file', 'STRU'))
                stru = AbacusStru.ReadStru(stru_file)
                if None in stru._velocity:
                    raise ValueError("Use md_tfirst < 0 should provide initial velocities in STRU")

        if input_params['calculation'] not in ['nvt', 'npt']:
            input_params.pop('md_tlast')

        WriteInput(input_params, os.path.join(work_path, "INPUT"))

        run_abacus(work_path)

        suffix = input_params.get('suffix', 'ABACUS')
        return {'md_work_path': work_path,
                'md_last_stru': get_last_md_stru(Path(os.path.join(work_path, f'OUT.{suffix}/STRU')).absolute()),
                'md_traj_file': convert_md_dump_to_ase_traj(Path(os.path.join(work_path, f'OUT.{suffix}/MD_dump')).absolute())}

    except Exception as e:
        return {'md_work_path': Path(""),
                'md_last_stru': Path(""),
                'md_traj_file': Path(""),
                "message": f"Error occured during the running md: {e}"}

if __name__ == '__main__':
    get_last_md_stru(Path('/tmp/abacusagent/20250805162511/20250805162511.abacus_run_md.8eb74ccf/OUT.Si/STRU'))
    convert_md_dump_to_ase_traj(Path('/tmp/abacusagent/20250805162511/20250805162511.abacus_run_md.8eb74ccf/OUT.Si/MD_dump'))