'''
workflow of calculating Bader charges.
'''
import os
import re
import unittest
import subprocess
from typing import List, Dict, Optional

import numpy as np

from abacusagent.init_mcp import mcp
from abacusagent.util.control import FlowEnvironment

def parse_abacus_cmd(cmd: str) -> Dict[str, str|int]:
    '''
    parse the abacus command to get parallelization
    options, such as `mpirun`, `mpiexec`, `OMP_NUM_THREADS`, etc.
    A typical command looks like:
    ```bash
    OMP_NUM_THREADS=4 /path/to/mpirun -np 8 /path/to/abacus
    ```
    
    Parameters:
    cmd (str): The command string to parse.
    
    Returns:
    dict: A dictionary containing parsed options and their values.
    '''
    pat = r'^(?:OMP_NUM_THREADS=(\d+)\s+)?' \
          r'(?:([\w/.-]*mpirun|mpiexec)\s+-[np]\s+(\d+)\s+)?' \
          r'(.+abacus.*)$'
    match = re.match(pat, cmd)
    if not match:
        raise ValueError(f"Failed to parse command: {cmd}")
    return {
        'openmp': match.group(1) is not None,
        'nthreads': int(match.group(1)) if match.group(1) else 1,
        'mpi': match.group(2),
        'nproc': int(match.group(3)) if match.group(3) else 1,
        'abacus': match.group(4)
    }

def ver_cmp(v1: str|tuple[int], v2: str|tuple[int]) -> int:
    """
    Compare two version strings or tuples. For example, 
    "1.0" < "1.0.1" returns -1, "1.0.1" > "1.0" returns 1,
    
    Parameters:
    v1 (str|tuple[int]): First version to compare.
    v2 (str|tuple[int]): Second version to compare.
    
    Returns:
    int: -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2.
    """
    if isinstance(v1, str):
        v1 = list(map(int, re.split(r'\D+', v1.replace('v', ''))))
    if isinstance(v2, str):
        v2 = list(map(int, re.split(r'\D+', v2.replace('v', ''))))
    # Ensure both versions are of the same length
    max_len = max(len(v1), len(v2))
    v1 = tuple(list(v1) + [0] * (max_len - len(v1)))
    v2 = tuple(list(v2) + [0] * (max_len - len(v2)))
    
    return (v1 > v2) - (v1 < v2)  # Returns 1, 0, or -1

@FlowEnvironment.static_decorate()
@mcp.tool()
def calculate_charge_densities_with_abacus(
    abacus: str,
    jobdir: str
) -> Optional[List[str]]:
    """
    Calculate the charge density using ABACUS in the specified job directory.
    
    Parameters:
    abacus (str): Path to the abacus executable.
    jobdir (str): Directory where the job files are located.
    
    Returns:
    list: List of file names for the charge density cube files.
    """
    # get the abacus version with `abacus --version`
    
    cmd = parse_abacus_cmd(abacus)['abacus'] + ' --version'
    version = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if version.returncode != 0:
        raise RuntimeError(f"Failed to get ABACUS version: {version.stderr}")
    version = version.stdout.strip()
    print(f"ABACUS version: {version}")
    
    cwd = os.getcwd()
    os.chdir(jobdir)
    cmd = f"{abacus}"
    print(f"Running command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Abacus command failed with error: {result.stderr}")
    os.chdir(cwd)
    
    dftparam: Dict = parse_abacus_param(os.path.join(jobdir, 'INPUT'))
    
    nspin = int(dftparam.get('nspin', 1))
    # the file name has been changed to chgs1.cube in ABACUS v3.9.0.6
    fcube = [f'chgs{i+1}.cube' if ver_cmp(version, '3.9.0.6') >= 0
                               else f'SPIN{i+1}_CHG.cube'
             for i in range(nspin)]
    
    outdir = f'OUT.{dftparam.get("suffix", "ABACUS")}'
    outdir = os.path.join(jobdir, outdir)
    return [os.path.join(outdir, f) for f in fcube]

@FlowEnvironment.static_decorate()
@mcp.tool()
def parse_abacus_param(
    fn: str
) -> Dict[str, str]:
    """
    Parse the ABACUS parameter file to extract relevant parameters.
    
    Parameters:
    fn (str): Path to the ABACUS parameter file.
    
    Returns:
    dict: A dictionary containing the parsed parameters.
    """
    with open(fn) as f:
        data = [l.strip() for l in f.readlines() 
                if l.strip() and \
                    not l.startswith("#") and \
                    not 'INPUT_PARMETERS' in l]
    data = [l.split() for l in data]
    
    return dict(zip([d[0] for d in data], [' '.join(d[1:]) for d in data]))

@FlowEnvironment.static_decorate()
@mcp.tool()
def merge_charge_densities_of_different_spin(
    cube_manipulator: str,
    fcube: List[str]
) -> str:
    """
    Run the cube manipulator to process cube files.
    
    Parameters:
    cube_manipulator (str): Path to the cube manipulator executable.
    fcube (list): List of file names for the cube files to be manipulated.
    
    Returns:
    str: Output cube file path.
    """
    assert 0 < len(fcube) <= 2, "fcube should contain 1 or 2 cube files."
    if len(fcube) == 1:
        return fcube[0]
    
    dir_ = os.path.dirname(fcube[0])
    prefix_ = os.path.basename(fcube[0]).replace('.cube', '')
    fout = os.path.join(dir_, f"{prefix_}_merged.cube")
    cmd = f'python3 {cube_manipulator} -i {fcube[0]} -o {fout} -p {fcube[1]}'
    print(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Cube manipulator command failed with error: {result.stderr}")
    print(f"Cube manipulator output: {result.stdout}")
    return fout

@FlowEnvironment.static_decorate()
@mcp.tool()
def make_charge_density_cube(
    cube_manipulator: str,
    fcube: List[str]
) -> str:
    """
    Create a charge density cube file optinally using the cube manipulator.
    
    Parameters:
    cube_manipulator (str): Path to the cube manipulator executable.
    fcube (list): List of file names for the cube files to be manipulated.
    
    Returns:
    str: Output from the cube manipulator command.
    """
    return merge_charge_densities_of_different_spin(cube_manipulator, fcube)

@FlowEnvironment.static_decorate()
@mcp.tool()
def read_bader_acf(
    fn: str
) -> List[float]:
    """
    Read Bader charges from a file.
    
    Parameters:
    fn (str): Path to the file containing Bader charges.
    
    Returns:
    list: A list of Bader charges.
    """
    with open(fn) as f:
        data = f.readlines()[2:-4]  # Skip header and footer
    data = [l.strip().split() for l in data if l.strip()]
    data = np.array(data, dtype=float)
    return data[:, 4].tolist()  # Return the Bader charges

@FlowEnvironment.static_decorate()
@mcp.tool()
def calculate_bader_charges(
    bader: str,
    fcube: str
) -> List[str]:
    """
    Calculate Bader charges using the bader executable.
    
    Parameters:
    bader (str): Path to the bader executable.
    fcube (str): Path to the cube file containing charge density.
    
    Returns:
    list: A list of file names generated by the Bader analysis.
    """
    cmd = f'{bader} {fcube}'
    print(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Bader command failed with error: {result.stderr}")

    with open('bader.out', 'w') as f:
        f.write(result.stdout)
        
    files = [os.path.join(os.getcwd(), f) 
             for f in ['ACF.dat', 'AVF.dat', 'BCF.dat', 'bader.out']]
    if not all(os.path.exists(f) for f in files):
        raise FileNotFoundError("Incomplete Bader analysis output files.")
    
    return files

@FlowEnvironment.static_decorate()
@mcp.tool()
def postprocess_charge_densities(
    fcube: List[str]|str,
    cube_manipulator: str,
    bader: str
) -> List[float]:
    """
    postprocess the charge density to obtain Bader charges.
    
    Parameters:
    jobdir (list|str): List of file names for the cube files or a single file name.
    cube_manipulator (str): Path to the cube manipulator executable.
    bader (str): Path to the bader executable.
    
    Returns:
    list: A list of Bader charges.
    """
    
    _ = merge_charge_densities_of_different_spin(cube_manipulator, fcube)
    _ = calculate_bader_charges(bader, fcube)

    return read_bader_acf(os.path.join(os.getcwd(), 'ACF.dat'))

@FlowEnvironment.static_decorate()
@mcp.tool() # make it visible to the MCP server
def calculate(
    jobdir: str,
    abacus: str,
    cube_manipulator: str,
    bader: str
) -> List[float]:
    """
    Calculate Bader charges for a given job directory, with ABACUS as
    the dft software to calculate the charge density, and then postprocess
    the charge density with the cube manipulator and Bader analysis.
    
    Parameters:
    jobdir (str): Directory where the job files are located.
    abacus (str): Path to the abacus executable.
    cube_manipulator (str): Path to the cube manipulator executable.
    bader (str): Path to the bader executable.
    
    Returns:
    list: A list of Bader charges.
    """

    # Run ABACUS to calculate charge density
    jobdir = calculate_charge_densities_with_abacus(abacus=abacus, jobdir=jobdir)
    
    # Postprocess the charge density to obtain Bader charges
    return postprocess_charge_densities(jobdir, cube_manipulator, bader)

class TestBaderChargeWorkflow(unittest.TestCase):
    
    def test_parse_abacus_cmd(self):
        cmd = "OMP_NUM_THREADS=4 /path/to/mpirun -n 8 /path/to/abacus"
        expected = {
            'openmp': True,
            'nthreads': 4,
            'mpi': '/path/to/mpirun',
            'nproc': 8,
            'abacus': '/path/to/abacus'
        }
        result = parse_abacus_cmd(cmd)
        self.assertDictEqual(result, expected)

    def test_ver_cmp(self):
        self.assertEqual(ver_cmp("1.0.0", "1.0.1"), -1)
        self.assertEqual(ver_cmp("1.0.1", "1.0.0"), 1)
        self.assertEqual(ver_cmp("1.0.0", "1.0.0"), 0)
        self.assertEqual(ver_cmp((1, 0, 0), (1, 0, 1)), -1)
        self.assertEqual(ver_cmp((1, 0, 1), (1, 0, 0)), 1)
        self.assertEqual(ver_cmp((1, 0, 0), (1, 0, 0)), 0)
        self.assertEqual(ver_cmp("1.0", "1.0.0"), 0)
        self.assertEqual(ver_cmp("v3.10.0", "v3.9.0.4"), 1)

if __name__ == "__main__":
    unittest.main(exit=True)
    
    # Example prompt to invoke the Bader charge calculation
    '''
    Hello, I want to calculate Bader charges for the system in the directory at `/home/xxx/abacus-develop/representation/examples/scf/lcao_Si2`, could you please help me do this job? I think you will need to run ABACUS first to calculate the charge density, and then if there are two spin channels, you will need to merge the charge density cube files and then run Bader analysis on the merged file. If there is only one spin channel, you can directly run Bader analysis on the charge density cube file. There are several executables you will need to complete the whole process. You can run the ABACUS executable directly with `abacus`, the cube manipulator is a Python script that you can find it at `/home/xxx/abacus-develop/representation/tools/plot-tools/cube_manipulator.py`. And the Bader analysis program is at `/home/xxx/soft/bader`.
    '''
    