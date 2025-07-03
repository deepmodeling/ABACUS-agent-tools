import subprocess
import select
from pathlib import Path
from typing import List, Tuple, Union, Optional
import os
import time
import json
import traceback
import uuid
import glob

from abacustest.lib_prepare.abacus import ReadInput

def run_command(
        cmd,
        shell=True
):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=shell,
        executable='/bin/bash'
    )
    out = ""
    err = ""
    while True:
        readable, _, _ = select.select(
            [process.stdout, process.stderr], [], [])

        for fd in readable:
            if fd == process.stdout:
                line = process.stdout.readline()
                print(line.decode()[:-1])
                out += line.decode()
            elif fd == process.stderr:
                line = process.stderr.readline()
                print("STDERR:", line.decode()[:-1])
                err += line.decode()

        return_code = process.poll()
        if return_code is not None:
            break
    return return_code, out, err

def remove_comm_prefix(paths: Union[List[Path], List[str]]) -> List[str]:
    """
    Remove the common prefix from a list of paths.
    This is useful for displaying relative paths in logs.
    """
    if not paths:
        return []

    if len(paths) == 1:
        return [os.path.basename(str(paths[0]))]
    
    # Convert all paths to absolute paths
    abs_paths = [Path(p).absolute() for p in paths]
    
    # Find the common prefix
    common_prefix = os.path.commonpath(abs_paths)
    
    # Remove the common prefix from each path
    relative_paths = [str(p.relative_to(common_prefix)) for p in abs_paths]
    
    return relative_paths

def get_physical_cores():
    """
    """
    # 对于Linux系统，解析/proc/cpuinfo文件
    with open('/proc/cpuinfo', 'r') as f:
        cpuinfo = f.read()
    
    # 统计物理ID的数量和每个物理ID下的核心数
    physical_ids = set()
    cores_per_socket = {}
    
    for line in cpuinfo.split('\n'):
        if line.startswith('physical id'):
            phys_id = line.split(':')[1].strip()
            physical_ids.add(phys_id)
        elif line.startswith('cpu cores'):
            cores = int(line.split(':')[1].strip())
            if phys_id in cores_per_socket:
                cores_per_socket[phys_id] = max(cores_per_socket[phys_id], cores)
            else:
                cores_per_socket[phys_id] = cores
    
    # 计算总物理核心数
    if physical_ids and cores_per_socket:
        return sum(cores_per_socket.values())
    else:
        # 备选方法：使用lscpu命令
        output = subprocess.check_output('lscpu', shell=True).decode()
        for line in output.split('\n'):
            if line.startswith('Core(s) per socket:'):
                cores_per_socket = int(line.split(':')[1].strip())
            elif line.startswith('Socket(s):'):
                sockets = int(line.split(':')[1].strip())
        return cores_per_socket * sockets

def run_abacus(job_paths: Union[str, List[str], Path, List[Path]],
               log_file: Optional[str] = "abacus.log") -> None:
    """
    Run the Abacus on the given job paths.
    If job_paths is a list, it will run the command for each path.
    If job_paths is a single Path, it will run the command for that path.
    """
    if isinstance(job_paths, (str, Path)):
        job_paths = [job_paths]
        
    try:
        job_paths = [Path(job_path).absolute() for job_path in job_paths]
    except Exception as e:
        raise ValueError(f"Invalid job path(s): {job_paths}. Error: {str(e)}")
    
    cwd = os.getcwd()
    
    submit_type = os.environ.get("ABACUSAGENT_SUBMIT_TYPE", "local").lower()
    
    if submit_type == "local":
        physical_cores = get_physical_cores()
        command_cmd = os.environ.get("ABACUS_COMMAND", f"OMP_NUM_THREADS=1 mpirun -np {physical_cores} abacus") + f" > {log_file} 2>&1"     

        for job_path in job_paths:
            if not job_path.is_dir():
                raise ValueError(f"{job_path} is not a valid directory.")
            
            os.chdir(job_path)           
            return_code, out, err = run_command([command_cmd])
            os.chdir(cwd)
            if return_code != 0:
                raise RuntimeError(f"ABACUS command failed with error: {err}")
            
    elif submit_type == "bohrium":
        # use abacustest to submit the job to bohrium
        # check the environment variables is not ""
        key_envs = [
            "BOHRIUM_USERNAME", "BOHRIUM_PASSWORD", "BOHRIUM_PROJECT_ID",
            "BOHRIUM_ABACUS_IMAGE", "BOHRIUM_ABACUS_MACHINE", "BOHRIUM_ABACUS_COMMAND"
        ]
        if not all(os.environ.get(var,"").strip() for var in key_envs):
            msg = "\n".join([
                f"{var}: '{os.environ.get(var, '')}'" for var in key_envs
            ])
            raise ValueError("Bohrium environment variables are not set correctly:\n" + msg)
        
        pwd = os.getcwd()
        
        work_path = os.environ.get("ABACUSAGENT_WORK_PATH")
        os.makedirs(work_path, exist_ok=True)
        # create a temporary directory in work_path to submit the job
        current_time = time.strftime("%Y%m%d%H%M%S")
        work_path_abacustest = Path(work_path) / f"abacustest.{current_time}"
        os.makedirs(work_path_abacustest, exist_ok=True)
        # make a soft link to the job paths in the work_path_abacustest
        jobs = remove_comm_prefix(job_paths)
        for idx, job_path in enumerate(jobs):
            if not Path(job_paths[idx]).is_dir():
                raise ValueError(f"{job_paths[idx]} is not a valid directory.")
            os.symlink(Path(job_paths[idx]), work_path_abacustest / job_path)
            
        os.chdir(work_path_abacustest)
        setting = {
            "save_path": "results",
            "bohrium_group_name": f"abacus-agent.abacustest.{current_time}",
            "run_dft": 
                {
                    "example": jobs,
                    "command": f"{os.environ['BOHRIUM_ABACUS_COMMAND']} > {log_file} 2>&1",
                    "image": os.environ["BOHRIUM_ABACUS_IMAGE"],
                    "bohrium":{
                        "scass_type": os.environ["BOHRIUM_ABACUS_MACHINE"],
                        "job_type": "container",
                        "platform": "ali"
                    }
                }
        }
        json.dump(setting, open("abacustest.json", "w"), indent=4)
        cmd = f"abacustest submit -p abacustest.json"
        return_code, out, err = run_command(cmd)
        
        # link the results to the original job paths
        if return_code != 0:
            os.chdir(pwd)
            raise RuntimeError(f"abacustest command failed with error: {err}")
        
        for idx, job_path in enumerate(jobs):
            result_path = work_path_abacustest / "results" / job_path
            if not result_path.exists():
                print(f"Warning: Result path {result_path} does not exist. Skipping.")
                continue
            # copy the result to the original job path
            os.system(f"cp -r {result_path}/* {job_paths[idx]}/")
        
        os.chdir(pwd)
    else:
        raise ValueError("Invalid ABACUSAGENT_SUBMIT_TYPE. Must be 'local' or 'bohrium'.")
            

def link_abacusjob(src: str, 
                   dst: str, 
                   include:Optional[List[str]]=None, 
                   exclude:Optional[List[str]]=None,
                   copy_files = ["INPUT", "STRU", "KPT"],
                   overwrite: Optional[bool] = True,
                   exclude_directories: Optional[bool] = False
                   ):
    """
    Link the ABACUS job files from src to dst.
    
    Parameters:
    src (str): Source directory containing the ABACUS job files.
    dst (str): Destination directory where the job files will be linked.
    include (Optional[List[str]]): List of files to include. If None, all files are included.
    exclude (Optional[List[str]]): List of files to exclude. If None, no files are excluded.
    copy_files (List[str]): List of files to copy from src to dst. Default is ["INPUT", "STRU", "KPT"].
    overwrite (bool): If True, existing files in the destination will be overwritten. Default is True.
    exclude_directories (bool): If True, directories will be excluded from linking. Default is False.
    
    Notes: 
        - If somes files are included in both include and exclude, the file will be excluded.
        - glob.glob is used to match the files in the source directory.
    """
    src = Path(src).absolute()
    dst = Path(dst).absolute()
    
    if not src.is_dir():
        raise ValueError(f"{src} is not a valid directory.")
    
    if dst.is_file():
        raise ValueError(f"{dst} is a file, not a directory.")
    os.makedirs(dst, exist_ok=True)
    
    if include is None:
        include = ["*"]
    include_files = []
    for pattern in include:
        include_files.extend(src.glob(pattern))
        
    if exclude is None:
        exclude = []
    exclude_files = []
    for pattern in exclude:
        exclude_files.extend(src.glob(pattern))
    
    # Remove excluded files from included files
    include_files = [f for f in include_files if f not in exclude_files]
    if not include_files:
        traceback.print_stack()
        print("No files to link after applying include and exclude patterns.\n",
              f"Include patterns: {include}, Exclude patterns: {exclude}, Source: {src}, Destination: {dst}\n",
              f"Files in source: {list(src.glob('*'))}"
              )
    else:
        for file in include_files:
            if exclude_directories and os.path.isdir(file):
                continue
            
            dst_file = dst / file.name
            if dst_file.exists():
                if overwrite:
                    dst_file.unlink()
                else:
                    continue
            if str(file.name) in copy_files:
                os.system(f"cp {file} {dst_file}")
            else:
                os.symlink(file, dst_file)
            
def generate_work_path(create: bool = True) -> str:
    """
    Generate a unique working directory path based on call function and current time.
    
    directory = calling function name + current time + random string.
    
    Returns:
        str: The path to the working directory.
    """
    calling_function = traceback.extract_stack(limit=2)[-2].name
    current_time = time.strftime("%Y%m%d%H%M%S")
    random_string = str(uuid.uuid4())[:8]
    work_path = f"{current_time}.{calling_function}.{random_string}"
    if create:
        os.makedirs(work_path, exist_ok=True)
    
    return work_path


def has_chgfile(abacus_jobpath:Path) -> bool:
    """
    Check if the charge file exists in the given ABACUS job path. Used to determine if an abacus job has writen the charge file.
    
    Parameters:
        abacus_jobpath (Path): The path to the ABACUS job directory.
        
    Returns:
        bool: True if the charge file exists, False otherwise.
    """
    input_file = os.path.join(abacus_jobpath, "INPUT")
    if not os.path.isfile(input_file):
        suffix = "ABACUS"
    else:
        suffix = ReadInput(input_file).get("suffix", "ABACUS")

    chgfile = os.path.join(abacus_jobpath, "OUT." + suffix, "SPIN1_CHG.cube")
    return os.path.isfile(chgfile)


def has_pyatb_matrix_files(work_path: Path) -> bool:
    """
    Check if the necessary files for Pyatb calculation exist in the given work path.
    
    Parameters:
        work_path (Path): The path to the working directory.
        
    Returns:
        bool: True if the necessary files exist, False otherwise.
    """
    input_file = os.path.join(work_path, "INPUT")
    if not os.path.isfile(input_file):
        suffix = "ABACUS"
    else:
        suffix = ReadInput(input_file).get("suffix", "ABACUS")
        
    return (os.path.isfile(os.path.join(work_path, "OUT." + suffix, "data-HR-sparse_SPIN0.csr")) and
            os.path.isfile(os.path.join(work_path, "OUT." + suffix, "data-SR-sparse_SPIN0.csr")) and 
            os.path.isfile(os.path.join(work_path, "OUT." + suffix, "data-rR-sparse.csr")))