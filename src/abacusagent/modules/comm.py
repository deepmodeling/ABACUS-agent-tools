import subprocess
import select
from pathlib import Path
from typing import List, Tuple, Union
import os
import time
import json

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

def run_abacus(job_paths: Union[str, List[str], Path, List[Path]]):
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
    
    if os.environ.get("ABACUSAGENT_SUBMIT_TYPE") == "local":
        for job_path in job_paths:
            if not job_path.is_dir():
                raise ValueError(f"{job_path} is not a valid directory.")
            
            os.chdir(job_path)
            cmd = f"{os.environ['ABACUS_COMMAND']} > abacus.log 2>&1"
            return_code, out, err = run_command(cmd)
            os.chdir(cwd)
            if return_code != 0:
                raise RuntimeError(f"ABACUS command failed with error: {err}")
            
    elif os.environ.get("ABACUSAGENT_SUBMIT_TYPE") == "bohrium":
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
                    "command": f"{os.environ['BOHRIUM_ABACUS_COMMAND']} > abacus.log 2>&1",
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
            
            
            
            

    