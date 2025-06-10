import os
import json

ENVS = {
    "ABACUSAGENT_WORK_PATH": "/tmp/abacusagent",
    "ABACUSAGENT_SUBMIT_TYPE": "local",  # local, bohrium

    # connection settings
    "ABACUSAGENT_HOST": "localhost",
    "ABACUSAGENT_PORT": "50001", 
    "ABACUSAGENT_MODEL": "fastmcp",  # fastmcp, abacus, bohrium

    # bohrium settings
    "BOHRIUM_USERNAME": "",
    "BOHRIUM_PASSWORD": "",
    "BOHRIUM_PROJECT_ID": "",
    "BOHRIUM_ABACUS_IMAGE": "registry.dp.tech/dptech/abacus-stable:LTSv3.10", # THE bohrium image for abacus calculations, 
    "BOHRIUM_ABACUS_MACHINE": "c32_m64_cpu",  # THE bohrium machine for abacus calculations, c32_m64_cpu
    "BOHRIUM_ABACUS_COMMAND": "OMP_NUM_THREADS=1 mpirun -np 16 abacus",
    
    # abacus pp orb settings
    "ABACUS_COMMAND": "abacus",  # abacus executable command
    "ABACUS_PP_PATH": "",  # abacus pseudopotential library path
    "ABACUS_ORB_PATH": "",  # abacus orbital library path
    
    "_comments":{
        "ABACUS_WORK_PATH": "The working directory for AbacusAgent, where all temporary files will be stored.",
        "ABACUS_SUBMIT_TYPE": "The type of submission for ABACUS, can be local or bohrium.",
        "ABACUSAGENT_HOST": "The host address for the AbacusAgent server.",
        "ABACUSAGENT_PORT": "The port number for the AbacusAgent server.",
        "ABACUSAGENT_MODEL": "The model to use for AbacusAgent, can be 'fastmcp', 'test', or 'dp'.",
        "BOHRIUM_USERNAME": "The username for Bohrium.",        
        "BOHRIUM_PASSWORD": "The password for Bohrium.",
        "BOHRIUM_PROJECT_ID": "The project ID for Bohrium.",
        "BOHRIUM_ABACUS_IMAGE": "The image for Abacus on Bohrium.",
        "BOHRIUM_ABACUS_MACHINE": "The machine type for Abacus on Bohrium.",
        "BOHRIUM_ABACUS_COMMAND": "The command to run Abacus on Bohrium",
        "ABACUS_COMMAND": "The command to execute Abacus on local machine.",
        "ABACUS_PP_PATH": "The path to the pseudopotential library for Abacus.",
        "ABACUS_ORB_PATH": "The path to the orbital library for Abacus.",
        "_comments": "This dictionary contains the default environment variables for AbacusAgent."
    }
}

def set_envs(model_input=None, port_input=None, host_input=None):
    """
    Set environment variables for AbacusAgent.
    
    The priority of environment variables is as follows:
    1. The value of `port_input` and `host_input` has the highest priority.
    2. The existing environment reading from `os.environ`.
    3. The values in the `~/.abacusagent/env.json` file.

    If the `~/.abacusagent/env.json` file does not exist, it will be created with default values.
    """
    # read setting in ~/.abacusagent/env.json
    envjson_file = os.path.expanduser("~/.abacusagent/env.json")
    if os.path.isfile(envjson_file):
        envjson = json.load(open(envjson_file, "r"))
    else:
        envjson = {}
        
    update_envjson = False    
    for key, value in ENVS.items():
        if key not in envjson:
            envjson[key] = value
            update_envjson = True
    
    if port_input is not None:
        os.environ["ABACUSAGENT_PORT"] = str(port_input)
    if host_input is not None:
        os.environ["ABACUSAGENT_HOST"] = str(host_input)
    if model_input is not None:
        os.environ["ABACUSAGENT_MODEL"] = str(model_input)
        
    for key, value in envjson.items():
        if key.startswith("_"):
            continue
        elif key not in os.environ:
            os.environ[key] = str(value)
    
    if update_envjson:
        # write envjson to ~/.abacusagent/env.json
        os.makedirs(os.path.dirname(envjson_file), exist_ok=True)
        json.dump(
            envjson,
            open(envjson_file, "w"),
            indent=4
        )
    
    