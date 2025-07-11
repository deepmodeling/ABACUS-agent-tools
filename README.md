# ABACUS Agent Tools

ABACUS-agent-tools is a Python package that provides the Model Context Protocol (MCP) tools to connect large language models (LLMs) to ABACUS computational jobs. It serves as a bridge between AI models and first principles calculations, enabling intelligent interaction with ABACUS workflows.

## Installation
To use ABACUS agent tools with Google Agent Development Kit (ADK), follow the recommended installation process:

### Configuring environment using uv.lock
There's a `uv.lock` file in the root directory of this project. You can use it to prepare python packages that needed by running
tools in ABACUS-agent-tools.

For packages that cannot be installed by `pip` or `conda`, please refer to **Other dependencies`=**.

### Install from scratch manually

1. Create and activate a conda enviroment:
```bash
conda create -n abacus-agent python=3.11
conda activate abacus-agent
```
2. Install necessary dependencies:
```bash
pip install mcp google-adk litellm bohr-agent-sdk pymatgen abacustest==0.4.24
```
3. Install ABACUS-agent-tools:
```bash
cd ..
git clone -b develop https://github.com/pxlxingliang/ABACUS-agent-tools.git
cd abacus-agent
pip install .
```
If you haven't installed ABACUS, you can install ABACUS to the same environment by conda:
```bash
conda install abacus "libblas=*=*mkl" mpich -c conda-forge
```

### Other dependencies

#### Bader
Follow the given step to download, compile and install `bader` executable required in Bader charge analysis:
```bash
apt install subversion
svn co https://theory.cm.utexas.edu/svn/bader
cd bader
cp makefile.lnx_ifort makefile
vi makefile # Choose `ifort` or `gfortran` as compiler
make
cp bader /usr/local/bin/bader
```

## Using ABACUS agent tools with Google ADK

### Use ABACUS agent tools and Google ADK on local machine

#### Starting ABACUS agent tools
Before launching `abacusagent`, you must provide the necessary configurations in the `~/.abacusagent/env.json` file. This file defines how the ABACUS agent tools generate input files and manage ABACUS calculation workflows.
Note: When running `abacusagent`, it will automatically check if the file exists. If not, `abacusagent` will create it and set some default values. It is recommended to run `abacusagent` once first before modifying this file.
```
{
    "_comments": {
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
```
Then you can start `abacusagent`.
```bash
>>> abacusagent
✅ Successfully loaded: abacusagent.modules.abacus
INFO:     Started server process [25487]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:50001 (Press CTRL+C to quit)
```
#### Preparing Google ADK Agent
Organize your agent files in the following structure:
```
name_of_your_agent/
├── __init__.py
└── agent.py
```
See example files in `examples\my_agent` to prepare `__init__.py` and `agent.py`.
#### Starting Google ADK
```bash
>>> adk web
INFO:     Started server process [25799]
INFO:     Waiting for application startup.

+-----------------------------------------------------------------------------+
| ADK Web Server started                                                      |
|                                                                             |
| For local testing, access at http://localhost:8000.                         |
+-----------------------------------------------------------------------------+

INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```
#### Accessing the agent
1. Open your browser and navigate to the provided ADK address.
2. Select the agent directory name you configured.
3. Interact with the LLM, which can now leverage ABACUS agent tools for computational tasks.

### Use ABACUS agent tools and Google ADK on remote server

After installing ABACUS agent tools and Google ADK on a remote server, use the exposed ports for configuration.

#### Example for Bohrium Nodes
```bash
# Start ABACUS agent tools with public host and port
abacusagent --host "0.0.0.0" --port 50001
# Start Google ADK with public host and port
adk web --host "0.0.0.0" --port 50002
```
#### Accessing Remotely
Visit http://your-node-address.dp.tech:50002 in your browser, where:

- `your-node-address.dp.tech` is the remote node URL
- `50002` is the configured port for Google ADK

## Supported functions
Functions of ABACUS Agent tools are in active development. Currently, the following functions are exposed as 
MCP tools and can be 
- Prepare ABACUS input files (INPUT, STRU, KPT, pseudopotential and orbital files)
- Modify INPUT and STRU
- Bader charge
- Density of states (DOS) and projected density of states (PDOS)
- Band calculation
