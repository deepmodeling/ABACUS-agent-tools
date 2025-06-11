# ABACUS Agent Tools

ABACUS-agent-tools is a Python package that provides the Model Context Protocol (MCP) tools to connect large language models (LLMs) to ABACUS computational jobs. It serves as a bridge between AI models and first principles calculations, enabling intelligent interaction with ABACUS workflows.

## Installation
To use ABACUS agent tools with Google Agent Development Kit (ADK), follow the recommended installation process:

1. Create and activate a conda enviroment:
```bash
conda create -n abacus-agent python=3.11
conda activate abacus-agent
```
2. Install necessary dependencies:
```bash
pip install mcp google-adk litellm science-agent-sdk
```
3. Install abacustest
```bash
git clone -b develop https://github.com/pxlxingliang/abacus-test.git
cd abacus-test
pip install .
```
4. Install ABACUS-agent-tools:
```bash
cd ..
git clone -b develop https://github.com/pxlxingliang/ABACUS-agent-tools.git
cd abacus-agent
pip install .
```

## Using ABACUS agent tools with Google ADK

### Use ABACUS agent tools and Google ADK on local machine

#### Starting ABACUS agent tools
Before launching `abacusagent`, you must provide the necessary configurations in the `~/.abacusagent/env.json` file. This file defines how the ABACUS agent tools generate input files and manage ABACUS calculation workflows.
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
##### Example `__init__.py`
```python
from . import agent
```
##### Example `agent.py`
```python
import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_toolset import SseServerParams
from dp.agent.adapter.adk import CalculationMCPToolset

# Set environments about LLM you want to use and API key here. 
os.environ['DEEPSEEK_API_KEY'] = ""
model = LiteLlm(model='deepseek/deepseek-chat')

instruction = "Provide your prompts to LLM here"

# Specify the URL and port for connecting to the SSE server running the ABACUS agent.
toolset = CalculationMCPToolset(
    connection_params=SseServerParams(
        url="http://localhost:50001/sse",
    )
)

root_agent = Agent(
    name="Abacus_agent",
    model=model,
    instruction=instruction,
    tools=[toolset]
)
```
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

