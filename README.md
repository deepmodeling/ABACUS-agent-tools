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
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import MCPTool, MCPToolset
from google.adk.tools.mcp_tool.mcp_toolset import SseServerParams
#from dp.agent.adapter.adk import CalculationMCPToolset

os.environ["ABACUSAGENT_MODEL"] = "fastmcp"
from abacusagent.modules.abacus import *

# Provide LLM you want to use and API key here
os.environ['DEEPSEEK_API_KEY'] = ""
model = LiteLlm(model='deepseek/deepseek-chat')

instruction = "Provide your prompts to LLM here"

tools = [abacus_prepare, abacus_modify_input, 
         abacus_modify_stru, abacus_collect_data]

"""
abacus_agent_url = "https://127.0.0.1:50001/mcp"
toolset = MCPToolset(
    connection_params=SseServerParams(
        url=abacus_agent_url,
    ),
)
"""

root_agent = Agent(
    name="Abacus_agent",
    model=model,
    instruction=instruction,
    tools=tools
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

