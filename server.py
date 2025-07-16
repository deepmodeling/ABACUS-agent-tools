
import sys
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.abacusagent.main import load_tools
from src.abacusagent.env import set_envs

mcp = FastMCP("calculator", host="0.0.0.0", port=50001)

load_tools()

if __name__ == "__main__":
    set_envs()
    mcp.run(transport='sse')
