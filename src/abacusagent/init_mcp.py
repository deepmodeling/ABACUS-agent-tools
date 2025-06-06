import os

port = os.environ.get("ABACUSAGENT_PORT", "50001")
host = os.environ.get("ABACUSAGENT_HOST", "0.0.0.0")
model = os.environ.get("ABACUSAGENT_MODEL", "fastmcp")

if model == "dp":
    from dp.agent.server import CalculationMCPServer
    mcp = CalculationMCPServer("ABACUSAGENT", port=port, host=host)
elif model == "fastmcp":
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("ABACUSAGENT", port=port, host=host)
elif model == "test": # For unit test of models
    class MCP:
        def tool(self):
            def decorator(func):
                return func
            return decorator
    mcp = MCP()
else:
    print("Please set the environment variable ABACUSAGENT_MODEL to dp, fastmcp or test.")
    raise ValueError("Invalid ABACUSAGENT_MODEL. Please set it to dp, fastmcp or test.")