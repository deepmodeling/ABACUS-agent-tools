import os

if os.getenv("ABACUSAGENT_MODEL") == "dp":
    from dp.agent.server import CalculationMCPServer
    mcp = CalculationMCPServer("Demo", port=50001)
elif os.getenv("ABACUSAGENT_MODEL") == "fastmcp":
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("research", port=50001)
elif os.getenv("ABACUSAGENT_MODEL") == "test": # For unit test of models
    class MCP:
        def tool(self):
            def decorator(func):
                return func
            return decorator
    mcp = MCP()
else:
    print("Please set the environment variable ABACUSAGENT_MODEL to dp, fastmcp or test.")
    raise ValueError("Invalid ABACUSAGENT_MODEL. Please set it to dp, fastmcp or test.")