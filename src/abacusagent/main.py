from pathlib import Path
import importlib
import os
import argparse

def load_tools():
    """
    Load all tools from the abacusagent package.
    """
    module_dir = Path(__file__).parent / "modules"
    
    for py_file in module_dir.glob("*.py"):
        if py_file.name.startswith("_") or py_file.stem in ["utils", "comm"]: 
            continue  # skipt __init__.py and utils.py
        
        module_name = f"abacusagent.modules.{py_file.stem}"
        try:
            module = importlib.import_module(module_name)
            print(f"✅ Successfully loaded: {module_name}")
        except Exception as e:
            print(f"⚠️ Failed to load {module_name}: {str(e)}")


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="AbacusAgent Command Line Interface")
    
    parser.add_argument(
        "--transport",
        type=str,
        default="sse",
        choices=["sse", "streamable-http"],
        help="Transport protocol to use (default: sse), choices: sse, streamable-http"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["fastmcp", "test", "dp"],
        help="Model to use (default: fastmcp), choices: fastmcp, test, dp"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the MCP server on (default: 50001)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to run the MCP server on (default: localhost)"
    )
    
    args = parser.parse_args()
    
    return args

def main():
    """
    Main function to run the MCP tool.
    """
    args = parse_args()  # Parse command line arguments
    
    #if args.model in ["dp"]:
    #    raise ValueError(f"Model '{args.model}' is not supported yet.")
    # Set environment variables based on arguments
    from abacusagent.env import set_envs
    set_envs(model_input=args.model,
             port_input=args.port, 
             host_input=args.host)

    from abacusagent.init_mcp import mcp
    load_tools()  
    mcp.run(transport=args.transport)

if __name__ == "__main__":
    main()
