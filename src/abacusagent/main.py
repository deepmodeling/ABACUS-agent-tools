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
        if py_file.name.startswith("_") or py_file.stem == "utils":
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
        "--model",
        type=str,
        default="fastmcp",
        help="Model to use (default: fastmcp)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50001,
        help="Port to run the MCP server on (default: 50001)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to run the MCP server on (default: localhost)"
    )
    
    args = parser.parse_args()
    
    return args

def set_environment_variables(args):
    # set model environment variable
    if os.environ.get("ABACUSAGENT_MODEL") is None:
        os.environ["ABACUSAGENT_MODEL"] = args.model
        print(f"ABACUSAGENT_MODEL is set to {os.environ['ABACUSAGENT_MODEL']}.")
    else:
        print(f"ABACUSAGENT_MODEL is already set to {os.environ['ABACUSAGENT_MODEL']}, using it.")
        
    if os.environ.get("ABACUSAGENT_PORT") is None:
        os.environ["ABACUSAGENT_PORT"] = str(args.port)
        print(f"ABACUSAGENT_PORT is set to {os.environ['ABACUSAGENT_PORT']}.")
    else:
        print(f"ABACUSAGENT_PORT is already set to {os.environ['ABACUSAGENT_PORT']}, using it.")
    
    if os.environ.get("ABACUSAGENT_HOST") is None:
        os.environ["ABACUSAGENT_HOST"] = args.host
        print(f"ABACUSAGENT_HOST is set to {os.environ['ABACUSAGENT_HOST']}.")
    else:
        print(f"ABACUSAGENT_HOST is already set to {os.environ['ABACUSAGENT_HOST']}, using it.")

def main():
    """
    Main function to run the MCP tool.
    """
    args = parse_args()  # Parse command line arguments
    set_environment_variables(args)  # Set environment variables based on arguments

    from abacusagent.init_mcp import mcp
    load_tools()  # Load all tools
    mcp.run(transport="streamable-http")

if __name__ == "__main__":
    main()
