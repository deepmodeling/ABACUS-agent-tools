from pathlib import Path
import importlib
import os

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


def main():
    """
    Main function to run the MCP tool.
    """
    os.environ["ABACUSAGENT_MODEL"] =  "dp"  # Set the model to dp
    
    from abacusagent.init_mcp import mcp
    load_tools()  # Load all tools
    mcp.run(transport="streamable-http")

if __name__ == "__main__":
    main()
