from pathlib import Path
from typing import Union, Optional
import os, sys

def create_google_adk_template(path: Optional[str | Path] = "."):
    tpath = os.path.join(path, "abacus-agent")
    os.makedirs(tpath, exist_ok=True)
    current_file_path = Path(__file__).parent / "google-adk-agent-template.py"
    
    # copy the template file to the target directory    
    os.system(f"cp {current_file_path} {tpath}/agent.py")
    with open(os.path.join(tpath, "__init__.py"), "w") as file:
        file.write("from . import agent")