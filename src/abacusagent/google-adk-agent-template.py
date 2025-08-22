from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from dp.agent.adapter.adk import CalculationMCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams, StreamableHTTPServerParams

import os, json

# Set the secret key in ~/.abacusagent/env.json or as an environment variable, or modify the code to set it directly.
env_file = os.path.expanduser("~/.abacusagent/env.json")
if os.path.isfile(env_file):
    env = json.load(open(env_file, "r"))
else:
    env = {}
model_name = env.get("LLM_MODEL", os.environ.get("LLM_MODEL", ""))
model_api_key = env.get("LLM_API_KEY", os.environ.get("LLM_API_KEY", ""))
model_base_url = env.get("LLM_BASE_URL", os.environ.get("LLM_BASE_URL", ""))
bohrium_username = env.get("BOHRIUM_USERNAME", os.environ.get("BOHRIUM_USERNAME", ""))
bohrium_password = env.get("BOHRIUM_PASSWORD", os.environ.get("BOHRIUM_PASSWORD", ""))
bohrium_project_id = env.get("BOHRIUM_PROJECT_ID", os.environ.get("BOHRIUM_PROJECT_ID", ""))

instruction = """You are an expert in materials science and computational chemistry. "
                "Help users perform ABACUS including single point calculation, structure optimization, molecular dynamics and property calculations. "
                
                abacus_prepare 能够将结构文件转换为 ABACUS 计算所需的全套输入文件，包括 INPUT、STRU 文件以及对应的赝势和轨道文件。

                在使用该工具的大部分函数时，请注意将生成的 ABACUS 输入文件所在路径传递给 abacus_inputs_dir 参数。

                对于部分工具函数，它们会根据计算结果生成新的 ABACUS 输入文件，此时这些函数的返回参数需命名为 new_abacus_inputs_dir 或 modified_abacus_inputs_dir。需要注意的是，这两个返回参数也可以作为 abacus_inputs_dir 参数的输入值使用。

                请确保在调用相关函数时遵循上述参数传递规则，以保证 ABACUS 计算的正确执行和输入文件的有效管理。
                
                Here we briefly introduce functions of avaliable tool functions and suggested use method below:

                ABACUS input files generation:
                - abacus_prepare: Prepare ABACUS input file directory from structure file and provided information.
                    Should only be used when a structure file is avaliable (in cif, poscar or abacus/stru format)
                    and generating ABACUS input file directory is explicity requested,
                - abacus_modify_input: Modify ABACUS INPUT file in prepared ABACUS input file directory.
                    Should only be used when abacus_prepare is finished or path to a prepared ABACUS input file directory is explicitly given.
                - abacus_modify_stru: Modify ABACUS STRU file in prepared ABACUS input file directory.
                    Should only be used when abacus_prepare is finished or path to a prepared ABACUS input file directory is explicitly given.
                
                Result collection;
                - abacus_collect_data: Collect data from finished ABACUS job directory. **Should only be used** after an ABACUS job is finished.
                
                Property calculation:
                - abacus_do_relax: Do relax (only relax the position of atoms in a cell) or cell-relax (relax the position of atoms and lattice parameters simutaneously)
                    for a given structure. abacus_phonon_dispersiton should only be used after using this function to do a cell-relax calculation,
                    and abacus_vibrational_analysis should only be used after using this function to do a cell-relax calculation. 
                    This function will give a new ABACUS input file directory containing the relaxed structure in STRU file, and keep input parameters in
                    original ABACUS input directory. Calculating properties should use the new directory.
                    It is not necessary but strongly suggested using this tool function before calculating other properties like band, 
                    Bader charge, DOS/PDOS and elastic properties 
                - abacus_prepare_inputs_from_relax_results: This function will collect new ABACUS input file directory containing
                    relaxed structure. Since abacus_do_relax has used this function and returned the path to new ABACUS input directory,
                    this function has limited usage in current example and suggested not to use proactively.
                - abacus_badercharge_run: Calculate the Bader charge of given structure.
                - abacus_cal_band: Calculate the electronic band of given structure. Support two modes: `nscf` mode, do a nscf calculation
                    after a scf calculation as normally done; `pyatb` mode, use PYATB to plot the band after a scf run. The default is PYATB.
                    Currently 2D material is not supported.
                - abacus_cal_elf: Calculate the electroic localization function of given system and return a cube file containing ELF.
                - abacus_cal_charge_density_difference: Calculate the charge density difference of a given system divided into to subsystems.
                    Atom indices should be explicitly requested if not certain.
                - abacus_cal_spin_density: Calculate the spin density of given  structure. A cube file containing the spin density will be returned.
                - abacus_dos_run: Calculate the DOS and PDOS of the given structure. Support non-magnetic and collinear spin-polarized now. 
                    Support 3 modes to plot PDOS: 1. Plot PDOS for each element; 2. Plot PDOS for each shell of each element (d orbital for Pd for example),
                    3. Plot PDOS for each orbital of each element (p_x, p_y and p_z for O for example). Path to plotted DOS and PDOS will be returned.
                - abacus_cal_elastic: Calculate elastic tensor (in Voigt notation) and related bulk modulus, shear modulus and young's modulus and
                    Poisson ratio from elastic tensor.
                - abacus_eos: Fit Birch-Murnaghan equation of state for cubic crystal. This function should only be used for cubic crystal.
                - abacus_phonon_dispersion: Calculate phonon dispersion curve for bulk material. Currently 2D material is not supported.
                    Should only be used after using abacus_do_relax to do a cell-relax calculation is finished.
                - abacus_vibrational_analysis: Do vibrational analysis using finite-difference method. Should only be used after using abacus_do_relax
                    to do a relax calculation is finished. Indices of atoms considerer should be explicitly requested if not certain.
                - abacus_run_md: Run ab-inito molecule dynamics calculation using ABACUS.

                A typical workflow is: 
                1. Using abacus_prepare to generate ABACUS input file directory;
                2. (Optional) using abacus_modify_input and abacus_modify_stru to modify INPUT and STRU file in given ABACUS input file directory,
                3. Using abacus_do_relax to do a cell-relax calculation for given material,
                4. Do property calculations like phonon dispersion, band, etc."
"""

executor = {
    "bohr": {
        "type": "dispatcher",
        "machine": {
            "batch_type": "Bohrium",
            "context_type": "Bohrium",
            "remote_profile": {
                "email": bohrium_username,
                "password": bohrium_password,
                "program_id": bohrium_project_id,
                "input_data": {
                    "image_name": "registry.dp.tech/dptech/dp/native/prod-22618/abacus-agent-tools:v0.0.3-20250814",
                    "job_type": "container",
                    "platform": "ali",
                    "scass_type": "c32_m64_cpu",
                },
            },
        }
    },
    "local": {"type": "local",}
}

EXECUTOR_MAP = {
    "generate_bulk_structure": executor["local"],
    "generate_molecule_structure": executor["local"],
    "abacus_prepare": executor["local"],
    "abacus_modify_input": executor["local"],
    "abacus_modify_stru": executor["local"],
    "abacus_collect_data": executor["local"],
    "abacus_prepare_inputs_from_relax_results": executor["local"],
    "generate_bulk_structure_from_wyckoff_position": executor["local"],
}

STORAGE = {
    "type": "https",
    "plugin":{
        "type": "bohrium",
        "username": bohrium_username,
        "password": bohrium_password,
        "project_id": bohrium_project_id,
    }
}

toolset = CalculationMCPToolset(
    connection_params=SseServerParams(
        url="http://localhost:50001/sse", # Or any other MCP server URL
        sse_read_timeout=3000,  # Set SSE timeout to 3000 seconds
    ),
    executor_map = EXECUTOR_MAP,
    executor=executor["local"],
    storage=STORAGE,
)

root_agent = Agent(
    name='agent',
    model=LiteLlm(
        model=model_name,
        base_url=model_base_url,
        api_key=model_api_key
    ),
    description=(
        "Do ABACUS calculations."
    ),
    instruction=instruction,
    tools=[toolset]
)