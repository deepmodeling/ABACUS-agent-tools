import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from google.genai.types import GenerateContentConfig
from dp.agent.adapter.adk import CalculationMCPToolset

load_dotenv()
#Use deepseek
#Please set your DeepSeek API key to DEEPSEEK_API_KEY in your environment variable
#model = LiteLlm(model='deepseek/deepseek-chat')

# Use qwen series LLM. See https://help.aliyun.com/zh/model-studio/models to get avaliable qwen series LLM.
#model = LiteLlm(
#    model="openai/qwen-turbo",
#    api_key="",
#    base_url=""
#)

# Use Doubao series LLM. See https://www.volcengine.com/docs/82379/1330310 to get avaliable Doubao LLMs.
#model = LiteLlm(
#    model="openai/doubao-seed-1-6-250615",
#    api_key=os.getenv("ARK_API_KEY"),
#    base_url="https://ark.cn-beijing.volces.com/api/v3"
#)

# Use Hunyuan LLM. See https://hunyuan.tencent.com/modelSquare/home/list to get avaliable Hunyuan LLMs.
#model = LiteLlm(
#    model='openai/hunyuan-turbos-latest',
#    api_key=os.getenv("HUNYUAN_API_KEY"),
#    base_url="https://api.hunyuan.cloud.tencent.com/v1")

instruction = """You are an expert in materials science and computational chemistry. "
                "Help users perform ABACUS including single point calculation, structure optimization, molecular dynamics and property calculations. "
                "The website of ABACUS documentation is at https://abacus.deepmodeling.com/en/latest/, please read it if necessary." 
                "Use default parameters if the users do not mention, but let users confirm them before submission. "
                "Always prepare an directory containing ABACUS input files before use specific tool functions."
                "Always verify the input parameters to users and provide clear explanations of results."
                "Do not try to modify the input files without explicit permission when errors occured."
                "The LCAO basis is prefered."
                "If path to output files are provided, always tell the users the path to output files in the response."
"""

EXECUTORS = {
    'abacus': {
        "type": "dispatcher",
        "machine": {
            "batch_type": "Bohrium",
            "context_type": "Bohrium",
            "remote_profile": {
                "email": "",
                "password": "",
                "project_id": 123456,
                "input_data": {
                    "image_name": "registry.dp.tech/dptech/dp/native/prod-22618/abacusagenttools:v0.1-20250709-2", 
                    "job_type": "container",
                    "platform": "ali",
                    "scass_type": "c4_m8_cpu",
                },
            },
        },
    },
    "local": {
        "type": "local"
    }
}
STORAGE = {
    "type": "bohrium",
    "username": "",
    "password": "",
    "project_id": 123456,
}

EXECUTOR_MAP = {
    "generate_bulk_structure": EXECUTORS["local"],
    "abacus_prepare": EXECUTORS["local"],
    "abacus_modify_input": EXECUTORS["local"],
    "abacus_modify_stru": EXECUTORS["local"],
    "abacus_collect_data": EXECUTORS["local"],
    "abacus_prepare_inputs_from_relax_results": EXECUTORS["local"],
}

toolset = CalculationMCPToolset(
    connection_params=SseServerParams(
        url="http://localhost:50001/sse", # 订阅ABACUS-Agent-tools时提供的SSE，或者启动adk web的服务器IP和端口
        sse_read_timeout = 3600,
    ),
    executor = EXECUTORS['abacus'],
    executor_map = EXECUTOR_MAP,
    storage = STORAGE
)

print([toolset])

root_agent = Agent(
    name="Abacus_agent",
    model=model,
    instruction=instruction,
    tools=[toolset],
    generate_content_config=GenerateContentConfig(
        temperature=0.5,
        )
)
