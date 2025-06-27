import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_toolset import SseServerParams
from dp.agent.adapter.adk import CalculationMCPToolset

# Set environments about LLM you want to use and API key here. 
os.environ['DEEPSEEK_API_KEY'] = ""
model = LiteLlm(model='deepseek/deepseek-chat')

instruction = "Provide your prompts to LLM here"

# Specify the URL and port for connecting to the SSE server running the ABACUS agent.
toolset = CalculationMCPToolset(
    connection_params=SseServerParams(
        url="http://localhost:50001/sse",
    )
)

root_agent = Agent(
    name="Abacus_agent",
    model=model,
    instruction=instruction,
    tools=[toolset]
)
