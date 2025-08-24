import asyncio
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
import os
os.environ['GEMINI_API_KEY'] = "AIzaSyA12-2shZJdz0BYLRs-7vHIjIlwuN3Xd3M"
from agents import create_design_analyst, get_model_client, create_report_saver, create_competitor_analyst, create_seketching_agent, create_cd_value_agent
from agents import initialize_sketch_file_tools, _get_sketch_file_system_params, get_clippasso_tool
from autogen_ext.tools.mcp import McpWorkbench

async def main():
    
    # Initialize the model client
    client = get_model_client()

    # Create assistant agent
    agent = await create_cd_value_agent(client)

    # Specify the car models to search for
    task = "计算以下STL文件的CD值：E_S_WW_WM_001.stl, E_S_WW_WM_002.stl"
    task2 = "列出你的工具"
    
    # Run the agent with the specified task
    result = await Console(agent.run_stream(task=task))
if __name__ == "__main__":
    
    asyncio.run(main())
 