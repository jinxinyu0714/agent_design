import asyncio
import argparse
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
import os
import re
os.environ['GEMINI_API_KEY'] = "AIzaSyA12-2shZJdz0BYLRs-7vHIjIlwuN3Xd3M"
from agents import create_design_analyst, get_model_client, create_report_saver, create_competitor_analyst, create_seketching_agent, create_cd_value_agent
from agents import initialize_sketch_file_tools, _get_sketch_file_system_params, get_clippasso_tool
from cd_utils.pipeline import predict_cd_value, print_multiple_prediction_results
from autogen_ext.tools.mcp import McpWorkbench
import ast
import numpy as np
import random


DEFAULT_TASK = ["/home/j/桌面/agent_design/113b3f4198658771b1ccc3d8efa5a23f.stl"]
# DEFAULT_TASK = ["/home/j/桌面/agent_design/cd_utils/data/stl_in/E_S_WW_WM_001.stl", "/home/j/桌面/agent_design/cd_utils/data/stl_in/E_S_WW_WM_002.stl", "/home/j/桌面/agent_design/0a96a34f61d9f9fd86ed358be624eda6.stl", "/home/j/桌面/agent_design/hunyuan3d_api/history/10-16-16.00/113b3f4198658771b1ccc3d8efa5a23f.stl"]

async def main(task):
    a = random.randint(1, 10000)
    summary = predict_cd_value(file_list=args.task, random_seed=a)
    print_multiple_prediction_results(summary)
    # client = get_model_client()
    # agent = await create_cd_value_agent(client)
    # output = ""
    # async for msg in agent.run_stream(task=task):
    #     output += str(msg) + "\n"
    # # 提取markdown部分，从第一个#开始，到“请随时告诉我”前为止
    # match = re.search(r"(#.*?)(?:请随时告诉我|如果您还有其他问题|如有其他需求)[^\n]*", output, re.DOTALL)
    # markdown_result = match.group(1).strip() if match else ""
    # #print(markdown_result)
    # return markdown_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CD value agent with a custom task.")
    parser.add_argument('--task', type=ast.literal_eval, default=DEFAULT_TASK,
                    help='Task description for the CD value agent (as Python list string)')
    args = parser.parse_args()
    asyncio.run(main(args))