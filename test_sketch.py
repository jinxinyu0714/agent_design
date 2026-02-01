import os
import argparse
import dotenv
dotenv.load_dotenv()
from autogen_agentchat.agents import AssistantAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
from agents import get_model_client, create_seketching_agent, get_openai_model_client
import asyncio
from autogen_agentchat.ui import Console
import re
import ast

DEFAULT_TASK = "创建一个关于['奥迪A3']的草图"


def ensure_directory_permissions(path):
    """Ensure the directory exists and has correct permissions"""
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, mode=0o755, exist_ok=True)
    # Set directory permissions to allow writing
    os.chmod(directory, 0o755)


async def main(task):
    client = get_model_client()
    sketch_agent = await create_seketching_agent(client)
    output = ""
    async for msg in sketch_agent.run_stream(task=task):
        output += str(msg) + "\n"
    # 提取output_sketch_path
    match = re.search(r"'output_sketch_path':\s*'([^']+)'", output)
    if match:
        sketch_path = match.group(1)
        # Ensure proper permissions before writing
        ensure_directory_permissions(sketch_path)
        return sketch_path

    #return None
    print("/home/j/桌面/agent_design/clippasso_utils/CLIPasso/output_sketches/理想MEGA/理想MEGA_100strokes_seed0_best.png")

    return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sketch agent with a custom task.")
    parser.add_argument('--task', type=str, default=DEFAULT_TASK, help='Task description for the sketch agent')
    args = parser.parse_args()

    try:
        result = asyncio.run(main(args.task))
        print(result)
    except PermissionError as e:
        print(f"权限错误: {e}")
        print("请确保程序有足够的写入权限")
    except Exception as e:
        print(f"发生错误: {e}")