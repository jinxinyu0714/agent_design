"""
汽车设计分析智能体定义
"""
import os
import asyncio
import requests
from pathlib import Path
from typing import List, Dict, Any 
from config import config
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams, mcp_server_tools
from utils.cli_research import web_deep_research
from utils.fetch_webpage import fetch_webpage_tool
from utils.google_search import google_search_tool
from clippasso_utils.clippasso_tool import run_clippasso_sketching_sync
from cd_utils.pipeline import evaluate_cd_value

from prompt import (analyst_prompt, file_saver_prompt, get_car_list_prompt, sketch_gen_prompt, cd_value_prompt)


async def get_deep_research_tool():
    """获取深度研究工具"""
    return FunctionTool(
        web_deep_research,  # 确保这是同步版本的函数
        description="""Perform a deep web research on the given question using the LangGraph.
        Args:
            question (str): The question to research.
        Returns:
            str: The final answer after research.
        """,
    )

async def get_clippasso_tool():
    """获取CLIPasso素描工具"""
    return FunctionTool(
        run_clippasso_sketching_sync,
        description="""Run CLIPasso sketching on the given image.
        Args:
            image_name (str): The name of the target image file (e.g., "camel.png").
            num_strokes (int): Number of strokes for sketching, controlling abstraction level (default 100).
            mask_object (Optional[int]): Whether to mask the background, enabled when value is 1 (suitable for images with background).
            fix_scale (Optional[int]): Whether to fix the image scale, enabled when value is 1 (suitable for non-square images).
            num_sketches (int): Number of sketches generated in parallel, default is 1 (recommended value).
        Returns:
            Dict[str, Any]: Dictionary containing execution results.
        """,
    )

async def get_cd_value_tool():
    """获取CD值计算工具"""
    return FunctionTool(
        evaluate_cd_value,  # 使用简化版函数
        description="""Evaluate the CD value of a list of STL files.
        Args:
            file_list: List of STL file names to evaluate (e.g., ["car1.stl", "car2.stl"]).
        Returns:
            Dict[str, Any]: Dictionary containing the evaluation results including individual file results and summary statistics.
        """,
    )


def get_model_client():
    """获取模型客户端"""
    return OpenAIChatCompletionClient(
        model=config.MODEL_NAME,
        api_key=config.MODEL_API_KEY,
        base_url=config.MODEL_BASE_URL,
        model_info={
            "json_output": False,
            "function_calling": True,
            "vision": False,
            "family": "unknown",
            "structured_output": False,
        },
    )

# def get_model_client():
#     """获取模型客户端"""
#     return OpenAIChatCompletionClient(
#         model=config.OPENAI_MODEL_NAME,
#         api_key=config.OPENAI_API_KEY,
#     )



async def create_design_analyst(client):
    """创建汽车设计需求分析智能体"""
    deep_research_tool = await get_deep_research_tool()
    return AssistantAgent(
        name="DesignAnalyst",
        description="汽车设计需求分析专家，您的目标是根据用户输入的模糊设计需求实现竞品分析，生成一系列复杂多样的网络搜索查询。这些查询旨在用于高级自动化网络研究工具，该工具能够分析复杂结果、跟踪链接和综合信息。",
        model_client=client,
        tools=[deep_research_tool],
        system_message=analyst_prompt,
    )


async def create_report_saver(client):
    """创建报告保存智能体"""
    file_mcp = await initialize_file_tools()
    return AssistantAgent(
        name="ReportSaver",
        description="汽车设计报告保存专家，负责将生成的设计报告保存到指定位置",
        model_client=client,
        tools=file_mcp,
        system_message=file_saver_prompt,
    )

async def create_competitor_analyst(client):
    """创建竞品分析智能体"""
    
    return AssistantAgent(
        name="CompetitorAnalyst",
        description="竞品分析专家，负责根据用户输入的竞品信息进行分析和总结",
        model_client=client,
        tools= [],
        system_message=get_car_list_prompt,
    )

async def create_seketching_agent(client):
    """创建CLIPasso素描生成智能体"""
    clippasso_tool = await get_clippasso_tool()
    sketch_file_mcp = await initialize_sketch_file_tools()
    return AssistantAgent(
        name="CLIPassoSketcher",
        description="CLIPasso素描生成专家，负责根据用户提供的图片生成素描",
        model_client=client,
        tools=[clippasso_tool] + sketch_file_mcp,
        system_message=sketch_gen_prompt,
        max_tool_iterations=20,
    )

async def create_cd_value_agent(client):
    """创建CD值计算智能体"""
    cd_value_tool = await get_cd_value_tool()
    return AssistantAgent(
        name="CDValueCalculator",
        description="CD值计算专家，负责根据用户提供的STL文件列表计算CD值",
        model_client=client,
        tools=[cd_value_tool],
        system_message=cd_value_prompt,
        max_tool_iterations=20,
    )





# def _get_web_search_params() -> StdioServerParams:
#     """获取网络搜索MCP服务器参数"""
#     # 注意：实际的包名可能不同，这里使用一个通用的搜索实现
#     return StdioServerParams(
#         command="node",
#         args=["/Users/jinxinyu/Desktop/资料/easy_agent/analy/web-search/build/index.js"],
#         read_timeout_seconds=60,
#     )

def _get_file_system_params() -> StdioServerParams:
    """获取文件系统MCP服务器参数"""
    return StdioServerParams(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/student/jxy/agent_design/automotive_output"  # 修正路径
        ],
        read_timeout_seconds=30,
    )

def _get_sketch_file_system_params() -> StdioServerParams:
    """获取文件系统MCP服务器参数"""
    # 使用当前工作目录，而不是输出目录
    current_dir = Path.cwd()
    return StdioServerParams(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/student/jxy/agent_design/clippasso_utils/CLIPasso/target_images",
            "/student/jxy/agent_design/clippasso_utils/CLIPasso",
        ],
        read_timeout_seconds=30,
    )
async def initialize_file_tools():
    file_system_params = _get_file_system_params()
    file_server_tools = await mcp_server_tools(file_system_params)
    return file_server_tools

async def initialize_sketch_file_tools():
    sketch_file_system_params = _get_sketch_file_system_params()
    sketch_file_server_tools = await mcp_server_tools(sketch_file_system_params)
    return sketch_file_server_tools

def _get_img_save_params() -> StdioServerParams:
    """获取图片保存MCP服务器参数"""
    output_dir = config.get_images_dir()

    return StdioServerParams(
        command="npx",
        args=["mcp-image-downloader"],
        env={
            "DEFAULT_SAVE_PATH": str(output_dir),
            "DEFAULT_FORMAT": "original",
            "DEFAULT_COMPRESS": "false",
            "DEFAULT_CONCURRENCY": "3",
        },
        read_timeout_seconds=30,
    )

# async def initialize_web_tools():
#     web_search_params = _get_web_search_params()
#     web_server_tools = await mcp_server_tools(web_search_params)
#     return web_server_tools

async def initialize_image_tools():
    """初始化图片工具，带有超时处理"""
    img_save_params = _get_img_save_params()
    img_server_tools = await mcp_server_tools(img_save_params)
    return img_server_tools

