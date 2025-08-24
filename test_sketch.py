import os
from autogen_agentchat.agents import AssistantAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
os.environ['GEMINI_API_KEY'] = "AIzaSyA12-2shZJdz0BYLRs-7vHIjIlwuN3Xd3M"
from agents import get_model_client, create_seketching_agent
import asyncio
from autogen_agentchat.ui import Console


async def main():
    client = get_model_client()
    sketch_agent = await create_seketching_agent(client)
    await Console(sketch_agent.run_stream(task="创建一个关于['广汽传祺E8', '荣威iMAX8 DMH', '传祺M8']的草图"))

if __name__ == "__main__":
    asyncio.run(main())