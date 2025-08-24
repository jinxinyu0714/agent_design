import os
from autogen_agentchat.agents import AssistantAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
os.environ['GEMINI_API_KEY'] = "AIzaSyA12-2shZJdz0BYLRs-7vHIjIlwuN3Xd3M"
from agents import create_design_analyst, get_model_client, create_report_saver, create_competitor_analyst
from agents import initialize_file_tools
import asyncio
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console


async def main():
    client = get_model_client()
    
    DesignAnalyst = await create_design_analyst(client)
    ReportSaver = await create_report_saver(client)
    CompetitorAnalyst = await create_competitor_analyst(client)

    # Apply message filtering
    filtered_analyst = MessageFilterAgent(
        name="ReportSaver",
        wrapped_agent=ReportSaver,
        filter=MessageFilterConfig(per_source=[PerSourceFilter(source="DesignAnalyst", position="last", count=1)]),
    )

    filtered_presenter = MessageFilterAgent(
        name="CompetitorAnalyst",
        wrapped_agent=CompetitorAnalyst,
        filter=MessageFilterConfig(per_source=[PerSourceFilter(source="DesignAnalyst", position="last", count=1)]),
    )


    # 创建工作流
    builder = DiGraphBuilder()
    builder.add_node(DesignAnalyst).add_node(filtered_analyst).add_node(filtered_presenter)
    # Fan-out from DesignAnalyst to ReportSaver and CompetitorAnalyst
    builder.add_edge(DesignAnalyst, filtered_analyst)
    builder.add_edge(DesignAnalyst, filtered_presenter)
    graph = builder.build()
    # 执行工作流
    flow = GraphFlow(
    participants=builder.get_participants(),
    graph=graph,
)


    await Console(flow.run_stream(task="我想要一个运动感的SUV，希望外观看起来比较有攻击性，价格在30-50万之间。完成中文竞品分析"))

if __name__ == "__main__":
    asyncio.run(main())