import argparse
from langchain_core.messages import HumanMessage
from utils.agent.graph import graph
import asyncio
from langchain_core.runnables import RunnableConfig

async def web_deep_research_async(question: str):
    """异步执行深度网络研究"""
    state = {
        "messages": [HumanMessage(content=question)],
        "initial_search_query_count": 10,
        "max_research_loops": 5,
        "reasoning_model": "gpt-4o",
        "provider": "openai", 
    }

    config = RunnableConfig(
        configurable={
            "provider": "openai",
            "reasoning_model": "gpt-4o",
        }
    )
    
    try:
        # 使用异步调用 graph
        result = await graph.ainvoke(state, config)
        messages = result.get("messages", [])
        if messages:
            return messages[-1].content
        return "No research results found."
    except Exception as e:
        return f"Research failed: {str(e)}"

def web_deep_research(question: str):
    """同步包装器 - 这是给工具调用的"""
    def run_in_new_loop():
        """在新的事件循环中运行异步函数"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(web_deep_research_async(question))
        finally:
            loop.close()
    
    try:
        # 检查当前是否有运行的事件循环
        try:
            current_loop = asyncio.get_running_loop()
            # 如果有运行的事件循环，使用线程池执行
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_new_loop)
                return future.result(timeout=300)  # 5分钟超时
        except RuntimeError:
            # 没有运行的事件循环，直接运行
            return asyncio.run(web_deep_research_async(question))
    except Exception as e:
        return f"Error in web_deep_research: {str(e)}"
