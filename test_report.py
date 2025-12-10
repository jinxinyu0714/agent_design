import os
import argparse
from autogen_agentchat.agents import AssistantAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
from agents import create_design_analyst, get_model_client, get_openai_model_client, create_report_saver, create_competitor_analyst
from agents import initialize_file_tools
import asyncio
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
import re
import requests
import json



# 使用--task参数传入任务描述
# python test_report.py --task "运动感suv"
def get_image_url(content):
    try:
        url = "https://qianfan.baidubce.com/v2/ai_search/web_search"
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "edition": "standard",
            "search_source": "baidu_search_v2",
            "resource_type_filter": [
                {
                    "type": "web",
                    "top_k": 0
                },
                {
                    "type": "video",
                    "top_k": 0
                },
                {
                    "type": "image",
                    "top_k": 0
                },
                {
                    "type": "aladdin",
                    "top_k": 1
                }
            ],
            "search_recency_filter": "week"
        }, ensure_ascii=False)
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer bce-v3/ALTAK-NwUMimwToaztlQLv3VKkY/0e773a1be58c8be63382897534f8668bfdb66225'
        }
        response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))
        result = json.loads(response.text)
        url = result["references"][0]["aladdin"]["img_list"][0]
        # 去除调url中@后面的内容，包括@
        url = re.sub(r'@.*$', '', url)
        
        return url
    except Exception as e:
        
        return None

def save_image(img_url, content, folder_name):
    try:
        # 发出图片请求
        img_response = requests.get(img_url)
        
        # 如果响应状态码是200，表示图片下载成功
        # image_name保留.文件后缀后用content作为文件名
        if img_response.status_code == 200:
            img_name = f"{content}{os.path.splitext(img_url)[-1]}"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            img_path = os.path.join(folder_name, img_name)
            
            # 保存图片
            with open(img_path, 'wb') as file:
                file.write(img_response.content)
    except Exception as e:
        print(f"Error downloading {img_url}: {e}")

def save_all_images(content, folder_name):
    valid_content = []
    for c in content:
        # c 中的空格替换为下划线
        c = c.replace(" ", "_")
        img_url = get_image_url(c)
        if img_url:
            save_image(img_url, c, folder_name)
            valid_content.append(c)
        # 不再直接删除元素
    return valid_content

async def main(task):
    for i in range(3):
        client = get_openai_model_client()

        DesignAnalyst = await create_design_analyst(client)
        ReportSaver = await create_report_saver(client)
        CompetitorAnalyst = await create_competitor_analyst(client)

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

        builder = DiGraphBuilder()
        builder.add_node(DesignAnalyst).add_node(filtered_analyst).add_node(filtered_presenter)
        builder.add_edge(DesignAnalyst, filtered_analyst)
        builder.add_edge(DesignAnalyst, filtered_presenter)
        graph = builder.build()
        flow = GraphFlow(
            participants=builder.get_participants(),
            graph=graph,
        )
        # await Console(flow.run_stream(task=task))
        # 收集所有输出
        output = ""
        async for msg in flow.run_stream(task=task):
            output += str(msg) + "\n"

        # 用正则提取车型列表
        match = re.search(r'"extracted_vehicle_models"\s*:\s*\[(.*?)\]', output, re.DOTALL)
        if match:
            models_str = match.group(1)
            # 提取每个车型名
            models = re.findall(r'"([^"]+)"', models_str)
        else:
            models = []

        models = save_all_images(models, "/home/j/桌面/agent_design/clippasso_utils/CLIPasso/target_images")
        if models:
            print("车型列表：", models)
            return models
        elif i == 2:
            models = ["瑞虎7"]
            print("车型列表：", models)
            return models
        else:
            # sleep 5分钟后重试
            await asyncio.sleep(300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agent workflow with a custom task.")
    parser.add_argument('--task', type=str, required=True, help='Task description for the workflow')
    args = parser.parse_args()
    asyncio.run(main(args.task))