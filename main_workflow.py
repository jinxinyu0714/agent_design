import asyncio
import os
import sys
import argparse
import shutil
import random
import json
from pathlib import Path
from typing import List, Dict, Any

# Add current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from agents import create_intent_understander, get_model_client

# Import workflow components
from test_report import main as get_competitors
from test_sketch import main as generate_sketch_agent
from cd_utils.pipeline import predict_cd_value
from hunyuan3d_api.test_api import main as generate_3d_model
from render_utils.controlnet import generate_rendering
from clippasso_utils.clippasso_tool import run_clippasso_sketching

# --- Intent Classification ---
async def classify_intent(user_query: str, has_file: bool) -> Dict[str, Any]:
    """
    使用 IntentUnderstander 智能体对用户意图进行分类。
    
    Args:
        user_query: 用户的自然语言输入
        has_file: 是否提供了文件
        
    Returns:
        Dict: 包含 'intent' (意图代码), 'extracted_prompt' (提取的关键信息), 'reason' (推理理由) 的字典
    """
    client = get_model_client()
    agent = await create_intent_understander(client)
    
    context = f"Query: {user_query}\nHas File: {has_file}"
    try:
        # Run the agent with the context
        response = await agent.run(task=context)
        
        # Extract JSON from the last message
        content = response.messages[-1].content
        
        # Clean up markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
             content = content.split("```")[1].split("```")[0].strip()
             
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing intent: {e}")
        return None

# --- Workflow Execution Functions ---
async def execute_param_design_loop(task_description: str):
    """
    Case 1 & 5 (COMPETITOR_ANALYSIS / DESIGN_LOOP): 
    执行闭环设计工作流：竞品分析 -> 草图生成 -> 渲染图 -> 3D模型 -> CD预测。
    此函数处理纯文本需求。
    
    Args:
        task_description: 用户的设计需求描述 (例如: "设计一款运动型SUV")
    """
    print(f"Executing Full Design Loop for: {task_description}")
    
    # Existing logic from main_workflow.py adapted here
    if "SUV" in task_description.upper():
        cd_threshold = 0.4
    else:
        cd_threshold = 0.35
        
    print(f"目标 Cd <= {cd_threshold}")
    
    max_iterations = max_iterations
    iteration = 0
    
    excluded_models = []
    high_cd_dict = {}
    
    success = False  # 保证 success 总是有定义
    while iteration < max_iterations:
        iteration += 1
        print(f"\n=== 迭代 {iteration} ===")
        print(f"当前任务: {task_description}")
        
        # Step 1: Get Competitor Models
        print("步骤1：搜索竞品模型...")
        try:
            models = await get_competitors(task_description)
        except Exception as e:
            print(f"获取竞品模型出错: {e}")
            success = False
            break
        
        if not models:
            print("未找到模型。停止。")
            success = False
            break
        
        print(f"找到模型: {models}")
        
        # Filter out already excluded models
        candidates = [m for m in models if m not in excluded_models]
        
        if not candidates:
            print("所有找到的模型均已排除。调整搜索...")
            task_description += " find different models"
            continue
        
        # Process each candidate
        success = False
        for model in candidates:
            print(f"\n处理模型: {model}")
            
            # Step 2: Generate Sketch
            print("步骤2：生成草图...")
            sketch_task = f"Create a sketch for {model}"
            try:
                sketch_path = await generate_sketch(sketch_task)
            except Exception as e:
                print(f"生成草图出错: {e}")
                continue
                
            if not sketch_path or not os.path.exists(sketch_path):
                # Fallback if sketch generation fails or returns None (mocking behavior from test_sketch.py)
                print("草图生成返回为空或路径无效，使用备用示例路径（如可用）。")
                sketch_path = "/Volumes/HP P900/agent_design/clippasso_utils/CLIPasso/output_sketches/理想L7/理想L7_100strokes_seed0_best.png"
            
            print(f"草图路径: {sketch_path}")

            # Step 3: Generate Rendering
            print("步骤3：生成渲染...")
            # user inputs could be added here if needed
            prompt = input("请输入渲染提示语（或直接回车使用默认）: ")
            if prompt:
                rendering_path = generate_rendering(sketch_path, prompt=prompt)
            else:
                rendering_path = generate_rendering(sketch_path)

            print(f"渲染路径: {rendering_path}")
            
            # Step 4: Generate 3D Model
            print("步骤4：生成3D模型...")
            model_path = generate_3d_model(rendering_path)

            stl_path = model_path["stl_path"]
            vtk_path = model_path["vtk_path"]

            
            # Step 5: Predict Cd
            print("步骤5：预测Cd...")
            # Use a random seed to get some variation if the mock STL is the same every time
            # (Though predict_cd_value might be deterministic depending on implementation)
            cd_summary = predict_cd_value([stl_path], random_seed=random.randint(1, 1000))
            
            if not cd_summary['individual_results']:
                print("Cd 预测失败。")
                continue
                
            predicted_cd = cd_summary['individual_results'][0]['predicted_value']
            print(f"预测{model}的 Cd 为 {predicted_cd}")
            
            # Step 6: Evaluate
            if predicted_cd > cd_threshold:
                print(f"Cd {predicted_cd} > {cd_threshold}。超过阈值。")
                print(f"排除模型 {model} 并细化搜索。")
                high_cd_dict[model] = predicted_cd
                excluded_models.append(model)
                # Update task to exclude this model and emphasize low drag
                task_description += f", exclude {model}, with their predict cd value {high_cd_dict}, find lower drag coefficient models"
                # Break inner loop to restart search with new constraints
                break 
            else:
                print(f"Cd {predicted_cd} <= {cd_threshold}。成功！")
                print(f"最终选定模型: {model}")
                print(f"最终3D模型: {stl_path}")
                success = True
                break
        
        if success:
            break
            
    if not success:
        print("达到最大迭代次数或未找到合适模型。")

if __name__ == "__main__":
    asyncio.run(main())
