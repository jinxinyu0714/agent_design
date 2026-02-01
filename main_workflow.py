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
    
    # Step 1: Competitor Analysis
    print("\n=== 步骤1：竞品分析 ===")
    try:
        # We call the competitor agent logic
        # Assuming get_competitors prints the report or saves it
        models = await get_competitors(task_description)
    except Exception as e:
        print(f"竞品分析出错: {e}")
        return
    # For the loop, we just take one model to proceed with the flow as an example
    model = models[0] if models else "Generic Car"
    print(f"基于分析结果，生成车型: {model}")
    print("\n=== 步骤2：生成草图 ===")
    sketch_task = f"Create a sketch for {model}"
    try:
        # generate_sketch_agent (test_sketch.main) returns a path
        sketch_path = await generate_sketch_agent(sketch_task)
    except Exception as e:
        print(f"草图生成出错: {e}")
        # Fallback
        sketch_path = "/home/j/桌面/agent_design/clippasso_utils/CLIPasso/output_sketches/理想MEGA/理想MEGA_100strokes_seed0_best.png"

    if not sketch_path or not os.path.exists(sketch_path):
         print("草图路径无效，无法继续。")
         return
         
    # Continue with Sketch -> CD flow
    # 手动输入prompt关键词
    prompt_keywords = input("请输入设计风格关键词（用于渲染），或直接回车使用默认: ").strip()
    await execute_sketch_to_cd(sketch_path, prompt_keywords=prompt_keywords, is_sketch_input=True)

async def execute_sketch_to_cd(sketch_path: str, prompt_keywords: str = "", is_sketch_input: bool = True):
    """
    Case 3 (SKETCH_TO_CD): 
    草图 -> 渲染图 -> 3D模型 -> CD预测。
    使用 ControlNet 将草图根据提示词渲染为逼真的汽车图像。
    
    Args:
        sketch_path: 草图文件路径
        prompt_keywords: 用于渲染的提示词
        is_sketch_input: 是否直接是用户输入的草图 (用于控制日志输出)
    """
    if is_sketch_input:
        print(f"\n=== 执行: 草图转CD预测 (SKETCH_TO_CD) ===")
    
    # 3. Sketch -> Render
    print("步骤2：生成渲染图 (ControlNet)...")
    prompt = prompt_keywords if prompt_keywords else "A futuristic car, high quality, realistic"
    try:
        render_path = generate_rendering(str(sketch_path), prompt=prompt)
        print(f"渲染图已生成: {render_path}")
        
        await execute_render_to_cd(render_path)
    except Exception as e:
        print(f"渲染生成出错: {e}")

async def execute_photo_to_cd(file_path: str, prompt_keywords: str):
    """
    Case 2 (PHOTO_TO_CD): 
    实车图/照片 -> 草图 -> 渲染图 -> 3D模型 -> CD预测。
    先使用 CLIPasso 将照片转换为线条草图。
    
    Args:
        file_path: 输入图片的路径
        prompt_keywords: 提取的设计风格关键词
    """
    print(f"\n=== 执行: 实车图转CD预测 (PHOTO_TO_CD) ===")
    
    # 1. Prepare Image
    target_dir = config.CLIPPASSO_TARGET_IMAGES_DIR
    if not target_dir.exists():
        target_dir.mkdir(parents=True)
        
    filename = os.path.basename(file_path)
    # Ensure filename is unique or safe
    dest_path = target_dir / filename
    if os.path.abspath(file_path) != os.path.abspath(dest_path):
        shutil.copy(file_path, dest_path)
    print(f"图片已准备: {dest_path}")
    
    # 2. Run CLIPasso
    print("步骤1：生成草图 (CLIPasso)...")
    
    result = await run_clippasso_sketching(
        image_name=filename,
        workspace_path=str(config.CLIPPASSO_DIR),
        num_sketches=1
    )
    
    sketch_path = result.get("output_sketch_path")
    if not sketch_path or not os.path.exists(sketch_path):
        print(f"Sketch generation failed: {result}")
        return
        
    print(f"草图已生成: {sketch_path}")
    
    # Proceed to next steps
    await execute_sketch_to_cd(sketch_path, prompt_keywords, is_sketch_input=False)

async def execute_render_to_cd(render_path: str):
    """
    Case 4 (RENDER_TO_CD): 
    渲染图 -> 3D模型 -> CD预测。
    使用 Hunyuan3D 将单张渲染图转换为 3D 网格模型 (.stl)。
    
    Args:
        render_path: 渲染图/效果图路径
    """
    print(f"\n=== 执行: 渲染图转CD预测 (RENDER_TO_CD) ===") 
    
    # 4. Render -> 3D
    print("步骤3：生成3D模型 (Hunyuan3D)...")
    try:
        model_paths = generate_3d_model(str(render_path))
        
        if not model_paths:
            print("3D模型生成失败。")
            return

        stl_path = model_paths["stl_path"]
        print(f"3D模型已生成: {stl_path}")
        
        execute_model_to_cd(stl_path)
    except Exception as e:
        print(f"3D模型生成出错: {e}")

def execute_model_to_cd(model_path: str):
    """
    Case 5 (MODEL_TO_CD): 
    3D模型 -> CD预测。
    使用深度学习模型预测 3D 文件的风阻系数 (Cd)。
    
    Args:
        model_path: 3D模型文件路径 (.stl)
    """
    print(f"\n=== 执行: 3D模型CD预测 (MODEL_TO_CD) ===")
    
    # 5. Predict Cd
    print("步骤4：预测CD值...")
    try:
        # Validate file exist
        if not os.path.exists(model_path):
            print(f"错误: 3D模型文件不存在: {model_path}")
            return
            
        result = predict_cd_value([model_path])
        print("\n=== CD值预测结果 ===")
        # Pretty print the important parts
        if 'summary_statistics' in result:
             print("Summary:", json.dumps(result['summary_statistics'], indent=2, ensure_ascii=False))
        if 'individual_results' in result:
             for res in result['individual_results']:
                 print(f"File: {res['file_name']}, CD: {res['predicted_value']}")
    except Exception as e:
        print(f"CD预测出错: {e}")


# --- Main ---

async def main():
    parser = argparse.ArgumentParser(description="Intelligent Vehicle Design Agent System")
    parser.add_argument('query', nargs='?', type=str, help="User instruction text", default="")
    parser.add_argument('--file', type=str, help="Path to input file", default=None)
    
    # Interactive mode check
    args = parser.parse_args()
    
    user_query = args.query
    file_path = args.file
    
    # If no args provided, ask interactively
    if not user_query and not file_path:
        print("欢迎使用智能汽车设计助手。")
        print("\n本系统支持以下功能：")
        print("1. 竞品分析与设计闭环 (输入需求文本，如：'设计一款运动型SUV')")
        print("2. 竞品实车图/照片 转 全流程设计 (提供照片)")
        print("3. 草图 转 渲染图/3D/风阻分析 (提供草图)")
        print("4. 渲染图 转 3D模型/风阻分析 (提供渲染图)")
        print("5. 3D模型 风阻(Cd)预测 (提供.stl/.obj模型文件)")
        print("-" * 40)
        
        user_query = input("请输入您的需求 (或直接回车跳过): ")
        file_input = input("请输入文件路径 (可选): ").strip()
        if file_input:
            file_path = file_input.replace("'","").replace('"','') # simplistic cleanup
    
    if not user_query and not file_path:
        print("未提供输入。退出。")
        return

    has_file = file_path is not None and os.path.exists(file_path)
    if file_path and not has_file:
        print(f"警告: 文件路径不存在: {file_path}")
        # Continue as if no file? Or stop?
        # Let's ask user to confirm no file or stop
        cont = input("文件未找到。是否继续作为纯文字任务处理? (y/n): ")
        if cont.lower() != 'y':
            return
        has_file = False
        file_path = None

    print(f"\n正在分析意图... [Query: '{user_query}', Has File: {has_file}]")
    intent_data = await classify_intent(user_query, has_file)
    
    if not intent_data:
        print("意图识别失败。")
        return

    intent = intent_data.get("intent")
    extracted_prompt = input("请输入设计风格关键词（用于渲染），或直接回车使用默认: ").strip()
    if extracted_prompt is None:
        extracted_prompt = intent_data.get("extracted_prompt", "")
    reason = intent_data.get("reason", "")
    
    print("\n" + "="*40)
    print(f"【任务确认】")
    print(f"检测到的意图: {intent}")
    print(f"核心提取内容: {extracted_prompt}")
    print(f"分析理由: {reason}")
    if has_file:
        print(f"处理文件: {file_path}")
    print("="*40 + "\n")
    
    confirm = input("确认执行此任务流程吗? (y/n): ")
    if confirm.lower() != 'y':
        print("任务已取消。")
        return

    print("\n开始执行任务...\n")
    
    try:
        if intent == "COMPETITOR_ANALYSIS" or intent == "DESIGN_LOOP":
             # Case 1
             await execute_param_design_loop(user_query)
             
        elif intent == "PHOTO_TO_CD":
             # Case 2
             if not has_file:
                 print("错误: 此意图需要文件输入。")
                 return
             await execute_photo_to_cd(file_path, extracted_prompt)
             
        elif intent == "SKETCH_TO_CD":
             # Case 3
             if not has_file:
                 print("错误: 此意图需要文件输入。")
                 return
             await execute_sketch_to_cd(file_path, extracted_prompt)
             
        elif intent == "RENDER_TO_CD":
             # Case 4
             if not has_file:
                 print("错误: 此意图需要文件输入。")
                 return
             await execute_render_to_cd(file_path)
             
        elif intent == "MODEL_TO_CD":
             # Case 5
             if not has_file:
                 print("错误: 此意图需要文件输入。")
                 return
             execute_model_to_cd(file_path)
        
        else:
            print(f"未处理的意图类型: {intent}")
            # Fallback to competitor analysis logic if it's generic text
            if not has_file:
                print("尝试执行通用设计/分析流程...")
                await execute_param_design_loop(user_query)

    except KeyboardInterrupt:
        print("\n用户中断任务。")
    except Exception as e:
        print(f"\n任务执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
