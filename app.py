import streamlit as st
import asyncio
import os
import sys
import json
import shutil
import time
from pathlib import Path

# 添加当前目录导致 sys.path 以确保导入正常
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入原有模块
from config import config
from agents import create_intent_understander, get_model_client

# 导入业务逻辑
from test_report import main as get_competitors
from test_sketch import main as generate_sketch_agent
from cd_utils.pipeline import predict_cd_value
from hunyuan3d_api.test_api import main as generate_3d_model
from clippasso_utils.clippasso_tool import run_clippasso_sketching
# Lazy load controlnet to avoid streamlit/torch reload issues immediately
# from render_utils.controlnet import generate_rendering
import render_utils.controlnet as controlnet_module

# --- 设置页面配置 ---
st.set_page_config(
    page_title="智能汽车设计助手",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 工具函数 ---

async def st_classify_intent(user_query: str, has_file: bool):
    """前端使用的意图识别"""
    client = get_model_client()
    agent = await create_intent_understander(client)
    context = f"Query: {user_query}\nHas File: {has_file}"
    try:
        response = await agent.run(task=context)
        content = response.messages[-1].content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
             content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)
    except Exception as e:
        st.error(f"意图识别错误: {e}")
        return None

def save_uploaded_file(uploaded_file):
    """保存上传的文件到临时目录"""
    if uploaded_file is not None:
        save_dir = Path("temp_uploads")
        save_dir.mkdir(exist_ok=True)
        file_path = save_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return str(file_path)
    return None

# --- 执行逻辑 (适配 Streamlit 输出) ---

async def run_competitor_analysis(task_description, status_container):
    status_container.write("🔍 正在进行竞品分析...")
    try:
        models = await get_competitors(task_description)
        status_container.write("✅ 竞品分析完成")
        return models
    except Exception as e:
        status_container.error(f"竞品分析出错: {e}")
        return None

async def run_sketch_generation(model_name, status_container):
    status_container.write(f"🎨 正在为 {model_name} 生成草图...")
    sketch_task = f"创建一个关于{model_name}的汽车草图"
    try:
        sketch_path = await generate_sketch_agent(sketch_task)
        if sketch_path and os.path.exists(sketch_path):
            status_container.write("✅ 草图生成成功")
            return sketch_path
        else:
            status_container.warning("⚠️ 草图生成未返回有效路径，使用默认示例。")
            return None
    except Exception as e:
        status_container.error(f"草图生成出错: {e}")
        return None

async def run_photo_to_sketch(image_path, status_container):
    status_container.write("🖼️ 正在处理实车照片 (CLIPasso)...")
    try:
        # 准备目录
        target_dir = config.CLIPPASSO_TARGET_IMAGES_DIR
        if not target_dir.exists():
            target_dir.mkdir(parents=True)
        
        filename = os.path.basename(image_path)
        dest_path = target_dir / filename
        
        # 复制文件到目标目录
        if os.path.abspath(image_path) != os.path.abspath(dest_path):
            shutil.copy(image_path, dest_path)
            
        result = await run_clippasso_sketching(
            image_name=filename,
            workspace_path=str(config.CLIPPASSO_DIR),
            num_sketches=1
        )
        
        sketch_path = result.get("output_sketch_path")
        if sketch_path and os.path.exists(sketch_path):
            status_container.write("✅ 线条草图提取成功")
            return sketch_path
        else:
            status_container.error(f"CLIPasso 失败: {result}")
            return None
    except Exception as e:
        status_container.error(f"图片转草图出错: {e}")
        return None

def run_rendering(sketch_path, prompt, status_container):
    status_container.write("✨ 正在生成渲染图 (ControlNet)...")
    prompt = prompt if prompt else "A futuristic car, high quality, realistic"
    try:
        # Use Cached Pipeline via module
        render_path = controlnet_module.generate_rendering(str(sketch_path), prompt=prompt)
        status_container.write("✅ 渲染完成")
        return render_path
    except Exception as e:
        status_container.error(f"渲染出错: {e}")
        return None

def run_3d_generation(render_path, status_container):
    status_container.write("🧊 正在生成 3D 模型 (Hunyuan3D)...")
    try:
        model_paths = generate_3d_model(str(render_path))
        if model_paths and "stl_path" in model_paths:
            status_container.write("✅ 3D 模型生成成功")
            return model_paths["stl_path"]
        else:
            status_container.error("3D 模型生成返回空结果")
            return None
    except Exception as e:
        status_container.error(f"3D 生成出错: {e}")
        return None

def run_cd_prediction(model_path, status_container):
    status_container.write("💨 正在预测风阻系数 (Cd)...")
    try:
        if not os.path.exists(model_path):
            status_container.error(f"文件不存在: {model_path}")
            return None
        
        result = predict_cd_value([model_path])
        status_container.write("✅ CD 预测完成")
        return result
    except Exception as e:
        status_container.error(f"CD 预测出错: {e}")
        return None

# --- 主界面 ---

def main():
    st.title("🚗 智能汽车设计助手")
    st.markdown("""
    本系统集成了意图识别、竞品分析、草图生成、AI 渲染、3D 建模及风阻预测全流程。
    """)

    # 侧边栏：输入区
    with st.sidebar:
        st.header("🏁 任务输入")
        
        # 1. 文本输入
        user_query = st.text_area("请输入您的需求描述", height=100, placeholder="例如：设计一款运动型SUV，或者帮我分析一下Model Y的竞品...")
        
        # 2. 文件上传
        uploaded_file = st.file_uploader("上传图片或3D模型 (可选)", type=['png', 'jpg', 'jpeg', 'stl', 'obj'])

        # --- FIX: 检测输入变化，重置状态 ---
        # 生成当前输入的指纹，用于检测变化
        current_input_id = f"{user_query}_{uploaded_file.name if uploaded_file else 'nofile'}_{uploaded_file.size if uploaded_file else 0}"
        
        if "last_input_id" not in st.session_state:
            st.session_state.last_input_id = current_input_id
            
        if st.session_state.last_input_id != current_input_id:
            # 输入发生变化，清除意图缓存，强制重新分析
            if "intent_data" in st.session_state:
                del st.session_state.intent_data
            st.session_state.last_input_id = current_input_id
        # ------------------------------------
        
        file_path = None
        has_file = False
        if uploaded_file:
            file_path = save_uploaded_file(uploaded_file)
            has_file = True
            st.success(f"文件已上传: {uploaded_file.name}")
            
            # 显示预览
            if uploaded_file.type.startswith('image'):
                st.image(uploaded_file, caption="上传的图片", use_container_width=True)
        
        analyze_btn = st.button("🔍 分析意图", type="primary", use_container_width=True)

    # 主区域：状态与结果
    if "intent_data" not in st.session_state:
        st.session_state.intent_data = None

    # Step 1: 意图分析
    if analyze_btn:
        if not user_query and not has_file:
            st.warning("⚠️ 请输入需求描述或上传文件。")
        else:
            with st.spinner("🤖 正在分析您的意图..."):
                # 运行异步意图识别
                intent_res = asyncio.run(st_classify_intent(user_query, has_file))
                st.session_state.intent_data = intent_res
                st.session_state.file_path = file_path # 保存路径到 session
                st.session_state.has_file = has_file

    # Step 2: 任务确认与执行
    if st.session_state.intent_data:
        st.divider()
        st.subheader("📋 任务确认")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"**识别意图**: {st.session_state.intent_data.get('intent')}")
            st.write(f"**分析理由**: {st.session_state.intent_data.get('reason')}")
            
            # 允许用户修改提取的 Prompt
            default_prompt = st.session_state.intent_data.get('extracted_prompt', '')
            final_prompt = st.text_input("提取的设计风格/关键词 (可修改):", value=default_prompt)

        with col2:
            st.write(" ") 
            st.write(" ")
            confirm_btn = st.button("🚀 开始执行", type="primary", use_container_width=True)

        # Step 3: 执行工作流
        if confirm_btn:
            intent = st.session_state.intent_data.get('intent')
            file_path = st.session_state.get('file_path')
            
            st.divider()
            st.subheader("⚙️ 执行进度")
            
            # 使用 status 容器显示步骤
            with st.status("正在处理任务...", expanded=True) as status:
                success = False
                # --- 工作流分发 ---
                
                # Case 1: 竞品分析 / 全流程设计
                if intent in ["COMPETITOR_ANALYSIS", "DESIGN_LOOP", "OTHER"]:
                    models = asyncio.run(run_competitor_analysis(user_query, status))
                    
                    if models:
                        model_name = models[0]
                        sketch_path = asyncio.run(run_sketch_generation(model_name, status))
                        
                        if sketch_path:
                            st.image(sketch_path, caption=f"生成的草图: {model_name}")
                            # 使用子流程
                            success = execute_render_flow(sketch_path, final_prompt, status)
                        else:
                            status.error("草图生成失败，无法继续流程。") # 明确错误

                # Case 2: 实车图 -> CD
                elif intent == "PHOTO_TO_CD":
                    if not file_path:
                        status.error("需上传文件") # 明确错误
                    else:
                        sketch_path = asyncio.run(run_photo_to_sketch(file_path, status))
                        if sketch_path:
                            st.image(sketch_path, caption="提取的线条草图")
                            # 继续后续流程
                            success = execute_render_flow(sketch_path, final_prompt, status)
                        else:
                             status.error("图片转线条失败")

                # Case 3: 草图 -> CD
                elif intent == "SKETCH_TO_CD":
                    if not file_path:
                        status.error("需上传文件")
                    else:
                        # 继续后续流程
                        st.image(file_path, caption="上传的草图", width=300)
                        success = execute_render_flow(file_path, final_prompt, status)

                # Case 4: 渲染图 -> CD
                elif intent == "RENDER_TO_CD":
                    if not file_path:
                         status.error("需上传文件")
                    else:
                        st.image(file_path, caption="上传的渲染图", width=300)
                        stl_path = run_3d_generation(file_path, status)
                        if stl_path:
                           success = display_3d_and_cd(stl_path, status)
                        # 如果没有生成stl，run_3d_generation 内部应该报错了，这里 success 保持 False 即可

                # Case 5: 3D模型 -> CD
                elif intent == "MODEL_TO_CD":
                     if not file_path:
                        status.error("需上传文件")
                     else:
                        cd_res = run_cd_prediction(file_path, status)
                        if cd_res:
                            display_cd_result(cd_res)
                            success = True
                        # 如果失败，run_cd_prediction 内部报错

                if success:
                    status.update(label="✅ 任务执行完毕", state="complete", expanded=True)
                else:
                    # 只有在没有明确报错但流程未完成时才真正显示错误状态
                    status.update(label="❌ 任务执行中断或出错", state="error", expanded=True)

# --- 辅助流程函数 ---

def execute_render_flow(sketch_path, prompt, status):
    """复用的子流程: Sketch -> Render -> 3D -> CD"""
    render_path = run_rendering(sketch_path, prompt, status)
    if render_path:
        st.image(render_path, caption="AI 渲染结果")
        stl_path = run_3d_generation(render_path, status)
        if stl_path:
            return display_3d_and_cd(stl_path, status)
    return False

def display_3d_and_cd(stl_path, status):
    """显示3D模型下载和CD预测结果"""
    st.success(f"3D 模型生成完毕: {os.path.basename(stl_path)}")
    with open(stl_path, "rb") as f:
        st.download_button("⬇️ 下载 .stl 模型文件", f, file_name=os.path.basename(stl_path))
    
    cd_res = run_cd_prediction(stl_path, status)
    if cd_res:
        display_cd_result(cd_res)
        return True
    return False

def display_cd_result(result):
    """可视化展示CD预测结果"""
    st.divider()
    st.subheader("🌪️ 风阻分析结果")
    
    if 'individual_results' in result and len(result['individual_results']) > 0:
        res = result['individual_results'][0]
        val = res['predicted_value']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("预测 Cd 值", f"{val:.3f}")
        
        if val < 0.30:
            col2.success("优秀的空气动力学表现")
        elif val < 0.35:
            col2.warning("表现良好，有优化空间")
        else:
            col2.error("风阻较高，建议优化")
            
        st.caption("详细数据:")
        st.json(result)

if __name__ == "__main__":
    main()
