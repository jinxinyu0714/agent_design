#!/usr/bin/env python3

import asyncio
import subprocess
import os
from typing import Dict, Any, Optional


async def run_clippasso_sketching(
    image_name: str,
    workspace_path: str = "/student/jxy/agent_design/clippasso_utils/CLIPasso",
    container_name: str = "clippasso-env:py310-cuda",
    gpu_device: str = "0",
    num_strokes: int = 100,
    mask_object: Optional[int] = None,
    fix_scale: Optional[int] = None,
    num_sketches: int = 1,
    use_cpu: bool = False,
) -> Dict[str, Any]:
    """
    LLM工具调用函数：运行CLIPasso对象素描
    
    Args:
        image_name: 目标图片文件名 (例如: "camel.png")
        workspace_path: CLIPasso工作空间路径
        container_name: Docker容器镜像名称
        gpu_device: GPU设备编号
        num_strokes: 素描笔画数量，控制抽象程度 (默认100，推荐16)
        mask_object: 是否遮罩背景，值为1时启用 (适用于有背景的图片)
        fix_scale: 是否自动修复图片比例，值为1时启用 (适用于非正方形图片)
        num_sketches: 并行生成的素描数量，默认1个 (推荐值，CPU运行时必须设为1)
        use_cpu: 是否使用CPU运行 (不推荐，速度较慢)
        
    Returns:
        Dict[str, Any]: 包含执行结果的字典
    """
    name = image_name.split('.')[0]  # 提取图片名称（不含扩展名）
    
    # 验证输入参数
    if not image_name:
        return {
            "success": False,
            "error": "图片名称不能为空",
            "output": "",
            "stderr": ""
        }
    
    # 检查工作空间是否存在
    if not os.path.exists(workspace_path):
        return {
            "success": False,
            "error": f"工作空间路径不存在: {workspace_path}",
            "output": "",
            "stderr": ""
        }
    

    
    # 构建命令行参数
    cmd_args = f"--target_file {image_name} --num_strokes {num_strokes} --num_sketches {num_sketches}"
    
    if mask_object == 1:
        cmd_args += " --mask_object 1"
    if fix_scale == 1:
        cmd_args += " --fix_scale 1"
    if use_cpu:
        cmd_args += " --cpu"
    
    # 构建Docker命令
    docker_cmd = [
        "docker", "run"
    ]
    
    # 添加GPU支持（如果不使用CPU）
    if not use_cpu:
        docker_cmd.extend([f"--gpus=device={gpu_device}"])
    
    docker_cmd.extend([
        "--ipc=host",
        "-v", f"{workspace_path}:/home/CLIPasso",
        "--rm", 
        container_name,
        "/bin/bash", "-c",
        f"""
        source /home/miniconda/etc/profile.d/conda.sh && \
        conda activate clippasso_py310 && \
        cd /home/CLIPasso && \
        python run_object_sketching.py {cmd_args}
        """
    ])
    
    try:
        print(f"开始运行CLIPasso素描生成，目标图片: {image_name}")
        print(f"使用容器: {container_name}")
        print(f"工作空间: {workspace_path}")
        print(f"命令参数: {cmd_args}")
        print(f"使用{'CPU' if use_cpu else 'GPU'}模式")
        
        # 执行Docker命令
        process = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.path.dirname(workspace_path)
        )
        
        # 等待执行完成
        stdout, stderr = await process.communicate()
        
        # 解码输出
        stdout_text = stdout.decode('utf-8', errors='ignore')
        stderr_text = stderr.decode('utf-8', errors='ignore')
        
        # 判断执行是否成功
        success = process.returncode == 0
        
        # 生成输出路径
        if num_sketches == 1:
            output_sketch_path = f"{workspace_path}/output_sketches/{name}/{name}_{num_strokes}strokes_seed0_best.svg"
        else:
            output_sketch_path = f"{workspace_path}/output_sketches/{name}/{name}_{num_strokes}strokes_best.svg"
        
        result = {
            "success": success,
            "returncode": process.returncode,
            "output": stdout_text,
            "stderr": stderr_text,
            "image_name": image_name,
            "workspace_path": workspace_path,
            "output_sketch_path": output_sketch_path,
            "parameters": {
                "num_strokes": num_strokes,
                "mask_object": mask_object,
                "fix_scale": fix_scale,
                "num_sketches": num_sketches,
                "use_cpu": use_cpu
            }
        }
        
        if success:
            print(f"CLIPasso执行成功!")
            print(f"素描保存在: {output_sketch_path}")
        else:
            print(f"CLIPasso执行失败，返回码: {process.returncode}")
            print(f"错误信息: {stderr_text}")
            
        return result
        
    except Exception as e:
        error_msg = f"执行CLIPasso时发生错误: {str(e)}"
        print(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "output": "",
            "stderr": "",
            "image_name": image_name,
            "workspace_path": workspace_path
        }


def run_clippasso_sketching_sync(
    image_name: str,
    workspace_path: str = "/student/jxy/agent_design/clippasso_utils/CLIPasso",
    container_name: str = "clippasso-env:py310-cuda",
    gpu_device: str = "0",
    num_strokes: int = 100,
    mask_object: Optional[int] = None,
    fix_scale: Optional[int] = None,
    num_sketches: int = 1,
    use_cpu: bool = False
) -> Dict[str, Any]:
    """
    同步版本的CLIPasso工具调用函数
    """
    def run_in_new_loop():
        """在新的事件循环中运行异步函数"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(run_clippasso_sketching(
                image_name=image_name,
                workspace_path=workspace_path,
                container_name=container_name,
                gpu_device=gpu_device,
                num_strokes=num_strokes,
                mask_object=mask_object,
                fix_scale=fix_scale,
                num_sketches=num_sketches,
                use_cpu=use_cpu
            ))
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
            return asyncio.run(run_clippasso_sketching(
                image_name=image_name,
                workspace_path=workspace_path,
                container_name=container_name,
                gpu_device=gpu_device,
                num_strokes=num_strokes,
                mask_object=mask_object,
                fix_scale=fix_scale,
                num_sketches=num_sketches,
                use_cpu=use_cpu
            ))
    except Exception as e:
        return {
            "success": False,
            "error": f"Error in CLIPasso sync wrapper: {str(e)}",
            "output": "",
            "stderr": "",
            "image_name": image_name,
            "workspace_path": workspace_path
        }


# 测试函数
async def test_clippasso_tool():
    for name in ["neo.png", "mengshi.png", "benz.png"]:
        for stroke in [10]:
            result = await run_clippasso_sketching(
                image_name=name,
                workspace_path="/student/jxy/easy_agent/analy/clippasso_utils/CLIPasso",
                container_name="clippasso-env:py310-cuda",
                gpu_device="0",
                num_strokes=stroke,
                mask_object=1,
                fix_scale=1,
                num_sketches=1
            )
        


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_clippasso_tool())

