# test_api.py
import os
import argparse
from .client import Hunyuan3DClient
from .models import ImageTo3DRequest, OutputFormat
from .config import Config, APIMode, PolygonLevel
import trimesh
import vtk

def convert_obj_to_stl(obj_file_path, stl_file_path):
    """
    Converts an OBJ file to an STL file.

    :param obj_file_path: Path to the input OBJ file.
    :param stl_file_path: Path to the output STL file.
    """
    # Load the OBJ file
    mesh = trimesh.load(obj_file_path)
    
    # Export the mesh to STL format
    mesh.export(stl_file_path)
    return stl_file_path

def convert_stl_to_vtk(stl_file_path, vtk_file_path):
    """
    Converts an STL file to a VTK file.

    :param stl_file_path: Path to the input STL file.
    :param vtk_file_path: Path to the output VTK file.
    """
    # Read the STL file
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_file_path)
    reader.Update()

    # Write to VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(vtk_file_path)
    writer.SetInputData(reader.GetOutput())
    writer.Write()

    return vtk_file_path



def test_pro_mode(image_path, polygon_level, pbr_material):
    """
    测试专业版模式
    """
    secret_id, secret_key = Config.get_credentials()
    client = Hunyuan3DClient(secret_id, secret_key)
    
    if not os.path.exists(image_path):
        return None  # 添加返回值以处理错误情况

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    Config.set_api_mode(APIMode.PRO)
    Config.set_polygon_level(polygon_level)  # 直接使用 polygon_level
    Config.set_pbr_material(pbr_material)
    
    request = ImageTo3DRequest(
        image_path=image_path,
        output_format=OutputFormat.OBJ,
        resolution="1024x1024",
        style="realistic",
        polygon_level=polygon_level,  # 直接使用 polygon_level
        pbr_material=pbr_material
    )
    
    result = client.submit_image_to_3d_job(request)
    # 处理 result 以确保返回值
    
    if not result["success"]:
        return None  # 修改为返回 None，以便在主函数中处理

    job_id = result["job_id"]

    final_status = client.wait_for_job_completion(job_id, timeout=1800, poll_interval=45)
    
    if final_status and final_status.is_success and final_status.download_url:
        # 专业版：下载并保存原始文件
        # print(f"\n=== 下载专业版模型原始文件 ===")
        result_files = client.download_pro_model(final_status.download_url, base_name)
        
        if result_files and "history_path" in result_files:
            return os.path.abspath(result_files['obj_files'][0])
        else:
          
            return None  # 修改为返回 None，以便在主函数中处理
    else:
        return None  # 修改为返回 None，以便在主函数中处理

def main(image_path, polygon_level="medium", pbr_material=False):
    """主函数 - 支持命令行参数"""
    if not os.path.exists(image_path):
        return None

    # 如果需要临时将 Blender 二进制目录加入 PATH（在 Python 进程内生效）
    blender_dir = "/home/j/桌面/blender-4.5.3-linux-x64"
    if os.path.isdir(blender_dir):
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + blender_dir

    path = test_pro_mode(image_path, polygon_level, pbr_material)
    if not path:
        return None

    base = os.path.splitext(path)[0]
    stl_path = base + ".stl"
    stl_path = convert_obj_to_stl(path, stl_path)
    vtk_path = base + ".vtk"
    vtk_path = convert_stl_to_vtk(stl_path, vtk_path)
    path_dict = {"stl_path": stl_path, "vtk_path": vtk_path}
    return path_dict



