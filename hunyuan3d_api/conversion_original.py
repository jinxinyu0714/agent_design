# # test_conversion.py
# import os
# import sys
# import argparse
# import time
# import shutil
# from client import Hunyuan3DClient
# from config import Config, OutputFormat

# def test_blender_conversion():
#     """测试Blender是否可用"""
#     print(" 测试Blender可用性")
    
#     if not Config.BLENDER_PATH:
#         print("❌ 未找到Blender，请检查安装")
#         return False
    
#     print(f" Blender路径: {Config.BLENDER_PATH}")
    
#     try:
#         import subprocess
#         result = subprocess.run([Config.BLENDER_PATH, "--version"], 
#                               capture_output=True, text=True)
#         if result.returncode == 0:
#             print("✅ Blender可用性测试通过")
#             version_line = result.stdout.split('\n')[0]
#             print(f" Blender版本: {version_line}")
#             return True
#         else:
#             print(f"❌ Blender测试失败: {result.stderr}")
#             return False
#     except Exception as e:
#         print(f"❌ Blender测试异常: {e}")
#         return False
    

# try:
#     import trimesh
#     TRIMESH_AVAILABLE = True
# except ImportError:
#     TRIMESH_AVAILABLE = False
#     print("⚠️  trimesh库未安装，STL转换将使用Blender")
#     print(" 如需使用trimesh进行STL转换，请运行: pip install trimesh numpy-stl")
    
# def test_trimesh_availability():
#     """测试trimesh是否可用"""
#     print(" 测试trimesh可用性")
    
#     if TRIMESH_AVAILABLE:
#         print("✅ trimesh可用")
#         return True
#     else:
#         print("❌ trimesh不可用")
#         print(" 请运行: pip install trimesh numpy-stl")
#         return False

# test_conversion.py
import os
import sys
import argparse
import time
import shutil
from client import Hunyuan3DClient
from config import Config, OutputFormat

def test_blender_conversion():
    """测试Blender是否可用"""
    # print(" 测试Blender可用性")
    
    if not Config.BLENDER_PATH:
        print("❌ 未找到Blender，请检查安装")
        return False
    
    # print(f" Blender路径: {Config.BLENDER_PATH}")
    
    try:
        import subprocess
        result = subprocess.run([Config.BLENDER_PATH, "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            # print("✅ Blender可用性测试通过")
            version_line = result.stdout.split('\n')[0]
            # print(f" Blender版本: {version_line}")
            return True
        else:
            print(f"❌ Blender测试失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Blender测试异常: {e}")
        return False
    

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("⚠️  trimesh库未安装，STL转换将使用Blender")
    print(" 如需使用trimesh进行STL转换，请运行: pip install trimesh numpy-stl")
    
def test_trimesh_availability():
    """测试trimesh是否可用"""
    # print(" 测试trimesh可用性")
    
    if TRIMESH_AVAILABLE:
        # print("✅ trimesh可用")
        return True
    else:
        print("❌ trimesh不可用")
        print(" 请运行: pip install trimesh numpy-stl")
        return False



def test_meshio_availability():
    """测试meshio是否可用"""
    try:
        import meshio
        # print(f"✅ meshio可用，版本: {meshio.__version__}")
        return True
    except ImportError as e:
        print(f"❌ meshio导入失败: {e}")
        print("�� 请运行: pip install meshio")
        return False
    except Exception as e:
        print(f"❌ 检查meshio时发生错误: {e}")
        return False

def normalize_path(path):
    """规范化路径，处理Windows反斜杠问题"""
    return os.path.abspath(os.path.normpath(path))

def wait_for_file(file_path, timeout=30):
    """等待文件出现并有内容"""
    start_time = time.time()
    file_path = normalize_path(file_path)
    
    while time.time() - start_time < timeout:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size > 0:
                return True
        time.sleep(1)
    
    return False

def _convert_to_glb(obj_file: str, output_path: str) -> bool:
    """
    使用Blender将OBJ转换为GLB（保留材质）
    """
    if Config.BLENDER_PATH:
        return _run_blender_conversion(obj_file, output_path, 'glb')
    else:
        return False

def _convert_to_stl(obj_file: str, output_path: str) -> bool:
    """
    使用trimesh将OBJ转换为STL（白模）
    """
    if TRIMESH_AVAILABLE:
        return _convert_with_trimesh(obj_file, output_path, 'stl')
    else:
        if Config.BLENDER_PATH:
            return _run_blender_conversion(obj_file, output_path, 'stl')
        else:
            return False

def _convert_stl_to_vtk(stl_file: str, output_path: str) -> bool:
    """
    使用meshio将STL转换为VTK（仅含结构信息）
    """
    # print(f"\n�� 开始STL到VTK转换: {stl_file} -> {output_path}")
    
    # 检查meshio是否可用
    try:
        import meshio
        # print(f"✅ meshio导入成功，版本: {meshio.__version__}")
    except ImportError as e:
        print(f"❌ meshio导入失败: {e}")
        return False
    
    try:
        import meshio
        
        # print(f"�� 读取STL文件: {stl_file}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 检查输入文件是否存在
        if not os.path.exists(stl_file):
            print(f"❌ STL文件不存在: {stl_file}")
            return False

        # 使用meshio读取STL文件
        # print("�� 使用meshio读取STL文件...")
        mesh = meshio.read(stl_file)
        # print(f"✅ STL文件读取成功")
        
        # 使用meshio写入VTK文件
        # print(f"�� 保存VTK文件: {output_path}")
        meshio.write(
            output_path,
            mesh,
            file_format='vtk',
            binary=True  # 使用二进制格式，文件更小
        )
        # print(f"✅ VTK文件保存成功: {output_path}")
        
        # 验证输出文件
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            file_size = os.path.getsize(output_path)
            # print(f"�� VTK文件大小: {file_size} 字节")
            return True
        else:
            print("❌ VTK文件生成失败")
            return False
            
    except Exception as e:
        print(f"❌ STL到VTK转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def _convert_with_trimesh(obj_path: str, output_path: str, target_format: str) -> bool:
    """
    使用trimesh库进行格式转换
    """
    if not TRIMESH_AVAILABLE:
        return False
    
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 加载OBJ文件
        import trimesh
        mesh = trimesh.load_mesh(obj_path)
        
        if target_format.lower() == 'stl':
            # 导出STL格式（白模）
            mesh.export(output_path, file_type='stl')
        else:
            return False
        
        # 验证输出文件
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            return False
            
    except Exception as e:
        return False

def _create_blender_script(obj_path: str, output_path: str, target_format: str) -> str:
    """
    创建Blender转换脚本
    """
    obj_path = obj_path.replace('\\', '/')
    output_path = output_path.replace('\\', '/')
    
    if target_format.lower() == 'glb':
        script = f"""
import bpy
import os
import sys

# 清除默认场景
bpy.ops.wm.read_factory_settings(use_empty=True)

# 设置路径
obj_file = r"{obj_path}"
output_file = r"{output_path}"

# 确保输出目录存在
os.makedirs(os.path.dirname(output_file), exist_ok=True)

try:
    # 删除默认对象
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # 导入OBJ文件（包含材质）
    bpy.ops.wm.obj_import(filepath=obj_file)
    
    # 检查是否成功导入
    imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not imported_objects:
        sys.exit(1)
    
    # 选择所有对象
    bpy.ops.object.select_all(action='SELECT')
    
    # 导出GLB（保留材质）
    bpy.ops.export_scene.gltf(
        filepath=output_file,
        export_format='GLB',
        export_yup=True,
        export_apply=True,
        export_materials='EXPORT'
    )
    
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    elif target_format.lower() == 'stl':
        script = f"""
import bpy
import os
import sys

# 清除默认场景
bpy.ops.wm.read_factory_settings(use_empty=True)

# 设置路径
obj_file = r"{obj_path}"
output_file = r"{output_path}"

# 确保输出目录存在
os.makedirs(os.path.dirname(output_file), exist_ok=True)

try:
    # 删除默认对象
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # 导入OBJ文件
    bpy.ops.wm.obj_import(filepath=obj_file)
    
    # 检查是否成功导入
    imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not imported_objects:
        sys.exit(1)
    
    # 选择所有对象
    bpy.ops.object.select_all(action='SELECT')
    
    # 导出STL
    bpy.ops.wm.stl_export(
        filepath=output_file,
        use_selection=True,
        global_scale=1.0,
        use_scene_unit=False,
        ascii=False,
        apply_modifiers=True
    )
    
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    else:
        raise ValueError(f"不支持的格式: {target_format}")
    
    return script

def _run_blender_conversion(obj_path: str, output_path: str, target_format: str) -> bool:
    """
    运行Blender转换
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 创建Blender脚本
        script_content = _create_blender_script(obj_path, output_path, target_format)
        
        # 使用临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            script_path = f.name
            f.write(script_content)
        
        # 执行Blender命令
        import subprocess
        process = subprocess.Popen(
            [Config.BLENDER_PATH, "--background", "--python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(timeout=300)  # 5分钟超时
        
        # 清理临时脚本
        try:
            os.unlink(script_path)
        except:
            pass
        
        success = (process.returncode == 0)
        if success:
            # 等待文件生成
            for i in range(30):  # 等待最多30秒
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return True
                time.sleep(1)
            
            return False
        else:
            return False
            
    except subprocess.TimeoutExpired:
        process.kill()
        return False
    except Exception as e:
        return False

def test_obj_to_glb_conversion(obj_file_path, output_dir="temp"):
    """测试OBJ到GLB转换（使用Blender）"""
    obj_file_path = normalize_path(obj_file_path)
    
    if not os.path.exists(obj_file_path):
        return None
    
    try:
        file_size = os.path.getsize(obj_file_path)
        if file_size == 0:
            return None
    except Exception as e:
        return None
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(obj_file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.glb")
    output_path = normalize_path(output_path)
    
    # 如果输出文件已存在，先删除
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception as e:
            pass
    
    success = _convert_to_glb(obj_file_path, output_path)
    
    if success:
        if wait_for_file(output_path, timeout=30):
            return output_path
        else:
            return None
    else:
        return None

def test_obj_to_stl_conversion(obj_file_path, output_dir="temp"):
    """测试OBJ到STL转换（使用trimesh）"""
    obj_file_path = normalize_path(obj_file_path)
    
    if not os.path.exists(obj_file_path):
        return None
    
    try:
        file_size = os.path.getsize(obj_file_path)
        if file_size == 0:
            return None
    except Exception as e:
        return None
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(obj_file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.stl")
    output_path = normalize_path(output_path)
    
    # 如果输出文件已存在，先删除
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception as e:
            pass
    
    success = _convert_to_stl(obj_file_path, output_path)
    
    if success:
        if wait_for_file(output_path, timeout=30):
            return output_path
        else:
            return None
    else:
        return None

def test_stl_to_vtk_conversion(stl_file_path, output_dir="temp"):
    """测试STL到VTK转换（使用meshio）"""
    stl_file_path = normalize_path(stl_file_path)
    # print(f"\n�� 测试STL到VTK转换: {stl_file_path}")
    
    if not os.path.exists(stl_file_path):
        print(f"❌ STL文件不存在: {stl_file_path}")
        return None
    
    try:
        file_size = os.path.getsize(stl_file_path)
        # print(f"�� 输入文件大小: {file_size} 字节")
        if file_size == 0:
            print("❌ 输入文件为空")
            return None
    except Exception as e:
        print(f"❌ 无法读取输入文件: {e}")
        return None
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(stl_file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.vtk")
    output_path = normalize_path(output_path)
    
    # print(f"�� 输入文件: {stl_file_path}")
    # print(f"�� 输出文件: {output_path}")
    # print(f"�� 转换方法: meshio（结构信息）")
    
    # 如果输出文件已存在，先删除
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            # print("��️ 已删除已存在的输出文件")
        except Exception as e:
            print(f"⚠️ 无法删除已存在文件: {e}")
    
    success = _convert_stl_to_vtk(stl_file_path, output_path)
    
    if success:
        # print("✅ VTK转换命令执行成功，等待文件生成...")
        if wait_for_file(output_path, timeout=30):
            file_size = os.path.getsize(output_path)
            # print(f"✅ STL到VTK转换成功!")
            # print(f"�� 输出文件大小: {file_size} 字节")
            return output_path
        else:
            print("❌ 转换成功但输出文件未生成或为空")
            return None
    else:
        print("❌ STL到VTK转换失败")
        return None

def _clear_output_directory():
    """清空output文件夹"""
    try:
        for filename in os.listdir(Config.OUTPUT_DIR):
            file_path = os.path.join(Config.OUTPUT_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                pass
    except Exception as e:
        pass

def find_latest_history_folder():
    """查找时间最近的history文件夹"""
    if not os.path.exists(Config.HISTORY_DIR):
        return None
    
    history_dirs = []
    for item in os.listdir(Config.HISTORY_DIR):
        item_path = os.path.join(Config.HISTORY_DIR, item)
        if os.path.isdir(item_path):
            history_dirs.append(item)
    
    if not history_dirs:
        return None
    
    # 按时间戳排序（文件夹名称格式: MM-DD-HH.MM）
    history_dirs.sort(reverse=True)
    latest_dir = history_dirs[0]
    return os.path.join(Config.HISTORY_DIR, latest_dir)

def test_with_latest_history():
    """使用最新的history文件夹进行测试"""
    latest_history = find_latest_history_folder()
    if not latest_history:
        return False
    
    # 查找原始文件目录中的OBJ文件
    original_dir = os.path.join(latest_history, "original", "extracted")
    if not os.path.exists(original_dir):
        return False
    
    # 查找OBJ文件
    obj_files = []
    for root, dirs, files in os.walk(original_dir):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_files.append(os.path.join(root, file))
    
    if not obj_files:
        return False
    
    obj_file = normalize_path(obj_files[0])
    base_name = os.path.splitext(os.path.basename(obj_file))[0]
    
    # 清空output文件夹
    _clear_output_directory()
    
    result_files = {}
    
    # 测试GLB转换（Blender）
    glb_output_path = test_obj_to_glb_conversion(obj_file, Config.OUTPUT_DIR)
    if glb_output_path:
        result_files["glb"] = glb_output_path
        # 复制到history文件夹
        glb_history_path = os.path.join(latest_history, f"{base_name}.glb")
        shutil.copy2(glb_output_path, glb_history_path)
    
    # 测试STL转换（trimesh）
    stl_output_path = test_obj_to_stl_conversion(obj_file, Config.OUTPUT_DIR)
    if stl_output_path:
        result_files["stl"] = stl_output_path
        # 复制到history文件夹
        stl_history_path = os.path.join(latest_history, f"{base_name}.stl")
        shutil.copy2(stl_output_path, stl_history_path)
        
        # 使用STL文件转换为VTK
        vtk_output_path = test_stl_to_vtk_conversion(stl_output_path, Config.OUTPUT_DIR)
        if vtk_output_path:
            result_files["vtk"] = vtk_output_path
            # 复制到history文件夹
            vtk_history_path = os.path.join(latest_history, f"{base_name}.vtk")
            shutil.copy2(vtk_output_path, vtk_history_path)
    
    # 打印输出文件的绝对路径
    for format_type, file_path in result_files.items():
        if file_path and os.path.exists(file_path):
            abs_path = os.path.abspath(file_path)
            print(f"{format_type}:{abs_path}")
    
    return len(result_files) > 0

def test_direct_conversion():
    """直接测试转换功能"""
    # 查找所有可能的OBJ文件
    obj_candidates = []
    
    # 在history目录中查找
    if os.path.exists(Config.HISTORY_DIR):
        for root, dirs, files in os.walk(Config.HISTORY_DIR):
            for file in files:
                if file.lower().endswith('.obj'):
                    obj_candidates.append(normalize_path(os.path.join(root, file)))
    
    if obj_candidates:
        if len(obj_candidates) == 1:
            selected_obj = obj_candidates[0]
        else:
            try:
                choice = int(input(f"请选择要测试的文件 (1-{len(obj_candidates)}): "))
                if 1 <= choice <= len(obj_candidates):
                    selected_obj = obj_candidates[choice-1]
                else:
                    return False
            except:
                return False
        
        # 测试三种转换，结果放到temp文件夹
        glb_path = test_obj_to_glb_conversion(selected_obj, "temp")
        stl_path = test_obj_to_stl_conversion(selected_obj, "temp")
        
        # 如果STL转换成功，再转换为VTK
        vtk_path = None
        if stl_path:
            vtk_path = test_stl_to_vtk_conversion(stl_path, "temp")
        
        # 输出结果路径
        if glb_path and os.path.exists(glb_path):
            print(f"glb:{os.path.abspath(glb_path)}")
        if stl_path and os.path.exists(stl_path):
            print(f"stl:{os.path.abspath(stl_path)}")
        if vtk_path and os.path.exists(vtk_path):
            print(f"vtk:{os.path.abspath(vtk_path)}")
        
        return glb_path is not None and stl_path is not None and vtk_path is not None
    else:
        return False

def cleanup_temp_dir():
    """清理temp目录"""
    temp_dir = "temp"
    if os.path.exists(temp_dir):
        try:
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            pass

def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description='格式转换测试工具')
    parser.add_argument('--direct', action='store_true', help='直接测试，自动查找OBJ文件，结果放到temp文件夹')
    parser.add_argument('--history', action='store_true', help='使用最新的历史记录进行转换，结果放到output文件夹')
    parser.add_argument('--cleanup', action='store_true', help='清理temp目录')
    
    args = parser.parse_args()
    
    # 清理选项
    if args.cleanup:
        cleanup_temp_dir()
        return
    
    # 测试转换工具可用性
    blender_available = test_blender_conversion()
    trimesh_available = test_trimesh_availability()
    
    if not blender_available:
        return
    
    # 测试meshio可用性
    meshio_available = test_meshio_availability()
    
    test_success = False
    
    if args.history:
        # 使用最新的历史记录测试
        test_success = test_with_latest_history()
    
    elif args.direct:
        # 直接测试
        test_success = test_direct_conversion()
    
    else:
        # 默认使用历史记录测试
        test_success = test_with_latest_history()

if __name__ == "__main__":
    main()