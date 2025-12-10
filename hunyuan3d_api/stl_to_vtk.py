import os
import meshio

# --- 配置 ---
# 定义输入和输出文件夹路径
# os.path.dirname(__file__) 获取当前脚本所在的目录，确保路径正确
# 如果文件夹不存在，则创建它
output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义输入和输出文件的完整路径
input_stl_path = os.path.join(output_folder, '3.stl')
output_vtk_path = os.path.join(output_folder, 'output.vtk')

# --- 主程序 ---
def convert_stl_to_vtk(stl_path, vtk_path):
    """
    使用 meshio 将 STL 文件转换为 VTK 文件 (仅包含几何结构)。

    Args:
        stl_path (str): 输入的 STL 文件路径。
        vtk_path (str): 输出的 VTK 文件路径。
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(stl_path):
            print(f"错误：输入文件不存在于 '{stl_path}'")
            return

        # 使用 meshio 读取 STL 文件
        # STL 文件本身只包含顶点和面（结构信息），不包含矢量场数据
        print(f"正在读取文件: {stl_path}...")
        mesh = meshio.read(stl_path)

        # 使用 meshio 写入 VTK 文件
        # write_points_cells=True 是默认行为，会写入结构信息
        # file_format='vtk' 指定输出格式
        # binary=True 可以让输出文件更小，读取更快
        print(f"正在写入文件: {vtk_path}...")
        meshio.write(
            vtk_path,
            mesh,
            file_format='vtk',
            binary=True  # 设置为 False 可以输出可读的ASCII格式
        )

        print("\n转换成功！")
        print(f"输入: {stl_path}")
        print(f"输出: {vtk_path}")

    except Exception as e:
        print(f"\n转换过程中发生错误: {e}")

# --- 运行转换 ---
if __name__ == "__main__":
    convert_stl_to_vtk(input_stl_path, output_vtk_path)