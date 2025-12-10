import glob
import os
import time
import argparse
import numpy as np
import torch
import pyvista as pv
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
import multiprocessing as mp

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', type=str, required=True, help="原始VTK目录")
    p.add_argument('--output_dir', type=str, required=True, help="输出图数据目录")
    p.add_argument('--target_points', type=int, default=100000, help="简化后目标点数")
    return p.parse_args()

def read_vtk_file(vtk_file, target_points=100000):
    """读取VTK文件并进行网格简化"""
    mesh = pv.read(vtk_file)
    print(f"读取VTK文件: {os.path.basename(vtk_file)}")
    print(f"  原始节点数: {mesh.n_points}")
    print(f"  原始单元数: {mesh.n_cells}")

    # 处理体网格
    if isinstance(mesh, pv.UnstructuredGrid):
        print("  检测到体网格，提取表面...")
        mesh = mesh.extract_surface()
        print(f"  表面提取完成: 节点数 {mesh.n_points}, 单元数 {mesh.n_cells}")

    # 转换为三角形网格
    if not mesh.is_all_triangles:
        print("  网格包含非三角形单元，三角化中...")
        mesh = mesh.triangulate()
        print(f"  三角化完成: 节点数 {mesh.n_points}, 单元数 {mesh.n_cells}")

    # 计算简化比例
    original_points = mesh.n_points
    if original_points <= target_points:
        print(f"  节点数({original_points})已小于目标值({target_points})，跳过简化")
    else:
        # 计算简化比例
        ratio = round(target_points / original_points, 6)
        print()
        reduction_ratio = 1.0 - ratio
        print(f"  目标简化比例: {reduction_ratio:.6f}")

        mesh = mesh.decimate_pro(reduction_ratio, preserve_topology=True)
        print(f"  简化后节点数: {mesh.n_points}")
        print(f"  简化后单元数: {mesh.n_cells}")

    # 提取点和场数据
    points = mesh.points
    field_data = {}
    # for field in ['p']:
    ###qirui
    for field in ['Pressure']:
    # for field in ['p', 'WallShearStressMagnitude', 'WallShearStress_0', 'WallShearStress_1',
    #               'WallShearStress_2']:
        if field in mesh.point_data:
            field_data[field] = mesh.point_data[field].astype(np.float32)
        else:
            print(f"  警告: 缺少{field}场数据，使用零填充")
            field_data[field] = np.zeros(mesh.n_points, dtype=np.float32)

    return mesh, points.astype(np.float32), field_data


def build_knn_graph(positions, k=10):
    """使用KNN方法构建图结构"""
    # 使用sklearn的kneighbors_graph构建邻接矩阵
    adj_matrix = kneighbors_graph(positions, k, mode='connectivity', include_self=False)
    # 转换为边列表
    edges = np.vstack(adj_matrix.nonzero())
    # 确保图是无向的（添加反向边）
    edges = np.hstack([edges, edges[::-1, :]])
    # 移除重复边
    edges = np.unique(edges, axis=1)
    return edges.astype(np.int64)


def process_field_data(field_data):
    processed = {}
    for name, data in field_data.items():
        min_val, max_val = np.percentile(data, [1, 99])
        processed[name] = (data - min_val) / (max_val - min_val + 1e-8)
    return processed



def save_graph_data(positions, edges, targets, savename):
    ###QIRUI
    y = torch.stack([torch.from_numpy(targets['Pressure'])], dim=1)
    # y = torch.stack([torch.from_numpy(targets['p'])], dim=1)
    graph = Data(
        x=torch.from_numpy(positions),
        y=y,
        edge_index=torch.from_numpy(edges),
        pos=torch.from_numpy(positions)
    )
    torch.save(graph, savename)


def process_one_vtk(vtk_file, output_dir, target_points):
    try:
        base_name = os.path.basename(vtk_file).split('.')[0]
        print(f"\n开始处理: {base_name}")
        start = time.time()

        mesh, positions, field_data = read_vtk_file(vtk_file)
        processed_fields = process_field_data(field_data)
        edges = build_knn_graph(positions, k=10)

        pt_dir = os.path.join(output_dir, "graphs")
        vtk_dir = os.path.join(output_dir)
        os.makedirs(pt_dir, exist_ok=True)
        os.makedirs(vtk_dir, exist_ok=True)

        mesh.save(os.path.join(vtk_dir, f"{base_name}.vtk"))
        save_graph_data(positions, edges, processed_fields,
                        os.path.join(pt_dir, f"{base_name}_graph_data.pt"))

        print(f"完成: {base_name} | 耗时: {time.time() - start:.2f}s")

    except Exception as e:
            print(f"处理失败 {vtk_file}: {str(e)}")


def main():
    args = parse_args()
    vtk_files = glob.glob(os.path.join(args.input_dir, "*.vtk"))
    print(f"共发现 {len(vtk_files)} 个VTK文件")

    num_workers = min(mp.cpu_count(), 32)  
    print(f"进程数: {num_workers}")

    args_list = [(vtk_file, args.output_dir, args.target_points) for vtk_file in vtk_files]

    with mp.Pool(processes=num_workers) as pool:
        pool.starmap(process_one_vtk, args_list)

if __name__ == "__main__":
    main()

