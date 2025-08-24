import argparse
import os
import time, csv
import torch
import trimesh
import numpy as np
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import yaml
from test_1stl import set_random_seed

# 读取STL文件并提取点和法向量
def load_stl_with_normals(file_path):
    stl_mesh = trimesh.load_mesh(file_path, force='mesh')
    vertices = stl_mesh.vertices
    normals = stl_mesh.face_normals  # 面的法向量
    return vertices, normals, stl_mesh


# 计算每个点的曲率
def compute_curvature(vertices, k=10):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(vertices)
    distances, indices = knn.kneighbors(vertices)
    neighbors = vertices[indices]
    covariance_matrices = np.array([np.cov(neigh.T) for neigh in neighbors])
    eigvals = np.linalg.eigvalsh(covariance_matrices)
    curvatures = eigvals[:, 0]  # 最小特征值表示曲率
    curvatures = np.round(curvatures, 10)  # 保留10位小数
    return curvatures


# 计算每个点的局部密度（邻居的平均距离）
def compute_density(vertices, k=10):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(vertices)
    distances, _ = knn.kneighbors(vertices)
    density = 1 / (distances[:, 1:].mean(axis=1) + 1e-6)
    return density


# 计算迎风面的网格数目
def compute_windward_faces(normals, direction=(1, 0, 0)):
    windward = normals @ direction  # 计算法向量与风向的点积
    windward_count = np.sum(windward < 0)  # 点积大于0表示迎风面
    return windward_count


# 计算基于曲率的熵
def compute_entropy(values):
    hist, bin_edges = np.histogram(values, bins=20, density=True)
    probabilities = hist / hist.sum()
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))
    return entropy


# 计算基于法向量的熵
def compute_normal_entropy(normals):
    # 将法向量归一化到球面投影
    normal_magnitudes = np.linalg.norm(normals, axis=1)
    normalized_normals = normals / (normal_magnitudes[:, np.newaxis] + 1e-9)

    # 球面分区 (bins 分区)
    theta = np.arctan2(normalized_normals[:, 1], normalized_normals[:, 0])  # 投影到XY平面
    phi = np.arccos(normalized_normals[:, 2])  # Z方向角
    hist, _ = np.histogramdd((theta, phi), bins=20)

    probabilities = hist / hist.sum()
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))
    return entropy


# 上下采样
# def sample_points(vertices, vert_normal, curvatures, densities, vertical_axis=2, upper_samples=25000,
#                   lower_samples=25000):
#     pointmax = np.max(vertices[:, vertical_axis])
#     pointmin = np.min(vertices[:, vertical_axis])
#     midpoint = np.mean([pointmax, pointmin])
#     upper_mask = vertices[:, vertical_axis] > midpoint
#     lower_mask = ~upper_mask
#     upper_points = vertices[upper_mask]
#     lower_points = vertices[lower_mask]
#     upper_normals = vert_normal[upper_mask]
#     lower_normals = vert_normal[lower_mask]
#     upper_curvatures = curvatures[upper_mask]
#     lower_curvatures = curvatures[lower_mask]
#     upper_densities = densities[upper_mask]
#     lower_densities = densities[lower_mask]
#     upper_weight = np.exp(upper_curvatures)
#     upper_weight /= upper_weight.sum()
#     lower_weight = 1 / (lower_densities + 1e-6)
#     lower_weight /= lower_weight.sum()
#     upper_indices = np.random.choice(len(upper_points), size=upper_samples, p=upper_weight)
#     lower_indices = np.random.choice(len(lower_points), size=lower_samples, p=lower_weight)
#     upper_sampled_points = upper_points[upper_indices]
#     upper_sampled_normals = upper_normals[upper_indices]
#     upper_sampled_curvatures = upper_curvatures[upper_indices]
#     lower_sampled_points = lower_points[lower_indices]
#     lower_sampled_normals = lower_normals[lower_indices]
#     lower_sampled_curvatures = lower_curvatures[lower_indices]
#     upper_sampled_data = np.hstack(
#         [upper_sampled_points, upper_sampled_normals, upper_sampled_curvatures.reshape(-1, 1)])
#     lower_sampled_data = np.hstack(
#         [lower_sampled_points, lower_sampled_normals, lower_sampled_curvatures.reshape(-1, 1)])
#     sampled_data = np.vstack([upper_sampled_data, lower_sampled_data])
#     return sampled_data

# #只根据曲率采样
def sample_points(vertices, vert_normal, curvatures, sampled_points=20000):
    weight = np.exp(curvatures)
    weight /= weight.sum() 
    indices = np.random.choice(len(vertices), size=sampled_points, p=weight)
    sampled_points = vertices[indices]
    sampled_normals =vert_normal[indices]
    sampled_curvatures = curvatures[indices]
    sampled_data = np.hstack([sampled_points, sampled_normals, sampled_curvatures.reshape(-1, 1)])
    return sampled_data


# 单个文件的处理函数
def append_to_csv(csv_path, row):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()  
        writer.writerow(row)


### 得到Cd 值
def get_cd_from_csv(csv_path, input_name):
 
    try:
        # 读取CSV文件
        data = pd.read_csv(csv_path)

        # 确保CSV文件包含至少两列：文件名和Cd值
        if data.shape[1] < 2:
            print("CSV文件格式错误，应包含至少两列：文件名和Cd值。")
            return None

        # 确保列名正确，假设第一列为文件名，第二列为Cd值
        file_column = data.columns[0]  # 第一列：文件名
        cd_column = data.columns[1]  # 第二列：Cd值

        # 查找文件名对应的Cd值
        result = data[data[file_column] == input_name]

        if result.empty:
            print(f"文件名 '{input_name}' 不存在于CSV文件中。")
            return None

        cd_value = result[cd_column].values[0]
        return cd_value
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return None


def process_single_stl(stl_file, input_dir, output_dir, csv_path, cd_path, wind_direction=(1, 0, 0)):
    try:
        start_time = time.time()
        stl_path = os.path.join(input_dir, stl_file)

        vertices, face_normals, mesh = load_stl_with_normals(stl_path)
        vert_normal = mesh.vertex_normals

        timeC1 = time.time()
        curvatures = compute_curvature(vertices)
        timeC2 = time.time()
        densities = compute_density(vertices)

        curvature_entropy = compute_entropy(curvatures)
        normal_entropy = compute_normal_entropy(vert_normal)

        windward_count = compute_windward_faces(face_normals, direction=wind_direction)

        volume = mesh.volume
        bounding_box = mesh.bounding_box.bounds 
        length = bounding_box[1, 0] - bounding_box[0, 0]  # x 轴的范围
        width = bounding_box[1, 1] - bounding_box[0, 1]  # y 轴的范围
        height = bounding_box[1, 2] - bounding_box[0, 2]  # z 轴的范围
        boundary_area = mesh.area
        interg_mean_convx = mesh.integral_mean_curvature

        time1 = time.time()
        
        # sampled_data = sample_points(
        #     vertices, vert_normal, curvatures,densities, vertical_axis=2, 
        #     upper_samples=25000,lower_samples=25000 )

        sampled_data = sample_points(
            vertices, vert_normal, curvatures,
            sampled_points=20000
        )
        time2 = time.time()

        output_path = os.path.join(output_dir, os.path.splitext(stl_file)[0] + '.txt')
        np.savetxt(output_path, sampled_data, fmt='%.10f', delimiter=' ')

        sampled_curvatures = sampled_data[:, -1]  # 曲率在最后一列
        sampled_normals = sampled_data[:, 3:6]  # 法向量在第4~6列

        timeD0 = time.time()
        sampled_curvature_entropy = compute_entropy(sampled_curvatures)
        timeD1 = time.time()
        sampled_normal_entropy = compute_normal_entropy(sampled_normals)
        timeD2 = time.time()
        file_name = os.path.splitext(stl_file)[0]
        cd = get_cd_from_csv(cd_path, file_name)

        row = {
            "file_name": file_name,
            "Cd": cd,
            "length": length,
            "width": width,
            "height": height,
            "volume": volume,
            "area": boundary_area,
            "interg_mean_convx": interg_mean_convx,
            "smp_curvature_entropy": sampled_curvature_entropy,
            "smp_normal_entropy": sampled_normal_entropy
        }

        append_to_csv(csv_path, row)

        elapsed_time = time.time() - start_time
        print(f"convx entro in {timeD1 - timeD0:.2f} seconds")
        print(f"normal entro in {timeD2 - timeD1:.2f} seconds")
        print(f"compute {stl_file} conv in {timeC2 - timeC1:.2f} seconds")
        print(f"sampling {stl_file} in {time2 - time1:.2f} seconds")
        print(f"Processed {stl_file} in {elapsed_time:.2f} seconds")
    except Exception as e:
        print(f"Failed to process {stl_file}: {e}")


# 处理文件夹中的所有STL文件（并行版本）
def process_stl_directory_parallel(input_dir, output_dir, csv_path, cd_path, max_workers=32):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    stl_files = [f for f in os.listdir(input_dir) if f.endswith(".stl")]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_stl, stl_file, input_dir, output_dir, csv_path, cd_path) for stl_file
                   in stl_files]

    for future in futures:
        future.result()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    from config import config
    torch.cuda.empty_cache()
    set_random_seed(42)
    # 创建解析器，并设置描述
    # parser = argparse.ArgumentParser(description="测试模型")
    # parser.add_argument("config_path", type=str, help="配置文件的路径")
    # # 解析命令行参数
    # args = parser.parse_args()
    # config = load_config(args.config_path)
    input_dir = config['paths']['stl_path']
    output_dir = config['paths']['data_process_path']
    csv_paths = config['paths']['csv_process_path']
    csv_process_path = os.path.join(csv_paths, "csv_processed.csv")  # 生成csv_processed.csv
    cd_path = config['paths']['csv_path']
    
    process_stl_directory_parallel(input_dir, output_dir, csv_process_path, cd_path, max_workers=8)
