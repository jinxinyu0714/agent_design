import os
import numpy as np
from accelerate.commands.to_fsdp2 import load_config
from torch.utils.data import Dataset
import torch
from scipy.spatial.distance import cdist
import time, trimesh
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import random

def set_random_seed(seed=42):
    """设置所有相关库的随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pc_normalize(pc):
    centroid = torch.mean(pc, axis=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, axis=1)))
    pc = pc / (m + 1e-10)
    return pc

def min_max_normalize(sample):
    min_val = -2.5
    max_val = 2.5
    normalized_sample = (sample - min_val) / (max_val - min_val) if max_val != min_val else sample
    return np.array(normalized_sample)

def voxel_downsampling_np(point_cloud, target_num_points, voxel_size):
    # 计算体素网格的索引
    voxel_grid = np.floor(point_cloud[:, :3] / voxel_size).astype(np.int32)

    # 创建一个字典以存储每个体素的点
    voxel_dict = {}
    for idx, voxel in enumerate(voxel_grid):
        voxel_key = tuple(voxel)  # 将体素索引转换为元组作为字典的键
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = []
        voxel_dict[voxel_key].append(point_cloud[idx])

    # 从每个体素中随机选择一个点
    downsampled_points = []
    for points in voxel_dict.values():
        # 从体素中的点中随机选择一个
        downsampled_points.append(points[np.random.choice(len(points))])

    # 如果下采样后点数少于目标数量，随机选择
    downsampled_points = np.array(downsampled_points)

    if downsampled_points.shape[0] < target_num_points:
        additional_points = np.random.choice(downsampled_points.shape[0],
                                             target_num_points - downsampled_points.shape[0], replace=True)
        downsampled_points = np.vstack((downsampled_points, downsampled_points[additional_points]))

    return downsampled_points[:target_num_points]

def voxel_downsampling(point_cloud, target_num_points, voxel_size):
    # 计算体素网格的索引
    voxel_grid = torch.floor(point_cloud[:, :3] / voxel_size).to(torch.int32)  # 转换为整数类型

    # 创建一个字典以存储每个体素的点
    voxel_dict = {}
    for idx, voxel in enumerate(voxel_grid):
        voxel_key = tuple(voxel.tolist())  # 将体素索引转换为元组作为字典的键
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = []
        voxel_dict[voxel_key].append(point_cloud[idx])

    # 从每个体素中随机选择一个点
    downsampled_points = []
    for points in voxel_dict.values():
        # 从体素中的点中随机选择一个
        random_idx = torch.randint(0, len(points), (1,)).item()  # 随机选择一个索引
        downsampled_points.append(points[random_idx])

    # 转换为 tensor
    downsampled_points = torch.stack(downsampled_points)

    # 如果下采样后点数少于目标数量，随机选择
    if downsampled_points.shape[0] < target_num_points:
        additional_points = torch.randint(0, downsampled_points.shape[0],
                                          (target_num_points - downsampled_points.shape[0],))  # 选择额外的点
        downsampled_points = torch.cat((downsampled_points, downsampled_points[additional_points]))

    # 返回目标数量的点
    return downsampled_points[:target_num_points]

def min_max_normalize_coord(arr):
    # 计算每列的最小值和最大值
    col_min = arr.min(axis=0)
    col_max = arr.max(axis=0)

    # 进行归一化
    normalized_arr = (arr - col_min) / (col_max - col_min + 1e-10)

    return normalized_arr

def calculate_normals_with_angle(point_cloud):
    # 假设输入的点云数据形状为 [N, 6]
    # 前三列为坐标，最后一列为攻角
    coordinates = point_cloud[:, :3]  # 提取坐标
    angles = point_cloud[:, -1]  # 提取攻角

    # 初始化法向量数组
    normals = np.zeros((point_cloud.shape[0], 3))

    # 计算法向量
    for i in range(len(angles)):
        angle_rad = np.radians(angles[i])  # 将角度转换为弧度

        # 计算法向量 (法向量可能基于具体情况而定)
        # 假设法向量在水平面上的分量为 (cos, sin)，垂直分量为 0
        normals[i] = [
            np.cos(angle_rad),  # 法向量的x分量
            np.sin(angle_rad),  # 法向量的y分量
            0  # 假设法向量的z分量为0（可以根据需求调整）
        ]

    # 将法向量作为第七个特征添加到原始数据中
    augmented_point_cloud = np.hstack((point_cloud, normals))

    return augmented_point_cloud

def compute_normals(points, k=10):
    n_points = points.shape[0]
    normals = np.zeros_like(points)  # 初始化法向量

    # 计算所有点之间的距离矩阵
    distances = cdist(points, points)  # 计算距离矩阵，形状为 (N, N)

    # 对每一行进行处理
    for i in range(n_points):
        # 获取当前点及其邻域（k个最近邻）
        neighbor_indices = np.argsort(distances[i])[1:k + 1]  # 排序并取k个最近邻（跳过自身）

        # 获取邻域点
        neighbors = points[neighbor_indices]

        # 使用PCA计算邻域点的法向量
        pca = PCA(n_components=3)
        pca.fit(neighbors)  # 对邻域点进行PCA拟合
        normal = pca.components_[-1]  # 获取最小方差方向作为法向量

        # 如果法向量指向错误方向（反向），就反转法向量
        if np.dot(normal, points[i] - np.mean(neighbors, axis=0)) < 0:
            normal = -normal

        # 将法向量保存到对应位置
        normals[i] = normal

    return normals

def standardize(y, y_mean, y_std):
    return (y - y_mean) / y_std

def pc_normalize1(xyz, global_mean=None, global_std=None):
    if global_mean is None or global_std is None:
        # 如果没有全局均值和标准差，则使用局部均值和标准差
        return (xyz - xyz.mean(axis=0)) / xyz.std(axis=0)
    else:
        # 使用全局均值和标准差
        return (xyz - global_mean) / global_std

#多stl测试
class DrivAerDataset_stl4pre_random_cd(Dataset):
    def __init__(self, data_dir=None, filename_list=None, transform=None, random_seed=42, use_csv_labels=True, csv_path=None):
        self.data_dir = data_dir
        self.transform = transform
        self.random_seed = random_seed
        self.use_csv_labels = use_csv_labels

        set_random_seed(self.random_seed)

        self.sample_list = None
        if self.use_csv_labels:
            self.data_list_file = csv_path or "/student/jxy/easy_agent/analy/cd_utils/data/DrivAer_model_TrainingData_100.csv"
            if not os.path.exists(self.data_list_file):
                raise FileNotFoundError(f"CSV 文件未找到，但 use_csv_labels=True。请检查路径: {self.data_list_file}")
            self.sample_list = pd.read_csv(self.data_list_file)
        else:
            self.data_list_file = None

        # 判断 data_dir 是否为空
        if not self.data_dir:
            # 如果 data_dir 为空，则 filename_list 必须提供，其中为文件的绝对路径
            if not filename_list:
                raise ValueError("当 data_dir 为空时，必须提供 filename_list，其中应包含 STL 文件的绝对路径。")
            self.stl_files = filename_list
        else:
            all_stl_files = [f for f in os.listdir(data_dir) if f.endswith('.stl')]
            if not all_stl_files:
                raise ValueError(f"No STL files found in {data_dir}")
            if filename_list is not None:
                self.stl_files = [os.path.join(data_dir, f) if not os.path.isabs(f) else f for f in filename_list if f in all_stl_files or os.path.isabs(f)]
            else:
                self.stl_files = [os.path.join(data_dir, f) for f in all_stl_files]

    def __len__(self):
        return len(self.stl_files)

    def load_stl_with_normals_new(self, file_path):
        try:
            stl_mesh = trimesh.load_mesh(file_path, file_type='stl')
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None

        vertices = stl_mesh.vertices  # 原始顶点，不采样
        bounding_box = stl_mesh.bounding_box.bounds 
        length = bounding_box[1, 0] - bounding_box[0, 0]
        width = bounding_box[1, 1] - bounding_box[0, 1]
        height = bounding_box[1, 2] - bounding_box[0, 2]
        lwh = np.array([length, width, height], dtype=np.float32)
        
        return vertices, lwh

    ###曲率采样
    def sample_points(self, vertices, curvatures, sampled_points=20000):
        # 确保每次调用时使用相同的随机状态
        np.random.seed(self.random_seed)
        
        weight = np.exp(curvatures)
        weight /= weight.sum() 
        indices = np.random.choice(len(vertices), size=sampled_points, p=weight)
        sampled_points = vertices[indices]
        sampled_curvatures = curvatures[indices]
        sampled_data = np.hstack([sampled_points,  sampled_curvatures.reshape(-1, 1)])
        return sampled_data

    # 计算每个点的曲率
    def compute_curvature(self, vertices, k=10):
        # 在函数开始设置随机种子
        np.random.seed(self.random_seed)
        
        # 移除random_state参数，改为手动设置随机种子
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(vertices)
        distances, indices = knn.kneighbors(vertices)
        neighbors = vertices[indices]
        covariance_matrices = np.array([np.cov(neigh.T) for neigh in neighbors])
        eigvals = np.linalg.eigvalsh(covariance_matrices)
        curvatures = eigvals[:, 0]  # 最小特征值表示曲率
        return curvatures   

    def pc_norm(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    # 计算每个点的局部密度（邻居的平均距离）
    def compute_density(self, vertices, k=10):
        # 在函数开始设置随机种子
        np.random.seed(self.random_seed)
        
        # 移除random_state参数，改为手动设置随机种子
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(vertices)
        distances, _ = knn.kneighbors(vertices)
        density = 1 / (distances[:, 1:].mean(axis=1) + 1e-6)
        return density

    def __getitem__(self, idx):
        # 在每次获取项目时重新设置随机种子，确保一致性
        set_random_seed(self.random_seed + idx)
        
        stl_file_path = self.stl_files[idx]
        stl_filename = os.path.basename(stl_file_path)
        # ...后续逻辑保持不变，stl_file_path直接用...
        if not os.path.exists(stl_file_path):
            print(f"Warning: STL file '{stl_filename}' not found at {stl_file_path}")
            return self.__getitem__((idx + 1) % len(self))

        vertices, cubesize = self.load_stl_with_normals_new(stl_file_path)
        curvatures = self.compute_curvature(vertices)
        sampled_data = self.sample_points(vertices, curvatures, sampled_points=20000)
  
        xyzsize = np.array([
            [4, 1.8, 1],
            [6, 2.6, 2]
        ], dtype=np.float32)

        # 计算最小值和最大值
        min_vals = xyzsize.min(axis=0)
        max_vals = xyzsize.max(axis=0)
        # 进行最大最小归一化
        cubesize = 2 * (cubesize - min_vals) / (max_vals - min_vals) - 1 # -1,1 
        cubesize = cubesize.astype(np.float32)
        N,C = sampled_data.shape
        cubesize = torch.tensor(cubesize)
        cubesize = cubesize.unsqueeze(0)  # 变为  [1, 3]
        cubesize = cubesize.repeat(N, 1)  # 扩展为 [N, 3]
        
        sampled_data[:, :3] = self.pc_norm(sampled_data[:, :3])
        sampled_data = torch.tensor(sampled_data, dtype=torch.float32)

        filename = os.path.splitext(stl_filename)[0]

        # 推理场景：不使用 CSV 标签
        if not self.use_csv_labels or self.sample_list is None:
            # 不返回真实标签
            dummy_label = torch.tensor([0.0], dtype=torch.float32)
            return sampled_data[:, :3], cubesize, dummy_label, filename

        # 训练/评估有标签场景
        matching_rows = self.sample_list[self.sample_list['ID'] == filename]
        if len(matching_rows) == 0:
            raise ValueError(f"未在 CSV 文件中找到 {filename} 的匹配条目。")
        label_data = matching_rows['Drag_Value'].values[0]
        label_data = torch.tensor([label_data], dtype=torch.float32)

        return sampled_data[:,:3], cubesize, label_data, filename
    
    def process_stl_file(self, stl_filename):
        set_random_seed(self.random_seed)
        stl_file_path = stl_filename if os.path.isabs(stl_filename) or not self.data_dir else os.path.join(self.data_dir, stl_filename)
    
        if not os.path.exists(stl_file_path):
            raise FileNotFoundError(f"STL file '{stl_filename}' not found in {self.data_dir}")
        
        vertices, cubesize = self.load_stl_with_normals_new(stl_file_path)
        if vertices is None or cubesize is None:
            raise ValueError(f"Failed to load STL file: {stl_filename}")
            
        curvatures = self.compute_curvature(vertices) 
        sampled_data = self.sample_points(vertices, curvatures, sampled_points=20000)
    
        xyzsize = np.array([
            [4, 1.8, 1],
            [6, 2.6, 2]
        ], dtype=np.float32)
    
        # 计算最小值和最大值
        min_vals = xyzsize.min(axis=0)
        max_vals = xyzsize.max(axis=0)
        # 进行最大最小归一化
        cubesize = 2 * (cubesize - min_vals) / (max_vals - min_vals) - 1  # -1,1 
        cubesize = cubesize.astype(np.float32)
        N, C = sampled_data.shape
        cubesize = torch.tensor(cubesize)
        cubesize = cubesize.unsqueeze(0)  # 变为 [1, 3]
        cubesize = cubesize.repeat(N, 1)  # 扩展为 [N, 3]
        
        sampled_data[:, :3] = self.pc_norm(sampled_data[:, :3])
        sampled_data = torch.tensor(sampled_data, dtype=torch.float32)
    
        filename = os.path.splitext(stl_filename)[0]

        # 若不使用 CSV，则仅返回特征与文件名（label 返回一个占位符，调用方可忽略）
        if not self.use_csv_labels or self.sample_list is None:
            dummy_label = torch.tensor([0.0], dtype=torch.float32)
            return sampled_data[:, :3], cubesize, dummy_label, filename

        # 使用 CSV 标签的情况
        matching_rows = self.sample_list[self.sample_list['ID'] == filename]
        if len(matching_rows) == 0:
            raise ValueError(f"未在 CSV 文件中找到 {filename} 的匹配条目。")
        label_data = matching_rows['Drag_Value'].values[0]
        label_data = torch.tensor([label_data], dtype=torch.float32)

        return sampled_data[:,:3], cubesize, label_data, filename

    def process_stl_file_no_label(self, stl_filename):
        set_random_seed(self.random_seed)
        stl_file_path = stl_filename if os.path.isabs(stl_filename) or not self.data_dir else os.path.join(self.data_dir, stl_filename)
        if not os.path.exists(stl_file_path):
            raise FileNotFoundError(f"STL file '{stl_filename}' not found in {self.data_dir}")

        vertices, cubesize = self.load_stl_with_normals_new(stl_file_path)
        if vertices is None or cubesize is None:
            raise ValueError(f"Failed to load STL file: {stl_filename}")

        curvatures = self.compute_curvature(vertices)
        sampled_data = self.sample_points(vertices, curvatures, sampled_points=20000)

        xyzsize = np.array([
            [4, 1.8, 1],
            [6, 2.6, 2]
        ], dtype=np.float32)

        # 归一化边界尺寸到 [-1, 1]
        min_vals = xyzsize.min(axis=0)
        max_vals = xyzsize.max(axis=0)
        cubesize = 2 * (cubesize - min_vals) / (max_vals - min_vals) - 1
        cubesize = cubesize.astype(np.float32)
        N, C = sampled_data.shape
        cubesize = torch.tensor(cubesize)
        cubesize = cubesize.unsqueeze(0).repeat(N, 1)

        sampled_data[:, :3] = self.pc_norm(sampled_data[:, :3])
        sampled_data = torch.tensor(sampled_data, dtype=torch.float32)

        filename = os.path.splitext(stl_filename)[0]
        return sampled_data[:, :3], cubesize, filename