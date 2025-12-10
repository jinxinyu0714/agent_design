import time
import pandas as pd
import torch
import torch_geometric
import vtk
import os
import itertools
import random
import numpy as np
from torch_geometric import nn as nng
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import k_hop_subgraph, subgraph
from vtk.util.numpy_support import vtk_to_numpy

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")###处理数据使用的gpu


def load_metadata(metadata_file):
    # metadata_file 是 CSV 文件，包含文件名和 CD 值
    import pandas as pd
    df = pd.read_csv(metadata_file,header=None)
    filenames = df.iloc[:, 0].values   
    return filenames

def load_split(split_file, filenames):
    # split_file 是 CSV 文件，包含样本索引
    import pandas as pd
    df = pd.read_csv(split_file,header=None)
    indices = df.iloc[:, 0].values  # 第一列为索引
    return indices

def load_unstructured_grid_data(file_name):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    output = reader.GetOutput()
    points = output.GetPoints()

    if points is not None:
      points_array = vtk_to_numpy(points.GetData())
      print(points_array.shape)
    else:
      print("No points found in PolyData")
    return output

def load_unstructured_grid_data_binary(file_name):
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(file_name)
    reader.ReadAllScalarsOn() 
    reader.Update()
    output = reader.GetOutput()
    points = output.GetPoints()

    if points is not None:
        points_array = vtk_to_numpy(points.GetData())
        print(points_array.shape)
    else:
        print("No points found in PolyData")
    return output


def unstructured_grid_data_to_poly_data(unstructured_grid_data):
    filter = vtk.vtkDataSetSurfaceFilter()
    filter.SetInputData(unstructured_grid_data)
    filter.Update()
    poly_data = filter.GetOutput()
    return poly_data, filter


def get_sdf(target, boundary):
    nbrs = NearestNeighbors(n_neighbors=1).fit(boundary)
    dists, indices = nbrs.kneighbors(target)
    neis = np.array([boundary[i[0]] for i in indices])
    dirs = (target - neis) / (dists + 1e-8)
    return dists.reshape(-1), dirs


def get_normal(unstructured_grid_data):
    poly_data, surface_filter = unstructured_grid_data_to_poly_data(unstructured_grid_data)
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(poly_data)
    normal_filter.SetAutoOrientNormals(1)
    normal_filter.SetConsistency(1)
    normal_filter.SetComputeCellNormals(1)
    normal_filter.SetComputePointNormals(0)
    normal_filter.Update()

    unstructured_grid_data.GetCellData().SetNormals(normal_filter.GetOutput().GetCellData().GetNormals())
    c2p = vtk.vtkCellDataToPointData()
   
    c2p.SetInputData(unstructured_grid_data)
    c2p.Update()
    unstructured_grid_data = c2p.GetOutput()
    
    normal = vtk_to_numpy(c2p.GetOutput().GetPointData().GetNormals()).astype(np.double)
    
    normal /= (np.max(np.abs(normal), axis=1, keepdims=True) + 1e-8)
    normal /= (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8)
    if np.isnan(normal).sum() > 0:
        print(np.isnan(normal).sum())
        print("recalculate")
        return get_normal(unstructured_grid_data)  # re-calculate
   
    return normal

def visualize_poly_data(poly_data, surface_filter, normal_filter=None):
    if normal_filter is not None:
        mask = vtk.vtkMaskPoints()
        mask.SetInputData(normal_filter.GetOutput())
        # mask.RandomModeOn()
        mask.Update()
        arrow = vtk.vtkArrowSource()
        arrow.Update()
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(mask.GetOutput())
        glyph.SetSourceData(arrow.GetOutput())
        glyph.SetVectorModeToUseNormal()
        glyph.SetScaleFactor(0.1)
        glyph.Update()
        norm_mapper = vtk.vtkPolyDataMapper()
        norm_mapper.SetInputData(normal_filter.GetOutput())
        glyph_mapper = vtk.vtkPolyDataMapper()
        glyph_mapper.SetInputData(glyph.GetOutput())
        norm_actor = vtk.vtkActor()
        norm_actor.SetMapper(norm_mapper)
        glyph_actor = vtk.vtkActor()
        glyph_actor.SetMapper(glyph_mapper)
        glyph_actor.GetProperty().SetColor(1, 0, 0)
        norm_render = vtk.vtkRenderer()
        norm_render.AddActor(norm_actor)
        norm_render.SetBackground(0, 1, 0)
        glyph_render = vtk.vtkRenderer()
        glyph_render.AddActor(glyph_actor)
        glyph_render.AddActor(norm_actor)
        glyph_render.SetBackground(0, 0, 1)

    scalar_range = poly_data.GetScalarRange()

    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(surface_filter.GetOutputPort())
    mapper.SetScalarRange(scalar_range)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)  # Set background to white

    renderer_window = vtk.vtkRenderWindow()
    renderer_window.AddRenderer(renderer)
    if normal_filter is not None:
        renderer_window.AddRenderer(norm_render)
        renderer_window.AddRenderer(glyph_render)
    renderer_window.Render()

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderer_window)
    interactor.Initialize()
    interactor.Start()



def compute_curvature_torch(vertices_np, k=10, device=device):
    vertices = torch.tensor(vertices_np, dtype=torch.float32, device='cpu')  # [N, 3]
    N = vertices.shape[0]
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(vertices_np)
    _, indices_np = knn.kneighbors(vertices_np)  # [N, k]
    indices = torch.tensor(indices_np, device='cpu')
    neighbors = vertices[indices]
    mean = neighbors.mean(dim=1, keepdim=True)         # [N, 1, 3]
    centered = neighbors - mean                        # [N, k, 3]
    cov = torch.matmul(centered.transpose(1, 2), centered) / (k - 1)
    eigvals = torch.linalg.eigvalsh(cov)
    curvatures = eigvals[:, 0]  # [N,]
    return curvatures.detach().cpu().numpy()


# 计算每个点的局部密度（邻居的平均距离）
def compute_density(vertices, k=10):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(vertices)
    distances, _ = knn.kneighbors(vertices)
    density = 1 / (distances[:, 1:].mean(axis=1) + 1e-6)
    return density


# 曲率采样
def sample_points(vertices, vert_normal, curvatures, sampled_points):
    weight = np.exp(curvatures)
    weight /= weight.sum() 
    indices = np.random.choice(len(vertices), size=sampled_points, p=weight)
    sampled_vertices = vertices[indices]
    sampled_normals =vert_normal[indices]
    sampled_curvatures = curvatures[indices]
    return sampled_vertices, sampled_normals, sampled_curvatures, indices



def get_datalist(root, samples, metadata_file, split_file,norm=False, coef_norm=None, savedir=None, preprocessed=False):
    start_time=time.time()
    # 加载元数据
    filenames= load_metadata(metadata_file)
    # 获取划分索引
    split_indices = load_split(split_file, filenames)
    dataset = []
    mean_in, mean_out = 0, 0
    std_in, std_out = 0, 0
    # for k, s in enumerate(samples):
    for idx, s in enumerate(split_indices):
        print(f"[INFO] 正在读取/处理第 {idx + 1}/{len(split_indices)} 个样本: {s}")

        if preprocessed and savedir is not None:
            save_path = os.path.join(savedir, s)
            if not os.path.exists(save_path):
                continue

            init = np.load(os.path.join(save_path, 'x.npy'))  # [N_points, 7]
            target = np.load(os.path.join(save_path, 'y.npy'))  # [N_points, 1]
            pos = np.load(os.path.join(save_path, 'pos.npy'))  # [N_points, 3]
            surf = np.load(os.path.join(save_path, 'surf.npy'))  # [N_points]
            faces = np.load(os.path.join(save_path, 'faces.npy'))  
            
        else:
          
            file_name_press = os.path.join(root, f'{s}.vtk')
            if not os.path.exists(file_name_press):
                print(f"VTK file {file_name_press} not found, skipping...")
                continue

            # 加载 VTK 文件（表面网格）
            time_1=time.time()
            unstructured_grid_data_press = load_unstructured_grid_data_binary(file_name_press)
            # 从 VTK 数据中提取出所有点坐标，并转成 NumPy 数组
            points_press = vtk_to_numpy(unstructured_grid_data_press.GetPoints().GetData())
            time_2=time.time()
            print(f'读取点坐标：{time_2-time_1}')
            point_data = unstructured_grid_data_press.GetPointData()
            scalars = point_data.GetArray("Pressure")
            if scalars is None:
                raise ValueError(f"No scalar field named 'Pressure' found in VTK file.")
            press = vtk_to_numpy(scalars)
            time_3=time.time()
            print(f'读取压力：{time_3-time_2}')
            # 计算SDF和法向量
            time_4=time.time()
            sdf_press = np.zeros(points_press.shape[0])
            time_5=time.time()
            print(f'计算SDF：{time_5-time_4}')
            normal_press = get_normal(unstructured_grid_data_press)
            time_6=time.time()
            print(f'法向量计算：{time_6-time_5}')
            # 输入和目标
            pos = points_press  # [N_points, 3]
            surf = np.ones(points_press.shape[0]) # [N_points], 全为表面点
            time_65=time.time()
            print(f'输入点：{time_65-time_6}')
            faces = get_faces(unstructured_grid_data_press, cell_size=3)
            time_7=time.time()
            print(f'获取面：{time_7-time_65}')
            ###拼接
            init = np.c_[pos, sdf_press, normal_press]  # [N_points, 7]
            target = press.reshape(-1, 1)  # [N_points, 1], 压力
            time_9=time.time()

            if savedir is not None:
                save_path = os.path.join(savedir, s)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                np.save(os.path.join(save_path, 'x.npy'), init)
                np.save(os.path.join(save_path, 'y.npy'), target)
                np.save(os.path.join(save_path, 'pos.npy'), pos)
                np.save(os.path.join(save_path, 'surf.npy'), surf)
                np.save(os.path.join(save_path, 'faces.npy'), faces)  
    

        # 转换为张量
        surf = torch.tensor(surf, dtype=torch.float32)
        pos = torch.tensor(pos, dtype=torch.float32)
        x = torch.tensor(init, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)

        if norm and coef_norm is None:
            if idx == 0:
                old_length = init.shape[0]
                mean_in = init.mean(axis=0)
                mean_out = target.mean(axis=0)
            else:
                new_length = old_length + init.shape[0]
                mean_in += (init.sum(axis=0) - init.shape[0] * mean_in) / new_length
                mean_out += (target.sum(axis=0) - init.shape[0] * mean_out) / new_length
                old_length = new_length
        data = Data(pos=pos, x=x, y=y, surf=surf.bool())
        dataset.append(data)
  

    if norm and coef_norm is None:
        for k, data in enumerate(dataset):
            if k == 0:
                old_length = data.x.numpy().shape[0]
                std_in = ((data.x.numpy() - mean_in) ** 2).sum(axis=0) / old_length
                std_out = ((data.y.numpy() - mean_out) ** 2).sum(axis=0) / old_length
            else:
                new_length = old_length + data.x.numpy().shape[0]
                std_in += (((data.x.numpy() - mean_in) ** 2).sum(axis=0) - data.x.numpy().shape[
                    0] * std_in) / new_length
                std_out += (((data.y.numpy() - mean_out) ** 2).sum(axis=0) - data.x.numpy().shape[
                    0] * std_out) / new_length
                old_length = new_length

        std_in = np.sqrt(std_in)
        std_out = np.sqrt(std_out)

        for data in dataset:
            data.x = ((data.x - mean_in) / (std_in + 1e-8)).float()
            data.y = ((data.y - mean_out) / (std_out + 1e-8)).float()

        coef_norm = (mean_in, std_in, mean_out, std_out)
        dataset = (dataset, coef_norm)

    elif coef_norm is not None:
        for data in dataset:
            data.x = ((data.x - coef_norm[0]) / (coef_norm[1] + 1e-8)).float()
            data.y = ((data.y - coef_norm[2]) / (coef_norm[3] + 1e-8)).float()

    end_time=time.time()
    print(f'数据处理总耗时:{end_time-start_time}')

    return dataset

def get_faces(unstructured_grid_data, cell_size=3):
    polys = vtk_to_numpy(unstructured_grid_data.GetPolys().GetData())
    faces = polys.reshape(-1, cell_size + 1)[:, 1:]   # 去掉首列“cell_size”
    return faces


def get_edges(unstructured_grid_data, points, cell_size=4):
    edge_indeces = set()
    polys = vtk_to_numpy(unstructured_grid_data.GetPolys().GetData())
    num_points_per_cell = cell_size + 1 
    if len(polys) % num_points_per_cell != 0:
        print(f"Warning: The data cannot be evenly reshaped with cell_size={cell_size}")
        valid_cells = len(polys) - len(polys) % num_points_per_cell
        polys = polys[:valid_cells]
    cells = polys.reshape(-1, num_points_per_cell)
    for i in range(len(cells)):
        for j, k in itertools.product(range(1, cell_size + 1), repeat=2):
            edge_indeces.add((cells[i][j], cells[i][k]))
            edge_indeces.add((cells[i][k], cells[i][j]))
    edges = [[], []]
    for u, v in edge_indeces:
        edges[0].append(tuple(points[u]))
        edges[1].append(tuple(points[v]))
    return edges



def get_edge_index(pos, edges_press):
    indices = {tuple(pos[i]): i for i in range(len(pos))}
    edges = set()
    for i in range(len(edges_press[0])):
        edges.add((indices[edges_press[0][i]], indices[edges_press[1][i]]))
    edge_index = np.array(list(edges)).T
    return edge_index


def get_induced_graph(data, idx, num_hops):
    subset, sub_edge_index, _, _ = k_hop_subgraph(node_idx=idx, num_hops=num_hops, edge_index=data.edge_index,
                                                  relabel_nodes=True)
    return Data(x=data.x[subset], y=data.y[idx], edge_index=sub_edge_index)


def pc_normalize(pc):
    centroid = torch.mean(pc, axis=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def get_shape(data, max_n_point=8192, normalize=True, use_height=False):
    surf_indices = torch.where(data.surf)[0].tolist()

    if len(surf_indices) > max_n_point:
        surf_indices = np.array(random.sample(range(len(surf_indices)), max_n_point))

    shape_pc = data.pos[surf_indices].clone()

    if normalize:
        shape_pc = pc_normalize(shape_pc)

    if use_height:
        gravity_dim = 1
        height_array = shape_pc[:, gravity_dim:gravity_dim + 1] - shape_pc[:, gravity_dim:gravity_dim + 1].min()
        shape_pc = torch.cat((shape_pc, height_array), axis=1)

    return shape_pc


def create_edge_index_radius(data, r, max_neighbors=32):
    data.edge_index = nng.radius_graph(x=data.pos, r=r, loop=True, max_num_neighbors=max_neighbors)
    return data


class GraphDataset(Dataset):
    def __init__(self, datalist, use_height=False, use_cfd_mesh=True, r=None):
        super().__init__()
        self.datalist = datalist
        self.use_height = use_height

    def len(self):
        return len(self.datalist)

    def get(self, idx):
        data = self.datalist[idx]
        shape = get_shape(data, use_height=self.use_height)
        return self.datalist[idx], shape

