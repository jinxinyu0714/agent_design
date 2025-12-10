import os
import sys
import torch
import numpy as np
import time
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from .Transolver import Model
import types
import pickle


def load_unstructured_grid_data_binary(file_name):
    """加载VTK文件"""
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(file_name)
    reader.ReadAllScalarsOn()
    reader.Update()
    output = reader.GetOutput()
    return output

def unstructured_grid_data_to_poly_data(unstructured_grid_data):
    """将非结构化网格转换为多边形数据"""
    filter = vtk.vtkDataSetSurfaceFilter()
    filter.SetInputData(unstructured_grid_data)
    filter.Update()
    poly_data = filter.GetOutput()
    return poly_data, filter

def get_normal(unstructured_grid_data):
    """计算法向量 - 严格按照dataset.py的方式"""
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
    
    # 检查法向量是否成功计算
    normals_array = c2p.GetOutput().GetPointData().GetNormals()
    if normals_array is None:
       # print("⚠️  法向量计算失败，使用默认法向量")
        # 创建默认法向量 (z方向)
        num_points = c2p.GetOutput().GetNumberOfPoints()
        normal = np.zeros((num_points, 3))
        normal[:, 2] = 1.0  # z方向向上
        return normal
    
    normal = vtk_to_numpy(normals_array).astype(np.double)
    
    # 严格按照dataset.py的归一化方式
    normal /= (np.max(np.abs(normal), axis=1, keepdims=True) + 1e-8)
    normal /= (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8)
    
    # 检查NaN并重新计算（如原始代码）
    if np.isnan(normal).sum() > 0:
        #print(f"检测到{np.isnan(normal).sum()}个NaN值，使用默认法向量")
        # 如果归一化失败，使用默认法向量
        num_points = normal.shape[0]
        normal = np.zeros((num_points, 3))
        normal[:, 2] = 1.0
   
    return normal

def get_faces(unstructured_grid_data, cell_size=3):
    """提取面信息 - 改进版，支持不同VTK格式"""
    try:
        # 首先尝试原始dataset.py的方式
        polys = vtk_to_numpy(unstructured_grid_data.GetPolys().GetData())
        
        # 检查是否能正确reshape
        expected_shape = cell_size + 1
        if polys.size % expected_shape == 0:
            faces = polys.reshape(-1, expected_shape)[:, 1:]   # 去掉首列"cell_size"
            return faces
        else:
            pass
            # print(f"ℹ️  VTK格式不标准，使用兼容方法提取面信息...")
            
    except Exception:
        # print(f"ℹ️  使用兼容方法提取面信息...")
        pass
    # 替代方法：直接从PolyData提取面
    try:
        # 转换为PolyData
        poly_data, surface_filter = unstructured_grid_data_to_poly_data(unstructured_grid_data)
        
        faces = []
        num_cells = poly_data.GetNumberOfCells()
        
        for i in range(num_cells):
            cell = poly_data.GetCell(i)
            if cell.GetNumberOfPoints() == cell_size:
                face = [cell.GetPointId(j) for j in range(cell_size)]
                faces.append(face)
        
        if len(faces) > 0:
            return np.array(faces)
        else:
            print("⚠️  未找到匹配的面，返回空数组")
            return np.array([]).reshape(0, cell_size)
        
    except Exception as e:
        print(f"⚠️  面信息提取失败: {e}")
        return np.array([]).reshape(0, cell_size)

def save_vtk_with_prediction(filename, original_vtk_path, predicted_pressure):
    """
    在原VTK文件基础上添加预测压力字段
    
    Args:
        filename: 输出文件名
        original_vtk_path: 原始VTK文件路径
        predicted_pressure: 预测的压力值数组
    """
    # print(f"🔄 正在保存预测结果到: {filename}")
    
    # 读取原始VTK文件
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(original_vtk_path)
    reader.ReadAllScalarsOn()
    reader.Update()
    original_data = reader.GetOutput()
    
    # 复制原始数据
    output_data = original_data.NewInstance()
    output_data.DeepCopy(original_data)
    
    # 添加预测压力字段
    pressure_array = numpy_to_vtk(predicted_pressure.astype(np.float32))
    pressure_array.SetName("pressure_prediction")  # 使用指定的字段名
    
    # 添加到点数据中
    output_data.GetPointData().AddArray(pressure_array)
    
    # 如果没有其他标量数据，将预测压力设为活动标量
    if output_data.GetPointData().GetScalars() is None:
        output_data.GetPointData().SetScalars(pressure_array)
    
    # 保存文件
    writer = vtk.vtkGenericDataObjectWriter()
    writer.SetFileName(filename)
    writer.SetInputData(output_data)
    writer.Write()
    
    return filename
    # print(f"   - 保持了原始VTK文件的所有数据")
    # print(f"   - 添加了 'pressure' 字段（预测值）")
    # print(f"   - 压力值已反归一化到真实物理量")

def load_vtk_data(vtk_file_path):
    """
    从VTK文件加载数据 - 严格按照dataset.py的数据处理流程
    
    Returns:
        pos: 点坐标 [N, 3]
        x: 特征向量 [N, 7] (pos + sdf + normal)
        y: 目标压力值 [N, 1] (如果存在)
        faces: 面信息
        surf: 表面掩码
        has_pressure: 是否包含压力数据
        coef_norm: 归一化系数 (从当前文件计算得出)
    """
    # print(f"📂 正在加载VTK文件: {vtk_file_path}")
    
    if not os.path.exists(vtk_file_path):
        raise FileNotFoundError(f"VTK文件不存在: {vtk_file_path}")
    
    # 按照dataset.py的方式加载VTK数据
    time_1 = time.time()
    unstructured_grid_data_press = load_unstructured_grid_data_binary(vtk_file_path)
    
    # 从VTK数据中提取出所有点坐标，并转成NumPy数组
    points_press = vtk_to_numpy(unstructured_grid_data_press.GetPoints().GetData())
    time_2 = time.time()
    # print(f'读取点坐标：{time_2-time_1:.4f}秒，点数: {points_press.shape[0]}')
    
    # 读取压力数据
    point_data = unstructured_grid_data_press.GetPointData()
    scalars = point_data.GetArray("Pressure")
    has_pressure = False
    press = None
    
    if scalars is not None:
        press = vtk_to_numpy(scalars)
        has_pressure = True
        time_3 = time.time()
        # print(f'读取压力：{time_3-time_2:.4f}秒')
        # print(f"✅ 检测到压力数据，范围: [{press.min():.6f}, {press.max():.6f}]")
    else:
        # print(f"ℹ️  未检测到压力数据，将进行纯预测")
        time_3 = time.time()
    
    # 计算SDF和法向量 - 按照dataset.py的方式
    time_4 = time.time()
    sdf_press = np.zeros(points_press.shape[0])  # 表面网格，SDF设为0
    time_5 = time.time()
    # print(f'计算SDF：{time_5-time_4:.4f}秒')
    
    # print(f"🔄 正在计算法向量...")
    normal_press = get_normal(unstructured_grid_data_press)
    time_6 = time.time()
    # print(f'法向量计算：{time_6-time_5:.4f}秒')
    
    # 输入和目标 - 按照dataset.py的方式
    pos = points_press  # [N_points, 3]
    surf = np.ones(points_press.shape[0])  # [N_points], 全为表面点
    time_65 = time.time()
    # print(f'输入点：{time_65-time_6:.4f}秒')
    
    # print(f"🔄 正在提取面信息...")
    faces = get_faces(unstructured_grid_data_press, cell_size=3)
    time_7 = time.time()
    # print(f'获取面：{time_7-time_65:.4f}秒，面数: {faces.shape[0]}')
    
    # 拼接特征 - 按照dataset.py的方式
    init = np.c_[pos, sdf_press, normal_press]  # [N_points, 7]
    target = press.reshape(-1, 1) if press is not None else None  # [N_points, 1], 压力
    
    # 计算归一化系数 - 按照dataset.py的方式
    coef_norm = None
    if has_pressure and scalars is not None:
        # print(f"📊 正在计算归一化系数...")
        
        # 输入特征归一化
        mean_in = init.mean(axis=0)  # [7,]
        std_in = init.std(axis=0) + 1e-8  # [7,] 避免除0
        
        # 输出压力归一化  
        mean_out = target.mean(axis=0)  # [1,]
        std_out = target.std(axis=0) + 1e-8  # [1,] 避免除0
        
        coef_norm = (mean_in, std_in, mean_out, std_out)
        
        # print(f"📊 归一化参数:")
        # print(f"   - 输入特征: 均值范围 [{mean_in.min():.4f}, {mean_in.max():.4f}]")
        # print(f"   - 输入特征: 标准差范围 [{std_in.min():.4f}, {std_in.max():.4f}]") 
        # print(f"   - 输出压力: 均值 {mean_out[0]:.4f}, 标准差 {std_out[0]:.4f}")
    
    # print(f"✅ 数据加载完成，总耗时: {time_7-time_1:.4f}秒")
    return pos, init, target, faces, surf, has_pressure, coef_norm

def load_model(model_path, device):
    """加载训练好的模型 - 兼容 PyTorch 2.6 的 weights_only 变更，并处理可能的模块路径。"""
    # print(f"🔄 正在加载模型: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 模型配置 (与原始项目保持一致)
    model_config = {
        'n_hidden': 256,
        'n_layers': 8,
        'space_dim': 7,
        'fun_dim': 0,
        'n_head': 8,
        'mlp_ratio': 2,
        'out_dim': 1,
        'slice_num': 32,
        'unified_pos': 0
    }

    # 创建空模型容器
    model = Model(**model_config).to(device)

    def alias_missing_modules():
        """为老检查点里引用的 modules.Transolver 建立别名，避免反序列化失败。"""
        from .Transolver import Model as _Model, MLP as _MLP, Transolver_block as _Block, Physics_Attention_Irregular_Mesh as _PAIM
        # 构造 models 与 models.Transolver 模块
        if 'models' not in sys.modules:
            sys.modules['models'] = types.ModuleType('models')
        if 'models.Transolver' not in sys.modules:
            transolver_module = types.ModuleType('models.Transolver')
            transolver_module.Model = _Model
            transolver_module.MLP = _MLP
            transolver_module.Transolver_block = _Block
            transolver_module.Physics_Attention_Irregular_Mesh = _PAIM
            sys.modules['models.Transolver'] = transolver_module
            # 也挂到父模块上，偶发环境会通过属性访问
            setattr(sys.modules['models'], 'Transolver', transolver_module)

    # 1) 首先尝试安全地只加载权重（若是 state_dict 保存）
    try:
        obj = torch.load(model_path, map_location=device)  # 在 PyTorch 2.6 默认 weights_only=True
    except (pickle.UnpicklingError, Exception) as e:
        msg = str(e)
        if 'Weights only load failed' in msg or 'Unsupported global' in msg:
            # 2) 退回到允许完整反序列化（需信任来源）
            # print("ℹ️  退回到 weights_only=False 以兼容旧检查点（仅在信任来源时使用）")
            try:
                alias_missing_modules()  # 可能需要的模块别名
                obj = torch.load(model_path, map_location=device, weights_only=False)
            except ModuleNotFoundError:
                alias_missing_modules()
                obj = torch.load(model_path, map_location=device, weights_only=False)
        else:
            raise

    # 处理不同保存格式
    if isinstance(obj, torch.nn.Module):
        # print("✅ 检测到直接保存的模型对象，采用该对象")
        model = obj.to(device)
        model.eval()
        return model

    if isinstance(obj, dict):
        state_dict = obj
        # 常见封装键
        for key in ('model_state_dict', 'state_dict'): 
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break

        # 兼容 DataParallel 前缀
        if state_dict and all(isinstance(k, str) for k in state_dict.keys()):
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

        # 加载到当前模型
        try:
            model.load_state_dict(state_dict, strict=True)
            # print("✅ 模型权重严格加载成功")
        except Exception as e:
            # print(f"⚠️  严格加载失败: {e}，尝试非严格加载…")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"   缺失键数量: {len(missing)} 示例: {missing[:5]}")
            if unexpected:
                print(f"   未预期键数量: {len(unexpected)} 示例: {unexpected[:5]}")
            # print("✅ 模型权重已部分加载")

        model.eval()
        return model

    # 兜底：未知格式
    raise RuntimeError("无法识别的模型文件格式（既不是 Module 也不是包含权重的字典）")

def predict_pressure(model, pos, x, surf, device, coef_norm=None):
    """预测压力场 - 严格按照dataset.py的数据格式"""
    # print(f"🔄 正在进行压力场预测...")
    
    # 转换为张量 - 按照dataset.py的方式
    surf_tensor = torch.tensor(surf, dtype=torch.float32)
    pos_tensor = torch.tensor(pos, dtype=torch.float32)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    
    # 如果提供了归一化系数，需要对输入进行归一化
    if coef_norm is not None:
        mean_in = coef_norm[0]  # 输入均值
        std_in = coef_norm[1]   # 输入标准差
        x_tensor = ((x_tensor - torch.tensor(mean_in)) / (torch.tensor(std_in) + 1e-8)).float()
        # print(f"✅ 已应用输入归一化")
    
    # 构建Data对象 - 按照dataset.py的格式
    from torch_geometric.data import Data
    
    data = Data(
        pos=pos_tensor, 
        x=x_tensor, 
        surf=surf_tensor.bool()
    ).to(device)
    
    # 构建几何形状 - 按照dataset.py中get_shape的方式
    def get_shape(data, max_n_point=8192, normalize=True):
        surf_indices = torch.where(data.surf)[0].tolist()
        
        if len(surf_indices) > max_n_point:
            import random
            surf_indices = np.array(random.sample(range(len(surf_indices)), max_n_point))
        
        shape_pc = data.pos[surf_indices].clone()
        
        if normalize:
            # pc_normalize函数
            centroid = torch.mean(shape_pc, axis=0)
            shape_pc = shape_pc - centroid
            m = torch.max(torch.sqrt(torch.sum(shape_pc ** 2, axis=1)))
            shape_pc = shape_pc / m
        
        return shape_pc
    
    geom = get_shape(data)
    
    # 预测
    with torch.no_grad():
        start_time = time.time()
        
        # 模型前向传播 - 按照原始评估脚本的方式
        output = model((data, geom))
        
        inference_time = time.time() - start_time
        # print(f"⏱️  推理时间: {inference_time:.4f} 秒")
    
    # 提取表面点预测结果 - 按照原始评估脚本的方式
    pred_pressure = output[data.surf, -1].cpu().numpy()
    
    # 反归一化 (如果提供了归一化系数)
    if coef_norm is not None:
        mean_out = coef_norm[2]  # 输出均值
        std_out = coef_norm[3]   # 输出标准差
        pred_pressure = pred_pressure * std_out[-1] + mean_out[-1]
        # print(f"✅ 已应用输出反归一化")
    
    # print(f"✅ 预测完成")
    # print(f"📊 预测压力范围: [{pred_pressure.min():.6f}, {pred_pressure.max():.6f}]")
    
    return pred_pressure, inference_time

def eval_pressure(input_vtk=None, output_vtk=None, model_path=None):
    """主函数"""
    coef_norm = [
        [
            1.357179175228297,
            0.000129043087459653,
            0.31800198033976484,
            0.0,
            -0.03849754638122156,
            9.781867553876436e-05,
            0.06872401083064958
        ],
        [
            1.5465418824572785,
            0.7671167429656915,
            0.4571207281599481,
            0.0,
            0.5040860059375848,
            0.6572056243997649,
            0.5547725026287595
        ],
        [
            -185.67654522992396
        ],
        [
            365.2276555973486
        ]
    ]
    
    
    input_vtk = input_vtk if input_vtk is not None else "examples/data/cylinder.vtk"
    model_path = model_path if model_path is not None else "examples/pretrained_models/transolver_cylinder.pth"
    output_vtk = output_vtk if output_vtk is not None else "examples/results/"

    # 防御性处理：避免传入 tuple/list 造成 os.path 操作报错
    if isinstance(input_vtk, (tuple, list)):
        input_vtk = input_vtk[0]
    if isinstance(output_vtk, (tuple, list)):
        output_vtk = output_vtk[0]
    if isinstance(model_path, (tuple, list)):
        model_path = model_path[0]

    # 生成输出文件名
    input_filename = os.path.basename(input_vtk)
    base_name = os.path.splitext(input_filename)[0]
    
    # 确保输出目录存在
    os.makedirs(output_vtk, exist_ok=True)
    
    # 生成输出文件路径
    output_prediction_file = os.path.join(output_vtk, f"{base_name}_with_prediction.vtk")
    output_comparison_file = os.path.join(output_vtk, f"{base_name}_with_truth.vtk")
    
    # print("="*60)
    # print("🚀 压力场预测脚本启动")
    # print("="*60)
    # print(f"📂 输入VTK: {input_vtk}")
    # print(f"🤖 模型路径: {model_path}")
    # print(f"📁 输出目录: {output_vtk}")
    # print(f"📄 预测文件: {base_name}_with_prediction.vtk")
    # print(f"📄 对比文件: {base_name}_with_truth.vtk (如有真实值)")
    # print("-"*60)
    
    # 设备配置
   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    
    try:
        # 1. 加载VTK数据 - 按照dataset.py的方式，同时计算归一化参数
        pos, x, gt_pressure, faces, surf, has_pressure, local_coef_norm = load_vtk_data(input_vtk)
        
        # 2. 确定使用哪个归一化参数
        # 优先使用外部提供的归一化参数，否则使用从当前文件计算的参数
        final_coef_norm = coef_norm if coef_norm is not None else local_coef_norm
        
        # if final_coef_norm is not None:
        #     print(f"✅ 使用归一化参数: {'外部提供' if coef_norm is not None else '当前文件计算'}")
        # else:
        #     print(f"⚠️  无归一化参数，预测结果将是归一化后的值")
        
        # 3. 加载模型
        model = load_model(model_path, device)
        
        # 4. 预测压力场
        pred_pressure, inference_time = predict_pressure(model, pos, x, surf, device, final_coef_norm)
        
        # 5. 保存预测结果 - 在原VTK文件基础上添加预测压力字段
        path = save_vtk_with_prediction(output_prediction_file, input_vtk, pred_pressure)
        
        # # 5. 输出统计信息
        # print("\n" + "="*60)
        # print("📊 预测统计信息")
        # print("="*60)
        # print(f"输入点数: {pos.shape[0]}")
        # print(f"面数: {faces.shape[0]}")
        # print(f"预测压力范围: [{pred_pressure.min():.6f}, {pred_pressure.max():.6f}]")
        # print(f"推理时间: {inference_time:.4f} 秒")
        
        # 如果有真实压力数据，计算误差
        if has_pressure and gt_pressure is not None:
            gt_pressure_denorm = gt_pressure.flatten()

            mse = np.mean((pred_pressure - gt_pressure_denorm) ** 2)
            mae = np.mean(np.abs(pred_pressure - gt_pressure_denorm))
            rel_l2_error = np.linalg.norm(pred_pressure - gt_pressure_denorm) / np.linalg.norm(gt_pressure_denorm)
            
            # print("\n📈 预测精度:")
            # print(f"MSE: {mse:.6e}")
            # print(f"MAE: {mae:.6e}")
            # print(f"相对L2误差: {rel_l2_error*100:.2f}%")
            
            # 保存对比文件：在原文件基础上添加真实值和误差字段
            gt_output = output_comparison_file
            
            # 读取原始文件并添加真实值和误差字段
            reader = vtk.vtkGenericDataObjectReader()
            reader.SetFileName(input_vtk)
            reader.Update()
            compare_data = reader.GetOutput().NewInstance()
            compare_data.DeepCopy(reader.GetOutput())
            
            # 添加预测值
            pred_array = numpy_to_vtk(pred_pressure.astype(np.float32))
            pred_array.SetName("pressure_predicted")
            compare_data.GetPointData().AddArray(pred_array)
            
            # 添加真实值
            truth_array = numpy_to_vtk(gt_pressure_denorm.astype(np.float32))
            truth_array.SetName("pressure_truth")
            compare_data.GetPointData().AddArray(truth_array)
            
            # 添加误差
            error_array = numpy_to_vtk((gt_pressure_denorm - pred_pressure).astype(np.float32))
            error_array.SetName("pressure_error")
            compare_data.GetPointData().AddArray(error_array)
            
            # 保存对比文件
            writer = vtk.vtkGenericDataObjectWriter()
            writer.SetFileName(gt_output)
            writer.SetInputData(compare_data)
            writer.Write()
            
        #     print(f"✅ 对比文件已保存到: {gt_output}")
        #     print(f"   - 包含字段: pressure_predicted, pressure_truth, pressure_error")
        
        # print("\n🎉 预测完成！")
        # print("="*60)
        return path
    except Exception as e:
        print(f"\n❌ 预测过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    input_vtk = '/student/jxy/agent_design/press_utils/data/vtk_in/F_S_WWS_WM_145_gt.vtk'
    output_vtk = '/student/jxy/agent_design/press_utils/data/output'
    model_path  = '/student/jxy/agent_design/press_utils/model_300.pth'
    eval_pressure(input_vtk=input_vtk, output_vtk=output_vtk, model_path=model_path)


