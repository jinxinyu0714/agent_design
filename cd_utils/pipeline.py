from .model import phsoffNet
import torch
import time
import numpy as np
from .dataLoader import DrivAerDataset_stl4pre_random_cd
from config import config
import pandas as pd
import random
import asyncio
from typing import List, Dict, Any, Union

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

def evaluate_single_stl(stl_filename, model_path, data_dir, config, dim=384, random_seed=42):
    """
    评估单个STL文件，计算完整的评估指标
    
    Args:
        stl_filename (str): STL文件名
        model_path (str): 模型权重路径
        data_dir (str): STL文件目录
        config: 配置对象
        dim (int): 模型维度
        random_seed (int): 随机种子
    
    Returns:
        dict: 包含所有评估指标的字典
    """
    # 设置随机种子
    set_random_seed(random_seed)
    
    # 初始化数据集和模型
    stlfile_dataset = DrivAerDataset_stl4pre_random_cd(data_dir, config, random_seed=random_seed)
    
    # 处理单个STL文件
    x, cubesize, cd_value, filename = stlfile_dataset.process_stl_file(stl_filename)
    
    # 增加batch维度
    x = x.unsqueeze(0)  # (1, N, 3)
    cubesize = cubesize.unsqueeze(0)  # (1, N, 3)
    cd_value = cd_value.unsqueeze(0)  # (1, 1)
    
    # 初始化模型
    model = phsoffNet(dim)
    
    # 加载模型权重
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 移除module前缀（如果存在）
    if 'module.' in list(state_dict.keys())[0]:
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    else:
        new_state_dict = state_dict
    
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = x.to(device)
    cubesize = cubesize.to(device)
    cd_value = cd_value.to(device)
    
    # 进行预测并计算时间
    with torch.no_grad():
        start_time = time.time()
        output = model(x, cubesize)
        inference_time = time.time() - start_time
    
    # 将数据移回CPU进行计算
    cd_value_cpu = cd_value.cpu().flatten()  # 确保是1维张量
    output_cpu = output.cpu().flatten()      # 确保是1维张量
    
    # 计算各种评估指标
    mask_threshold = 1e-4
    mask = (cd_value_cpu.abs() >= mask_threshold)
    
    if mask.sum() > 0:
        filtered_y = cd_value_cpu[mask]
        filtered_output = output_cpu[mask]
        
        # 绝对误差
        absolute_error = torch.abs(filtered_y - filtered_output)
        mae = absolute_error.mean().item()
        
        # 平方误差
        squared_error = (filtered_y - filtered_output) ** 2
        mse = squared_error.mean().item()
        
        # 相对误差
        relative_error = absolute_error / torch.clamp(filtered_y.abs(), min=1e-10)
        mre = relative_error.mean().item()
        max_re = relative_error.max().item()
        
        # R²计算
        if len(filtered_y) > 1:
            y_mean = filtered_y.mean()
            ss_tot = ((filtered_y - y_mean) ** 2).sum()
            ss_res = squared_error.sum()
            r2 = 1 - (ss_res / torch.clamp(ss_tot, min=1e-10))
            r2 = r2.item()
        else:
            # 单个样本无法计算R²
            r2 = 0.0
            
        # 获取单个值用于显示
        abs_error_value = absolute_error.item() if absolute_error.numel() == 1 else absolute_error.mean().item()
        rel_error_value = relative_error.item() if relative_error.numel() == 1 else relative_error.mean().item()
        
    else:
        mae = mse = mre = max_re = r2 = 0.0
        abs_error_value = rel_error_value = 0.0
    
    # 构建结果字典
    results = {
        'filename': filename,
        'true_value': cd_value_cpu.item(),
        'predicted_value': output_cpu.item(),
        'absolute_error': abs_error_value,
        'relative_error': rel_error_value,
        'mae': mae,
        'mse': mse,
        'mre': mre,
        'max_re': max_re,
        'r2': r2,
        'inference_time': inference_time
    }
    
    return results

def evaluate_multiple_stl(stl_filenames, model_path, data_dir, config, dim=384, random_seed=42):
    """
    评估多个STL文件，计算完整的评估指标
    
    Args:
        stl_filenames (list or str): STL文件名列表或单个文件名
        model_path (str): 模型权重路径
        data_dir (str): STL文件目录
        config: 配置对象
        dim (int): 模型维度
        random_seed (int): 随机种子
    
    Returns:
        dict: 包含所有评估指标的字典
    """
    # 设置随机种子
    set_random_seed(random_seed)
    
    # 如果输入是单个文件名，转换为列表
    if isinstance(stl_filenames, str):
        stl_filenames = [stl_filenames]
    
    all_results = []
    total_inference_time = 0.0
    
    # 初始化数据集和模型（只初始化一次）
    stlfile_dataset = DrivAerDataset_stl4pre_random_cd(data_dir, config, random_seed=random_seed)
    
    # 初始化模型
    model = phsoffNet(dim)
    
    # 加载模型权重
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 移除module前缀（如果存在）
    if 'module.' in list(state_dict.keys())[0]:
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    else:
        new_state_dict = state_dict
    
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"开始评估 {len(stl_filenames)} 个STL文件...")
    
    for i, stl_filename in enumerate(stl_filenames):
        try:
            print(f"正在处理 ({i+1}/{len(stl_filenames)}): {stl_filename}")
            
            # 处理STL文件
            x, cubesize, cd_value, filename = stlfile_dataset.process_stl_file(stl_filename)
            
            # 增加batch维度
            x = x.unsqueeze(0).to(device)
            cubesize = cubesize.unsqueeze(0).to(device)
            cd_value = cd_value.unsqueeze(0).to(device)
            
            # 进行预测并计算时间
            with torch.no_grad():
                start_time = time.time()
                output = model(x, cubesize)
                inference_time = time.time() - start_time
                total_inference_time += inference_time
            
            # 将数据移回CPU进行计算
            cd_value_cpu = cd_value.cpu().flatten()
            output_cpu = output.cpu().flatten()
            
            # 计算各种评估指标
            mask_threshold = 1e-4
            mask = (cd_value_cpu.abs() >= mask_threshold)
            
            if mask.sum() > 0:
                filtered_y = cd_value_cpu[mask]
                filtered_output = output_cpu[mask]
                
                # 绝对误差
                absolute_error = torch.abs(filtered_y - filtered_output)
                mae = absolute_error.mean().item()
                
                # 平方误差
                squared_error = (filtered_y - filtered_output) ** 2
                mse = squared_error.mean().item()
                
                # 相对误差
                relative_error = absolute_error / torch.clamp(filtered_y.abs(), min=1e-10)
                mre = relative_error.mean().item()
                max_re = relative_error.max().item()
                
                # 获取单个值用于显示
                abs_error_value = absolute_error.item() if absolute_error.numel() == 1 else absolute_error.mean().item()
                rel_error_value = relative_error.item() if relative_error.numel() == 1 else relative_error.mean().item()
                
            else:
                mae = mse = mre = max_re = 0.0
                abs_error_value = rel_error_value = 0.0
            
            # 构建结果字典
            result = {
                'filename': filename,
                'true_value': cd_value_cpu.item(),
                'predicted_value': output_cpu.item(),
                'absolute_error': abs_error_value,
                'relative_error': rel_error_value,
                'mae': mae,
                'mse': mse,
                'mre': mre,
                'max_re': max_re,
                'inference_time': inference_time
            }
            
            all_results.append(result)
            
        except Exception as e:
            print(f"处理文件 {stl_filename} 时出错: {e}")
            # 添加错误结果
            error_result = {
                'filename': stl_filename,
                'true_value': 0.0,
                'predicted_value': 0.0,
                'absolute_error': 0.0,
                'relative_error': 0.0,
                'mae': 0.0,
                'mse': 0.0,
                'mre': 0.0,
                'max_re': 0.0,
                'inference_time': 0.0,
                'error': str(e)
            }
            all_results.append(error_result)
    
    # 计算总体统计信息
    valid_results = [r for r in all_results if 'error' not in r]
    
    if valid_results:
        avg_mae = np.mean([r['mae'] for r in valid_results])
        avg_mse = np.mean([r['mse'] for r in valid_results])
        avg_mre = np.mean([r['mre'] for r in valid_results])
        max_re_overall = max([r['max_re'] for r in valid_results])
        
        # 计算整体R²
        true_values = [r['true_value'] for r in valid_results]
        pred_values = [r['predicted_value'] for r in valid_results]
        
        if len(true_values) > 1:
            true_mean = np.mean(true_values)
            ss_tot = np.sum([(y - true_mean)**2 for y in true_values])
            ss_res = np.sum([(true_values[i] - pred_values[i])**2 for i in range(len(true_values))])
            r2_overall = 1 - (ss_res / max(ss_tot, 1e-10))
        else:
            r2_overall = 0.0
    else:
        avg_mae = avg_mse = avg_mre = max_re_overall = r2_overall = 0.0
    
    # 汇总结果
    summary = {
        'total_files': len(stl_filenames),
        'successful_files': len(valid_results),
        'failed_files': len(stl_filenames) - len(valid_results),
        'avg_mae': avg_mae,
        'avg_mse': avg_mse,
        'avg_mre': avg_mre,
        'max_re_overall': max_re_overall,
        'r2_overall': r2_overall,
        'total_inference_time': total_inference_time,
        'avg_inference_time': total_inference_time / len(stl_filenames) if stl_filenames else 0.0,
        'individual_results': all_results
    }
    
    return summary

def print_evaluation_results(results):
    """打印单个文件评估结果"""
    print(f"文件名: {results['filename']}")
    print(f"真实值: {results['true_value']:.6f}")
    print(f"预测值: {results['predicted_value']:.6f}")
    print(f"绝对误差: {results['absolute_error']:.6f}")
    print(f"相对误差: {results['relative_error']:.6f}")
    print(f"平均绝对误差 (MAE): {results['mae']:.6f}")
    print(f"均方误差 (MSE): {results['mse']:.6f}")
    print(f"平均相对误差 (MRE): {results['mre']:.6f}")
    print(f"最大相对误差: {results['max_re']:.6f}")
    print(f"R² 分数: {results['r2']:.6f}")
    print(f"推理时间: {results['inference_time']:.6f} 秒")

def print_multiple_evaluation_results(summary):
    """打印多个文件评估结果汇总"""
    print("=" * 60)
    print("批量评估结果汇总")
    print("=" * 60)
    print(f"总文件数: {summary['total_files']}")
    print(f"成功处理: {summary['successful_files']}")
    print(f"处理失败: {summary['failed_files']}")
    print("-" * 40)
    print(f"平均MAE: {summary['avg_mae']:.6f}")
    print(f"平均MSE: {summary['avg_mse']:.6f}")
    print(f"平均MRE: {summary['avg_mre']:.6f}")
    print(f"最大相对误差: {summary['max_re_overall']:.6f}")
    print(f"整体R²: {summary['r2_overall']:.6f}")
    print(f"总推理时间: {summary['total_inference_time']:.6f} 秒")
    print(f"平均推理时间: {summary['avg_inference_time']:.6f} 秒")
    print("=" * 60)
    
    # 打印每个文件的详细结果
    print("\n详细结果:")
    for i, result in enumerate(summary['individual_results']):
        print(f"\n文件 {i+1}: {result['filename']}")
        if 'error' in result:
            print(f"  错误: {result['error']}")
        else:
            print(f"  真实值: {result['true_value']:.6f}")
            print(f"  预测值: {result['predicted_value']:.6f}")
            print(f"  相对误差: {result['relative_error']:.6f}")

def save_results_to_csv(summary, output_path):
    """将结果保存到CSV文件"""
    df = pd.DataFrame(summary['individual_results'])
    df.to_csv(output_path, index=False)
    print(f"结果已保存到: {output_path}")



def evaluate_cd_value(file_list: List[str]) -> Dict[str, Any]:
    """
    评估多个STL文件的CD值
    Args:
        file_list (List[str]): STL文件名列表
    
    Returns:
        Dict[str, Any]: 包含所有评估指标的字典
    """
    set_random_seed(42)
    results = evaluate_multiple_stl(file_list, model_path="/student/jxy/easy_agent/analy/cd_utils/best_model_mre_0.0209375933.pt",
        data_dir="/student/jxy/easy_agent/analy/cd_utils/data/stl_in",
        config=config,
        dim=384,
        random_seed=42)
    save_results_to_csv(results, "/student/jxy/easy_agent/analy/cd_utils/evaluation_results.csv")
    print_multiple_evaluation_results(results)
    return results

if __name__ == "__main__":
    result = evaluate_cd_value(file_list=["E_S_WW_WM_001.stl", "E_S_WW_WM_002.stl", "E_S_WW_WM_003.stl", "E_S_WW_WM_004.stl", "E_S_WW_WM_005.stl", "E_S_WW_WM_006.stl", "E_S_WW_WM_007.stl", "E_S_WW_WM_008.stl", "E_S_WW_WM_009.stl", "E_S_WW_WM_010.stl"])
    print_evaluation_results(result)