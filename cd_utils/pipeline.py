from .model import phsoffNet
import torch
import time
import numpy as np
from .dataLoader import DrivAerDataset_stl4pre_random_cd
from .config import config
import pandas as pd
import random
import asyncio
from typing import List, Dict, Any, Union
import os

def set_random_seed(seed=42):
    """设置所有相关库的随机种子"""
    pass
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # # 确保使用确定性算法
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def predict_single_stl(stl_filename: str, model_path: str, data_dir: str = None, dim=384, random_seed=random.randint) -> Dict[str, Any]:
    """
    仅预测单个 STL 的 Cd（不读取 CSV、不计算误差）。

    返回: { filename, predicted_value, inference_time }
    """
    #set_random_seed(random_seed)

    # 数据集（不读取 CSV 标签）
    # 支持 stl_filename 为绝对路径，此时 data_dir 可为 None
    if os.path.isabs(stl_filename):
        dataset = DrivAerDataset_stl4pre_random_cd(
            data_dir=None,
            filename_list=[stl_filename],
            random_seed=random_seed,
            use_csv_labels=False
        )
    else:
        dataset = DrivAerDataset_stl4pre_random_cd(
            data_dir=data_dir,
            filename_list=[stl_filename] if stl_filename else None,
            random_seed=random_seed,
            use_csv_labels=False
        )

    # 处理单个 STL
    x, cubesize, filename = dataset.process_stl_file_no_label(stl_filename)

    # 增加 batch 维
    x = x.unsqueeze(0)
    cubesize = cubesize.unsqueeze(0)

    # 模型
    model = phsoffNet(dim)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(state_dict, dict) and state_dict and 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict):
        state_dict = state_dict['state_dict']
    if isinstance(state_dict, dict) and state_dict and 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"[warn] 严格加载权重失败，尝试非严格加载: {e}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[warn] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    x, cubesize = x.to(device), cubesize.to(device)

    with torch.no_grad():
        start = time.time()
        output = model(x, cubesize)
        inference_time = time.time() - start

    pred = output.squeeze().detach().cpu().item()
    return {
        'filename': filename,
        'predicted_value': pred,
        'inference_time': inference_time
    }

def predict_multiple_stl(stl_filenames: List[str], model_path: str, data_dir: str = None, dim=384, random_seed=42) -> Dict[str, Any]:
    """
    批量预测 STL 的 Cd（不读取 CSV、不计算误差）。

    返回: { total_files, successful_files, failed_files, total_inference_time, avg_inference_time, individual_results[] }
    individual_results: { filename, predicted_value, inference_time, error? }
    """
    #set_random_seed(random_seed)
    
    if isinstance(stl_filenames, str):
        # 这里stl_filenames为一个字符串列表
        stl_filenames = [stl_filenames]
        

    all_results: List[Dict[str, Any]] = []
    total_inference_time = 0.0

    # 数据集与模型（不读取 CSV 标签）
    dataset = DrivAerDataset_stl4pre_random_cd(
        data_dir=data_dir,
        filename_list=stl_filenames,
        random_seed=random_seed,
        use_csv_labels=False
    )
    model = phsoffNet(dim)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(state_dict, dict) and state_dict and 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict):
        state_dict = state_dict['state_dict']
    if isinstance(state_dict, dict) and state_dict and 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"[warn] 严格加载权重失败，尝试非严格加载: {e}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[warn] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    
    for i, stl_filename in enumerate(stl_filenames):
        try:
            # print(f"处理({i+1}/{len(stl_filenames)}): {stl_filename}")
            x, cubesize, filename = dataset.process_stl_file_no_label(stl_filename)
            x = x.unsqueeze(0).to(device)
            cubesize = cubesize.unsqueeze(0).to(device)
            with torch.no_grad():
                start = time.time()
                output = model(x, cubesize)
                inf_t = time.time() - start
                total_inference_time += inf_t
            pred = output.squeeze().detach().cpu().item()
            # if pred > 0.4:
            #     pred = np.random.uniform(0.28, 0.36)
            all_results.append({
                'filename': filename,
                'predicted_value': pred,
                'inference_time': inf_t
            })
        except Exception as e:
            print(f"处理文件 {stl_filename} 出错: {e}")
            all_results.append({
                'filename': stl_filename,
                'error': str(e)
            })

    valid_results = [r for r in all_results if 'error' not in r]
    summary = {
        'total_files': len(stl_filenames),
        'successful_files': len(valid_results),
        'failed_files': len(stl_filenames) - len(valid_results),
        'total_inference_time': total_inference_time,
        'avg_inference_time': (total_inference_time / len(valid_results)) if valid_results else 0.0,
        'individual_results': all_results
    }
    return summary

def print_prediction_result(result: Dict[str, Any]):
    """打印单个文件预测结果"""
    print(f"文件名: {result['filename']}")
    if 'error' in result:
        print(f"错误: {result['error']}")
        return
    print(f"预测Cd: {result['predicted_value']:.6f}")
    print(f"推理时间: {result['inference_time']:.6f} 秒")

def print_multiple_prediction_results(summary: Dict[str, Any]):
    """打印多个文件预测结果汇总"""
    print("=" * 60)
    print("批量预测结果汇总")
    print("=" * 60)
    print(f"总文件数: {summary['total_files']}")
    print(f"成功处理: {summary['successful_files']}")
    print(f"处理失败: {summary['failed_files']}")
    print("-" * 40)
    print(f"总推理时间: {summary['total_inference_time']:.6f} 秒")
    print(f"平均推理时间: {summary['avg_inference_time']:.6f} 秒")
    print("=" * 60)
    
    # 打印每个文件的详细结果
    print("\n详细结果:")
    for i, result in enumerate(summary['individual_results']):
        print(f"\n文件 {i+1}: {result.get('filename','?')}")
        if 'error' in result:
            print(f"  错误: {result['error']}")
        else:
            print(f"  预测Cd: {result['predicted_value']:.6f}")
            print(f"  推理时间: {result['inference_time']:.6f} 秒")

def save_predictions_to_csv(summary: Dict[str, Any], output_path: str):
    """将预测结果保存到CSV文件（仅包含文件名、预测值、推理时间、错误信息）。"""
    rows = []
    for r in summary['individual_results']:
        rows.append({
            'filename': r.get('filename'),
            'predicted_value': r.get('predicted_value', None),
            'inference_time': r.get('inference_time', None),
            'error': r.get('error', None),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"{output_path}")



def predict_cd_value(file_list=None, model_path: str = None, data_dir: str = None, dim: int = None, random_seed: int = 2, save_csv: str = None) -> Dict[str, Any]:
    """
    对多个 STL 文件进行 Cd 预测（不读取 CSV、无真实值对比）。

    Args:
        file_list: STL 文件名（相对 data_dir）列表
        model_path: 模型权重路径，默认取 config['paths']['model_path']
        data_dir: STL 目录，默认取 config['paths']['stl_path']
        dim: 模型维度，默认取 config['model']['dim']
        random_seed: 随机种子
        save_csv: 可选，保存预测结果的 CSV 路径

    Returns:
        summary 字典，见 predict_multiple_stl
    """
    #set_random_seed(random_seed)
    model_path = model_path or config['paths']['model_path']
    #data_dir = data_dir or config['paths']['stl_path']
    data_dir = data_dir
    dim = dim or config['model']['dim']
    save_csv = save_csv or config['paths']['output']

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"模型权重不存在: {model_path}. 请传入有效的 model_path 参数或在 config['paths']['model_path'] 中配置正确路径。")
    # 只在 data_dir 不为 None 时检查目录
    if data_dir is not None and not os.path.isdir(data_dir):
        raise ValueError(f"data_dir {data_dir} 不是有效目录")
    
    

    summary = predict_multiple_stl(file_list, model_path=model_path, data_dir=data_dir, dim=dim, random_seed=random_seed)
    if save_csv:
        save_predictions_to_csv(summary, save_csv)
    return summary

def evaluate_cd_value(file_list: List[str]) -> Dict[str, Any]:
    """
    兼容旧接口：评估多个 STL 文件的 Cd（无真实值、不会读取 CSV）。
    等价于调用 predict_cd_value(file_list)。
    """
    return predict_cd_value(file_list)

if __name__ == "__main__":
    # 示例：批量预测（请确保 config['paths'] 中 stl_path 与 model_path 指向可用路径）
    files = [
        "E_S_WW_WM_001.stl", "E_S_WW_WM_002.stl", "E_S_WW_WM_003.stl"
    ]
    summary = predict_cd_value(files, save_csv=None)
    print_multiple_prediction_results(summary)
    