import os
import torch
import argparse
import numpy as np
import time
from torch import nn
from torch_geometric.loader import DataLoader
from load_dataset import load_test_split
from dataset import GraphDataset
import scipy as sc
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from datetime import datetime
from Transolver import Model 
import json

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/student/ysm/data/car-pressure/10wvtk/sampled_vtk/')###简化后vtk文件路径
parser.add_argument('--save_dir', default='/student/ysm/Transolver-main/press_main/proce_f_100_10w/')###缓存路径
parser.add_argument('--gpu', default=0, type=int)###使用的GPU编号
parser.add_argument('--test_split',default='/student/ysm/data/car-pressure/test_20.csv')   ###test样本名称csv
parser.add_argument('--metadata_file',default='/student/ysm/data/car-pressure/csv/DrivAerNet-pressure-8119.csv')  ###所有样本名称csv
parser.add_argument('--preprocessed', type=int, default=1 )   # 是否使用npy缓存,preprocessed =1 使用缓存
parser.add_argument('--model_path', type=str, default='/student/ysm/press_7.14_qirui/logdir/press_100_10w_ddpcs_Aug07_15-30-50/ckpt/model_epoch40_val_mse0.224362.pth')###测试的模型
parser.add_argument('--test_outpath', type=str, default='/student/ysm/press_7.14_qirui/output')###测试输出文件路径
args = parser.parse_args()
print(args)

model_config = {
    'n_hidden': 256,
    'n_layers': 8,
    'space_dim': 7,
    'fun_dim': 0,
    'n_head': 8,
    'mlp_ratio': 2,
    'out_dim': 1,
    'slice_num': 32,#32
    'unified_pos': 0
}

def save_vtk_with_faces(filename, pos, scalar, faces, name='pressure'):
    pts = vtk.vtkPoints()
    for p in pos: pts.InsertNextPoint(*p)
    polys = vtk.vtkCellArray()
    for face in faces:
        polys.InsertNextCell(len(face))
        for idx in face: polys.InsertCellPoint(int(idx))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts); poly.SetPolys(polys)
    arr = numpy_to_vtk(scalar.astype(np.float32))
    arr.SetName(name)
    poly.GetPointData().AddArray(arr)
    poly.GetPointData().SetActiveScalars(name)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(poly)
    writer.Write()


n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')

test_data, coef_norm, vallst = load_test_split(args, preprocessed=args.preprocessed)
test_ds = GraphDataset(test_data)



model = Model(**model_config).to(device)
state_dict = torch.load(args.model_path, map_location=device)
if all(k.startswith('module.') for k in state_dict.keys()):
    state_dict = {k[7:]: v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

test_loader = DataLoader(test_ds, batch_size=1)
model_filename = os.path.splitext(os.path.basename(args.model_path))[0]

timestamp = datetime.now().strftime('%m-%d_%H-%M')
save_root = os.path.join(args.test_outpath, f'{model_filename}_{timestamp}')
os.makedirs(save_root, exist_ok=True)
vtk_dir = os.path.join(save_root, 'vtk')
os.makedirs(vtk_dir, exist_ok=True)
subdir = os.path.join(save_root, 'gt_pred_pos')
os.makedirs(subdir, exist_ok=True)
with torch.no_grad():
    model.eval()
    per_sample_metrics = []
    criterion_func = nn.MSELoss(reduction='none')
    l2errs_press = []
    mses_press = []
    l1errs=[]
    maes_norm=[]
    times = []
    max_aes_norm=[]
    index = 0
    for cfd_data, geom in test_loader:
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        tic = time.time()
        out = model((cfd_data, geom))
        toc = time.time()
        targets = cfd_data.y

        if coef_norm is not None:
            mean = torch.tensor(coef_norm[2]).to(device)
            std = torch.tensor(coef_norm[3]).to(device)
            pred_press = out[cfd_data.surf, -1] * std[-1] + mean[-1]
            gt_press = targets[cfd_data.surf, -1] * std[-1] + mean[-1]
            out_denorm = out * std + mean
            y_denorm = targets * std + mean

        sample_name = os.path.splitext(os.path.basename(vallst[index]))[0]

        np.save(os.path.join(subdir, f"{sample_name}_pos.npy"), cfd_data.pos.cpu().numpy())
        np.save(os.path.join(subdir, f"{sample_name}_pred.npy"), out_denorm.detach().cpu().numpy())###预测值
        np.save(os.path.join(subdir, f"{sample_name}_gt.npy"), y_denorm.detach().cpu().numpy())###真实值

        pred_np = out_denorm.detach().cpu().numpy()
        gt_np = y_denorm.detach().cpu().numpy()
        # pos_np = cfd_data.pos.cpu().numpy()
        pos_np = np.load(os.path.join(args.save_dir, sample_name,"pos.npy"))
        faces = np.load(os.path.join(args.save_dir, sample_name,"faces.npy"))


        save_vtk_with_faces(os.path.join(vtk_dir, f'{sample_name}_pred.vtk'),
                            pos_np, pred_np[:, -1], faces, name='pred_pressure')
        save_vtk_with_faces(os.path.join(vtk_dir, f'{sample_name}_gt.vtk'),
                            pos_np, gt_np[:, -1], faces, name='gt_pressure')
        save_vtk_with_faces(os.path.join(vtk_dir, f'{sample_name}_diff.vtk'),
                            pos_np, gt_np[:, -1] - pred_np[:, -1], faces,
                            name='error_pressure')

        l2err_press = torch.norm(pred_press - gt_press) / torch.norm(gt_press)
        mse_press = criterion_func(out[cfd_data.surf, -1], targets[cfd_data.surf, -1]).mean(dim=0)
        l1_error = torch.abs(pred_press - gt_press).sum() / torch.abs(gt_press).sum()
        
        mae_norm = torch.abs(out[cfd_data.surf, -1] - targets[cfd_data.surf, -1]).mean()
        max_ae_norm = torch.max(torch.abs(out[cfd_data.surf, -1] - targets[cfd_data.surf, -1]))
        maes_norm.append(mae_norm.cpu().numpy())
        max_aes_norm.append(max_ae_norm.cpu().numpy())
        l1errs.append(l1_error.cpu().numpy())
        l2errs_press.append(l2err_press.cpu().numpy())
        mses_press.append(mse_press.cpu().numpy())
        times.append(toc - tic)
        index += 1
        per_sample_metrics.append({
                'sample': sample_name,
                'mse': float(mse_press.cpu().numpy()),
                'mae_norm': float(mae_norm.cpu().numpy()),
                'max_ae_norm': float(max_ae_norm.cpu().numpy()),
                'l1_error': float(l1_error.cpu().numpy()),
                'l2_error': float(l2err_press.cpu().numpy()),
                'inference_time': float(toc - tic)
            })

    mean_mae_norm = np.mean(maes_norm)
    mean_max_ae_norm = np.mean(max_aes_norm)
    l1err_press = np.mean(l1errs)
    l2err_press = np.mean(l2errs_press)
    mean_mse = np.mean(mses_press)
    rmse_press = np.sqrt(np.mean(mses_press))
    if coef_norm is not None:
        rmse_press *= coef_norm[3][-1]
        
    results = {
        'MSE': float(mean_mse),
        'RMSE': float(rmse_press),
        'MAE_norm': float(mean_mae_norm),
        'MaxAE_norm': float(mean_max_ae_norm),
        'L1_error_percent': float(l1err_press * 100),
        'L2_error_percent': float(l2err_press * 100),
        'Average_inference_time_sec': float(np.mean(times))
    }
    all_metrics = {
        'per_sample': per_sample_metrics,
        'overall': results
    }
    with open(os.path.join(save_root, 'metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=4)


    print('MSE (mean squared error):', mean_mse)
    print('RMSE (root mean squared error):', rmse_press)
    print('MAE (normalized):', mean_mae_norm)
    print('Max AE (normalized):', mean_max_ae_norm)
    print('Relative L1 Error (%):', l1err_press * 100)
    print('Relative L2 Error (%):', l2err_press * 100)
    print('Average Inference Time (s):', np.mean(times))
    print(f"Saved metrics to {os.path.join(save_root, 'metrics.json')}")

