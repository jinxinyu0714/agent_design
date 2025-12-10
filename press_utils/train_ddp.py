import gc
import os, time, json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from datetime import datetime
import torch.distributed as dist
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

def set_seed(seed=42):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_nb_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(device, model, train_loader, optimizer, scheduler):
    # 单卡训练逻辑，不变
    model.train()
    criterion_mse = nn.MSELoss()
    all_preds, all_targets = [], []
    pbar = tqdm(train_loader, desc="Training", leave=True)
    for cfd_data, geom in pbar:
        cfd_data, geom = cfd_data.to(device), geom.to(device)
        optimizer.zero_grad()
        out = model((cfd_data, geom))
        targets = cfd_data.y

        pred_press = out[cfd_data.surf, -1]
        gt_press = targets[cfd_data.surf, -1]

        loss_press = criterion_mse(pred_press, gt_press)
        loss_press.backward()
        optimizer.step()
        scheduler.step()

        all_preds.append(pred_press.detach().cpu())
        all_targets.append(gt_press.detach().cpu())
        pbar.set_postfix(loss=loss_press.item())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    mse = criterion_mse(all_preds, all_targets).item()
    mae = torch.mean(torch.abs(all_preds - all_targets)).item()
    max_ae = torch.max(torch.abs(all_preds - all_targets)).item()
    rel_l2 = (torch.norm(all_preds - all_targets) / torch.norm(all_targets)).item() * 100
    rel_l1 = (torch.sum(torch.abs(all_preds - all_targets)) / torch.sum(torch.abs(all_targets))).item() * 100

    print(f"Training MAE:| MSE: {mse:.4f}  {mae:.4f} | Max-AE: {max_ae:.4f} | Rel L2 Error: {rel_l2:.4f}% | Rel L1 Error: {rel_l1:.4f}%")
    return mse, mae, max_ae, rel_l1, rel_l2


@torch.no_grad()
def test(device, model, test_loader):
    # 单卡验证逻辑，不变
    model.eval()
    all_preds, all_targets = [], []
    criterion_mse = nn.MSELoss()
    pbar = tqdm(test_loader, desc="Evaluating", leave=True)
    for cfd_data, geom in pbar:
        cfd_data, geom = cfd_data.to(device), geom.to(device)
        out = model((cfd_data, geom))
        targets = cfd_data.y

        pred_press = out[cfd_data.surf, -1].detach().cpu()
        gt_press = targets[cfd_data.surf, -1].detach().cpu()

        all_preds.append(pred_press)
        all_targets.append(gt_press)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    mse = criterion_mse(all_preds, all_targets).item()
    mae = torch.mean(torch.abs(all_preds - all_targets)).item()
    max_ae = torch.max(torch.abs(all_preds - all_targets)).item()
    rel_l2 = (torch.norm(all_preds - all_targets) / torch.norm(all_targets)).item() * 100
    rel_l1 = (torch.sum(torch.abs(all_preds - all_targets)) / torch.sum(torch.abs(all_targets))).item() * 100

    print(f"Validation MSE: {mse:.4f} | MAE: {mae:.4f} | Max-AE: {max_ae:.4f} | Rel L2 Error: {rel_l2:.2f}% | Rel L1 Error: {rel_l1:.2f}%")
    return mse, mae, max_ae, rel_l1, rel_l2


def ddp_train(device, model, train_loader, optimizer, scheduler, epoch, coef_norm=None):
    # DDP 训练
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model.train()
    criterion = nn.MSELoss()
    all_preds, all_targets = [], []
    loader = tqdm(train_loader, desc=f" Train {epoch}", leave=True) if rank == 0 else train_loader

    for cfd_data, geom in loader:
        cfd_data, geom = cfd_data.to(device), geom.to(device)
        optimizer.zero_grad()
        out = model((cfd_data, geom))
        targets = cfd_data.y

        pred = out[cfd_data.surf, -1]
        gt = targets[cfd_data.surf, -1]

        loss = criterion(pred, gt)
        loss.backward()
        optimizer.step()
        scheduler.step()

        all_preds.append(pred.detach())
        all_targets.append(gt.detach())

    preds = torch.cat(all_preds).to(device)
    targets = torch.cat(all_targets).to(device)

    # 聚合指标
    sq_err = torch.sum((preds - targets) ** 2)
    l1_err = torch.sum(torch.abs(preds - targets))
    count = torch.tensor([preds.numel()], device=device)
    dist.all_reduce(sq_err, op=dist.ReduceOp.SUM)
    dist.all_reduce(l1_err, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)

    mse = (sq_err / count).item()
    mae = (l1_err / count).item()

    # 最大误差
    local_max = torch.max(torch.abs(preds - targets))
    max_err = local_max.clone()
    dist.all_reduce(max_err, op=dist.ReduceOp.MAX)
    max_ae = max_err.item()

    # 相对误差
    target_sq = torch.sum(targets ** 2)
    abs_target = torch.sum(torch.abs(targets))
    dist.all_reduce(target_sq, op=dist.ReduceOp.SUM)
    dist.all_reduce(abs_target, op=dist.ReduceOp.SUM)
    rel_l2 = (torch.sqrt(sq_err) / torch.sqrt(target_sq)).item() * 100
    rel_l1 = (l1_err / abs_target).item() * 100

    if rank == 0:
        print(f"[ E{epoch}] MSE: {mse:.4f} | MAE: {mae:.4f} | Max-AE: {max_ae:.4f} | RelL1: {rel_l1:.2f}% | RelL2: {rel_l2:.2f}%")
    return mse, mae, max_ae, rel_l1, rel_l2


@torch.no_grad()
def ddp_test(device, model, val_loader, epoch, args, coef_norm=None):
    # DDP 验证
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model.eval()
    all_preds, all_targets = [], []
    loader = tqdm(val_loader, desc=f" Eval {epoch}", leave=True) if rank == 0 else val_loader

    for cfd_data, geom in loader:
        cfd_data, geom = cfd_data.to(device), geom.to(device)
        out = model((cfd_data, geom))
        targets = cfd_data.y

        pred = out[cfd_data.surf, -1].detach()
        gt = targets[cfd_data.surf, -1].detach()

        all_preds.append(pred)
        all_targets.append(gt)

    preds = torch.cat(all_preds).to(device)
    targets = torch.cat(all_targets).to(device)

    criterion = nn.MSELoss()
    sq_err = torch.sum((preds - targets) ** 2)
    l1_err = torch.sum(torch.abs(preds - targets))
    count = torch.tensor([preds.numel()], device=device)
    dist.all_reduce(sq_err, op=dist.ReduceOp.SUM)
    dist.all_reduce(l1_err, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)

    mse = (sq_err / count).item()
    mae = (l1_err / count).item()

    local_max = torch.max(torch.abs(preds - targets))
    max_err = local_max.clone()
    dist.all_reduce(max_err, op=dist.ReduceOp.MAX)
    max_ae = max_err.item()

    target_sq = torch.sum(targets ** 2)
    abs_target = torch.sum(torch.abs(targets))
    dist.all_reduce(target_sq, op=dist.ReduceOp.SUM)
    dist.all_reduce(abs_target, op=dist.ReduceOp.SUM)
    rel_l2 = (torch.sqrt(sq_err) / torch.sqrt(target_sq)).item() * 100
    rel_l1 = (l1_err / abs_target).item() * 100

    if rank == 0:
        print(f"[ Eval E{epoch}] MSE: {mse:.4f} | MAE: {mae:.4f} | Max-AE: {max_ae:.4f} | RelL1: {rel_l1:.2f}% | RelL2: {rel_l2:.2f}%")
    return mse, mae, max_ae, rel_l1, rel_l2


def main(device, train_dataset, val_dataset, NetClass, model_config, hparams, path, val_iter=1, coef_norm=[]):
    set_seed(42)

    def is_main_process():
        return not is_initialized() or get_rank() == 0

    model = NetClass(**model_config).to(device)
   
    if torch.cuda.device_count() > 1 and not is_initialized():
        if is_main_process():
            print(f"[INFO] 使用 {torch.cuda.device_count()} 张 GPU 进行训练 (DataParallel)")
        model = torch.nn.DataParallel(model)

    train_loader = DataLoader(train_dataset,
                              batch_size=hparams['batch_size'],
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=hparams['batch_size'],
                            shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        total_steps=len(train_loader) * hparams['nb_epochs'],
        final_div_factor=1000.
    )

    log_dir = os.path.join(path, 'log')
    ckpt_dir = os.path.join(path, 'ckpt')
    if is_main_process():
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

    # 仅主进程写 TensorBoard
    if is_main_process():
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    start_time = time.time()
    best_models = []
    max_keep = 5
    train_mse = train_mae = train_max_ae = train_l1 = train_l2 = 1e5
    val_mse = val_mae = val_max_ae = val_l1 = val_l2 = 1e5

    for epoch in range(hparams['nb_epochs']):
        if is_main_process():
            print(f"\n[---------------------- Epoch {epoch} ------------------------]")

        # 训练
        train_mse, train_mae, train_max_ae, train_l1, train_l2 = train(
            device, model, train_loader, optimizer, scheduler)
        gc.collect()
        torch.cuda.empty_cache()

        # 验证
        if val_iter and (epoch % val_iter == 0 or epoch == hparams['nb_epochs'] - 1):
            val_mse, val_mae, val_max_ae, val_l1, val_l2 = test(
                device, model, val_loader)

        if is_main_process():
            # TensorBoard
            writer.add_scalar('Loss/train_mse', train_mse, epoch)
            writer.add_scalar('Loss/train_mae', train_mae, epoch)
            writer.add_scalar('Loss/train_max_ae', train_max_ae, epoch)
            writer.add_scalar('Loss/train_rel_l1', train_l1, epoch)
            writer.add_scalar('Loss/train_rel_l2', train_l2, epoch)

            if val_iter and (epoch % val_iter == 0 or epoch == hparams['nb_epochs'] - 1):
                writer.add_scalar('Loss/val_mse', val_mse, epoch)
                writer.add_scalar('Loss/val_mae', val_mae, epoch)
                writer.add_scalar('Loss/val_max_ae', val_max_ae, epoch)
                writer.add_scalar('Loss/val_rel_l1', val_l1, epoch)
                writer.add_scalar('Loss/val_rel_l2', val_l2, epoch)

                # 控制台打印
                print(f"[Epoch {epoch}] "
                      f"Train - MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, Max-AE: {train_max_ae:.4f}, "
                      f"Rel L1: {train_l1:.2f}%, Rel L2: {train_l2:.2f}% | "
                      f"Val   - MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, Max-AE: {val_max_ae:.4f}, "
                      f"Rel L1: {val_l1:.2f}%, Rel L2: {val_l2:.2f}%")

                # 保存最优模型
                model_path = os.path.join(ckpt_dir, f'model_epoch{epoch}_val_mse{val_mse:.6f}.pth')
                torch.save(
                    model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    model_path)
                best_models.append((val_mse, epoch, model_path))
                best_models.sort(key=lambda x: x[0])
                if len(best_models) > max_keep:
                    for _, _, old_path in best_models[max_keep:]:
                        if os.path.exists(old_path):
                            os.remove(old_path)
                    best_models = best_models[:max_keep]

    # 保存最终模型 & 日志
    if is_main_process():
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(state_dict,
                   os.path.join(ckpt_dir, f'model_{hparams["nb_epochs"]}_final.pth'))
        with open(os.path.join(log_dir, f'log_{hparams["nb_epochs"]}.json'), 'w') as f:
            json.dump({
                'nb_parameters': int(get_nb_trainable_params(model)),
                'time_elapsed': float(time.time() - start_time),
                'hparams': {k: str(v) if isinstance(v, (np.ndarray, np.number)) else v
                            for k, v in hparams.items()},
                'train_metrics': {
                    'mse': float(train_mse),
                    'mae': float(train_mae),
                    'max_ae': float(train_max_ae),
                    'rel_l1': float(train_l1),
                    'rel_l2': float(train_l2)
                },
                'val_metrics': {
                    'mse': float(val_mse),
                    'mae': float(val_mae),
                    'max_ae': float(val_max_ae),
                    'rel_l1': float(val_l1),
                    'rel_l2': float(val_l2)
                },
                'coef_norm': to_python(coef_norm),
            }, f, indent=4)
        writer.close()

    return model