import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning,module=r"timm\.models\.layers")
import time
import pandas as pd
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataLoader import DrivAerDataset
from model import phsoffNet
from datetime import datetime
import yaml
import glob
import random
import numpy as np
from timm.scheduler.cosine_lr import CosineLRScheduler
import multiprocessing
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

###设置使用gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

def set_random_seed(seed, deterministic=False):
    """Set random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


### mix loss
def train_one_epoch(data_loader, model, optimizer, device, epoch):
    model.train()
    total_loss_accumulated = 0.0
    total_sum = 0.0
    total_sq_sum = 0.0
    residual_sum = 0.0
    mse_Loss = 0.0
    mae_Loss = 0.0
    r2_sum = 0.0
    total_count = 0
    skipped_count = 0
    max_mae = 0.0
    re = 0.0
    max_re = -float('inf')  # 初始化为负无穷，确保任何误差都会更新它

    progress_bar = tqdm(enumerate(data_loader), 
                        total=len(data_loader), 
                        desc=f'Epoch {epoch+1} [Train]',
                        dynamic_ncols=True,
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    
    for batch_idx, (x, y, filename, cubesize) in progress_bar:
        x, y, cubesize = x.to(device), y.to(device), cubesize.to(device)
        optimizer.zero_grad()

        output = model(x, cubesize)
        mask = (y.abs() >= 1e-4)

        if mask.sum() > 0:
            filtered_y = y[mask]
            filtered_output = output[mask]
            re = ((filtered_y - filtered_output).abs()) / (filtered_y.abs())
            mse_error = ((filtered_y - filtered_output)) ** 2
            mae_error = ((filtered_y - filtered_output).abs())

            # 计算 MRE 和 max_re
            mre = re.mean()  # MRE 是相对误差的均值
            max_re = max(max_re, re.max().item())  # 更新全局的 max_re

            loss2 = mse_error.mean()
            loss3 = mae_error.mean()
            total_loss_accumulated += mre.item() * mask.sum().item()
            mse_Loss += loss2.item() * mask.sum().item()
            mae_Loss += loss3.item() * mask.sum().item()
            total_count += mask.sum().item()

            # 计算 R^2
            batch_sum = filtered_y.sum().item()
            batch_sq_sum = (filtered_y ** 2).sum().item()
            batch_residual_sum = ((filtered_y - filtered_output) ** 2).sum().item()
            
            total_sum += batch_sum
            total_sq_sum += batch_sq_sum
            residual_sum += batch_residual_sum
        else:
            skipped_count += 1

        # loss = mre + max_re  # 计算混合损失: MRE + max_re
        loss = mre
        # loss=loss2#mse
        loss.backward()
        optimizer.step()

    average_loss = total_loss_accumulated / total_count if total_count > 0 else 0.0
    mse = mse_Loss / total_count if total_count > 0 else 0.0
    mae = mae_Loss / total_count if total_count > 0 else 0.0
    mixloss = average_loss + max_re  # 最终返回的混合损失
    ###R2
    if total_count > 0:
        global_mean = total_sum / total_count
        # SS_tot = Σ(y_i²) - n·ȳ²
        ss_tot = total_sq_sum - total_count * global_mean ** 2
        # 防止除以零
        ss_tot = max(ss_tot, 1e-10) if ss_tot == 0 else ss_tot
        r2 = 1 - (residual_sum / ss_tot)
    else:
        r2 = 0.0
    print(f"Skipped count: {skipped_count}")
    print(f"Global max_re: {max_re}")  # 输出全局的 max_re
    # 关闭进度条
    progress_bar.close()
    return mixloss, average_loss, max_re, mse, mae, r2


def val(data_loader, model, device, epoch):
    model.eval()
    total_loss_accumulated = 0.0
    total_sum = 0.0  # 所有符合条件的 y 值之和
    total_sq_sum = 0.0  # 所有符合条件的 y 值平方和
    residual_sum = 0.0  # 残差平方和
    mse_Loss = 0.0
    mae_Loss = 0.0
    r2_sum = 0.0  # 用于累加 R^2 值
    total_count = 0  # 统计符合条件的样本数
    skipped_count = 0  # 统计被跳过的样本数
    max_mae = 0.0  # 初始化最大的 MAE
    re = 0.0
    max_re = 0.0  # 初始化最大的 MAE
    with torch.no_grad():
        for x, y, filename, cubesize in data_loader:
            x, y, cubesize = x.to(device), y.to(device), cubesize.to(device)
            start_time = time.time()
            output = model(x, cubesize)
            # 记录模型推理后的时间并计算时间差
            end_time = time.time()
            # 创建掩码：选择 y 的绝对值大于或等于 10^-4 的位置
            mask = (y.abs() >= 1e-4)

            if mask.sum() > 0:
                filtered_y = y[mask]
                filtered_output = output[mask]

                # 计算rere误差 MRE
                re = ((filtered_y - filtered_output).abs()) / (filtered_y.abs())
                mse_error = ((filtered_y - filtered_output)) ** 2
                mae_error = ((filtered_y - filtered_output).abs())

                loss1 = re.mean()  # 计算平均相对误ee
                loss2 = mse_error.mean()
                loss3 = mae_error.mean()

                total_loss_accumulated += loss1.item() * mask.sum().item()
                mse_Loss += loss2.item() * mask.sum().item()
                mae_Loss += loss3.item() * mask.sum().item()
                total_count += mask.sum().item()

                # 更新最大的 MAE
                max_re = max(max_re, re.max().item())  # 计算当前 batch 中的最大 re

                # 计算 R^2
                batch_sum = filtered_y.sum().item()
                batch_sq_sum = (filtered_y ** 2).sum().item()
                batch_residual_sum = mse_error.sum().item()
                
                total_sum += batch_sum
                total_sq_sum += batch_sq_sum
                residual_sum += batch_residual_sum

            else:
                skipped_count += 1  # 统计不符合条件的样本数

    # 计算平均损失
    average_loss = total_loss_accumulated / total_count if total_count > 0 else 0.0
    mse_Loss = mse_Loss / total_count if total_count > 0 else 0.0
    mae_Loss = mae_Loss / total_count if total_count > 0 else 0.0
    # 计算总的 R^2
    if total_count > 0:
        global_mean = total_sum / total_count
        ss_tot = total_sq_sum - total_count * global_mean ** 2
        # 防止除零
        ss_tot = max(ss_tot, 1e-10) if ss_tot == 0 else ss_tot
        r2 = 1 - (residual_sum / ss_tot)
    else:
        r2 = 0.0
    return average_loss, mse_Loss, mae_Loss, max_re, r2


if __name__ == "__main__":
    torch.cuda.empty_cache()
    set_random_seed(42)  ### set random seed to ensure the same results
    # 创建解析器，并设置描述
    parser = argparse.ArgumentParser(description="训练和测试模型")
    # 添加位置参数config_path，用来指定配置文件的路径
    parser.add_argument("config_path", type=str, help="配置文件的路径")
    try:
        args = parser.parse_args()
    except SystemExit as e:
        print(f"Error: {e}")
    config = load_config(args.config_path)

    # 从配置中获取变量
    train_csv_name = config['paths']['train_csv_name']
    val_csv_name = config['paths']['val_csv_name']
    test_csv_name = config['paths']['test_csv_name']
    log_dir = config['paths']['log_dir']
    checkpoint_path = config['paths']['checkpoint_path']
    dim = config['model']['dim']
    heads = config['model']['heads']
    group_size = config['model']['group_size']
    num_group = config['model']['num_group']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    epoch_number = config['training']['epoch_number']
    best_val_loss = config['training'].get('best_val_loss', float('inf'))

    # 检查显卡是否使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置tensorbaord
    experiment_name = config.get('experiment_name', "exp_with_custom_params")
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = f"{log_dir}{experiment_name}_{current_time}"
    writer = SummaryWriter(log_dir=log_dir)

    # 创建模型权重保存路径
    ckpt_path = os.path.join(checkpoint_path, experiment_name, current_time)
    os.makedirs(ckpt_path, exist_ok=True)
 
    train_dataset = DrivAerDataset(config, train_csv_name)
    val_dataset = DrivAerDataset(config, val_csv_name)

    # 记录结束时间
    end_time = time.time()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers,pin_memory=True)

    # 初始化模型、优化器和学习率调度器

    model = phsoffNet(dim=dim)

    if torch.cuda.device_count() > 1:
        # model = torch.nn.DataParallel(model)
        print(f"使用 {torch.cuda.device_count()} 块GPU")
        model = model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        model = model.to(device)

    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=epoch_number,
                                  lr_min=1e-7,
                                  warmup_lr_init=1e-7,
                                  warmup_t=5,
                                  cycle_limit=1,
                                  t_in_epochs=True)

    for epoch in range(epoch_number):
        start_time = time.time()
        mixloss, mre, max_re, mse, mae, r2 = train_one_epoch(train_loader, model, optimizer, device, epoch)
        start_val_time = time.time()
        val_mre_loss, val_mse_loss, val_mae_loss, val_max_re, val_r2 = val(val_loader, model, device, epoch)
        writeend_time = time.time()
        writer.add_scalar('Loss/Train-mix', mixloss, epoch)
        writer.add_scalar('Loss/Train-mre', mre, epoch)
        writer.add_scalar('Loss/Train-max-re', max_re, epoch)
        writer.add_scalar('Loss/Train-mse', mse, epoch)
        writer.add_scalar('Loss/Train-mae', mae, epoch)
        writer.add_scalar('Loss/Train-R2', r2, epoch)
        writer.add_scalar('Loss/val-mre', val_mre_loss, epoch)
        writer.add_scalar('Loss/val-mse', val_mse_loss, epoch)
        writer.add_scalar('Loss/val-mae', val_mae_loss, epoch)
        writer.add_scalar('Loss/val-max-re', val_max_re, epoch)
        writer.add_scalar('Loss/val-R2', val_r2, epoch)
        print(
            f"Train mix Loss: {mixloss:>15.10f}\n"
            f"Train mre Loss: {mre:>15.10f}\n"
            f"Train max-re Loss: {max_re:>15.10f}\n"
            f"Train mse Loss: {mse:>15.10f}\n"
            f"Train mae Loss: {mae:>15.10f}\n"
            f"Train R2: {r2:>15.10f}\n"
            f"val mre Loss: {val_mre_loss:>15.10f}\n"
            f"val mse Loss: {val_mse_loss:>15.5f}\n"
            f"val mae Loss: {val_mae_loss:>15.10f}\n"
            f"val max-re Loss: {val_max_re:>15.10f}\n"
            f"val R2: {val_r2:>15.10f}\n"
            f"lr: {optimizer.param_groups[0]['lr']:>15.10f}\n"
            f"Train Time: {start_val_time - start_time:>15.2f}s\n"
            f"val Time: {time.time() - start_val_time:>15.2f}s\n"
            f"Total Time: {time.time() - start_time:>15.2f}s"
        )
        if val_mre_loss < best_val_loss:
            best_val_loss = val_mre_loss
            save_path = os.path.join(ckpt_path, f"best_model_mre_{best_val_loss:.10f}.pt")
            torch.save(model.state_dict(), save_path)

            print("Model saved as best_model.pt")
        scheduler.step(epoch)
        print("-" * 80)

    writer.close()
    # 删除最旧的权重文件，只保留最低的两个权重文件
    ckpt_files = glob.glob(os.path.join(ckpt_path, '*.pt'))
    ckpt_files.sort(key=os.path.getmtime)  # 按修改时间排序，最旧的文件排在前面
    if len(ckpt_files) > 5:
        for i in range(len(ckpt_files) - 5):
            os.remove(ckpt_files[i])


