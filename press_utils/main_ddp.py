import os
import random
import time
import argparse
from datetime import datetime
import warnings
import torch
import json
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import train_ddp
from load_dataset import load_train_val_split
from dataset import GraphDataset
from Transolver import Model
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch.distributed import is_initialized, get_rank
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_main_process():
    return not is_initialized() or get_rank() == 0

def pyg_collate(batch):

    data_list, shape_list = zip(*batch)
    batched_data  = Batch.from_data_list(data_list)
    batched_shape = torch.stack(shape_list, dim=0)
    return batched_data, batched_shape

def main_worker(local_rank: int, args):
    set_seed(42)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # distributed init
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    dist.init_process_group("nccl", world_size=args.world_size, rank=local_rank)

    if args.preprocess and is_main_process():
        import subprocess
        print("[Rank-0] 正在执行 create_data_2.py 做预处理 …")
        ret = subprocess.call([
            "python", "create_data_2.py",
            "--input_dir",  args.input_dir,
            "--output_dir", args.output_dir,
            "--target_points", str(args.target_points)
        ])
        if ret != 0:
            raise RuntimeError("create_data_2.py 执行失败，请检查错误信息")

    if args.preprocess:
        args.data_dir = args.output_dir   
        dist.barrier() 


    train_data, val_data, coef_norm = load_train_val_split(args, preprocessed=args.preprocessed)
    train_ds = GraphDataset(train_data)
    val_ds = GraphDataset(val_data)

    from torch.utils.data.distributed import DistributedSampler
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=4, pin_memory=True, collate_fn=pyg_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=2, pin_memory=True, collate_fn=pyg_collate)

    # model
    model = Model(**args.model_config).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.nb_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    total_steps=total_steps, final_div_factor=1000.)

    # dirs
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    exp_root = os.path.join(args.log_dir, f"{args.exp_name}_{timestamp}")
    log_dir = os.path.join(exp_root, "log")
    ckpt_dir = os.path.join(exp_root, "ckpt")
    if is_main_process():
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    best_models = []  
    max_keep = 5
    start = time.time()

    for epoch in range(args.nb_epochs):
        train_sampler.set_epoch(epoch)

        train_mse, train_mae, train_max_ae, train_l1, train_l2 = train_ddp.ddp_train(
            device, model, train_loader, optimizer, scheduler, epoch, coef_norm)

        if epoch % args.val_iter == 0 or epoch == args.nb_epochs - 1:
            val_mse, val_mae, val_max_ae, val_l1, val_l2 = train_ddp.ddp_test(
                device, model, val_loader, epoch, args, coef_norm)
        else:                                  # 不验证时占位
            val_mse = val_mae = val_max_ae = val_l1 = val_l2 = float('inf')

        if is_main_process():
            writer.add_scalar('Loss/train_mse', train_mse, epoch)
            writer.add_scalar('Loss/train_mae', train_mae, epoch)
            writer.add_scalar('Loss/train_max_ae', train_max_ae, epoch)
            writer.add_scalar('Loss/train_rel_l1', train_l1, epoch)
            writer.add_scalar('Loss/train_rel_l2', train_l2, epoch)

            if val_mse != float('inf'):       
                writer.add_scalar('Loss/val_mse', val_mse, epoch)
                writer.add_scalar('Loss/val_mae', val_mae, epoch)
                writer.add_scalar('Loss/val_max_ae', val_max_ae, epoch)
                writer.add_scalar('Loss/val_rel_l1', val_l1, epoch)
                writer.add_scalar('Loss/val_rel_l2', val_l2, epoch)
                
                print(f"[Epoch {epoch}] Train MSE {train_mse:.4f} | Val MSE {val_mse:.4f}")

                # keep best k
                ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch}_val_mse{val_mse:.6f}.pth")
                torch.save(model.module.state_dict(), ckpt_path)
                best_models.append((val_mse, epoch, ckpt_path))
                best_models.sort(key=lambda x: x[0])
                for _, _, old in best_models[max_keep:]:
                    if os.path.exists(old):
                        os.remove(old)
                best_models = best_models[:max_keep]
                    
    if is_main_process():
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(state_dict,
                os.path.join(ckpt_dir, f"model_{args.nb_epochs}_final.pth"))

    
        import numpy as np
        def get_nb_trainable_params(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        def to_python(obj):
            if isinstance(obj, np.ndarray):          return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)): return obj.item()
            if isinstance(obj, dict):                return {k: to_python(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):       return [to_python(v) for v in obj]
            return obj

        log_path = os.path.join(log_dir, f"log_{args.nb_epochs}.json")
        with open(log_path, "w") as f:
            json.dump({
                "nb_parameters": int(get_nb_trainable_params(model)),
                "time_elapsed" : float(time.time() - start),
                "hparams"      : {k: str(v) for k, v in vars(args).items()},
                "train_metrics": {
                    "mse": train_mse, "mae": train_mae, "max_ae": train_max_ae,
                    "rel_l1": train_l1, "rel_l2": train_l2
                },
                "val_metrics": {
                    "mse": val_mse, "mae": val_mae, "max_ae": val_max_ae,
                    "rel_l1": val_l1, "rel_l2": val_l2
                },
                "coef_norm": to_python(coef_norm)
            }, f, indent=4)
        writer.close()

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='press_100_10w_ddpcs')
    # VTK 预处理参数
    parser.add_argument('--preprocess', type=int, default=0, help="是否预处理原始 VTK")  ###1=是，0=跳过
    parser.add_argument('--input_dir', type=str, default='/student/ysm/data/car-pressure/pressure/')###原始vtk文件路径
    parser.add_argument('--output_dir', type=str, default='/student/ysm/press_7.14_qirui/simple_vtk/')###简化之后vtk输出路径
    parser.add_argument('--target_points', type=int, default=100000)###简化之后vtk点数  
    # 数据路径
    parser.add_argument('--data_dir',default='/student/ysm/press_7.14_qirui/simple_vtk/')    ###简化之后vtk文件路径，与output_dir相同
    parser.add_argument('--metadata_file',default='/student/ysm/data/car-pressure/csv/100.csv')    # 总文件名称csv文件路径，无表头一列
    parser.add_argument('--train_split',default='/student/ysm/data/car-pressure/train_70.csv')        # train文件名称csv文件路径，无表头一列
    parser.add_argument('--val_split',default='/student/ysm/data/car-pressure/val_10.csv')          # val文件名称csv文件路径，无表头一列
    parser.add_argument('--save_dir',default='/student/ysm/press_7.14_qirui/process_vtk/')    # 预处理缓存目录
    parser.add_argument('--log_dir',default='/student/ysm/press_7.14_qirui/logdir/')   # 训练存放模型及数据文件路径
    parser.add_argument('--preprocessed', type=int, default=0 )   # 是否使用npy缓存,preprocessed =1 使用缓存
    parser.add_argument('--val_iter', default=5, type=int)   #训练多少轮验证一次
    # 训练超参
    parser.add_argument('--lr', default=0.001, type=float)###0.002
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--nb_epochs', default=500, type=int)

    # DDP 参数
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count())
    parser.add_argument('--master_addr', type=str, default='127.0.0.1')
    parser.add_argument('--master_port', type=str, default='29500')

    args = parser.parse_args()

    # 模型配置
    args.model_config = {
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

    # spawn 多进程
    mp.spawn(main_worker, nprocs=args.world_size, args=(args,))


if __name__ == '__main__':
    main()
