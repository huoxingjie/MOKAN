import os,sys

# Must be set before importing torch when deterministic algorithms are enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import joblib
import random
import torch.fft
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
import numpy as np
import time
from torch.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
try:
    from .wavelet_multiscale_model import Model
except ImportError:
    from wavelet_multiscale_model import Model
import torch
from types import SimpleNamespace
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pywt

from torch.optim.lr_scheduler import _LRScheduler

# 自定义分阶段学习率调度器
class StagedLRScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epochs, initial_lr, stages=4):
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr
        self.stage_length = total_epochs // stages  # 每个阶段的epoch数
        self.stage_lrs = [initial_lr]  # 存储每个阶段的初始学习率
        super(StagedLRScheduler, self).__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch
        stage = min(epoch // self.stage_length, 3)  # 确保阶段不超过4

        # 计算当前阶段的初始学习率
        if stage == 0:
            stage_lr = self.initial_lr  # 阶段1：从全局初始学习率开始
        elif stage == 1:
            # 阶段2、3、4：基于上一阶段初始学习率的75%
            stage_lr = self.stage_lrs[stage - 1] * 0.7
        elif stage == 2:
            stage_lr = self.stage_lrs[stage - 1] * 0.7
        else:
            stage_lr = self.stage_lrs[stage - 1] * 0.7


        # 计算阶段内的学习率（线性下降到50%）
        stage_epoch = epoch % self.stage_length  # 当前阶段内的epoch偏移
        fraction = stage_epoch / (self.stage_length - 1) if self.stage_length > 1 else 0  # 阶段内进度（0到1）
        if stage == 0:
            lr = stage_lr * (1.0 - 0.5 * fraction)  # 阶段1：下降70%
        elif stage == 1:
            lr = stage_lr * (1.0 - 0.5 * fraction)  # 阶段2：下降30%
        elif stage == 2:
            lr = stage_lr * (1.0 - 0.5 * fraction)  # 阶段3：下降10%
        else:
            if stage_epoch < 5:  # 前5个epoch
                # 前5个epoch线性下降到50%，fraction按前5个epoch重新计算（0到1）
                fraction = stage_epoch / 4  # 前5个epoch的进度
                lr = stage_lr * (1.0 - 0.5 * fraction)
            else:  # 后5个epoch
                lr = stage_lr * 0.5  # 保持前5个epoch结束时的学习率（即50%）
        # 更新阶段初始学习率列表（仅在阶段开始时更新）
        if epoch % self.stage_length == 0 and stage <= 3:
            if len(self.stage_lrs) <= stage:
                self.stage_lrs.append(stage_lr)

        return [lr for _ in self.optimizer.param_groups]
##################每次下降的低点比上一次更低#############################

# 温度也要调一下
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, x, neg, positive):
        batch_size = x.size(0)
        # print(f"x_shape:{x.shape}")
        # 重塑张量以确保形状正确
        x = x.view(batch_size, -1)  # [batch_size, dim]
        neg = neg.view(batch_size, -1)  # [batch_size, dim]

        # print(f"x_shape:{x.shape}")

        positive = positive.view(batch_size, -1)  # [batch_size, dim]

        def cosine_similarity(a, b):
            if b is neg:
                # 负样本：计算所有样本对的余弦相似度
                dot_product = torch.matmul(a, b.t())  # [batch_size, batch_size]
                norm_a = torch.norm(a, dim=-1)  # [batch_size]
                norm_b = torch.norm(b, dim=-1)  # [batch_size]
                norm = norm_a.unsqueeze(1) * norm_b.unsqueeze(0)  # [batch_size, batch_size]
                return dot_product / (norm + 1e-8)
            else:
                # 正样本：计算每个样本与其正样本的余弦相似度
                dot_product = torch.sum(a * b, dim=-1)  # [batch_size]
                norm_a = torch.norm(a, dim=-1)  # [batch_size]
                norm_b = torch.norm(b, dim=-1)  # [batch_size]
                norm = norm_a * norm_b  # [batch_size]
                return dot_product / (norm + 1e-8)

        # 计算相似度
        # 正样本相似度
        pos_sim = cosine_similarity(x, positive) / self.temperature  # [batch_size]
        # print(f"pos_sim.shape:{pos_sim.shape}")
        # 锚点与所有样本的相似度矩阵（包括自身）
        all_sim = cosine_similarity(x, neg) / self.temperature  # [batch_size, batch_size]
        # print(f"all_sim.shape:{all_sim.shape}")
        # 排除锚点与自身的相似度(即对角线元素，设置为负无穷，后续exp时结果为0排除影响)
        # mask = torch.eye(batch_size, device=x.device, dtype=torch.bool)
        # all_sim = all_sim.masked_fill(mask, float('-inf'))  # 将对角线设为负无穷

        # 对每个锚点，计算正样本概率
        pos_sim = pos_sim.unsqueeze(1) # [batch_size, 1]
        combined_sim = torch.cat([pos_sim, all_sim], dim=1)  # [batch_size, batch_size+1]
        pos_probs = F.softmax(combined_sim, dim=1)  # [batch_size, batch_size + 1]
        # print(f"pos_probs.shape:{pos_probs.shape}")

        # 取正样本的概率（第 0 列）
        pos_prob = pos_probs[:, 0]
        # print(f"pos_prob.shape:{pos_prob.shape}")
        # print(pos_prob)
        loss = -torch.log(pos_prob + 1e-8).mean()
        # print(loss)

        return loss



# 设置 PyTorch CUDA 内存分配配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def setup(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12359')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


# def set_seed(seed: int = 42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     try:
#         torch.use_deterministic_algorithms(True, warn_only=True)
#     except Exception:
#         pass






def _load_csv_numeric(file_path: str):
    df = pd.read_csv(file_path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] == 0:
        raise ValueError("No numeric columns found after preprocessing.")
    return df.values.astype(np.float32)


def _build_windows(data: np.ndarray, in_length: int, out_length: int):
    total = len(data) - in_length - out_length + 1
    if total <= 0:
        raise ValueError(
            f"Segment length={len(data)} is too short for in_length={in_length}, out_length={out_length}."
        )
    x = np.zeros((total, in_length, data.shape[1]), dtype=np.float32)
    y = np.zeros((total, out_length, data.shape[1]), dtype=np.float32)

    for i in range(total):
        x[i] = data[i : i + in_length]
        y[i] = data[i + in_length : i + in_length + out_length]
    return x, y


def _dataset_split_ends(file_path: str, total_len: int):
    name = os.path.basename(file_path).lower()

    if "etth" in name:
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
        test_end = train_end + 8 * 30 * 24
    elif "ettm" in name:
        train_end = 12 * 30 * 24 * 4
        val_end = train_end + 4 * 30 * 24 * 4
        test_end = train_end + 8 * 30 * 24 * 4
    elif "electricity" in name:
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.8)
        test_end = total_len
    else:
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.8)
        test_end = total_len

    train_end = max(1, min(train_end, total_len))
    val_end = max(train_end + 1, min(val_end, total_len))
    test_end = total_len if test_end > total_len else test_end
    return train_end, val_end, test_end


def load_and_preprocess_data(file_path, in_length, out_length,
                             wavelet="db1", level=1, isTraining=True):
    fname, _ = os.path.splitext(file_path)
    signs = ["train", "val", "test"]
    lb = signs[:2] if isTraining else signs[-1:]
    x_data = {}
    y_data = {}
    x_original_data = {}  # 用于存储未经过小波分解的初始数据

    cache_prefix = f"{fname}_waveletms_v2_in{in_length}_out{out_length}"
    scaler_path = f"{cache_prefix}_scaler.npz"
    split_paths = {
        "x_train": f"{cache_prefix}_x_train.npy",
        "y_train": f"{cache_prefix}_y_train.npy",
        "x_val": f"{cache_prefix}_x_val.npy",
        "y_val": f"{cache_prefix}_y_val.npy",
        "x_test": f"{cache_prefix}_x_test.npy",
        "y_test": f"{cache_prefix}_y_test.npy",
    }

    if os.path.exists(scaler_path) and all(os.path.exists(v) for v in split_paths.values()):
        x_data["train"] = np.load(split_paths["x_train"], allow_pickle=True).astype(np.float32)
        y_data["train"] = np.load(split_paths["y_train"], allow_pickle=True).astype(np.float32)
        x_data["val"] = np.load(split_paths["x_val"], allow_pickle=True).astype(np.float32)
        y_data["val"] = np.load(split_paths["y_val"], allow_pickle=True).astype(np.float32)
        x_data["test"] = np.load(split_paths["x_test"], allow_pickle=True).astype(np.float32)
        y_data["test"] = np.load(split_paths["y_test"], allow_pickle=True).astype(np.float32)
        for s in signs:
            x_original_data[s] = x_data[s].copy()
    else:
        try:
            raw = _load_csv_numeric(file_path)
        except Exception as e:
            print(f"载入数据文件{file_path}失败。\n失败原因：{e}")
            sys.exit(0)

        total_len = len(raw)
        train_end, val_end, test_end = _dataset_split_ends(file_path, total_len)

        min_required = in_length + out_length + 1
        if train_end < min_required:
            print("警告：训练集长度不足，无法生成样本")
            for s in signs:
                x_data[s] = np.empty((0, in_length, raw.shape[1]), dtype=np.float32)
                y_data[s] = np.empty((0, out_length, raw.shape[1]), dtype=np.float32)
                x_original_data[s] = x_data[s].copy()
            return x_data, y_data, x_original_data

        scaler = StandardScaler()
        scaler.fit(raw[:train_end])
        all_scaled = scaler.transform(raw).astype(np.float32)
        all_x, all_y = _build_windows(all_scaled, in_length, out_length)

        idx = np.arange(len(all_x))
        pred_start = idx + in_length
        pred_end = pred_start + out_length

        train_mask = pred_end <= train_end
        val_mask = (pred_start >= train_end) & (pred_end <= val_end)
        test_mask = (pred_start >= val_end) & (pred_end <= test_end)

        if not train_mask.any() or not val_mask.any() or not test_mask.any():
            print("警告：切分后有空集合，请检查 in_length/out_length")
            for s in signs:
                x_data[s] = np.empty((0, in_length, raw.shape[1]), dtype=np.float32)
                y_data[s] = np.empty((0, out_length, raw.shape[1]), dtype=np.float32)
                x_original_data[s] = x_data[s].copy()
            return x_data, y_data, x_original_data

        x_data["train"], y_data["train"] = all_x[train_mask], all_y[train_mask]
        x_data["val"], y_data["val"] = all_x[val_mask], all_y[val_mask]
        x_data["test"], y_data["test"] = all_x[test_mask], all_y[test_mask]

        for s in signs:
            x_original_data[s] = x_data[s].copy()

        os.makedirs(os.path.dirname(cache_prefix) or ".", exist_ok=True)
        np.save(split_paths["x_train"], x_data["train"])
        np.save(split_paths["y_train"], y_data["train"])
        np.save(split_paths["x_val"], x_data["val"])
        np.save(split_paths["y_val"], y_data["val"])
        np.save(split_paths["x_test"], x_data["test"])
        np.save(split_paths["y_test"], y_data["test"])
        np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)

    for k in lb:
        x_data[f"{k}_shape"] = x_data[k].shape
        y_data[f"{k}_shape"] = y_data[k].shape

    return x_data, y_data, x_original_data


def __loadData(rank, world_size, args, isTraining=True):
    # 加载数据
    print(f"加载数据, rank={rank}")
    file_path = args.file_path
    X, Y, X_original = load_and_preprocess_data(file_path,
                in_length=int(args.in_length), out_length=int(args.out_length),
                level=int(args.level), wavelet=args.wavelet, isTraining=isTraining)

    batch_size_per_gpu = max(1, int(args.batch_size) // max(int(world_size), 1))

    result = None
    if isTraining:
        X_train = torch.tensor(X["train"], dtype=torch.float32)
        X_val = torch.tensor(X["val"], dtype=torch.float32)

        X_original_train = torch.tensor(X_original["train"], dtype=torch.float32)  # 初始数据
        X_original_val = torch.tensor(X_original["val"], dtype=torch.float32)      # 初始数据

        y_train = torch.tensor(Y["train"], dtype=torch.float32)
        y_val = torch.tensor(Y["val"], dtype=torch.float32)
        # print(Y["train"].shape)
        # print(Y["val"].shape)

        train_dataset = TensorDataset(X_train, X_original_train, y_train)
        val_dataset = TensorDataset(X_val, X_original_val, y_val)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size_per_gpu, sampler=train_sampler, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_per_gpu, sampler=val_sampler, drop_last=False)
        result = (train_loader, val_loader, train_sampler, X["train_shape"], Y["train_shape"])
    else:
        X_test = torch.tensor(X["test"], dtype=torch.float32)
        X_original_test = torch.tensor(X_original["test"], dtype=torch.float32)  # 初始数据
        y_test = torch.tensor(Y["test"], dtype=torch.float32)
        test_dataset = TensorDataset(X_test, X_original_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_per_gpu, drop_last=False)
        result = (test_loader, X["test_shape"], Y["test_shape"])
    # end if
    print(f"数据加载完成, rank={rank}")
    return result
# end loadData
def _build_model_configs(args, num_features: int):
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=args.in_length,
        label_len=args.in_length,
        pred_len=args.out_length,
        down_sampling_window=args.down_sampling_window,
        down_sampling_layers=args.down_sampling_layers,
        channel_independence=0,
        e_layers=args.e_layers,
        moving_avg=args.moving_avg,
        enc_in=num_features,
        c_out=num_features,
        d_model=args.d_model,
        embed="timeF",
        freq="h",
        dropout=args.dropout,
        use_future_temporal_feature=False,
        use_norm=args.use_norm,
        begin_order=args.begin_order,
        wavelet_name=args.wavelet_name,
        wavelet_mode=args.wavelet_mode,
        num_blocks=int(getattr(args, "num_blocks", 2)),
        level=int(getattr(args, "level", -6)),
        feature_switch_threshold=int(getattr(args, "feature_switch_threshold", 100)),
    )
def __loadModel(args, gpuID, xShape, yShape):
    print("生成模型对象...")
    obj = None
    try:
        N = int(args.hippo_length)
        in_features = xShape[-1]
        in_length = yShape[1]
        out_length = yShape[1]
        num_blocks = int(args.num_blocks)
        level = int(args.level)

        print(f"Input shapes: xShape={xShape}, yShape={yShape}")
        print(f"Model params: N={N}, in_features={in_features}, out_length={out_length}, num_blocks={num_blocks}")

        num_features = xShape[-1]
        obj = Model(_build_model_configs(args, num_features))

        obj.to(gpuID)
    except Exception as e:
        print(f"载入模型失败。GPU[{gpuID}]退出运行。\n错误原因：{e}")

    # for name, module in obj.named_modules():
    #     params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    #     print(f"Module {name}: trainable params = {params}")
    return obj
        # obj = StackedFrequencyEnsembleModel(
        #     N=N,
        #     kan_layers=[128, 128, 128, 128],
        #     in_features=in_features,
        #     out_length=out_length,
        #     num_blocks=num_blocks
        # )

        # # 调试：检查每个模块的参数
        # print("Checking parameters:")
        # for name, module in obj.named_modules():
        #     params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        #     print(f"Module {name}: trainable params = {params}")
        #
        # trainable_params = sum(p.numel() for p in obj.parameters() if p.requires_grad)
        # print(f"模型总参数数量: {sum(p.numel() for p in obj.parameters())}")
        # print(f"可训练参数数量: {trainable_params}")
        # if trainable_params == 0:
        #     raise ValueError("模型没有可训练参数，请检查模块初始化或输入形状。")

    #     obj.to(gpuID)
    #
    # except Exception as e:
    #     print(f"载入模型失败。GPU[{gpuID}]退出运行。\n错误原因：{e}")
    # return obj

#加上MSEloss
def train(rank, world_size, args, gpuID):
    # set_seed(int(getattr(args, "seed", 42)))

    # 加载训练数据
    train_loader, val_loader, train_sampler, xShape, yShape = __loadData(rank, world_size, args)
    # print(f"yshape:{yShape}")
    print(f"开始初始化模型, rank={rank}, gpu={gpuID}")
    # 初始化模型
    model = __loadModel(args, gpuID, xShape=xShape, yShape=yShape)
    if model is None:
        return
    use_ddp = world_size > 1
    if use_ddp:
        model = DDP(model, device_ids=[gpuID], find_unused_parameters=False)
    print(f"模型初始化完成, rank={rank}, gpu={gpuID}")
    if rank == 0:
        model_for_name = model.module if use_ddp else model
        backend_name = getattr(model_for_name, "_impl_name", model_for_name.__class__.__name__)
        print(f"[rank0] 当前路由后端模型: {backend_name}")

    criterion_mse = nn.MSELoss()
    criterion_info = InfoNCELoss(temperature=0.1)
    criterion_mae = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=args.lr,  # 初始 lr 建议 1e-4（ETTh）或 5e-4（电力）
    #     weight_decay=1e-5,  # ETTh 上 1e-5 ~ 1e-4，电力可稍大
    # )
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.95)

    # 使用自定义的 StagedLRScheduler
    # scheduler = StagedLRScheduler(optimizer, total_epochs=args.epochs, initial_lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1
    )
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)  # 余弦退火

    # 初始化混合精度训练
    scaler = GradScaler()
    # TensorBoard如果路径权限不对，会报无法使用GPU的问题！

    if rank == 0:
        if not os.path.exists(args.tensorboard_path):
            os.mkdir(args.tensorboard_path)

        # log_dir = f'{args.tensorboard_path}/block_original_decomp_kan_y_wave_mix{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        log_dir = f'{args.tensorboard_path}/traffic_96_block_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        writer = SummaryWriter(log_dir=log_dir)
    # end if

    best_loss = float('inf')
    model_state = None
    # early_stop_counter = 0

    print(f"[GPU {gpuID}] 开始训练...")
    # 加上mseloss
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = 0.0
        train_samples = 0
        t0 = time.time()
        last_ybatch = None

        cnt = 0
        total = len(train_loader)
        for i, (X_batch, X_orginal_batch, y_batch) in enumerate(train_loader):
            tb0 = time.time()
            X_batch = X_batch.to(gpuID)
            X_orginal_batch = X_orginal_batch.to(gpuID)
            y_batch = y_batch.to(gpuID)

            optimizer.zero_grad()
            # with autocast(device_type='cuda'):    # 启用混合精度
            outputs = model(X_batch,None, None, None)
            # outputs = model(X_batch)

            # print(outputs)
            y = last_ybatch if i and (y_batch.shape == last_ybatch.shape) else y_batch

            # loss_info = criterion_info(outputs, y, y_batch)
            loss_mse = criterion_mse(outputs, y_batch)
            # loss = loss_info + loss_mse
            loss = loss_mse

            last_ybatch = y_batch
            # scaler.scale(loss).backward()  # 缩放梯度
            # scaler.step(optimizer)  # 更新优化器
            # scaler.update()  # 更新 scaler
            loss.backward()
            optimizer.step()

            # 将 loss 同步到所有 GPU 并取平均
            # 同步损失
            loss_reduced = loss.clone().detach()
            if world_size > 1:
                dist.all_reduce(loss_reduced, op=dist.ReduceOp.SUM)
                avg_loss = loss_reduced.item() / world_size
            else:
                avg_loss = loss.item()
            train_loss += avg_loss * X_batch.size(0)
            train_samples += X_batch.size(0)
            # train_loss += loss.item() * X_batch.size(0)
            if rank==0:
                cnt += 1
                print(f"Batch {cnt}/{total} time={time.time()-tb0:.4f}S.\r", end="")
        # end for

        train_loss /= train_samples
        scheduler.step()
        t1 = time.time()

        # 验证 只需要看MSEloss
        model.eval()
        val_loss = 0.0
        val_mae_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            # with autocast(device_type='cuda'):  # 验证也用混合精度
            for X_batch, X_orginal_batch, y_batch in val_loader:
                X_batch = X_batch.to(gpuID)
                X_orginal_batch = X_orginal_batch.to(gpuID)
                y_batch = y_batch.to(gpuID)

                outputs = model(X_batch, None, None, None)
                loss = criterion_mse(outputs, y_batch)
                loss_mae = criterion_mae(outputs, y_batch)
                # outputs = model(X_batch)
                # loss = criterion_mse(outputs, y_batch)
                # # --- MAE_loss ---
                # loss_mae = criterion_mae(outputs, y_batch)
                # val_loss += loss.item() * X_batch.size(0)
                # 同步验证 loss
                loss_reduced = loss.clone()
                if world_size > 1:
                    dist.all_reduce(loss_reduced, op=dist.ReduceOp.SUM)
                    avg_loss = loss_reduced.item() / world_size
                else:
                    avg_loss = loss.item()
                # --- 分布式同步 MAE ---
                mae_reduced = loss_mae.clone()
                if world_size > 1:
                    dist.all_reduce(mae_reduced, op=dist.ReduceOp.SUM)
                    avg_mae = mae_reduced.item() / world_size
                else:
                    avg_mae = loss_mae.item()

                val_mae_loss += avg_mae * X_batch.size(0)
                val_loss += avg_loss * X_batch.size(0)
                val_samples += X_batch.size(0)

        val_loss /= val_samples

        val_mae_loss /= val_samples  # 计算平均 MAE

        # 只在主进程中记录日志和早停
        if rank == 0:
            t2 = time.time()
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            print(f"\nEpoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val MAE: {val_mae_loss:.6f}, training time={t1-t0:.4f}S, valid time={t2-t1:.4f}S.")
            latest = train_loss if args.save_best=="train" else val_loss
            ##########################################################################
            # 保存最佳模型。
            ##########################################################################
            # # 下面应该只适用于数据分布，各节点复制模型的分布式训练方式。
            if best_loss > latest:
                # 在多卡分布式训练的情况下，下面是保存一张卡上的模型。
                # 以后载入的时候，先载入模型，然后再把模型分布到多卡上。
                if hasattr(model, 'module'):
                    model_state = model.module.state_dict()
                else:
                    model_state = model.state_dict()
                # "optimizer_state": optimizer.state_dict(),
                best_loss = latest
                # 为了避免频繁写硬盘，影响训练速度。
                # 所以先把最佳模型保存再内存中。等训练结束后再写硬盘。
            # end if

            # # 早停逻辑
            # if val_loss < best_loss - args.min_delta:
            #     best_loss = val_loss
            #     best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            #     early_stop_counter = 0
            # else:
            #     early_stop_counter += 1
            #     if early_stop_counter >= args.patience:
            #         print("早停触发，恢复最佳模型权重...")
            #         model.load_state_dict(best_model_state)
            #         break
    # end of training

    # 训练结束了，把最佳检查点写入硬盘文件。
    if rank == 0:
        writer.close()
        async_save(
            {
                "model_state": model_state,
                "loss": best_loss
            },
            args.checkpoint_path
        )


        print(f"正在加载最佳模型权重 (Best Loss: {best_loss:.6f})...")
        if model_state is not None:
            # 注意：训练时 model 被 DDP 包裹了，所以结构是 model.module
            # 我们保存的 model_state 是 model.module.state_dict() (没有 module. 前缀)
            # 所以这里要加载回 model.module
            if hasattr(model, 'module'):
                model.module.load_state_dict(model_state)
            else:
                model.load_state_dict(model_state)


        # 测试（只在主进程中执行）
        test(model, args, gpuID=gpuID)
    # end if
# end train

def test(model, # 待测试的模型对象，或者为空。
         args, # 选项参数。
         gpuID # 运行测试过程的gpu号。
         ):
    """
    测试只能在单进程中进行。因为没有进行多进程间的loss合并。
    """
    # 载入测试数据
    test_loader, xShape, yShape = __loadData(rank=0, world_size=1, args=args, isTraining=False)

    if model is None:
        # 生成模型对象
        model = __loadModel(args, gpuID, xShape=xShape, yShape=yShape)
    # end if

    print("开始测试...")
    model.eval()
    test_loss = 0.0
    test_mae_loss = 0.0
    test_samples = 0
    criterion = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    with torch.no_grad():
        for X_batch, X_orginal_batch, y_batch in test_loader:
            X_batch = X_batch.to(gpuID)
            X_orginal_batch = X_orginal_batch.to(gpuID)
            y_batch = y_batch.to(gpuID)
            outputs = model(X_batch,None, None, None)
            loss = criterion(outputs, y_batch)
            mae = criterion_mae(outputs, y_batch)

            avg_loss = loss.item()
            test_loss += avg_loss * X_batch.size(0)
            test_mae_loss += mae.item() * X_batch.size(0)  # 【新增】
            test_samples += X_batch.size(0)
    test_loss /= test_samples
    test_mae_loss /= test_samples
    print(f"Test MSE: {test_loss:.6f}, Test MAE: {test_mae_loss:.6f}")
# end test

def parallelTrain(rank, gpu, args):
    """
    用于多GPU并行训练的进程函数。
    """
    world_size = 1
    gpuID = 0
    if isinstance(gpu, int):
        # 使用指定单个GPU运行
        world_size = 1
        gpuID = gpu
    else:
        # 多GPU运行
        world_size = len(gpu)
        gpuID = gpu[rank]
    # end if

    # 设置分布式运行环境
    setup(rank, world_size)
    # 开始训练过程
    train(rank, world_size, args, gpuID)
    # 退出分布式运行
    cleanup()
# end parallelTrain

def save(data, path):
    """
    保存模型字典到指定硬盘文件中。
    """
    try:
        torch.save(data, path)
    except Exception as e:
        print(f"保存检查点文件{path}失败！\n错误原因：{e}")
# end save

# 异步保存
import threading
def async_save(data, path):
    threading.Thread(target=save, args=(data, path)).start()
