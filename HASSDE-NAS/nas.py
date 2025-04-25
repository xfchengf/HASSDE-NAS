import random
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
from einops import rearrange
from torch import nn
from yacs.config import CfgNode
from collections import defaultdict
from collections import deque
import math
from bisect import bisect_right
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import os
from graphviz import Digraph
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.utils.prune as prune

CN = CfgNode
_C = CN()
cfg = _C

# -----------------------------------------------------------------------------
# SEARCH
# -----------------------------------------------------------------------------
_C.SEARCH = CN()
_C.SEARCH.ARCH_START_EPOCH = 20
_C.SEARCH.SEARCH_ON = False

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = "searchnet"
_C.MODEL.NUM_LAYERS = 4
_C.MODEL.NUM_BLOCKS = 3  # cell里面基本块数量
_C.MODEL.NUM_STRIDES = 3  # ASPPModule中的膨胀率,strides不能超过layers
_C.MODEL.AFFINE = True
_C.MODEL.WEIGHT = ""
_C.MODEL.PRIMITIVES_DBS = "HSI_DBS"
_C.MODEL.PRIMITIVES_GEA = "HSI_GEA"
_C.MODEL.PRIMITIVES_BFA = "HSI_BFA"
_C.MODEL.ACTIVATION_F = 'Leaky'
_C.MODEL.ASPP_RATES = (2, 4, 6)
_C.MODEL.USE_ASPP = True

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.DATA_ROOT = "./dataset"
_C.DATASET.DATA_SET = "gd"
_C.DATASET.CATEGORY_NUM = 2
_C.DATASET.CROP_SIZE = 25
_C.DATASET.PATCHES_NUM = 400
_C.DATASET.MCROP_SIZE = [16, 32, 48]
_C.DATASET.OVERLAP = True
_C.DATASET.SHOW_ALL = True
_C.DATASET.DIST_MODE = 'per'
_C.DATASET.TRAIN_NUM = 30
_C.DATASET.VAL_NUM = 10
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4  # DBScify the number of subprocesses used for data loading, which can DBSed up the data loading process.
_C.DATALOADER.BATCH_SIZE_TRAIN = 2
_C.DATALOADER.BATCH_SIZE_TEST = 2
_C.DATALOADER.DATA_LIST_DIR = "./sample"

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 200
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.WEIGHT_DECAY_BIAS = 0
_C.SOLVER.WEIGHT_DECAY = 1e-4  # 降低权重衰减（原0.0001）
_C.SOLVER.MOMENTUM = 0.95  # 增加动量项
_C.SOLVER.SEARCH = CN()
_C.SOLVER.SEARCH.LR_START = 3e-4
_C.SOLVER.SEARCH.LR_END = 1e-6
_C.SOLVER.SEARCH.MOMENTUM = 0.9
_C.SOLVER.SEARCH.WEIGHT_DECAY = 0.01
_C.SOLVER.SEARCH.LR_A = 0.001  # 提高架构参数学习率
_C.SOLVER.SEARCH.LR_END = 1e-6  # 添加最低学习率限制
_C.SOLVER.SEARCH.WD_A = 0.001  # 降低正则化强度
_C.SOLVER.SEARCH.T_MAX = 10
_C.SOLVER.TRAIN = CN()
_C.SOLVER.TRAIN.INIT_LR = 5e-4  # 增大初始学习率（原3e-5过小）
_C.SOLVER.TRAIN.POWER = 0.5  # 减缓多项式衰减速度（原0.9衰减过快）
_C.SOLVER.TRAIN.MAX_ITER = 100000
_C.SOLVER.SCHEDULER = 'poly'  # 改用余弦退火，避免学习率归零
_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.VALIDATE_PERIOD = 1
_C.SOLVER.STEPS = [60, 120, 180]  # 学习率调整的epoch节点
_C.SOLVER.GAMMA = 0.1  # 学习率衰减系数
_C.SOLVER.GRAD_CLIP = 1.0  # 添加梯度裁剪参数
_C.SOLVER.COSINE = CN()
_C.SOLVER.COSINE.T_MAX = _C.SOLVER.TRAIN.MAX_ITER  # 半周期长度（通常设为总epoch数）
_C.SOLVER.COSINE.ETA_MIN = 1e-6  # 最小学习率
_C.SOLVER.WARMUP = CN()
_C.SOLVER.WARMUP.ENABLED = True  # 是否启用预热
_C.SOLVER.WARMUP.FACTOR = 1.0  # 初始学习率缩放因子
_C.SOLVER.WARMUP.ITERS = 500  # 预热迭代次数
_C.SOLVER.WARMUP.METHOD = "linear"  # 预热方法：["constant", "linear"]
_C.SOLVER.PATIENCE = 5  # 早停耐心次数
_C.SOLVER.COMPOSITE_WEIGHTS = CN()  # 必须显式初始化子节点
_C.SOLVER.COMPOSITE_WEIGHTS.IOU = 0.6
_C.SOLVER.COMPOSITE_WEIGHTS.F1 = 0.3
_C.SOLVER.COMPOSITE_WEIGHTS.SMALL_RECALL = 0.1
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.RESULT_DIR = "."

gd42 = {
    '1': [255, 255, 255],
    '0': [0, 0, 0],
}

hn32 = {
    '1': [255, 255, 255],
    '0': [0, 0, 0],
}

splice = {
    '1': [255, 255, 255],
    '0': [0, 0, 0],
}

color_dict = {
    "gd42": gd42,
    "hn32": hn32,
    "splice": splice,
}


class HSIdataset(Dataset):
    def __init__(self, hsi_h5_dir, dist_h5_dir, data_dict, mode='train', aug=False, rand_crop=False, rand_map=False,
                 crop_size=32):
        self.data_dict = data_dict
        with h5py.File(hsi_h5_dir, 'r') as f:
            data = f['data'][:]
        self.data = data / data.max()
        if mode == 'train':
            with h5py.File(dist_h5_dir, 'r') as f:
                label_map = f['train_label_map'][0]
        elif mode == 'val':
            with h5py.File(dist_h5_dir, 'r') as f:
                label_map = f['val_label_map'][0]
        elif mode == 'test':
            with h5py.File(dist_h5_dir, 'r') as f:
                label_map = f['test_label_map'][0]
        self.label_map = label_map
        self.aug = aug
        self.rand_crop = rand_crop
        self.height, self.width = self.label_map.shape  # Get dimensions of label map
        self.crop_size = crop_size
        self.rand_map = rand_map
        self.cache = {}  # 添加内存缓存

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        if self.rand_map:
            label_map_t = self.get_rand_map(self.label_map)
        else:
            label_map_t = self.label_map
        x1 = 0
        x2 = 0
        y1 = 0
        y2 = 0
        if self.rand_crop:
            flag = 0
            while flag == 0:

                # 修改循环条件
                x1 = random.randint(0, self.width - self.crop_size - 1)
                x2 = x1 + self.crop_size
                y1 = random.randint(0, self.height - self.crop_size - 1)
                y2 = y1 + self.crop_size
                if label_map_t[y1:y2, x1:x2].max() > 0:
                    flag = 1
            input_data = self.data[y1:y2, x1:x2]
            target = label_map_t[y1:y2, x1:x2]
        else:
            patch_info = self.data_dict[idx]
            x1, x2, y1, y2 = patch_info['x1'], patch_info['x2'], patch_info['y1'], patch_info['y2']
            input_data = self.data[y1:y2, x1:x2]
            target = label_map_t[y1:y2, x1:x2]
        if self.aug:
            input_data, target = self.random_flip_lr(input_data, target)
            input_data, target = self.random_flip_tb(input_data, target)
            input_data, target = self.random_rot(input_data, target)
        # print(np.unique(target))  # 应输出 [0, 1]
        return (torch.from_numpy(input_data).float().permute(2, 0, 1).unsqueeze(dim=0),
                torch.from_numpy(target).long())

    def __len__(self):
        return len(self.data_dict)

    @staticmethod
    def random_flip_lr(input_data, target):
        if np.random.randint(0, 2):
            h, w, d = input_data.shape
            index = np.arange(w, 0, -1) - 1
            return input_data[:, index, :], target[:, index]
        else:
            return input_data, target

    @staticmethod
    def random_flip_tb(input_data, target):
        if np.random.randint(0, 2):
            h, w, d = input_data.shape
            index = np.arange(h, 0, -1) - 1
            return input_data[index, :, :], target[index, :]
        else:
            return input_data, target

    @staticmethod
    def random_rot(input_data, target):
        rot_k = np.random.randint(0, 4)
        return np.rot90(input_data, rot_k, (0, 1)).copy(), np.rot90(target, rot_k, (0, 1)).copy()

    @staticmethod
    def get_rand_map(label_map, keep_ratio=0.6):
        label_map_t = label_map
        label_indices = np.where(label_map > 0)
        label_num = len(label_indices[0])
        shuffle_indices = np.random.permutation(int(label_num))
        dis_num = int(label_num * (1 - keep_ratio))
        dis_indices = (label_indices[0][shuffle_indices[:dis_num]], label_indices[1][shuffle_indices[:dis_num]])
        label_map_t[dis_indices] = 0
        return label_map_t


class HSIdatasettest(Dataset):
    def __init__(self, hsi_data, data_dict):
        self.HSI_data = hsi_data / hsi_data.max()
        self.data_dict = data_dict

    def __getitem__(self, idx):
        patch_info = self.data_dict[idx]
        x1, x2, y1, y2 = patch_info['x1'], patch_info['x2'], patch_info['y1'], patch_info['y2']
        input_data = self.HSI_data[y1:y2, x1:x2]
        return torch.from_numpy(input_data).float().permute(2, 0, 1).unsqueeze(dim=0), [x1, x2, y1, y2]

    def __len__(self):
        return len(self.data_dict)


def h5_dist_loader(data_dir):
    with h5py.File(data_dir, 'r') as f:
        height, width = f['height'][0], f['width'][0]
        category_num = f['category_num'][0]
        train_map, val_map, test_map = f['train_label_map'][0], f['val_label_map'][0], f['test_label_map'][0]
    return height, width, category_num, train_map, val_map, test_map


def get_patches_list(height, width, crop_size, label_map, patches_num, shuffle=True, overlap=False, min_water_pixels=5):
    patch_list = []
    count = 0
    if shuffle:
        while count < patches_num:
            x1 = random.randint(0, width - crop_size - 1)
            y1 = random.randint(0, height - crop_size - 1)
            x2, y2 = x1 + crop_size, y1 + crop_size
            patch_label = label_map[y1:y2, x1:x2]
            water_pixels = np.sum(patch_label == 1)

            # 小水体过采样：若包含小水体，提高被选中的概率
            if water_pixels > 0:
                is_small_water = (0 < water_pixels <= min_water_pixels)
                weight = 3.0 if is_small_water else 1.0

                if random.random() < weight / (weight + 1):
                    patch_list.append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})
                    count += 1
    else:
        # 测试模式：滑动窗口全覆盖
        slide_step = crop_size // 2 if overlap else crop_size
        x_steps = list(range(0, width - crop_size + 1, slide_step))
        y_steps = list(range(0, height - crop_size + 1, slide_step))

        # 确保覆盖边缘
        if (width - crop_size) % slide_step != 0:
            x_steps.append(width - crop_size)
        if (height - crop_size) % slide_step != 0:
            y_steps.append(height - crop_size)

        for y1 in y_steps:
            for x1 in x_steps:
                x2 = x1 + crop_size
                y2 = y1 + crop_size
                patch_label = label_map[y1:y2, x1:x2]

                # 测试时保留所有非空块（即使全为背景）
                if patch_label.max() >= 0:  # 兼容可能存在的-1忽略标签
                    patch_list.append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})

    return patch_list


def build_dataset(cfg):
    data_root = cfg.DATASET.DATA_ROOT
    data_set = cfg.DATASET.DATA_SET
    crop_size = cfg.DATASET.CROP_SIZE
    data_list_dir = cfg.DATALOADER.DATA_LIST_DIR
    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.DATALOADER.BATCH_SIZE_TRAIN
    search_on = cfg.SEARCH.SEARCH_ON
    dist_dir = os.path.join(data_list_dir, '{}_dist_{}_train-{}_val-{}.h5'.
                            format(data_set,
                                   cfg.DATASET.DIST_MODE,
                                   float(cfg.DATASET.TRAIN_NUM),
                                   float(cfg.DATASET.VAL_NUM)))
    height, width, category_num, train_map, val_map, test_map = h5_dist_loader(dist_dir)

    if search_on:
        w_data_list = get_patches_list(height, width, crop_size, train_map, cfg.DATASET.PATCHES_NUM // 2, shuffle=True)
        a_data_list = get_patches_list(height, width, crop_size, train_map, cfg.DATASET.PATCHES_NUM // 2, shuffle=True)
        v_data_list = get_patches_list(height, width, crop_size, val_map, cfg.DATASET.PATCHES_NUM, shuffle=False)
        dataset_w = HSIdataset(hsi_h5_dir=os.path.join(data_root, '{}.h5'.format(data_set)),
                               dist_h5_dir=dist_dir,
                               data_dict=w_data_list, mode='train')
        dataset_a = HSIdataset(hsi_h5_dir=os.path.join(data_root, '{}.h5'.format(data_set)),
                               dist_h5_dir=dist_dir,
                               data_dict=a_data_list, mode='train')
        dataset_v = HSIdataset(hsi_h5_dir=os.path.join(data_root, '{}.h5'.format(data_set)),
                               dist_h5_dir=dist_dir,
                               data_dict=v_data_list, mode='val')
        data_loader_w = torch.utils.data.DataLoader(
            dataset_w,
            shuffle=True,  # True 表示每次 epoch 都要重新打乱数据，默认 False
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,  # 启用内存锁页
            persistent_workers=True,  # 保持worker进程存活
            prefetch_factor=4,  # 添加预取优化
            # multiprocessing_context='fork'  # 多处理特有，在Windows上，PyTorch使用“GEAwn”方法启动进程，而在Linux上，通常使用“fork”方法。
        )
        data_loader_a = torch.utils.data.DataLoader(
            dataset_a,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,  # 启用内存锁页
            persistent_workers=True,  # 保持worker进程存活
            prefetch_factor=4,  # 添加预取优化
            # multiprocessing_context='fork'  # 优化多进程
        )
        data_loader_v = torch.utils.data.DataLoader(
            dataset_v,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,  # 启用内存锁页
            persistent_workers=True,  # 保持worker进程存活
            prefetch_factor=4,  # 添加预取优化
            # multiprocessing_context='fork'  # 优化多进程
        )
        return [data_loader_w, data_loader_a], data_loader_v
    else:
        tr_data_list = get_patches_list(height, width, crop_size, train_map, cfg.DATASET.PATCHES_NUM, shuffle=True)
        te_data_list = get_patches_list(height, width, crop_size, test_map, cfg.DATASET.PATCHES_NUM, shuffle=False)
        dataset_tr = HSIdataset(hsi_h5_dir=os.path.join(data_root, '{}.h5'.format(data_set)),
                                dist_h5_dir=dist_dir,
                                data_dict=tr_data_list, mode='train', aug=True, rand_crop=True, crop_size=crop_size)

        dataset_te = HSIdataset(hsi_h5_dir=os.path.join(data_root, '{}.h5'.format(data_set)),
                                dist_h5_dir=dist_dir,
                                data_dict=te_data_list, mode='test')
        data_loader_tr = torch.utils.data.DataLoader(
            dataset_tr,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,  # 启用内存锁页
            persistent_workers=True,  # 保持worker进程存活
            prefetch_factor=4,  # 添加预取优化
            # multiprocessing_context='fork'  # 优化多进程
        )
        data_loader_te = torch.utils.data.DataLoader(
            dataset_te,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,  # 启用内存锁页
            persistent_workers=True,  # 保持worker进程存活
            prefetch_factor=4,  # 添加预取优化
            # multiprocessing_context='fork'  # 优化多进程
        )
        return data_loader_tr, data_loader_te


class MixedOp(nn.Module):
    def __init__(self, c, primitives, affine=True):
        super().__init__()
        self._ops = nn.ModuleList()
        self.prune_threshold = 0.05
        self.prune_interval = 50
        self.register_buffer('op_activity', torch.ones(len(primitives)))

        # 初始化候选操作
        for primitive in primitives:
            op = OPS[primitive](c, affine)
            self._ops.append(op)

    def forward(self, x, weights):
        # 原有剪枝逻辑（无创新点）
        if self.training and getattr(self, '_step_counter', 0) % self.prune_interval == 0:
            self._prune_ops(weights)
        self._step_counter += 1

        active_weights = weights[self.op_activity > self.prune_threshold]
        active_ops = [op for op, active in zip(self._ops, self.op_activity > self.prune_threshold) if active]
        return sum(w * op(x) for w, op in zip(active_weights, active_ops))

    def _prune_ops(self, weights):
        self.op_activity = 0.9 * self.op_activity + 0.1 * weights.abs().mean(dim=0)
        for i, op in enumerate(self._ops):
            if self.op_activity[i] < self.prune_threshold:
                prune.l1_unstructured(op, 'weight', amount=1)
                for param in op.parameters():
                    param.requires_grad_(False)


class CellBase(nn.Module):
    def __init__(self, blocks, c, primitives, empty_a1=False, affine=True):
        super().__init__()
        self._steps = blocks
        self._multiplier = blocks

        # 创新点1：动态路径权重参数 (核心改动)
        self.path_alphas = nn.ParameterList([
            nn.Parameter(torch.ones(2 + i))  # 初始化等权重
            for i in range(self._steps)
        ])

        # 创新点2：残差跨层连接 (核心改动)
        self.res_conv = nn.Sequential(
            nn.Conv3d(c, 32, 1) if c != 32 else nn.Identity(),
            nn.BatchNorm3d(32)
        )

        # 原有操作初始化保持不变
        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                op = MixedOp(c, primitives, affine=affine)
                self._ops.append(op)

        # 保持原输出层结构
        self.conv_end = nn.Sequential(
            nn.Conv3d(24 * self._multiplier, 32, 1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2))

    def forward(self, s0, s1, weights):
        states = [s1, s0]
        offset = 0

        # 初始化残差连接
        residual = self.res_conv(s0)  # 通道对齐

        for i in range(self._steps):
            s = 0
            for j, h in enumerate(states):
                if not self._empty_h1 or j > 0:
                    # 应用动态路径权重（核心创新点）
                    op_out = self._ops[offset + j](h, weights[offset + j])
                    s += torch.sigmoid(self.path_alphas[i][j]) * op_out

            offset += len(states)
            states.append(s)

            # 累积残差（核心创新点）
            residual_channel = residual.shape[1]
            residual += s[:, :residual_channel]  # 自动对齐通道

        # 最终输出（保持维度不变）
        out = torch.cat(states[-self._multiplier:], dim=1)
        out = self.conv_end(out) + residual  # 残差融合
        return out

    def genotype(self, weights):
        gene = []
        n = 2
        start = 0
        for i in range(self._steps):
            end = start + n
            w = weights[start:end].clone().detach()
            edges = sorted(range(i + 2),
                           key=lambda x: -max(w[x][s]
                                              for s in range(len(w[x]))))[:2]
            for j in edges:
                k_best = None
                for k in range(len(w[j])):
                    if k_best is None or w[j][k] > w[j][k_best]:
                        k_best = k
                gene.append((self._primitives[k_best], j))
            start = end
            n += 1
        return gene


class CellDBS(CellBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CellGEA(CellBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CellBFA(CellBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def drop_path(x, drop_prob):
    if drop_prob > 0:
        keep_prob = 1 - drop_prob
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class FixCell(nn.Module):
    def __init__(self, genotype, c):
        super(FixCell, self).__init__()
        op_names = genotype
        self._steps = len(op_names) // 2
        self._multiplier = self._steps
        self._ops = nn.ModuleList()
        self._indices = []
        for name in op_names:
            if name[0] in OPS:  # 确保操作名称在 OPS 字典中有定义
                op = OPS[name[0]](c, True)
                self._ops.append(op)
                self._indices.append(name[1])
            else:
                raise ValueError(f"Operation {name[0]} not found in OPS dictionary")
        self._indices = tuple(self._indices)
        self.conv_end = nn.Sequential(
            nn.Conv3d(24, 32, (1, 1, 1), (1, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, s0, s1, drop_prob):
        states = [s1, s0]
        for i in range(self._steps):
            s = 0
            for ind in [2 * i, 2 * i + 1]:
                op = self._ops[ind]
                h = op(states[self._indices[ind]])
                if self.training and drop_prob > 0:
                    if not isinstance(op, Identity):
                        h = drop_path(h, drop_prob)
                s = s + h
            states.append(s)
        out = torch.cat(states[-self._multiplier:], dim=1)
        out = self.conv_end(out)
        return out


class ASPPModule(nn.Module):
    def __init__(self, inp, oup, rates, affine=True, use_gap=True, activate_f='ReLU'):
        super(ASPPModule, self).__init__()
        self.conv1 = conv_bn(inp, oup, 1, 1, 0, affine=affine, activate_f=activate_f)
        self.atrous = nn.ModuleList()
        self.use_gap = use_gap
        for rate in rates:
            self.atrous.append(sep_bn(inp, oup, rate))
        num_branches = 1 + len(rates)
        if use_gap:
            self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     conv_bn(inp, oup, 1, 1, 0, activate_f=activate_f))
            num_branches += 1
        self.conv_last = conv_bn(oup * num_branches, oup, 1, 1, 0, affine=affine, activate_f=activate_f)

    def forward(self, x):
        atrous_outs = [atrous(x) for atrous in self.atrous]
        atrous_outs.append(self.conv1(x))
        if self.use_gap:
            gap = self.gap(x)
            gap = F.interpolate(gap, size=x.size()[2:], mode='bilinear', align_corners=False)
            atrous_outs.append(gap)
        x = torch.cat(atrous_outs, dim=1)
        x = self.conv_last(x)
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, device=x.device).ge_(self.drop).div(
                1 - self.drop).detach()
        else:
            return x + self.m(x)


class LinearBN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)


class Transformer_vit(nn.Module):
    def __init__(self, dim_in, crop_size, dim_head=32, heads=8, dropout=0.1, emb_dropout=0.1):
        super(Transformer_vit, self).__init__()
        tokens_num = crop_size * crop_size
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.pos_embedding = nn.Parameter(torch.randn(1, tokens_num, dim_in))
        self.drop_out = nn.Dropout(emb_dropout)  # 用于embedding的dropout
        self.dropout = dropout  # 保存 dropout 概率
        inner_dim = dim_head * heads
        # attention
        self.attend = torch.nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim_in, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_in),
            nn.Dropout(dropout)
        )
        # FF
        self.ffnet = nn.Sequential(
            nn.Linear(dim_in, dim_head * heads * 2),
            nn.GELU(),
            # nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_head * heads * 2, dim_in),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)

    def forward(self, x):
        b, c, h, w = x.shape
        sque_x = x.view(b, c, h * w).permute(0, 2, 1)
        x = sque_x + self.pos_embedding
        x = self.drop_out(x)
        qkv = self.to_qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # 使用优化后的注意力计算
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,  # 直接使用dropout参数
            is_causal=False
        )
        out = rearrange(attn_output, 'b h n d -> b n (h d)')
        x = self.to_out(out) + x
        x = self.ffnet(self.norm2(x)) + x
        out = x.permute(0, 2, 1).view(b, c, h, w)
        return out


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.activate_f = cfg.MODEL.ACTIVATION_F
        self.use_aspp = cfg.MODEL.USE_ASPP
        bxf = 48
        inp = 32
        rates = cfg.MODEL.ASPP_RATES
        self.pre_conv = ASPPModule(inp, 32, rates, use_gap=False, activate_f='leaky')
        self.proj = conv_bn(bxf, 32, 1, 1, 0, activate_f='leaky')
        self.transformer = Transformer_vit(dim_in=64, crop_size=cfg.DATASET.CROP_SIZE)
        self.pre_cls = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64))
        self.cls = nn.Conv2d(64, cfg.DATASET.CATEGORY_NUM, kernel_size=1, stride=1)

    def forward(self, x):
        x0, x1 = x
        # print("x0.size=", x0.shape)
        # print("x1.size=", x1.shape)
        x0 = self.proj(x0)
        x1 = self.pre_conv(x1)
        # print("x02.size=", x0.shape)
        # print("x12.size=", x1.shape)
        x = torch.cat((x0, x1), dim=1)
        x = x.mean(dim=2)
        x = self.transformer(x)
        x = self.pre_cls(x)
        pred = self.cls(x)
        return pred


class AutoDecoder(nn.Module):
    def __init__(self, cfg, out_strides):
        super(AutoDecoder, self).__init__()
        self.aspps = nn.ModuleList()
        self.activate_f = cfg.MODEL.ACTIVATION_F
        bxf = 32
        affine = cfg.MODEL.AFFINE
        num_strides = len(out_strides)
        for i, out_stride in enumerate(out_strides):
            rate = out_stride
            inp = 32
            oup = bxf
            self.aspps.append(ASPPModule(inp, oup, [rate], affine=affine, use_gap=False, activate_f='leaky'))
            self.pre_cls = nn.Sequential(
                nn.Conv2d(bxf * num_strides, bxf * num_strides, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(bxf * num_strides),
                nn.Conv2d(bxf * num_strides, bxf * num_strides, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(bxf * num_strides),
            )
        self.cls = nn.Conv2d(bxf * num_strides, cfg.DATASET.CATEGORY_NUM, kernel_size=1, stride=1)

    def forward(self, x):
        x = [aspp(x_i) for aspp, x_i in zip(self.aspps, x)]
        x = torch.cat(x, dim=1).mean(dim=2)
        x = self.pre_cls(x)
        pred = self.cls(x)
        return pred


def build_decoder(cfg):
    if cfg.SEARCH.SEARCH_ON:
        out_strides = np.ones(cfg.MODEL.NUM_STRIDES, np.int16) * 2
        return AutoDecoder(cfg, out_strides)
    else:
        return Decoder(cfg)


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        # 损失权重系数
        self.d = 0.3  # CE权重
        self.e = 0.5  # Dice权重
        self.f = 0.2  # Focal权重
        self.ignore_lb = -100
        self.focal_gamma = 3.0  # Focal衰减系数
        self.pos_weight = 25.0  # 正样本权重(1/水体占比）
        # 基础损失函数
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=-100,
            weight=torch.tensor([1.0, self.pos_weight])  # 正样本加权
        )

    def dice_loss(self, logits, labels):
        probs = F.softmax(logits, dim=1)
        labels = labels.view(-1)
        valid_mask = labels != self.ignore_lb
        labels = labels[valid_mask]
        probs = probs[valid_mask]
        if labels.numel() == 0:
            return torch.tensor(0.0, device=logits.device)
        labels_one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
        intersection = torch.sum(probs * labels_one_hot, dim=0)
        union = torch.sum(probs, dim=0) + torch.sum(labels_one_hot, dim=0)
        dice = (2. * intersection + 1e-5) / (union + 1e-5)
        return 1 - dice.mean()

    def focal_loss(self, logits, labels):
        log_softmax = F.log_softmax(logits, dim=1)
        labels = labels.view(-1)
        valid_mask = labels != self.ignore_lb
        labels = labels[valid_mask]
        log_softmax = log_softmax[valid_mask]

        if labels.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

        # 高效Focal Loss实现
        ce_loss = F.nll_loss(log_softmax, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma * ce_loss).mean()
        return focal_loss

    def forward(self, logits, labels):
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, logits.size(1))
        labels = labels.view(-1)
        ce_loss = self.ce_loss(logits, labels)
        dice_loss = self.dice_loss(logits, labels)
        focal_loss = self.focal_loss(logits, labels)
        combined_loss = self.d * ce_loss + self.e * dice_loss + self.f * focal_loss
        return combined_loss


HSI_DBS = [
    'hamm_DBS',
    'esep_s3',
    'esep_s5',
    'dilate_3-1',
    'dilate_5-1',
    'water_enh',
    'spec_diff',
    'res_block'
]

HSI_GEA = [
    'hamm_GEA',  # 空间注意力
    'dilate_1-5',  # 大感受野
    'spat_sep3',  # 高效空间提取
    'spat_sep5',  # 大范围上下文
    'axial_sep5',  # 横向特征
    'axial_sep3',
    'dilate_1-3',  # 平衡细节
    'res_block'
]

HSI_BFA = [
    'hamm_BFA',  # 光谱-空间联合注意力
    'cross_conv3',  # 跨维度融合
    'dilated_5-5',  # 大范围上下文
    'multi_sep5',  # 多尺度分离
    'multi_sep3',
    'cross_conv5',  # 大核形态捕捉
    'dilated_3-3',  # 中等范围
    'res_block'
]

PRIMITIVES = {
    "HSI_DBS": HSI_DBS,
    "HSI_GEA": HSI_GEA,
    "HSI_BFA": HSI_BFA,
}

OPS = {
    'axial_sep3': lambda c, affine: LeakySepConv(c, c, 3, 1, affine=affine),
    'axial_sep5': lambda c, affine: LeakySepConv(c, c, 5, 1, affine=affine),
    'esep_s3': lambda c, affine: LeakySepConv(c, c, 1, 3, affine=affine),
    'esep_s5': lambda c, affine: LeakySepConv(c, c, 1, 5, affine=affine),
    'multi_sep3': lambda c, affine: LeakySepConv(c, c, 3, 3, affine=affine),
    'multi_sep5': lambda c, affine: LeakySepConv(c, c, 5, 5, affine=affine),
    'dilate_3-1': lambda c, affine: DilatedConv(c, c, 3, 2, 1, affine=affine),
    'dilate_5-1': lambda c, affine: DilatedConv(c, c, 5, 2, 1, affine=affine),
    'dilate_1-3': lambda c, affine: DilatedConv(c, c, 1, 2, 3, affine=affine),
    'dilate_1-5': lambda c, affine: DilatedConv(c, c, 1, 2, 5, affine=affine),
    'dilated_3-3': lambda c, affine: DilatedConv(c, c, 3, 2, 3, affine=affine),
    'dilated_5-5': lambda c, affine: DilatedConv(c, c, 5, 3, 5, affine=affine),
    'hamm_DBS': lambda c, affine: HAMM(c, mode='DBS'),
    'hamm_GEA': lambda c, affine: HAMM(c, mode='GEA'),
    'hamm_BFA': lambda c, affine: HAMM(c, mode='BFA'),
    'cross_conv3': lambda c, affine: CrossConv(c, c, kernel_size=3, affine=affine),
    'cross_conv5': lambda c, affine: CrossConv(c, c, kernel_size=5, affine=affine),
    'spat_sep3': lambda c, affine: SpatialSeparableConv(c, c, kernel_size=3, affine=affine),
    'spat_sep5': lambda c, affine: SpatialSeparableConv(c, c, kernel_size=5, affine=affine),
    'spec_diff': lambda c, affine: SpectralDiff(c, affine),
    'water_enh': lambda c, affine: WaterBandEnhance(c, affine),
    'res_block': lambda c, affine: ResidualBlock(c)
}


class ResidualBlock(nn.Module):
    """通道注意力残差块"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv3d(channels, channels, 1),
            nn.BatchNorm3d(channels),
            nn.ReLU()
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv3d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = x
        x = self.conv(x)
        x = x * self.attn(x)
        return x + res


class LeakySepConv(nn.Module):
    def __init__(self, c_in, c_out, spa_s, spe_s, affine=True, repeats=2):
        super(LeakySepConv, self).__init__()

        def basic_op():
            return nn.Sequential(
                nn.LeakyReLU(negative_slope=0.2, inplace=False),
                nn.Conv3d(c_in, c_in, (1, spa_s, spa_s), padding=(0, spa_s // 2, spa_s // 2), groups=c_in, bias=False),
                nn.Conv3d(c_in, c_in, (spe_s, 1, 1), padding=(spe_s // 2, 0, 0), groups=c_in, bias=False),
                nn.BatchNorm3d(c_out, affine=affine),
            )

        self.op = nn.Sequential()
        for idx in range(repeats):
            self.op.add_module('sep_{}'.format(idx), basic_op())

    def forward(self, x):
        return self.op(x)


class WaterBandEnhance(nn.Module):
    def __init__(self, c, affine=True):
        super().__init__()
        # 可学习的波段注意力（初始化为偏向水体敏感波段）
        self.band_weights = nn.Parameter(torch.ones(1, c, 1, 1, 1))
        self.conv = nn.Sequential(
            nn.Conv3d(c, c, (3, 1, 1), padding=(1, 0, 0), groups=c, bias=False),  # 光谱维轻量卷积
            nn.BatchNorm3d(c, affine=affine),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x * torch.sigmoid(self.band_weights))  # 软注意力加权


class SpectralDiff(nn.Module):
    def __init__(self, c, affine=True):
        super().__init__()
        self.diff_conv = nn.Conv3d(c, c, (2, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), groups=c, bias=False)
        self.bn = nn.BatchNorm3d(c, affine=affine)

    def forward(self, x):
        x_diff = self.diff_conv(x)  # [B, C, D-1, H, W]
        # 检查 x_diff.shape[2] == 241
        restored = torch.cat([x[:, :, :1], x_diff], dim=2)  # [B, C, D, H, W]
        assert restored.shape[2] == x.shape[2], "波段数未恢复！"
        return restored


class CrossConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, affine=True):
        super().__init__()
        # 光谱维卷积 (D方向)
        self.conv_spe = nn.Conv3d(
            c_in, c_out, (kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0), bias=False
        )
        # 空间维卷积 (H, W方向)
        self.conv_spa = nn.Conv3d(
            c_out, c_out, (1, kernel_size, kernel_size),
            padding=(0, kernel_size // 2, kernel_size // 2), bias=False
        )
        self.bn = nn.BatchNorm3d(c_out, affine=affine)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # 先光谱卷积，再空间卷积
        x = self.conv_spe(x)
        x = self.conv_spa(x)
        x = self.bn(x)
        return self.act(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    @staticmethod
    def forward(x):
        return x


class DilatedConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation, spatial_kernel_size=None, affine=True):
        super(DilatedConv, self).__init__()

        # 如果不传递空间卷积尺寸，则使用统一的 kernel_size
        if spatial_kernel_size is None:
            spatial_kernel_size = kernel_size

        # 定义卷积操作
        self.op = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            # 光谱维度卷积
            nn.Conv3d(c_in, c_out, (kernel_size, 1, 1),
                      padding=(kernel_size // 2 * dilation, 0, 0),
                      dilation=(dilation, 1, 1), bias=False),
            # 空间维度卷积
            nn.Conv3d(c_out, c_out, (1, spatial_kernel_size, spatial_kernel_size),
                      padding=(0, spatial_kernel_size // 2 * dilation, spatial_kernel_size // 2 * dilation),
                      dilation=(1, dilation, dilation), bias=False),
            nn.BatchNorm3d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SpectralAttention(nn.Module):
    def __init__(self, c_in):
        super(SpectralAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Conv3d(c_in, c_in // 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_in // 2, c_in, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        # 注意：输入为经过波段聚合后的2D特征，故这里使用Conv3d时通道数写死为2（即[avg, max]）
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


class HAMM(nn.Module):
    def __init__(self, c_in, mode='BFA'):
        super(HAMM, self).__init__()
        self.mode = mode  # 'DBS', 'GEA', or 'BFA'

        # 光谱注意力（仅在 DBS 或 BFA 模式下启用）
        if mode in ['DBS', 'BFA']:
            self.spectral_attention = SpectralAttention(c_in)

        # 空间注意力（仅在 GEA 或 BFA 模式下启用）
        if mode in ['GEA', 'BFA']:
            self.spatial_attention = SpatialAttention()

    def forward(self, x):
        if self.mode == 'DBS':
            # DBS 模式：仅光谱注意力
            x = self.spectral_attention(x)
        elif self.mode == 'GEA':
            # GEA 模式：仅空间注意力
            x = self.spatial_attention(x)
        elif self.mode == 'BFA':
            # BFA 模式：先光谱注意力，再空间注意力
            x = self.spectral_attention(x)
            x = self.spatial_attention(x)
        return x


class SpatialSeparableConv(nn.Module):  # 空间可分离卷积（Spatial Separable Conv）
    def __init__(self, c_in, c_out, kernel_size=3, affine=True):
        super().__init__()
        # 横向卷积 (H方向)
        self.conv_h = nn.Conv3d(
            c_in, c_out, (1, kernel_size, 1),
            padding=(0, kernel_size // 2, 0), bias=False
        )
        # 纵向卷积 (W方向)
        self.conv_w = nn.Conv3d(
            c_out, c_out, (1, 1, kernel_size),
            padding=(0, 0, kernel_size // 2), bias=False
        )
        self.bn = nn.BatchNorm3d(c_out, affine=affine)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv_h(x)  # 处理 H 方向
        x = self.conv_w(x)  # 处理 W 方向
        x = self.bn(x)
        return self.act(x)


def conv_bn(inp, oup, kernel, stride, padding, affine=True, activate_f='leaky'):
    layers = nn.ModuleList([
        nn.Conv3d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm3d(oup, affine=affine)
    ])
    if activate_f == 'leaky':
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
    return nn.Sequential(*layers)


def sep_bn(inp, oup, rate=1):
    return nn.Sequential(
        nn.Conv3d(inp, inp, 3, stride=1,
                  padding=rate, dilation=rate, groups=inp,
                  bias=False),
        nn.BatchNorm3d(inp),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(inp, oup, 1, bias=False),
        nn.BatchNorm3d(oup),
        nn.LeakyReLU(negative_slope=0.2, inplace=True))


class SmoothedValue(object):
    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            if v != v:
                continue
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return AttributeError("Attribute does not exist")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.avg, meter.global_avg)
            )
        return self.delimiter.join(loss_str)


class WarmupMultiStepLR(LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class PolynormialLR(LRScheduler):
    def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_iter) ** self.power
                for base_lr in self.base_lrs]


class PolyCosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, max_iter, t_max, eta_min=0, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        self.t_max = t_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.t_max)) / 2
                * (1 - self.last_epoch / self.max_iter) ** self.power
                for base_lr in self.base_lrs]


class OptimizerDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def state_dict(self):
        return [optim.state_dict() for optim in self.values()]

    def load_state_dict(self, state_dicts):
        for state_dict, optim in zip(state_dicts, self.values()):
            optim.load_state_dict(state_dict)
            for state in optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()


def make_optimizer(cfg, model):
    if cfg.SEARCH.SEARCH_ON:
        return make_search_optimizers(cfg, model)
    else:
        return make_normal_optimizer(cfg, model)


def make_normal_optimizer(cfg, model):
    """
        构建普通训练模式下的优化器（单个优化器）。
        """
    # 参数分组：区分权重和偏置
    params = [
        {"params": [], "lr": cfg.SOLVER.TRAIN.INIT_LR * 0.1, "weight_decay": 0.0},  # 偏置参数
        {"params": [], "weight_decay": cfg.SOLVER.WEIGHT_DECAY}  # 权重参数
    ]

    # 遍历模型参数，将权重和偏置分组
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 跳过不需要梯度的参数

        if "bias" in name:
            params[0]["params"].append(param)  # 偏置参数
        else:
            params[1]["params"].append(param)  # 权重参数

    # 返回单个优化器
    return torch.optim.AdamW(
        params,
        lr=cfg.SOLVER.TRAIN.INIT_LR,  # 初始学习率
        betas=(0.9, 0.999),  # AdamW 的 beta 参数
        eps=1e-8,  # AdamW 的 epsilon 参数
        weight_decay=cfg.SOLVER.WEIGHT_DECAY  # 权重衰减
    )


def make_search_optimizers(cfg, model):
    """
        构建搜索模式下的优化器（权重优化器和架构优化器），并返回 OptimizerDict。
        """
    # 权重优化器（optim_w）：优化模型权重
    optim_w = torch.optim.AdamW(
        model.w_parameters(),  # 仅优化模型权重参数
        lr=cfg.SOLVER.SEARCH.LR_START,  # 初始学习率
        betas=(0.9, 0.999),  # AdamW 的 beta 参数
        eps=1e-8,  # AdamW 的 epsilon 参数
        weight_decay=cfg.SOLVER.SEARCH.WEIGHT_DECAY  # 权重衰减
    )

    # 架构优化器（optim_a）：优化架构参数
    optim_a = torch.optim.AdamW(
        model.a_parameters(),  # 仅优化架构参数
        lr=cfg.SOLVER.SEARCH.LR_A,  # 架构参数的学习率
        betas=(0.9, 0.999),  # AdamW 的 beta 参数
        eps=1e-8,  # AdamW 的 epsilon 参数
        weight_decay=cfg.SOLVER.SEARCH.WD_A  # 架构参数的权重衰减
    )

    # 返回 OptimizerDict，包含两个优化器
    return OptimizerDict(optim_w=optim_w, optim_a=optim_a)


def make_search_lr_scheduler(cfg, optimizer_dict, train_data_loader_len):
    optim_w = optimizer_dict['optim_w']
    optim_a = optimizer_dict['optim_a']
    train_iters_per_epoch = train_data_loader_len  # 使用传入的训练数据加载器长度
    # 为权重优化器配置 OneCycle 策略
    scheduler_w = OneCycleLR(
        optim_w,
        max_lr=cfg.SOLVER.SEARCH.LR_START,
        total_steps=cfg.SOLVER.MAX_EPOCH * train_iters_per_epoch,
        pct_start=0.3,
        div_factor=25,  # 初始学习率=max_lr/25
        anneal_strategy='cos',
        final_div_factor=1e4
    )

    # 为架构参数优化器配置余弦退火
    scheduler_a = CosineAnnealingLR(
        optim_a,
        T_max=cfg.SOLVER.MAX_EPOCH * train_iters_per_epoch,
        eta_min=cfg.SOLVER.SEARCH.LR_END
    )
    return {'scheduler_w': scheduler_w, 'scheduler_a': scheduler_a}


class WarmupCosineScheduler(LRScheduler):
    def __init__(
            self,
            optimizer,
            base_scheduler,
            warmup_factor=1e-3,
            warmup_iters=1000,
            warmup_method="linear",
            total_iters=10000,
            last_epoch=-1
    ):
        self.base_scheduler = base_scheduler
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Warmup 阶段
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                alpha = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.warmup_factor + (1 - self.warmup_factor) * (self.last_epoch / self.warmup_iters)
            else:
                raise ValueError(f"Invalid warmup method: {self.warmup_method}")
            return [base_lr * alpha for base_lr in self.base_lrs]

        # 进入余弦退火阶段
        return self.base_scheduler.get_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_iters:
            # Warmup 阶段手动更新
            super().step()
        else:
            # 委托给基础调度器
            self.base_scheduler.step()


def make_lr_scheduler(cfg, optimizer, train_data_loader_len=None):
    scheduler = None  # 初始化 scheduler 变量
    if cfg.SEARCH.SEARCH_ON:
        return make_search_lr_scheduler(cfg, optimizer, train_data_loader_len)
    if cfg.SOLVER.SCHEDULER == 'poly':
        power = cfg.SOLVER.TRAIN.POWER
        max_iter = cfg.SOLVER.TRAIN.MAX_ITER
        return PolynormialLR(optimizer, max_iter, power)
    if cfg.SOLVER.SCHEDULER == 'one cycle':
        # 使用OneCycleLR策略
        total_steps = cfg.SOLVER.MAX_EPOCH * train_data_loader_len
        scheduler = OneCycleLR(
            optimizer,
            max_lr=cfg.SOLVER.TRAIN.INIT_LR,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1e4
        )
    return scheduler


class Checkpointer(object):
    def __init__(
            self,
            model,
            optimizer=None,
            scheduler=None,
            save_dir="",
            save_to_disk=None,
            logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return
        if not self.save_to_disk:
            return
        data = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            **kwargs
        }
        if isinstance(self.scheduler, dict):
            # 搜索模式下的 scheduler 是一个字典
            data["scheduler"] = {
                "scheduler_w": self.scheduler["scheduler_w"].state_dict() if self.scheduler else None,
                "scheduler_a": self.scheduler["scheduler_a"].state_dict() if self.scheduler else None,
            }
        else:
            # 普通模式下的 scheduler 是一个单一对象
            data["scheduler"] = self.scheduler.state_dict() if self.scheduler else None
        model = self.model
        # 生成并保存基因型文件
        if name == "model_best" and isinstance(self.model, HSIsearchnet):
            # 如果是搜索模型，生成基因型
            gene_list, cell_types = model.genotype()
            geno_dir = os.path.join(self.save_dir)
            os.makedirs(geno_dir, exist_ok=True)
            geno_file = os.path.join(geno_dir, "model_best.geno")
            cell_type_file = os.path.join(geno_dir, "model_best.cell_types")
            torch.save(gene_list, geno_file)
            torch.save(cell_types, cell_type_file)
            print(f"[Checkpointer] Saved genotype to: {geno_file}")
        # 保存模型检查点
        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None):
        if self.has_checkpoint():
            f = self.get_checkpoint_file()
        if not f:
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            if isinstance(self.scheduler, dict):
                self.scheduler["scheduler_w"].load_state_dict(checkpoint["scheduler"]["scheduler_w"])
                self.scheduler["scheduler_a"].load_state_dict(checkpoint["scheduler"]["scheduler_a"])
            else:
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    @staticmethod
    def _load_file(f):
        return torch.load(f, weights_only=True)

    def _load_model(self, checkpoint):
        model_state_dict = checkpoint.pop("model")
        try:
            self.model.load_state_dict(model_state_dict)
        except RuntimeError:
            # 如果失败，移除参数名称中的 `module.` 前缀
            model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
            self.model.load_state_dict(model_state_dict)


class HSIsearchnet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_layers = int(cfg.MODEL.NUM_LAYERS)  # 强制转换为整数
        self.num_blocks = cfg.MODEL.NUM_BLOCKS
        self.primitives_DBS = PRIMITIVES[cfg.MODEL.PRIMITIVES_DBS]
        self.primitives_GEA = PRIMITIVES[cfg.MODEL.PRIMITIVES_GEA]
        self.primitives_BFA = PRIMITIVES[cfg.MODEL.PRIMITIVES_BFA]
        self.activatioin_f = cfg.MODEL.ACTIVATION_F
        affine = cfg.MODEL.AFFINE
        self.stem0 = conv_bn(1, 32, (5, 3, 3), (1, 1, 1), (2, 1, 1), affine)
        self.stem1 = conv_bn(32, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1), affine)
        self.cells_DBS = nn.ModuleList()
        self.cells_GEA = nn.ModuleList()
        self.cells_BFA = nn.ModuleList()
        self.cell_router_DBS = nn.ModuleList()
        self.cell_router_GEA = nn.ModuleList()
        self._cache_features = []  # 预先声明属性
        # 初始化时添加重置方法
        self.selected_branches = []  # 初始化为空列表
        self.selected_counter = []  # 初始化为空列表
        self.val_acc_history = deque(maxlen=5)  # 在此初始化验证准确率历史
        self.reset_selection_cache()  # 调用初始化方法
        # 添加通道适配层
        self.channel_adapter = nn.Sequential(
            nn.Conv3d(32, 8, kernel_size=1, bias=False),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # === 强制选择控制参数 ===
        self.force_select = None  # 当某分支未被选中时的强制选择标记
        for s in range(0, self.num_layers):
            self.cells_DBS.append(CellDBS(self.num_blocks, 8, self.primitives_DBS, affine=affine))
            self.cells_GEA.append(CellGEA(self.num_blocks, 8, self.primitives_GEA, affine=affine))
            self.cells_BFA.append(CellBFA(self.num_blocks, 8, self.primitives_BFA, affine=affine))
            if s == 0:
                self.cell_router_DBS.append(
                    conv_bn(32, 8, (1, 1, 1), (1, 1, 1), (0, 0, 0), affine=affine, activate_f='None'))
                self.cell_router_GEA.append(
                    conv_bn(32, 8, (1, 1, 1), (1, 1, 1), (0, 0, 0), affine=affine, activate_f='None'))
            else:
                self.cell_router_GEA.append(
                    conv_bn(32, 8, (1, 1, 1), (1, 1, 1), (0, 0, 0), affine=affine, activate_f='None'))
        self.decoder = build_decoder(cfg)
        k = sum(2 + i for i in range(self.num_blocks))
        num_ops = len(self.primitives_DBS)
        init_std = 0.01  # 较小的标准差
        self.arch_alphas = nn.Parameter(torch.randn(self.num_layers, k, num_ops) * init_std)
        self.arch_betas = nn.Parameter(torch.randn(self.num_layers, k, num_ops) * init_std)
        self.arch_deltas = nn.Parameter(torch.randn(self.num_layers, k, num_ops) * init_std)
        self.arch_gammas = nn.Parameter(torch.randn(self.num_layers, 3) * init_std)
        self.criteria = CombinedLoss()

    def reset_selection_cache(self):
        """重置跨epoch的缓存和计数器"""
        self.selected_branches.clear()  # 清空而不是重新赋值
        # 保持维度结构但清空内容
        self.selected_counter = [defaultdict(int) for _ in range(self.num_layers)]

    def w_parameters(self):
        return [param for name, param in self.named_parameters() if 'arch' not in name and param.requires_grad]

    def a_parameters(self):
        return [param for name, param in self.named_parameters() if 'arch' in name]

    def scores(self):
        # temperature参数建议低于1使得softmax分布更尖锐
        temp = 0.6  # 可调超参数。温度太低容易过快决策
        alphas = F.softmax(self.arch_alphas / temp, dim=-1)
        betas = F.softmax(self.arch_betas / temp, dim=-1)
        deltas = F.softmax(self.arch_deltas / temp, dim=-1)
        gammas = F.softmax(self.arch_gammas, dim=-1)
        return alphas, betas, deltas, gammas

    def forward(self, images, targets=None):
        self._cache_features.clear()  # 清空列表内容，而非重新创建
        alphas, betas, deltas, gammas = self.scores()
        input_0 = self.stem0(images)
        input_1 = self.stem1(F.leaky_relu(input_0, negative_slope=0.2))
        hidden_states = []
        for s in range(self.num_layers):
            cell_weights_DBS = alphas[s]
            cell_weights_GEA = betas[s]
            cell_weights_BFA = deltas[s]
            cell_weights_arch = gammas[s]
            if s == 0:
                input_0 = self.cell_router_DBS[s](input_0)
            input_1 = self.cell_router_GEA[s](input_1)
            out0 = self.cells_DBS[s](input_0, input_1, cell_weights_DBS) * cell_weights_arch[0]
            out1 = self.cells_GEA[s](input_0, input_1, cell_weights_GEA) * cell_weights_arch[1]
            out2 = self.cells_BFA[s](input_0, input_1, cell_weights_BFA) * cell_weights_arch[2]
            fused_out = out0 + out1 + out2
            self._cache_features.append(fused_out)  # 缓存每层输出特征
            hidden_states.append(fused_out)
            input_0 = input_1
            input_1 = fused_out
        pred = self.decoder(hidden_states)
        assert torch.isfinite(pred).all(), "非数值出现在模型输出!"
        if self.training:
            assert torch.isfinite(targets).all(), "标签包含非法值!"
            loss = self.criteria(pred, targets)
            assert torch.isfinite(loss), "损失值非有限!"
            return pred, loss
        else:
            return pred

    def evaluate_cells(self, layer_idx):
        # 优化后的单元评估函数（引入启发式方法）
        # ===== 初始化检查 =====
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} is out of range [0, {self.num_layers - 1}]")
        # ===== 阶段1：架构参数分析 =====
        alphas, betas, deltas, gammas = self.scores()
        current_gamma = gammas[layer_idx]  # 直接取用已归一化的概率分布
        CELL_TYPES = ['DBS', 'GEA', 'BFA']

        # 1.1 维护验证准确率历史（问题1修复）
        if not hasattr(self, 'val_acc_history'):
            self.val_acc_history = deque(maxlen=5)  # 记录最近5个epoch的验证准确率

        # 1.2 操作参数质量分析（数值稳定性改进）
        def analyze_operation_params(params):
            """返回标准化后的参数质量元组（max_prob, entropy, sparsity）"""
            max_prob = params.max(dim=-1)[0].mean().item()  # 平均最大概率
            # 熵计算（带数值保护）
            log_probs = torch.log(torch.clamp(params, min=1e-10))  # 防止log(0)
            entropy = - ((params * log_probs).sum(dim=-1) / np.log(params.size(-1)))
            entropy = 1 - entropy.mean().item()  # 归一化熵
            sparsity = (params > 0.3).float().mean().item()  # 稀疏程度
            return (max_prob + entropy + sparsity) / 3  # 综合质量得分

        # 各单元参数质量
        param_quality = {
            'DBS': analyze_operation_params(alphas[layer_idx]),
            'GEA': analyze_operation_params(betas[layer_idx]),
            'BFA': analyze_operation_params(deltas[layer_idx])
        }

        # ========== 阶段2：特征动态分析 ==========
        current_feat = self._cache_features[layer_idx]  # [B,C,D,H,W]
        B, C, D, H, W = current_feat.shape

        # 2.1 特征响应多样性计算
        def compute_feature_diversity(feat, mode):
            """数值稳定的特征多样性计算"""
            with torch.no_grad():
                if mode == 'DBS':
                    # 光谱模式：沿空间维度平均 [B,C,D]
                    feat = feat.mean(dim=[3, 4])
                    feat_flat = feat.permute(1, 0, 2).reshape(C, -1)  # [C, B*D]
                elif mode == 'GEA':
                    # 空间模式：沿光谱维度平均 [B,C,H,W]
                    feat = feat.mean(dim=2)
                    feat_flat = feat.permute(1, 0, 2, 3).reshape(C, -1)  # [C, B*H*W]
                elif mode == 'BFA':
                    # 光谱-空间联合模式 [B,C,D,H,W]
                    feat_flat = feat.permute(1, 0, 2, 3, 4).reshape(C, -1)  # [C, B*D*H*W]
                else:
                    raise ValueError(f"Invalid mode: {mode}. Must be 'DBS', 'GEA' or 'BFA'")
                # 计算协方差矩阵
                # ==== 添加数值稳定性处理 ====
                # 带保护的归一化 (防止协方差矩阵数值不稳定)
                feat_flat = feat_flat - feat_flat.mean(dim=1, keepdim=True)
                std = torch.clamp(feat_flat.std(dim=1, keepdim=True), min=1e-4, max=1e4)
                feat_flat = feat_flat / (std + 1e-8)

                # 协方差计算（带正则化）
                cov = torch.cov(feat_flat.float())  # 确保float32
                cov_reg = cov + 1e-3 * torch.eye(C, device=feat.device)
                eigenvals = torch.linalg.svdvals(cov_reg)
                return eigenvals.mean().item()

        feat_diversity = {
            'DBS': compute_feature_diversity(current_feat, 'DBS'),
            'GEA': compute_feature_diversity(current_feat, 'GEA'),
            'BFA': compute_feature_diversity(current_feat, 'BFA')
        }

        # ========== 阶段3：梯度动态分析 ==========
        grad_sensitivity = {}
        for cell_type in CELL_TYPES:
            params = {'DBS': alphas, 'GEA': betas, 'BFA': deltas}[cell_type]

            # 显式计算梯度并立即释放资源
            with torch.enable_grad():
                virtual_loss = params[layer_idx].sum()
                grad = torch.autograd.grad(
                    outputs=virtual_loss,
                    inputs=params,
                    retain_graph=False  # 不再保留计算图
                )[0][layer_idx].norm().item()

            grad_sensitivity[cell_type] = grad / 1e5  # 归一化

        # ========== 阶段4：综合评分与选择策略 ==========

        final_scores = {}
        prev_selected = self.selected_branches[:layer_idx]  # 仅取当前层之前的选择
        used_types = set(prev_selected)

        for ct in CELL_TYPES:
            # 基础评分
            gamma_val = current_gamma[CELL_TYPES.index(ct)].item()
            param_val = param_quality[ct]
            feat_val = feat_diversity[ct] / 50.0  # 缩放至合理范围
            grad_val = grad_sensitivity[ct] * 100.0
            base_score = 0.4 * (0.7 * gamma_val + 0.3 * param_val) + 0.3 * feat_val + 0.2 * grad_val

            # --- 新增1：多样性奖励项 ---
            diversity_ratio = 1 - (len(used_types) / len(CELL_TYPES))  # 未使用类型占比
            diversity_bonus = 0.15 * diversity_ratio  # 奖励系数可调（0.1~0.2）
            if ct not in used_types:
                base_score += diversity_bonus  # 未使用类型获得奖励

            final_scores[ct] = base_score
        selected = max(final_scores, key=lambda x: final_scores[x])
        # 更新选择记录
        self._update_selection(layer_idx, selected)
        return selected

    def _update_selection(self, layer_idx, selected):
        """更新选择记录"""
        # 确保selected_branches长度与当前层对应
        if len(self.selected_branches) <= layer_idx:
            self.selected_branches.append(selected)
        else:
            self.selected_branches[layer_idx] = selected
        # 更新当前层的计数器（通过 layer_idx 访问对应层的 defaultdict）
        self.selected_counter[layer_idx][selected] += 1  # 正确索引方式

    def genotype(self):
        gene_list = []  # 仅保存基因结构
        cell_types = []  # 仅保存细胞类型（'DBS'/'GEA'/'BFA'）
        genes = []
        alphas, betas, deltas, _ = self.scores()
        for s in range(self.num_layers):
            branch = self.evaluate_cells(s)
            if branch == 'DBS':
                genes = self.cells_DBS[s].genotype(alphas[s])
            elif branch == 'GEA':
                genes = self.cells_GEA[s].genotype(betas[s])
            elif branch == 'BFA':
                genes = self.cells_BFA[s].genotype(deltas[s])
            gene_list.append(genes)
            cell_types.append(branch)
        return gene_list, cell_types


def model_visualize(save_dir):
    geno_dir = os.path.join(cfg.OUTPUT_DIR, str(cfg.DATASET.DATA_SET), "search/models")
    geno_file = os.path.join(geno_dir, "model_best.geno")
    cell_type_file = os.path.join(geno_dir, "model_best.cell_types")
    gene_list = torch.load(geno_file, weights_only=True)
    cell_types = torch.load(cell_type_file, weights_only=True)
    visualize(gene_list, cell_types, save_dir)


def visualize(gene_list, cell_types, save_dir):
    font_config = {
        'fontname': 'Arial',  # 字体
        'fontsize': '34',  # 字体大小
        'fontcolor': '#333333'  # 字体颜色
    }
    # 添加类型映射字典
    type_mapping = {
        'DBS': 'DBS',
        'GEA': 'GEA',
        'BFA': 'BFA'
    }
    # 边属性配置（重点修改部分）
    edge_attr = {
        'fontsize': font_config['fontsize'],
        'fontname': font_config['fontname'],
        'fontcolor': font_config['fontcolor'],
        'penwidth': '2',  # 加粗边线
        'arrowsize': '0.8'  # 箭头大小调整
    }
    g = Digraph(
        format='png',
        edge_attr=edge_attr,  # 应用边属性
        # ===== 新增尺寸控制参数 =====
        graph_attr={
            'size': "11,6",  # 标准A4纸尺寸比例（单位：英寸）,宽度,高度
            'ratio': 'compress',  # 压缩布局填充设定尺寸,fill/expand/compress
            'rankdir': 'TB',  # 拓扑方向(TB=竖向, LR=横向)
            'dpi': '300',  # 提高分辨率
            'nodesep': '0.5',  # 节点水平间距
            'ranksep': '0.7',  # 层级垂直间距
            'overlap': 'scalexy',  # 智能重叠处理
        },

        node_attr=dict(frontsize='22', style='filled', shape='rect', align='center', fontsize=font_config['fontsize'],
                       height='0.5', width='0.5', penwidth='2', fontname=font_config['fontname'],
                       fontcolor=font_config['fontcolor']),
        engine='dot'
    )
    g.body.extend(['rankdir=TB'])
    for cell_id, (original_cell_type, cell_genes) in enumerate(zip(cell_types, gene_list)):
        # 将原始cell_type转换为缩写
        cell_type = type_mapping.get(original_cell_type, original_cell_type.upper())
        # 处理前前层和前层节点的cell_type缩写
        if cell_id == 0:
            pre_pre_cell = 'stem0'
            pre_cell = 'stem1'
        elif cell_id == 1:
            pre_cell_type = cell_types[cell_id - 1]
            pre_cell_abbr = type_mapping.get(pre_cell_type, pre_cell_type.upper())
            pre_pre_cell = 'stem1'
            pre_cell = f'cell_{pre_cell_abbr}_{cell_id - 1}'
        else:
            # 处理前前层
            pre_pre_cell_type = cell_types[cell_id - 2]
            pre_pre_cell_abbr = type_mapping.get(pre_pre_cell_type, pre_pre_cell_type.upper())
            pre_pre_cell = f'cell_{pre_pre_cell_abbr}_{cell_id - 2}'
            # 处理前层
            pre_cell_type = cell_types[cell_id - 1]
            pre_cell_abbr = type_mapping.get(pre_cell_type, pre_cell_type.upper())
            pre_cell = f'cell_{pre_cell_abbr}_{cell_id - 1}'
        cur_cell = f'cell_{cell_type.upper()}_{cell_id}'
        g.node(pre_pre_cell, fillcolor='darkseagreen2')
        g.node(pre_cell, fillcolor='darkseagreen2')

        node_num = len(cell_genes) // 2
        for i in range(node_num):
            g.node(name='C{}_{}_N{}'.format(cell_type, cell_id, i), fillcolor='lightblue')

        for i in range(node_num):
            # 中间节点名称使用缩写后的cell_type
            g.node(name=f'C{cell_type}_{cell_id}_N{i}', fillcolor='lightblue')
        for i in range(node_num):
            for k in [2 * i, 2 * i + 1]:
                op, j = cell_genes[k]
                if op != 'none':
                    if j == 1:
                        u = pre_pre_cell
                        v = 'C{}_{}_N{}'.format(cell_type, cell_id, i)
                        g.edge(u, v, label=op, fillcolor='red')
                    elif j == 0:
                        u = pre_cell
                        v = 'C{}_{}_N{}'.format(cell_type, cell_id, i)
                        g.edge(u, v, label=op, fillcolor='red')
                    else:
                        u = 'C{}_{}_N{}'.format(cell_type, cell_id, j - 2)
                        v = 'C{}_{}_N{}'.format(cell_type, cell_id, i)
                        g.edge(u, v, label=op, fillcolor='gray')

        # 当前cell节点标签使用缩写
        g.node(cur_cell, fillcolor='darkseagreen2', label=f'cell_{cell_type}_{cell_id}')
        for i in range(node_num):
            g.edge(f'C{cell_type}_{cell_id}_N{i}', cur_cell, fillcolor='palegoldenrod')
    # 在渲染前添加后处理步骤
    g.body.append('''// 统一缩放配置
    size="11,8.5!";
    margin=0;
    ''')
    g.render(save_dir, view=False)


class HSItrainnet(nn.Module):
    def __init__(self, cfg):
        super(HSItrainnet, self).__init__()
        geno_file = os.path.join(cfg.OUTPUT_DIR, '{}'.format(cfg.DATASET.DATA_SET),
                                 'search/models/model_best.geno')
        if os.path.exists(geno_file):
            print(f"Loading genotype from {geno_file}")
            self.genotype = torch.load(geno_file, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(
                f"Genotype file {geno_file} not found. "
                "Run search training first to generate it."
            )
        self.activate_f = cfg.MODEL.ACTIVATION_F
        self.num_layers = cfg.MODEL.NUM_LAYERS
        self.num_blocks = cfg.MODEL.NUM_BLOCKS
        self.stem0 = conv_bn(1, 32, (5, 3, 3), (1, 1, 1), (0, 1, 1))
        self.stem1 = conv_bn(32, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        skip_conv = [conv_bn(32, 48, (1, 1, 1), (1, 1, 1), (0, 0, 0), affine=False)]
        for i in range(1, self.num_layers):
            skip_conv.append(conv_bn(48, 48, (1, 1, 1), (1, 1, 1), (0, 0, 0), affine=False))
        self.skip_conv = nn.Sequential(*skip_conv)
        self.cell = nn.ModuleList()
        self.cell_router = nn.ModuleList()
        for layer, genes in enumerate(self.genotype):
            self.cell.append(FixCell(genes, 8))
            self.cell_router.append(conv_bn(32, 8, (1, 1, 1), (1, 1, 1), (0, 0, 0), activate_f='None'))
        self.decoder = build_decoder(cfg)
        self.criteria = CombinedLoss()

    def genotype(self):
        return self.genotype

    def forward(self, images, targets=None, drop_prob=-1):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed.")
        h0 = self.stem0(images)
        h1 = self.stem1(F.leaky_relu(h0, negative_slope=0.2))
        endpoint = self.skip_conv(h1)
        for i, [cell, cell_router] in enumerate(zip(self.cell, self.cell_router)):
            h0 = h1
            h1 = cell(cell_router(h0), cell_router(h1), drop_prob)
        pred = self.decoder([endpoint, F.leaky_relu(h1, negative_slope=0.2)])

        # 在计算损失前添加检查
        if self.training:
            loss = self.criteria(pred, targets)
            return pred, loss
        else:
            return pred
