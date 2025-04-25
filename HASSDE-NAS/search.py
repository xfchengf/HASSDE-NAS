import argparse
import os
import logging
import time
import datetime
import torch
import errno
import sys
from tensorboardX import SummaryWriter
from nas import cfg
from nas import build_dataset
from nas import make_lr_scheduler
from nas import make_optimizer
from nas import Checkpointer
from nas import MetricLogger
from nas import HSItrainnet
from nas import HSIsearchnet
from nas import model_visualize
import torch.nn.functional as F
from skimage import measure
from torch import Tensor

ARCHITECTURES = {
    "searchnet": HSIsearchnet,
    "trainnet": HSItrainnet,
}


def build_model(cfga):
    meta_arch = ARCHITECTURES[cfga.MODEL.META_ARCHITECTURE]
    return meta_arch(cfga)


def setup_logger(name, save_dir, distributed_rank=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class ResultAnalysis(object):
    def __init__(self):
        # 保持原始变量名
        self.correct_pixel = 0  # 现在表示真正例(TP)
        self.total_pixel = 0  # 现在表示(TP+FP+FN)

        # 新增小目标统计
        self.small_gt = 0  # 小目标真实数量
        self.small_detected = 0  # 小目标检测数量
        self.iou = 0
        self.small_recall = 0
        self.loss_sum = 0
        # 保持原始损失统计
        self.loss_sum = 0

    def _is_small_object(self, mask, min_size=10):
        """判断是否为小水体对象（连通区域分析）"""
        labeled = measure.label(mask, connectivity=2)
        regions = measure.regionprops(labeled)
        return any(r.area <= min_size for r in regions)

    def __call__(self, pred, targets):
        # 保持原始参数不变
        pred_label = torch.argmax(pred, dim=1)
        water_pred: torch.Tensor = (pred_label == 1)  # 明确类型
        water_gt: Tensor = torch.as_tensor(targets == 1)
        # 保持原始变量名但修改含义
        tp = torch.sum(water_pred & water_gt).item()
        fp = torch.sum(water_pred & ~water_gt).item()
        fn = torch.sum(~water_pred & water_gt).item()

        self.correct_pixel += tp  # 现在correct_pixel实际存储TP
        self.total_pixel += (tp + fp + fn)  # total_pixel存储TP+FP+FN

        # 修改后（添加 detach() 和类型注释）

        pred_np = water_pred.detach().cpu().numpy().astype(bool)
        gt_np = water_gt.detach().cpu().numpy().astype(bool)

        batch_size = pred.shape[0]
        for i in range(batch_size):
            # 统计包含小目标的真实样本
            if self._is_small_object(gt_np[i]):
                self.small_gt += 1
                # 检查是否检测到小目标
                if self._is_small_object(pred_np[i] & gt_np[i]):
                    self.small_detected += 1

        # 保持原始损失计算
        logits = pred.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        labels = targets.view(-1)
        valid_mask: Tensor = torch.as_tensor(labels != -1)  # 明确类型
        self.loss_sum += F.cross_entropy(logits[valid_mask],
                                         labels[valid_mask]).item() * valid_mask.sum().item()

    def reset(self):
        self.iou = 0
        self.small_recall = 0
        self.loss_sum = 0

    def get_result(self):
        eps = 1e-10
        iou = self.correct_pixel / (self.total_pixel + eps)
        small_recall = self.small_detected / (self.small_gt + eps)

        # 保持返回元组格式，新增指标
        return (
            iou * 100,  # 原始accuracy位置替换为IoU
            small_recall * 100,  # 新增小目标召回率
            self.loss_sum / (self.total_pixel + eps)
        )


@torch.inference_mode()  # 替代torch.no_grad()，更彻底禁用计算图
def inference(model, val_loaders):
    """修改后的推理评估函数，返回元组（IoU, 小目标召回率, 平均损失）"""
    print('start_inference')
    model.eval()
    result_anal = ResultAnalysis()
    with torch.inference_mode():
        for images, targets in val_loaders:
            images = images.to(device='cuda', non_blocking=True)  # 新增代码
            targets = targets.to(device='cuda', non_blocking=True)  # 修改代码
            pred = model(images, targets)
            result_anal(pred, targets)
        iou, small_recall, avg_loss = result_anal.get_result()  # 修改返回值为（IoU，小目标召回率,平均损失）
        result_anal.reset()
    return iou, small_recall, avg_loss  # 返回格式：(iou, small_recall, avg_loss)


def do_search(
        model,
        train_loaders,
        val_loaders,
        max_epoch,
        arch_start_epoch,
        val_period,
        optimizer,
        scheduler,
        checkpointer,
        checkpointer_period,
        arguments,
        writer,
        cfgc,
        visual_dir):
    logger = logging.getLogger("nas.searcher")
    logger.info("Start searching")
    start_epoch = arguments["epoch"]
    # 最佳模型指标
    best_iou = 0.0
    best_small_recall = 0.0
    best_loss = float('inf')
    start_training_time = time.time()
    for epoch in range(start_epoch, max_epoch):
        model.reset_selection_cache()  # 新增代码：每个epoch开始时重置缓存
        epoch = epoch + 1
        arguments["epoch"] = epoch
        train(model, train_loaders, optimizer, scheduler, epoch, train_arch=epoch > arch_start_epoch)    # 传递scheduler
        if epoch > cfgc.SEARCH.ARCH_START_EPOCH:
            save_dir = os.path.join(visual_dir, 'visualize', f'arch_epoch{epoch}')
            model_visualize(save_dir)
        # 验证阶段（修改指标计算）
        if epoch % val_period == 0:
            # 使用修改后的inference函数
            iou, small_recall, avg_loss = inference(model, val_loaders)

            # 模型选择策略（优先级：IoU > 小目标召回率 > 损失）
            is_better = False
            if iou > best_iou:
                is_better = True
            elif iou == best_iou:
                if small_recall > best_small_recall:
                    is_better = True
                elif small_recall == best_small_recall and avg_loss < best_loss:
                    is_better = True

            # 更新并保存最佳模型
            if is_better:
                best_iou = iou
                best_small_recall = small_recall
                best_loss = avg_loss
                checkpointer.save("model_best", **arguments)
                best_model_dir = os.path.join(visual_dir, 'visualize', 'best_model')
                model_visualize(best_model_dir)

            # TensorBoard记录（兼容原始字段，新增指标）
            writer.add_scalar('Val/IoU', iou, epoch)
            writer.add_scalar('Val/SmallRecall', small_recall, epoch)
            writer.add_scalar('Val/Loss', avg_loss, epoch)

        # 保持原始检查点保存逻辑
        if epoch % checkpointer_period == 0:
            checkpointer.save(f"model_{epoch:03d}", **arguments)
        if epoch == max_epoch:
            checkpointer.save("model_final", **arguments)

    # 保持原始时间统计
    total_training_time = time.time() - start_training_time
    logger.info(f"Total training time: {datetime.timedelta(seconds=total_training_time)}")
    logger.info(f"[Final Best] IoU: {best_iou:.1f}% | SmallRecall: {best_small_recall:.1f}%")


def train(model, data_loaders, optimizer, schedulers, epoch, train_arch=False):
    data_loader_w = data_loaders[0]
    data_loader_a = data_loaders[1]
    optim_w = optimizer['optim_w']
    optim_a = optimizer['optim_a']
    scheduler_w = schedulers['scheduler_w']
    scheduler_a = schedulers['scheduler_a']
    logger = logging.getLogger("nas.searcher")
    max_iter = len(data_loader_w)
    model.train()
    meters = MetricLogger(delimiter="  ")
    end = time.time()
    for iteration, (images, targets) in enumerate(data_loader_w):
        images = images.to(device='cuda', non_blocking=True)  # 新增代码
        targets = targets.to(device='cuda', non_blocking=True)  # 修改代码
        data_time = time.time() - end
        if train_arch:
            images_a, targets_a = next(iter(data_loader_a))
            images_a = images_a.to(device='cuda', non_blocking=True)
            targets_a = targets_a.to(device='cuda', non_blocking=True)
            pred, loss = model(images_a, targets_a)
            optim_a.zero_grad()
            loss.backward()
            optim_a.step()
            scheduler_a.step()  # 更新 optim_a 学习率
        pred, loss = model(images, targets)
        meters.update(loss=loss.item())
        optim_w.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # 添加梯度裁剪
        optim_w.step()
        scheduler_w.step()  # 更新 optim_w 学习率
        # 记录损失和时间
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        # 计算剩余时间
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        if iteration % 50 == 0:
            logger.info(
                meters.delimiter.join(
                    ["eta: {eta}",
                     "iter: {epoch}/{iter}",
                     "{meters}",
                     "lr: {lr:.6f}"]).format(
                    eta=eta_string,
                    epoch=epoch,
                    iter=iteration,
                    meters=str(meters),
                    lr=optim_w.param_groups[0]['lr']))


def search(cfgb, output_dir):
    train_loaders, val_loaders = build_dataset(cfgb)
    model = build_model(cfgb)
    model = model.cuda()
    # model = torch.compile(model, dynamic=True)
    optimizer = make_optimizer(cfgb, model)
    scheduler = make_lr_scheduler(cfgb, optimizer, len(train_loaders[0]))
    checkpointer = Checkpointer(model, optimizer, scheduler, output_dir + '/models', save_to_disk=True)
    extra_checkpoint_data = checkpointer.load(cfgb.MODEL.WEIGHT)
    arguments = {
        "epoch": 0,
        **extra_checkpoint_data
    }
    checkpoint_period = cfgb.SOLVER.CHECKPOINT_PERIOD
    val_period = cfgb.SOLVER.VALIDATE_PERIOD
    max_epoch = cfgb.SOLVER.MAX_EPOCH
    arch_start_epoch = cfgb.SEARCH.ARCH_START_EPOCH
    writer = SummaryWriter(logdir=output_dir + '/log', comment=cfgb.DATASET.DATA_SET)
    do_search(
        model,
        train_loaders,
        val_loaders,
        max_epoch,
        arch_start_epoch,
        val_period,
        optimizer,
        scheduler,
        checkpointer,
        checkpoint_period,
        arguments,
        writer,
        cfg,
        visual_dir=output_dir
    )


def main():
    parser = argparse.ArgumentParser(description="neural architecture search for water body identification")
    parser.add_argument("--config-file", default='./configs/gd/search.yaml', metavar="FILE",
                        help="path to config file", type=str)
    parser.add_argument("--device", default='0', help="path to config file", type=str)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    # 设置TensorFloat32加速（新增代码）
    # torch.set_float32_matmul_precision('high')  # 启用TF32加速矩阵运算，是否混合精度都可以使用
    # 启用CuDNN Benchmark
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True  # 是否混合精度都需要使用
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    output_dir = os.path.join(cfg.OUTPUT_DIR, '{}'.format(cfg.DATASET.DATA_SET), 'search')
    mkdir(output_dir)
    mkdir(os.path.join(output_dir, 'models'))
    logger = setup_logger("nas", output_dir)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    search(cfg, output_dir)


if __name__ == "__main__":
    main()
