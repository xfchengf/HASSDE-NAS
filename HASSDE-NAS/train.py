import argparse
import os
import logging
import time
import datetime
import PIL.Image as Image
import h5py
import numpy as np
import torch
import torch.utils.data
import errno
import sys
from nas import cfg
import torch.nn.functional as F
from nas import build_dataset
from nas import make_lr_scheduler
from nas import make_optimizer
from nas import Checkpointer
from nas import MetricLogger
from tensorboardX import SummaryWriter
from nas import color_dict
from nas import HSIdatasettest
from nas import HSItrainnet
from nas import HSIsearchnet
from skimage import measure
from nas import model_visualize

ARCHITECTURES = {
    "searchnet": HSIsearchnet,
    "trainnet": HSItrainnet,
}


def compute_params(model):
    n_params = 0
    for m in model.parameters():
        n_params += m.numel()
    return n_params


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
        # 混淆矩阵元素初始化
        self.tp = 0  # 真正例（水体正确预测）
        self.tn = 0  # 真反例（背景正确预测）
        self.fp = 0  # 假正例（背景误报为水体）
        self.fn = 0  # 假反例（水体漏检）

        # 小目标统计
        self.small_gt = 0  # 小目标真实数量
        self.small_detected = 0  # 检测到的小目标数量

        # 损失统计
        self.loss_sum = 0.0  # 累计损失
        self.total_pixels = 0  # 有效像素总数（用于计算平均损失）

    def __call__(self, pred, targets):
        """更新统计指标"""
        # 计算混淆矩阵
        pred_label = torch.argmax(pred, dim=1)

        # 二分类假设：0=背景，1=水体
        water_pred = (pred_label == 1)
        water_gt = (targets == 1)
        background_pred = (pred_label == 0)
        background_gt = (targets == 0)

        # 累积统计
        self.tp += torch.sum(water_pred & water_gt).item()
        self.fp += torch.sum(water_pred & ~water_gt).item()
        self.fn += torch.sum(~water_pred & water_gt).item()
        self.tn += torch.sum(background_pred & background_gt).item()

        # 小目标统计
        self._update_small_objects(pred_label.cpu().numpy(), targets.cpu().numpy())

        # 损失计算（修复平均损失为0的问题）
        logits = pred.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        labels = targets.view(-1)
        valid_mask = (labels != -1)  # 排除无效像素
        loss = F.cross_entropy(logits[valid_mask], labels[valid_mask], reduction='sum')

        self.loss_sum += loss.item()
        self.total_pixels += valid_mask.long().sum().item()

    def _update_small_objects(self, pred_labels, target_labels):
        """更新小目标统计（假设小目标定义为面积<=50像素）"""
        min_size = 50
        for i in range(pred_labels.shape[0]):
            # 真实小目标
            gt_label = target_labels[i]
            gt_regions = measure.regionprops(measure.label(gt_label))
            for region in gt_regions:
                if region.area <= min_size and region.label == 1:  # 仅统计水体小目标
                    self.small_gt += 1

            # 预测小目标（需与真实小目标重叠）
            pred_label = pred_labels[i]
            pred_regions = measure.regionprops(measure.label(pred_label))
            for p_region in pred_regions:
                if p_region.area <= min_size and p_region.label == 1:
                    # 检查是否与真实小目标重叠
                    for g_region in gt_regions:
                        if (g_region.area <= min_size and
                                self._regions_overlap(p_region, g_region)):
                            self.small_detected += 1
                            break

    def _regions_overlap(self, region_a, region_b):
        """判断两个区域是否重叠"""
        a_coords = set(tuple(c) for c in region_a.coords)
        b_coords = set(tuple(c) for c in region_b.coords)
        return len(a_coords & b_coords) > 0

    def compute_metrics(self):
        eps = 1e-10  # 防止除零

        # 基础统计量
        total = self.tp + self.tn + self.fp + self.fn + eps

        # 1. Overall Accuracy (OA)
        oa = (self.tp + self.tn) / total

        # 2. Kappa Coefficient
        po = oa
        pe = ((self.tp + self.fp) * (self.tp + self.fn) +
              (self.fn + self.tn) * (self.fp + self.tn)) / (total ** 2)
        kappa = (po - pe) / (1 - pe + eps)

        # 3. IoU计算
        iou_water = self.tp / (self.tp + self.fp + self.fn + eps)
        iou_background = self.tn / (self.tn + self.fp + self.fn + eps)  # 修正原错误

        # 4. F1 Score
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)

        # 5. 平均损失
        avg_loss = self.loss_sum / (self.total_pixels + eps)

        # 6. 小目标召回率
        small_recall = self.small_detected / (self.small_gt + eps)

        return {
            'OA': oa,
            'Kappa': kappa,
            'IoU_Water': iou_water,
            'IoU_Background': iou_background,
            'F1': f1,
            'Small_Recall': small_recall,
            'Loss': avg_loss
        }

    def reset(self):
        """重置所有统计量"""
        self.tp = self.tn = self.fp = self.fn = 0
        self.small_gt = self.small_detected = 0
        self.loss_sum = 0.0
        self.total_pixels = 0


def save_results(cfg, result_anal, pred_map, test_map, save_dir):
    metrics = result_anal.compute_metrics()
    # 保存所有指标到文本文件
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        # 基础指标
        f.write(f"Overall Accuracy (OA): {metrics['OA'] * 100:.2f}%\n")
        f.write(f"Kappa Coefficient: {metrics['Kappa']:.3f}\n")
        f.write(f"Water IoU: {metrics['IoU_Water'] * 100:.2f}%\n")
        f.write(f"Background IoU: {metrics['IoU_Background'] * 100:.2f}%\n")

        # 高级指标
        f.write(f"F1 Score: {metrics['F1']:.3f}\n")
        f.write(f"Small Target Recall: {metrics['Small_Recall'] * 100:.2f}%\n")

        # 损失
        f.write(f"Average Loss: {metrics['Loss']:.4f}\n")
    # 保存预测图像
    img = labelmap_2_img(color_dict[cfg.DATASET.DATA_SET], pred_map)
    Image.fromarray(img).save(os.path.join(save_dir, 'prediction.png'))
    # 可选：如果需要同时保存与测试集重叠区域的可视化，可另存一个文件
    if cfg.DATASET.SHOW_ALL:
        overlap_map = pred_map.copy()
        overlap_map[test_map == 0] = 0  # 仅显示测试集标注区域的预测
        overlap_img = labelmap_2_img(color_dict[cfg.DATASET.DATA_SET], overlap_map)
        Image.fromarray(overlap_img).save(os.path.join(save_dir, 'prediction_overlap.png'))


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
        # 从compute_metrics获取指标字典
        metrics = result_anal.compute_metrics()
        # 提取需要的指标
        iou = metrics['IoU_Water']
        small_recall = metrics['Small_Recall']
        avg_loss = metrics['Loss']
        result_anal.reset()
    return iou, small_recall, avg_loss  # 返回格式：(iou, small_recall, avg_loss)


def h5_data_loader(data_dir):
    with h5py.File(data_dir, 'r') as g:
        data = g['data'][:]
        label = g['label'][:]
    return data, label


def h5_dist_loader2(data_dir):
    with h5py.File(data_dir, 'r') as h:
        height, width = h['height'][0], h['width'][0]
        category_num = h['category_num'][0]
        test_map = h['test_label_map'][0]
    return height, width, category_num, test_map


def get_patches_list2(height, width, crop_size, overlap):
    patch_list = []
    if overlap:
        slide_step = crop_size // 2
    else:
        slide_step = crop_size
    x1_list = list(range(0, width - crop_size, slide_step))
    y1_list = list(range(0, height - crop_size, slide_step))
    x1_list.append(width - crop_size)
    y1_list.append(height - crop_size)
    x2_list = [x + crop_size for x in x1_list]
    y2_list = [y + crop_size for y in y1_list]
    for x1, x2 in zip(x1_list, x2_list):
        for y1, y2 in zip(y1_list, y2_list):
            patch = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
            patch_list.append(patch)
    return patch_list


def labelmap_2_img(color_list, label_map):
    h, w = label_map.shape
    img = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            r, g, b = color_list[str(label_map[i, j])]
            img[i, j] = [r, g, b]
    return np.array(img, np.uint8)


def do_train(
        model,
        train_loader,
        val_loaders,
        max_iter,
        val_period,
        optimizer,
        scheduler,
        checkpointer,
        arguments,
        writer,
        visual_dir):
    logger = logging.getLogger("nas.trainer")
    logger.info(f"Model Params: {compute_params(model) / 1000:.2f}K")
    logger.info("Start training")
    # 初始化训练状态
    iou = 0.0
    best_iou = 0.0
    small_recall = 0.0
    best_small_recall = 0.0
    best_loss = float('inf')
    meters = MetricLogger(delimiter="  ")
    # 定义综合评分计算函数
    start_iter = arguments["iteration"]
    start_time = time.time()  # 新增此行
    for iteration in range(start_iter, max_iter):
        iteration = iteration + 1
        arguments["iteration"] = iteration
        model.train()
        # 训练步骤
        data_start_time = time.time()
        images, targets = next(iter(train_loader))
        data_time = time.time() - data_start_time
        images = images.to(device='cuda', non_blocking=True)
        targets = targets.to(device='cuda', non_blocking=True)
        pred, main_loss = model(images, targets)
        # --- 反向传播与优化 ---
        optimizer.zero_grad()
        main_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        # --- 指标记录 ---
        batch_time = time.time() - data_start_time - data_time
        meters.update(loss=main_loss.item(), time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        if iteration % (val_period // 4) == 0:
            logger.info(
                meters.delimiter.join(
                    ["eta: {eta}",
                     "iter: {iter}",
                     "{meters}",
                     "max_mem: {memory:.0f}"]).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))
        # --- 验证与保存 ---
        if iteration % val_period == 0 or iteration == max_iter:
            logger.info(f"Validating at iteration {iteration}")  # 调试日志
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

            # ==== 记录详细指标 ====
            writer.add_scalar('Val/IoU', iou, iteration)
            writer.add_scalar('Val/Samll_Recall', small_recall, iteration)
            writer.add_scalar('Val/Avg_Loss', avg_loss, iteration)

            # 强制保存第一个检查点（防止初始全零）
            if iteration == val_period:
                checkpointer.save("model_init", **arguments)

            # 定期日志输出
        if iteration % 50 == 0:
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                f"Iter: {iteration}/{max_iter} | "
                f"IoU: {iou*100:.1f}% | "
                f"SmallRecall: {small_recall:.1f}% | "
                f"Loss: {meters.loss.avg:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"ETA: {eta_str}"
            )

        if iteration % val_period == 0:
            checkpointer.save("model_{:06d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        # 训练结束处理
    total_time = time.time() - start_time
    logger.info(f"Training complete. Total time: {str(datetime.timedelta(seconds=total_time))}")
    writer.close()


def train2(cfgb, output_dir):
    train_loaders, val_loaders = build_dataset(cfgb)
    model = build_model(cfgb)
    model = model.cuda()  # 单卡环境#
    # model = torch.compile(model)  # 添加编译优化
    geno_file = os.path.join(cfgb.OUTPUT_DIR, '{}'.format(cfgb.DATASET.DATA_SET), 'search/models/model_best.geno')
    genotype = torch.load(geno_file, map_location=torch.device("cpu"), weights_only=True)
    gene_cell = genotype
    visual_dir = output_dir
    best_model_visualization_dir = os.path.join(visual_dir, 'visualize', 'best_model')
    model_visualize(best_model_visualization_dir)
    # 普通训练模式下，make_optimizer 返回单个优化器
    optimizer = make_optimizer(cfgb, model)
    # 构建学习率调度器
    scheduler = make_lr_scheduler(cfgb, optimizer, len(train_loaders))
    checkpointer = Checkpointer(model, optimizer, scheduler, os.path.join(output_dir, 'models'), save_to_disk=True)
    extra_checkpoint_data = checkpointer.load(cfgb.MODEL.WEIGHT)
    arguments = {
        "iteration": 0,
        "gene_cell": gene_cell,  # 传递完整的基因型列表
        **extra_checkpoint_data
    }
    arguments.update(extra_checkpoint_data)
    val_period = cfgb.SOLVER.VALIDATE_PERIOD
    max_iter = cfgb.SOLVER.TRAIN.MAX_ITER
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log'), comment=cfgb.DATASET.DATA_SET)
    do_train(
        model,
        train_loaders,
        val_loaders,
        max_iter,
        val_period,
        optimizer,
        scheduler,
        checkpointer,
        arguments,
        writer,
        visual_dir)


def evaluation(cfgc):
    print('model build')
    trained_model_dir = os.path.join(cfgc.OUTPUT_DIR, '{}'.format(cfgc.DATASET.DATA_SET),
                                     'train{}'.format(cfgc.DATASET.TRAIN_NUM), 'models/model_best.pth')
    if not os.path.exists(trained_model_dir):
        print('trained_model does not exist')
        return None, None
    model = build_model(cfgc)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.cuda()  # 单卡环境
    model_state_dict = torch.load(trained_model_dir, weights_only=True).pop("model")
    try:
        model.load_state_dict(model_state_dict)
    except RuntimeError:
        # 如果失败，移除参数名称中的 `module.` 前缀
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict)  # 重新加载修正后的参数
    model.eval()
    print('load test set')
    data_root = cfgc.DATASET.DATA_ROOT
    data_set = cfgc.DATASET.DATA_SET
    batch_size = cfgc.DATALOADER.BATCH_SIZE_TEST
    dataset_dir = os.path.join(data_root, f'{data_set}.h5')
    dataset_dist_dir = os.path.join(cfgc.DATALOADER.DATA_LIST_DIR, '{}_dist_{}_train-{}_val-{}.h5'
                                    .format(data_set, cfgc.DATASET.DIST_MODE, float(cfgc.DATASET.TRAIN_NUM),
                                            float(cfgc.DATASET.VAL_NUM)))
    test_data, label_map = h5_data_loader(dataset_dir)
    height, width, category_num, test_map = h5_dist_loader2(dataset_dist_dir)
    result_save_dir = os.path.join(cfgc.OUTPUT_DIR, '{}'.format(cfgc.DATASET.DATA_SET),
                                   'eval_{}'.format(cfgc.DATASET.TRAIN_NUM))
    mkdir(result_save_dir)
    crop_size = cfgc.DATASET.CROP_SIZE
    overlap = cfgc.DATASET.OVERLAP
    test_patches_list = get_patches_list2(height, width, crop_size, overlap)
    dataset_test = HSIdatasettest(hsi_data=test_data, data_dict=test_patches_list)
    test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size=batch_size, pin_memory=True,
                                              num_workers=4)
    print(f'Evaluating on {len(dataset_test)} patches...')
    # 6. 使用GPU加速的累计张量
    result_anal = ResultAnalysis()
    pred_score_map = torch.zeros(category_num, height, width, device=device)
    pred_count_map = torch.zeros(height, width, device=device)
    with torch.no_grad():
        for patches, indices in test_loader:
            patches = patches.to(device, non_blocking=True)
            pred = model(patches)  # 模型原始输出 (B, C, H, W)

            # --- 从完整test_map提取目标块 ---
            batch_size = pred.shape[0]
            targets = torch.zeros(batch_size, cfg.DATASET.CROP_SIZE, cfg.DATASET.CROP_SIZE,
                                  dtype=torch.long, device=device)

            for i in range(batch_size):
                x1 = indices[0][i].item()
                x2 = indices[1][i].item()
                y1 = indices[2][i].item()
                y2 = indices[3][i].item()
                # 从完整标签图提取对应区域
                targets[i] = torch.from_numpy(test_map[y1:y2, x1:x2]).to(device)

            # --- 更新指标统计 ---
            result_anal(pred, targets)  # 直接传递模型输出和targets

            # --- 后续拼接预测图逻辑保持不变 ---
            pred = torch.softmax(pred, dim=1)  # 转概率
            for i in range(pred.shape[0]):
                x1, x2 = indices[0][i].item(), indices[1][i].item()
                y1, y2 = indices[2][i].item(), indices[3][i].item()
                # 累加概率到全局图
                pred_score_map[:, y1:y2, x1:x2] += pred[i]
                pred_count_map[y1:y2, x1:x2] += 1

    # 9. 使用GPU进行最终计算
    pred_score_map /= pred_count_map.clamp(min=1e-7).unsqueeze(0)
    pred_map = torch.argmax(pred_score_map, dim=0).cpu().numpy().astype(np.uint8)
    # 获取所有指标
    metrics = result_anal.compute_metrics()
    # 12. 正确传递配置参数
    # 保存结果
    save_results(
        cfgc,
        result_anal,
        pred_map,
        test_map,
        result_save_dir
    )

    torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="NAS Training and evaluation")
    parser.add_argument("--config-file", default='./configs/Houston2018/train.yaml', metavar="FILE",
                        help="path to config file", type=str)
    parser.add_argument("--device", default='0', help="path to config file", type=str)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    # 设置TensorFloat32加速（新增代码）
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True  # 是否混合精度都需要使用
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    output_dir = os.path.join(cfg.OUTPUT_DIR, '{}'.format(cfg.DATASET.DATA_SET),
                              'train{}'.format(cfg.DATASET.TRAIN_NUM))
    mkdir(output_dir + '/models')
    logger = setup_logger("nas", output_dir)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    train2(cfg, output_dir)
    evaluation(cfg)


if __name__ == "__main__":
    main()
