# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from utils.config2 import get_config
from models import build_model
from data import build_loader
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.logger import create_logger
from utils.utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor

from fvcore.nn import FlopCountAnalysis, flop_count_str

from timm.utils import ModelEma as ModelEma
from utils.utils_ema import load_checkpoint_ema, load_pretrained_ema, save_checkpoint_ema
# print(f"||{torch.multiprocessing.get_start_method()}||", end="")
# torch.multiprocessing.set_start_method("spawn", force=True)

# HSI分类指标
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score

# 导入可视化库
import matplotlib.pyplot as plt



def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    # parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    # 不强制要求分布式
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel, default 0 for single GPU')

    # for acceleration
    # parser.add_argument('--fused_window_process', action='store_true',
                        # help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')

    # 控制是否使用分布式训练
    parser.add_argument('--use_distributed', type=str2bool, default=False, help='Whether to use distributed training. If False, use single GPU mode.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def calculate_hsi_metrics(all_predictions, all_targets):
    """
    计算高光谱图像分类的评估指标
    
    Args:
        all_predictions: 所有预测结果
        all_targets: 所有真实标签
    
    Returns:
        dict: 包含OA、AA、Kappa等指标的字典
    """
    # 转换为numpy数组
    predictions_np = all_predictions.cpu().numpy() if torch.is_tensor(all_predictions) else all_predictions
    targets_np = all_targets.cpu().numpy() if torch.is_tensor(all_targets) else all_targets
    
    # 计算混淆矩阵
    cm = confusion_matrix(targets_np, predictions_np)
    
    # 总体精度 (Overall Accuracy)
    oa = accuracy_score(targets_np, predictions_np)
    
    # 平均精度 (Average Accuracy)
    # 计算每个类别的精度
    class_accuracies = []
    for i in range(cm.shape[0]):
        if np.sum(cm[i, :]) > 0:
            class_acc = cm[i, i] / np.sum(cm[i, :])
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    aa = np.mean(class_accuracies)
    
    # Kappa系数
    kappa = cohen_kappa_score(targets_np, predictions_np)
    
    # 每类精度
    per_class_acc = class_accuracies
    
    return {
        'OA': oa * 100,  # 转换为百分比
        'AA': aa * 100,
        'Kappa': kappa * 100,
        'per_class_acc': per_class_acc,
        'confusion_matrix': cm
    }


def main(config):
    
    torch.cuda.empty_cache()

    # 添加记录指标的列表
    train_losses = []
    val_losses = []
    train_accuracies = []  # 如果训练时也记录精度的话
    val_accuracies = []
    val_oas = []  # 对于HSI模型的OA指标
    val_aas = []  # 对于HSI模型的AA指标
    val_kappas = []  # 对于HSI模型的Kappa指标
    
    # 设置设备
    if config.USE_DISTRIBUTED:
        # 分布式
        device = torch.device(f"cuda:{config.LOCAL_RANK}")
    else:
        # 单GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # build_loader已修改for单GPU和eval无训练数据模式
    # 只修了一个，simmim那俩没改，不知道用没用，要用再回来改
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    
    if not (config.EVAL_MODE):
        try:
            logger.info(flop_count_str(FlopCountAnalysis(model, (dataset_val[0][0][None],))))
        except Exception as e:
            logger.info(str(model))
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"number of params: {n_parameters}")
            if hasattr(model, 'flops'):
                flops = model.flops()
                logger.info(f"number of GFLOPs: {flops / 1e9}")

    # 使用统一的device设置
    # model.cuda()
    model.to(device)

    model_without_ddp = model

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            # device='cpu' if args.model_ema_force_cpu else '',
            device=device if not args.model_ema_force_cpu else 'cpu', # 使用device
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)


    optimizer = build_optimizer(config, model)
    
    if config.USE_DISTRIBUTED:
        # 分布式模式使用DDP
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[config.LOCAL_RANK], 
            broadcast_buffers=False
        )
    
    loss_scaler = NativeScalerWithGradNormCount()

    # 非单纯测试模式时
    if data_loader_train is not None:
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
        else:
            lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    else:
        lr_scheduler = None

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    max_accuracy_ema = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        model_without_ddp, max_accuracy, max_accuracy_ema = load_checkpoint_ema(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger, model_ema)
        if config.EVAL_MODE:
            try:
                logger.info(flop_count_str(FlopCountAnalysis(model_without_ddp.cpu(), (dataset_val[0][0][None],))))
                # 统一使用device
                # model_without_ddp.cuda()
                model_without_ddp.to(device)
            except Exception as e:
                logger.info(str(model))
                n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"number of params: {n_parameters}")
                if hasattr(model, 'flops'):
                    flops = model.flops()
                    logger.info(f"number of GFLOPs: {flops / 1e9}")
            # 根据模型类型调用相应验证函数
            if config.MODEL.TYPE == 'vHeat':
                acc1, acc5, loss = validate(config, data_loader_val, model)
                logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            elif config.MODEL.TYPE == 'vHeat4HSI':
                hsi_metrics = validate_hsi(config, data_loader_val, model)
                logger.info(f"HSI metrics of the network on the {len(dataset_val)} test samples:")
                logger.info(f"OA: {hsi_metrics['OA']:.2f}%, AA: {hsi_metrics['AA']:.2f}%, Kappa: {hsi_metrics['Kappa']:.2f}%")
        if model_ema is not None:
            # 根据模型类型调用相应的验证函数
            if config.MODEL.TYPE == 'vHeat':
                acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
                logger.info(f"Accuracy of the network ema on the {len(dataset_val)} test images: {acc1_ema:.1f}%")
            elif config.MODEL.TYPE == 'vHeat4HSI':
                hsi_metrics_ema = validate_hsi(config, data_loader_val, model_ema.ema)
                logger.info(f"HSI metrics of the network ema on the {len(dataset_val)} test samples:")
                logger.info(f"OA: {hsi_metrics_ema['OA']:.2f}%, AA: {hsi_metrics_ema['AA']:.2f}%, Kappa: {hsi_metrics_ema['Kappa']:.2f}%")
            
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained_ema(config, model_without_ddp, logger, model_ema)
        # 根据模型类型调用相应验证函数
        if config.MODEL.TYPE == 'vHeat':
            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        elif config.MODEL.TYPE == 'vHeat4HSI':
            hsi_metrics = validate_hsi(config, data_loader_val, model)
            logger.info(f"HSI metrics of the network on the {len(dataset_val)} test samples:")
            logger.info(f"OA: {hsi_metrics['OA']:.2f}%, AA: {hsi_metrics['AA']:.2f}%, Kappa: {hsi_metrics['Kappa']:.2f}%")
        if model_ema is not None:
            # 根据模型类型调用相应的验证函数
            if config.MODEL.TYPE == 'vHeat':
                acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
                logger.info(f"Accuracy of the network ema on the {len(dataset_val)} test images: {acc1_ema:.1f}%")
            elif config.MODEL.TYPE == 'vHeat4HSI':
                hsi_metrics_ema = validate_hsi(config, data_loader_val, model_ema.ema)
                logger.info(f"HSI metrics of the network ema on the {len(dataset_val)} test samples:")
                logger.info(f"OA: {hsi_metrics_ema['OA']:.2f}%, AA: {hsi_metrics_ema['AA']:.2f}%, Kappa: {hsi_metrics_ema['Kappa']:.2f}%")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        if model_ema is not None:
            throughput(data_loader_val, model_ema.ema, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # 修改sampler设置
        if config.USE_DISTRIBUTED:
            # 分布式模式需要设置sampler的epoch
            data_loader_train.sampler.set_epoch(epoch)
        
        # 接收返回的训练损失
        train_loss = train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema)
        train_losses.append(train_loss)
        
        # 修改checkpoint保存逻辑：单GPU总是保存，分布式只在rank 0保存
        if (not config.USE_DISTRIBUTED) or (config.USE_DISTRIBUTED and dist.get_rank() == 0):
            if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint_ema(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, model_ema, max_accuracy_ema)

        # 根据模型类型调用相应验证函数
        if config.MODEL.TYPE == 'vHeat':
            acc1, acc5, val_loss = validate(config, data_loader_val, model)
            val_losses.append(val_loss)
            val_accuracies.append(acc1)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        # 修改最佳模型保存逻辑：单GPU总是保存，分布式只在rank 0保存
            if (not config.USE_DISTRIBUTED) or (config.USE_DISTRIBUTED and dist.get_rank() == 0):
                if acc1 > max_accuracy:
                    save_checkpoint(config,
                                    epoch,
                                    model,
                                    acc1,
                                    optimizer,
                                    lr_scheduler,
                                    loss_scaler,
                                    logger,
                                    best='best')
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        elif config.MODEL.TYPE == 'vHeat4HSI':
            hsi_metrics = validate_hsi(config, data_loader_val, model)
            val_losses.append(hsi_metrics['loss'])
            val_oas.append(hsi_metrics['OA'])
            val_aas.append(hsi_metrics['AA'])
            val_kappas.append(hsi_metrics['Kappa'])
            logger.info(f"HSI metrics of the network on the {len(dataset_val)} test samples:")
            logger.info(f"OA: {hsi_metrics['OA']:.2f}%, AA: {hsi_metrics['AA']:.2f}%, Kappa: {hsi_metrics['Kappa']:.2f}%")
            
            # 使用OA作为主要指标保存最佳模型
            oa = hsi_metrics['OA']
            if (not config.USE_DISTRIBUTED) or (config.USE_DISTRIBUTED and dist.get_rank() == 0):
                if oa > max_accuracy:
                    save_checkpoint(config,
                                    epoch,
                                    model,
                                    oa,  # 保存OA值
                                    optimizer,
                                    lr_scheduler,
                                    loss_scaler,
                                    logger,
                                    best='best')
            max_accuracy = max(max_accuracy, oa)
            logger.info(f'Max OA: {max_accuracy:.2f}%')
        if model_ema is not None:
            # 根据模型类型调用相应的验证函数
            if config.MODEL.TYPE == 'vHeat':
                acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
                logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1_ema:.1f}%")
                
                # 单GPU总是保存，分布式只在rank 0保存
                if (not config.USE_DISTRIBUTED) or (config.USE_DISTRIBUTED and dist.get_rank() == 0):
                    if acc1_ema > max_accuracy_ema:
                        save_checkpoint(config,
                                        epoch,
                                        model_ema.ema,
                                        acc1_ema,
                                        optimizer,
                                        lr_scheduler,
                                        loss_scaler,
                                        logger,
                                        best='ema_best')
                
                max_accuracy_ema = max(max_accuracy_ema, acc1_ema)
                logger.info(f'Max accuracy ema: {max_accuracy_ema:.2f}%')
                
            elif config.MODEL.TYPE == 'vHeat4HSI':
                hsi_metrics_ema = validate_hsi(config, data_loader_val, model_ema.ema)
                logger.info(f"HSI metrics of the network ema on the {len(dataset_val)} test samples:")
                logger.info(f"OA: {hsi_metrics_ema['OA']:.2f}%, AA: {hsi_metrics_ema['AA']:.2f}%, Kappa: {hsi_metrics_ema['Kappa']:.2f}%")
                
                # 使用OA作为主要指标保存最佳模型
                oa_ema = hsi_metrics_ema['OA']
                if (not config.USE_DISTRIBUTED) or (config.USE_DISTRIBUTED and dist.get_rank() == 0):
                    if oa_ema > max_accuracy_ema:
                        save_checkpoint(config,
                                        epoch,
                                        model_ema.ema,
                                        oa_ema,  # 保存OA值
                                        optimizer,
                                        lr_scheduler,
                                        loss_scaler,
                                        logger,
                                        best='ema_best')
                
                max_accuracy_ema = max(max_accuracy_ema, oa_ema)
                logger.info(f'Max OA ema: {max_accuracy_ema:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    # 绘制训练曲线
    plot_training_curves(config, train_losses, val_losses, val_accuracies, val_oas, val_aas, val_kappas)


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()

    # 获取设备
    device = next(model.parameters()).device

    for idx, (samples, targets) in enumerate(data_loader):
        # 修改这里：使用统一的设备设置
        # samples = samples.cuda(non_blocking=True)
        # targets = targets.cuda(non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        data_time.update(time.time() - end)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    
    # 返回训练损失
    return loss_meter.avg


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        # 使用统一的设备设置
        # images = images.cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)
        device = next(model.parameters()).device  # 获取模型所在的设备
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if hasattr(config, 'USE_DISTRIBUTED') and config.USE_DISTRIBUTED:
            # 分布式模式：需要同步
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)
        else:
            # 单GPU模式：直接使用，不需要同步
            pass  # acc1, acc5, loss 已经是正确的值

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

@torch.no_grad()
def validate_hsi(config, data_loader, model):
    """
    高光谱图像分类验证函数（用于vHeat4HSI模型）
    """
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    
    # 收集所有预测和真实标签
    all_predictions = []
    all_targets = []

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        # 使用统一的设备设置
        device = next(model.parameters()).device
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # 记录损失
        loss = criterion(output, target)
        
        # 获取预测结果
        _, predicted = torch.max(output, 1)
        
        # 收集结果
        all_predictions.append(predicted.cpu())
        all_targets.append(target.cpu())

        if hasattr(config, 'USE_DISTRIBUTED') and config.USE_DISTRIBUTED:
            # 分布式模式：需要同步损失
            loss = reduce_tensor(loss)
        else:
            # 单GPU模式：直接使用
            pass

        loss_meter.update(loss.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB')
    
    # 合并所有批次的结果
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # 计算高光谱分类指标
    hsi_metrics = calculate_hsi_metrics(all_predictions, all_targets)
    
    # 添加损失到指标中
    hsi_metrics['loss'] = loss_meter.avg
    
    logger.info(f' * Loss: {loss_meter.avg:.4f}')
    logger.info(f' * OA: {hsi_metrics["OA"]:.2f}%, AA: {hsi_metrics["AA"]:.2f}%, Kappa: {hsi_metrics["Kappa"]:.2f}%')
    
    return hsi_metrics

@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    # 获取设备
    device = next(model.parameters()).device

    for idx, (images, _) in enumerate(data_loader):
        # 使用统一的设备设置
        # images = images.cuda(non_blocking=True)
        images = images.to(device, non_blocking=True)

        batch_size = images.shape[0]

        for i in range(50):
            model(images)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        logger.info(f"throughput averaged with 30 times")
        
        tic1 = time.time()
        for i in range(30):
            model(images)

        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        torch.cuda.synchronize()

        tic2 = time.time()

        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        logger.info(f'Mem {memory_used:.0f}MB')
        return


def plot_training_curves(config, train_losses, val_losses, val_accuracies=None, val_oas=None, val_aas=None, val_kappas=None):
    """
    绘制训练过程曲线图
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # 1. 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. 精度曲线（针对普通分类）
    if val_accuracies:
        plt.subplot(2, 2, 2)
        plt.plot(epochs, val_accuracies, 'g-', label='Val Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.grid(True)
    
    # 3. HSI指标曲线
    if val_oas:
        plt.subplot(2, 2, 3)
        plt.plot(epochs, val_oas, 'c-', label='OA')
        plt.plot(epochs, val_aas, 'm-', label='AA')
        plt.plot(epochs, val_kappas, 'y-', label='Kappa')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics (%)')
        plt.title('HSI Classification Metrics')
        plt.legend()
        plt.grid(True)
    
    # 4. 训练损失细节
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Detail')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(config.OUTPUT, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training curves saved to {save_path}")
    
    # 保存指标数据为txt文件
    metrics_path = os.path.join(config.OUTPUT, 'training_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Epoch\tTrain_Loss\tVal_Loss")
        if val_accuracies:
            f.write("\tVal_Acc")
        if val_oas:
            f.write("\tOA\tAA\tKappa")
        f.write("\n")
        
        for i in range(len(train_losses)):
            f.write(f"{i+1}\t{train_losses[i]:.4f}\t{val_losses[i]:.4f}")
            if val_accuracies:
                f.write(f"\t{val_accuracies[i]:.2f}")
            if val_oas:
                f.write(f"\t{val_oas[i]:.2f}\t{val_aas[i]:.2f}\t{val_kappas[i]:.2f}")
            f.write("\n")
    
    logger.info(f"Training metrics saved to {metrics_path}")


if __name__ == '__main__':
    args, config = parse_option()

    # 检查并设置 USE_DISTRIBUTED
    if not hasattr(config, 'USE_DISTRIBUTED'):
        config.defrost()
        config.USE_DISTRIBUTED = getattr(args, 'use_distributed', False)
        config.freeze()

    # 添加一个配置项来控制是否使用分布式
    # USE_DISTRIBUTED = False  # 从config中读取

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if config.USE_DISTRIBUTED:
        # 原分布式初始化代码
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
        else:
            rank = -1
            world_size = -1
        torch.cuda.set_device(config.LOCAL_RANK)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.distributed.barrier()

        # 原分布式随机种子设置
        seed = config.SEED + dist.get_rank()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        #cudnn.benchmark = True

        # 原分布式学习率缩放计算
        # linear scale the learning rate according to total batch size, may not be optimal
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    else:
        # 单GPU
        device = torch.device(f"cuda:{config.LOCAL_RANK}" if torch.cuda.is_available() else "cpu")

        # 单GPU随机种子设置
        seed = config.SEED
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 单GPU学习率缩放计算
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
    
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    # 修改日志创建，使其适用于单GPU
    os.makedirs(config.OUTPUT, exist_ok=True)
    if config.USE_DISTRIBUTED:
        logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
    else:
        logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    # 修改配置文件保存，使其适用于单GPU
    if config.USE_DISTRIBUTED:
        if dist.get_rank() == 0:
            path = os.path.join(config.OUTPUT, "config.json")
            with open(path, "w") as f:
                f.write(config.dump())
            logger.info(f"Full config saved to {path}")
    else:    
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
