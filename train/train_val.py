import torch
import logging
import numpy as np
from tqdm import tqdm
import numpy as np
from torch.nn.utils import clip_grad_norm_
from tool.utils import ModelManager
from min_norm_solvers import MinNormSolver
from models.spatial_block import freeze_decoder

def train_apply(model=None, dataloader=None, epoch=None, optimizer=None, grad_scaler=None, criterion=None, metric_collection=None, device=None,usage_model=None,
            std=None, mean=None, logger=None, wandb=None, max_grad_norm=20, total_step=None, warmup_lr=None, warm_up_step=None,algorithm_type ='normal',
            train=False,use_freeze_decoder=False,spatial_name=None,**kwargs):
    """
    model: model
    dataloader: dataloader
    epoch: epoch
    optimizer: optimizer
    grad_scaler: GradScaler
    criterion: criterion
    metric_collection: metric_collection
    device: device
    std: std
    mean: mean
    logger: logger
    wandb: wandb
    max_grad_norm: max_grad_norm
    total_step: total_step
    warmup_lr: warmup_lr
    warm_up_step: warm_up_step
    """
    model.train()
    epoch_loss = 0.0
    assert std is not None and mean is not None, 'std and mean should not be None.'
    logger.info(f'Starting training for epoch {epoch}')
    for _, (static_sample, day_sample, half_day_sample, hour_sample, mask_sample) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        static_sample, day_sample, hour_sample, mask_sample = static_sample.to(device), day_sample.to(device), hour_sample.to(device), mask_sample.to(device)
        target = half_day_sample.to(device)  # half_day_sample 是目标数据
        # 创建一个掩码，标识出非零值
        non_zero_mask = target != 0
        target[non_zero_mask] = target[non_zero_mask] * std + mean

        total_step += 1
        optimizer.zero_grad()

        # 温暖期学习率调整
        if warm_up_step is not None and total_step < warm_up_step:
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr[total_step]

        with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
            output = model(hour_input=hour_sample, day_input=day_sample, half_day_input=mask_sample, static_input=static_sample)
            output = tuple([item * std + mean for item in output]) if isinstance(output, tuple) else output * std + mean
            loss_dict_original = criterion(output, target)
            # 提取损失字典中的元组第一个值
            loss_dict = {key: value[0] if isinstance(value, tuple) else value for key, value in loss_dict_original.items()}

        # 计算损失
        loss_dict = {loss_name: loss_val for loss_name, loss_val in loss_dict.items() if loss_val != 0}
        loss_dict = loss_dict[list(loss_dict.keys())[0]] if len(loss_dict) == 1 else loss_dict
        total_scaled_loss = loss_combine(loss_dict=loss_dict,algorithm_type=algorithm_type,optimizer=optimizer,model=model)
        # 冻结采样层
        freeze_decoder(model,usage_model=usage_model,spatial_name=spatial_name,use_freeze_decoder=use_freeze_decoder,train=train)
        # 更新梯度
        grad_scaler = grad_scaler_apply(grad_scaler=grad_scaler,loss=total_scaled_loss,optimizer=optimizer,max_grad_norm=max_grad_norm,model=model)
        epoch_loss += total_scaled_loss.item()

        # 更新度量值
        if metric_collection:
            metric_collection.update(output[0], target) if isinstance(output, tuple) else metric_collection.update(output,target)
    # 在epoch结束时，计算和打印平均度量值
    epoch_loss /= len(dataloader)
    #循环输出loss_tuple的所有数据
    logger.info(f'Epoch {epoch} loss: {epoch_loss}')
    wandb.log({"epoch": epoch, "train_loss": epoch_loss}) if wandb else None
    if metric_collection:
        average_metrics = metric_collection.calculate_averages()
        for metric_name, average_value in average_metrics.items():
            if average_value is not None:
                logger.debug(f"Epoch {epoch} {metric_name}: {average_value}")
                wandb.log({"epoch": epoch, "train_" + metric_name: average_value}) if wandb else None
    metric_collection.reset()

    return model, optimizer, grad_scaler, total_step

def val_apply(project_name:str='ST-Conv',model=None,epoch=None,non_improved_epoch=None,logger=None,patience=5,
        checkpoint_path=None,best_model_path=None,save_interval=None,save_checkpoint=None,
        optimizer=None,best_metrics=None,factor=None,device=None,metric_collection=None,
        dataloader=None,wandb=None,std=None,mean=None,**kwargs):
    model.eval()
    logger.info(f'Starting validation for epoch {epoch}')
    for _, (static_sample, day_sample, half_day_sample, hour_sample, mask_sample) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        static_sample, day_sample, hour_sample, mask_sample = static_sample.to(device), day_sample.to(device), hour_sample.to(device), mask_sample.to(device)
        target = half_day_sample.to(device)
        non_zero_mask = target != 0
        target[non_zero_mask] = target[non_zero_mask] * std + mean

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(hour_input=hour_sample, day_input=day_sample, half_day_input=mask_sample, static_input=static_sample)
                output = tuple([item * std + mean for item in output]) if isinstance(output, tuple) else output * std + mean

        if metric_collection:
            metric_collection.update(output[0], target) if isinstance(output, tuple) else metric_collection.update(output,target)

    if metric_collection:
        average_metrics = metric_collection.calculate_averages()
        for metric_name, average_value in average_metrics.items():
            if average_value is not None:
                logger.debug(f"Epoch {epoch} {metric_name}: {average_value}")
                wandb.log({f"val_{metric_name}": average_value, "epoch": epoch}) if wandb else None

    metric_collection.reset()
    if best_metrics is not None:
        current_metric = average_metrics['RMSE']
        is_best = False
        # 更新指标
        if current_metric < best_metrics.get('best_RMSE', float('inf')):
            best_metrics['best_RMSE'] = current_metric
            is_best = True
            logger.info(f'New best metric {current_metric} at epoch {epoch}')
            non_improved_epoch = 0
        else:
            non_improved_epoch += 1
            logger.info(f'No improvement in metric {current_metric} for {non_improved_epoch} epochs')
            if non_improved_epoch == patience:
                logger.info(f'No improvement in metric {current_metric} for {non_improved_epoch} epochs, '
                                f'adjusting learning rate')
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= factor
                non_improved_epoch = 0
        # 保存模型
        if is_best and best_model_path is not None:
            ModelManager.save_model(file_name=project_name, model=model, checkpoint_path=checkpoint_path, best_model_path=best_model_path, epoch=epoch, mode='best', optimizer=optimizer)
        elif save_interval is not None and (epoch + 1) % save_interval == 0 and save_checkpoint:
            ModelManager.save_model(file_name=project_name, model=model, checkpoint_path=checkpoint_path, best_model_path=best_model_path, epoch=epoch, mode='checkpoint', optimizer=optimizer)

    return model, optimizer, best_metrics, non_improved_epoch

def test_apply(model=None, dataloader=None, metric_collection=None, device=None, logger=None, wandb=None, std=None, mean=None,epoch=None,**kwargs):

    model.eval()
    assert std is not None and mean is not None, 'std and mean should not be None.'
    for _, (static_sample, day_sample, half_day_sample, hour_sample, mask_sample) in enumerate(tqdm(dataloader, desc=f"Test")):
        static_sample, day_sample, hour_sample, mask_sample = static_sample.to(device), day_sample.to(device), hour_sample.to(device), mask_sample.to(device)
        target = half_day_sample.to(device)
        non_zero_mask = target != 0
        target[non_zero_mask] = target[non_zero_mask] * std + mean

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(hour_input=hour_sample, day_input=day_sample, half_day_input=mask_sample, static_input=static_sample)
                output = tuple([item * std + mean for item in output]) if isinstance(output, tuple) else output * std + mean

        if metric_collection:
            metric_collection.update(output[0], target) if isinstance(output, tuple) else metric_collection.update(output,target)

    if metric_collection:
        average_metrics = metric_collection.calculate_averages()
        for metric_name, average_value in average_metrics.items():
            if average_value is not None:
                logger.debug(f"test {metric_name}: {average_value}")
                wandb.log({"epoch": epoch, "test_" + metric_name: average_value}) if wandb else None

    metric_collection.reset()
    return average_metrics

def loss_combine(loss_dict=None,algorithm_type=None,optimizer=None,model=None):
    """
    loss_dict: dict - value single loss
    algorithm_type: str - mgda、Cross、normal、None
    optimizer: optimizer
    model: model
    """
    if isinstance(loss_dict, dict):
        if algorithm_type == 'mgda':
            # 计算每个损失的梯度并存储
            grads = {}
            for loss_name, loss_val in loss_dict.items():
                optimizer.zero_grad()
                loss_val.backward(retain_graph=True)
                grads[loss_name] = [param.grad.clone() for param in model.parameters() if param.grad is not None]

            sol, _ = MinNormSolver.find_min_norm_element([grads[loss_name] for loss_name in loss_dict])
            scale = {loss_name: sol[i] for i, loss_name in enumerate(loss_dict)}

            optimizer.zero_grad()
            total_scaled_loss = 0.0
            for loss_name, loss_val in loss_dict.items():
                scaled_loss = scale[loss_name] * loss_val
                total_scaled_loss += scaled_loss

        elif algorithm_type == 'Cross':
            # 假设字典的第一个键值对是基准损失
            base_loss_name, base_loss = next(iter(loss_dict.items()))
            total_scaled_loss = base_loss.clone()

            # 遍历其余的损失并根据比例累加到总损失
            for loss_name, loss_val in loss_dict.items():
                if loss_name != base_loss_name:
                    scaled_loss = loss_val / (loss_val / base_loss).detach()
                    total_scaled_loss += scaled_loss

        elif algorithm_type == 'normal':
            scales = [0.2, 0.4, 0.2]
            num_active_losses = min(len(loss_dict), len(scales))
            adjusted_scales = [scale / sum(scales[:num_active_losses]) for scale in scales[:num_active_losses]]
            total_scaled_loss = 0.0

            for i, (loss_name, loss_val) in enumerate(loss_dict.items()):
                if i < num_active_losses:
                    scaled_loss = adjusted_scales[i] * loss_val
                    total_scaled_loss += scaled_loss
        else:
            total_scaled_loss = sum(loss_dict.values())
    else:
        total_scaled_loss = loss_dict
    return total_scaled_loss

def grad_scaler_apply(grad_scaler=None,loss=None,optimizer=None,max_grad_norm=20,model=None):
    """
    grad_scaler: GradScaler
    loss: loss
    optimizer: optimizer
    """
    if grad_scaler:
        grad_scaler.scale(loss).backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
    return grad_scaler

