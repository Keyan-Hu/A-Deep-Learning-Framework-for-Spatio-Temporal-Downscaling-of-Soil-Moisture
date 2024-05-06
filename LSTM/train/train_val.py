import torch
import numpy as np
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from utils.utils import ModelManager

def train_val(mode=None,model=None,dataloader=None,epoch=None,non_improved_epoch=None,
              best_metrics=None,device=None,warmup_lr=None,grad_scaler=None,logger=None,
              max_grad_norm=20,criterion=None,metric_collection=None,patience=None,
              optimizer=None,checkpoint_path=None,factor=None,save_interval=None,
              save_checkpoint=None,total_step=None,warm_up_step=None, mean = None,std = None):
    """
    mode: train or val
    model: model
    dataloader: dataloader
    epoch: epoch
    best_metrics:dict
    """
    assert mode in ['train', 'val'], 'Mode should be either "train" or "val".'
    model.train() if mode == 'train' else model.eval()
    epoch_loss = 0.0
    n_iter = len(dataloader)
    logger.info(f'Starting {mode} for epoch {epoch}')
    for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        data, target = data.to(device), target.to(device)
        
        '''
        tensor_values = data[0, 1, :]  # 假设这是您要打印的张量

        # 将张量转换为 NumPy 数组（如果它还不是）
        np_values = tensor_values.cpu().detach().numpy()

        # 格式化每个元素并打印
        formatted_values = [f'{x:.2f}' for x in np_values]
        print(formatted_values)
        '''
        
        target = target * std + mean
        total_step += 1
        if mode == 'train':
            optimizer.zero_grad()
            if warm_up_step is not None and total_step < warm_up_step:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr[total_step]

            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                output = model(data)
                output = output.squeeze()
                output = output * std + mean
                loss = criterion(output, target)

            if grad_scaler:
                grad_scaler.scale(loss).backward()
                clip_grad_norm_(model.parameters(), max_grad_norm)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                    output = model(data)
                    output = output.squeeze()
                    output = output * std + mean
                    loss = criterion(output, target)
        epoch_loss += loss.item()

        # 更新度量值
        if metric_collection:
            metric_collection.update(output.detach(), target.detach())

    # 在epoch结束时，计算和打印平均度量值
    epoch_loss /= len(dataloader)
    logger.info(f'Epoch {epoch} Average Loss: {epoch_loss}')

    if metric_collection:
        average_metrics = metric_collection.calculate_averages()
        for metric_name, average_value in average_metrics.items():
            if average_value is not None:
                logger.debug(f"Epoch {epoch} {metric_name}: {average_value}")

    metric_collection.reset()

    if mode == 'val' and best_metrics is not None:
        current_metric = average_metrics['RMSE']
        is_best = False
        if current_metric < best_metrics.get('best_RMSE', float('inf')):
            best_metrics['best_RMSE'] = current_metric
            is_best = True
            logger.info(f'New best metric {current_metric} at epoch {epoch}')
        else:
            non_improved_epoch += 1
            logger.info(f'No improvement in metric {current_metric} for {non_improved_epoch} epochs')
            if non_improved_epoch == patience:
                logger.info(f'No improvement in metric {current_metric} for {non_improved_epoch} epochs, '
                             f'adjusting learning rate')
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= factor
                non_improved_epoch = 0
        if is_best and checkpoint_path is not None:
            ModelManager.save_model(model, checkpoint_path, epoch, 'best_metric', optimizer=optimizer)
            print('model_saved')
        if save_interval is not None and (epoch + 1) % save_interval == 0 and save_checkpoint:
            ModelManager.save_model(model, checkpoint_path, epoch, 'checkpoint', optimizer=optimizer)
            print('model_saved')
    if mode == 'train':
        return model, optimizer, grad_scaler, total_step
    else:
        return model, optimizer, best_metrics, non_improved_epoch, epoch_loss,total_step
    
def test(model=None, dataloader=None, criterion=None,metric_collection=None, device=None, logger=None, mean = None,std = None):
    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            target = target * std + mean
            output = model(data)
            output = output.squeeze()
            output = output * std + mean
            loss = criterion(output, target)
            # 更新指标
            metric_collection.update(output, target)

    # 计算平均指标
    average_metrics = metric_collection.calculate_averages()

    # 记录指标
    logger.info(f'Test Metrics:')
    for metric_name, average_value in average_metrics.items():
        if average_value is not None:
            logger.info(f"{metric_name}: {average_value}")

    return average_metrics
