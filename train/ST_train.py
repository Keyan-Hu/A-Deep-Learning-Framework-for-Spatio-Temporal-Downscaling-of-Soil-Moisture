import os
import wandb
import logging
import torch
import argparse
import numpy as np
from torch import optim
from torch.utils.data import DataLoader

import sys
sys.path.append(r"C:\Users\Administrator\Desktop\code\ST-Conv")
from dataset.dataset import STConvDataset
from log.logger import setup_logger
from train.train_val import test_apply,train_apply,val_apply
from tool.utils import Util,ModelManager
from metrics.metrics import CustomMetricCollection, CustomRMSELoss, CombinedLoss
from models.model import STConvNet
from models.block import BasicBlock,Bottleneck

os.environ["WANDB_MODE"] = 'offline'
##1!!!
def load_data(pre_train=True, train=True, seed=42, batch_size=256, missing_rate_threshold=1, **kwargs):
    half_day_data = np.load(kwargs['Half_Day_data_dir'], allow_pickle=True)
    #samples time_steps lat lon channels
    central_index = half_day_data.shape[1] // 2
    missing_rates = np.mean(half_day_data[:, central_index, :, :, 0] == 0, axis=(1, 2))
    valid_indices = np.where(missing_rates < missing_rate_threshold)[0]

    static_data = np.load(kwargs['Static_data_dir'], allow_pickle=True)[valid_indices]
    day_data = np.load(kwargs['Day_data_dir'], allow_pickle=True)[valid_indices]
    hour_data = np.load(kwargs['Hour_data_dir'], allow_pickle=True)[valid_indices]
    half_day_data = half_day_data[valid_indices]

    dataset_size = len(valid_indices)

    train_indices, val_indices, test_indices = Util.split_dataset_indices(dataset_size, seed=seed, train_ratio=kwargs.get('train_ratio', 0.7), val_ratio=kwargs.get('val_ratio', 0.15), test_ratio=kwargs.get('test_ratio', 0.15))

    train_dataset = STConvDataset(static_data, day_data, half_day_data, hour_data, pre_train=pre_train, train=train, 
                                  seed=seed, indices=train_indices, mode='train', **kwargs)
    val_dataset = STConvDataset(static_data, day_data, half_day_data, hour_data, pre_train=pre_train, train=False, 
                                seed=seed, indices=val_indices, mode='val', **kwargs)
    test_dataset = STConvDataset(static_data, day_data, half_day_data, hour_data, pre_train=False, train=False, 
                                 seed=seed, indices=test_indices, mode='test', **kwargs)

    loader_args = dict(batch_size=batch_size, num_workers=kwargs.get('num_workers', 1), prefetch_factor=kwargs.get('prefetch_factor', 2), persistent_workers=kwargs.get('persistent_workers', False))
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_args)
    del static_data, day_data, half_day_data, hour_data
    return train_loader, val_loader, test_loader

def train(project_name:str='ST-Conv',use_freeze_decoder:bool=True,spatial_name:str=None,usage_model:str=None,train_dict:dict=None,logger=None,model=None,criterion=CombinedLoss(),wandb=wandb,       # necessary parameter
          checkpoint_path:str=None,best_model_path:str=None,train_loader=None,val_loader=None,test_loader=None,                                                                 # Path & Dataloader
          algorithm_type ='normal',mean=None,std=None,train:bool=True,                                                                                                          # model parameter
          learning_rate:float=1e-3,weight_decay:float=1e-6,warm_up_step:int=1000,epochs:int=100,evaluate_epoch:int=0,test_epoch:int=10,amp:bool=True,**kwargs):                 # train parameter
    
    # 设定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    std = torch.tensor(np.load(std,allow_pickle=True), dtype=torch.float)[0].to(device)
    mean = torch.tensor(np.load(mean,allow_pickle=True), dtype=torch.float)[0].to(device)
    model = model.to(device)

    # 设置优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 优化器
    warmup_lr = np.arange(1e-7, learning_rate, (learning_rate - 1e-7) / warm_up_step)
    grad_scaler = torch.cuda.amp.GradScaler() if amp else None

    # 加载模型、指标收集器、最佳指标字典、无改进步计数、模型管理器、总步数
    Util.load_model_and_optimizer(model, optimizer, device, best_model_path, logger=logger, model_index=-1,model_name=project_name)
    metric_collection = CustomMetricCollection()
    best_metrics = dict.fromkeys(['best_RMSE', 'best_r2','best_r','best_MAE','best_MSE','best_ubRMSE'], 0.5)  # 最佳评估指标
    non_improved_epoch = 0
    ModelManager.start_new_training_session()
    total_step = 0
    
    # 预训练、验证、测试
    for epoch in range(epochs):
        if train and use_freeze_decoder:train_loader.dataset.update_dataset_parameters(epochs, epochs)
        else:train_loader.dataset.update_dataset_parameters(epoch, epochs)
        model, optimizer, _, _ = \
        train_apply(model=model, dataloader=train_loader, epoch=epoch, optimizer=optimizer, grad_scaler=grad_scaler, criterion=criterion,usage_model=usage_model,
                    metric_collection=metric_collection, device=device, std=std, mean=mean, logger=logger, wandb=wandb, total_step=total_step, 
                    warmup_lr=warmup_lr, algorithm_type=algorithm_type, use_freeze_decoder=use_freeze_decoder,spatial_name=spatial_name,train=train,**train_dict)
        with torch.no_grad():
            if epoch>=evaluate_epoch:
                    val_apply(project_name=project_name,model=model,epoch=epoch,non_improved_epoch=non_improved_epoch,logger=logger,optimizer=optimizer,
                            checkpoint_path=checkpoint_path,best_model_path=best_model_path,
                            best_metrics=best_metrics,device=device,metric_collection=metric_collection,dataloader=val_loader
                            ,wandb=wandb,std=std,mean=mean,**train_dict)
            if  epoch>=test_epoch:
                    test_metrics = test_apply(model=model, dataloader=test_loader, criterion=criterion,metric_collection=metric_collection, device=device,logger=logger,wandb=wandb,std=std,mean=mean)
            

def main(args):
    models_config = Util.load_config(args.models_config_path)
    train_config = models_config['train_shared_parameter'] 
    path_config = Util.load_config(args.path_config_path)
    logger = setup_logger(path_config['log'],append=True)
    return train_config, models_config, path_config, logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_config_path', type=str, default='./config/models_config.yaml')
    parser.add_argument('--path_config_path', type=str, default='./config/path_config.yaml')
    args = parser.parse_known_args()[0]

    train_dict, models_config, path_config, logger = main(args)
    Util.random_seed(seed=train_dict['seed'])
    # 解包model_config
    shared_parameters = models_config['Shared_parameter']       # 全局共享参数
    experiment_groups = models_config['experiment_groups']      # 实验组参数-实验组共享参数+实验组模型参数

    for group in experiment_groups:
        if group['group_name'] == 'Experiment Group 9':
            print(f"Running experiments for group: {group['group_name']}")
            experiment_shared_params = group.get('experiment_shared_parameter', {})

            # 如果模式是train==True,修改为False，保存为临时变量，先以false训练再以True训练
            temp_train = False
            if experiment_shared_params['train'] == True:
                experiment_shared_params['train'] = False
                experiment_shared_params['use_freeze_decoder'] = False
                temp_train = True

            # 定义损失函数、加载数据
            criterion = CombinedLoss(**experiment_shared_params)
            train_loader,val_loader,test_loader = load_data(**experiment_shared_params,**path_config,**train_dict)
            # 训练实验组中每个模型

            for model_config in group['models']:
                wandb_config = {'seed': train_dict['seed'],
                                'learning_rate': train_dict['learning_rate'],
                                'weight_decay': train_dict['weight_decay'],
                                'warm_up_step': train_dict['warm_up_step'],
                                'epochs': train_dict['epochs'],
                                'amp': train_dict['amp'],
                                'batch_size': train_dict['batch_size']}
                
                wandb_config.update(model_config)
                wandb.init(project='ST-Conv', config=wandb_config, name=model_config['model_name'], group=group['group_name'])
                model = STConvNet(**shared_parameters, **experiment_shared_params, **model_config['parameters'])
                train(project_name=model_config['model_name'],spatial_name = model_config['parameters']['spatial_block'],usage_model = model_config['parameters']['usage_model'],train_dict=train_dict,logger=logger,model = model,criterion=criterion,wandb=wandb,
                                    train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,**experiment_shared_params,**train_dict)
            
            if temp_train == True:
                # 微调decoder训练100次
                train_dict['epochs'] = 100
                experiment_shared_params['train'] = True
                experiment_shared_params['use_freeze_decoder'] = True
                # 训练实验组中每个模型
                for model_config in group['models']:
                    model = STConvNet(**shared_parameters, **experiment_shared_params, **model_config['parameters'])
                    # 设定设备
                    if torch.cuda.device_count() > 1:
                        print(f"Let's use {torch.cuda.device_count()} GPUs!")
                    # 封装模型以在多GPU上运行
                    model = nn.DataParallel(model)
                    train(project_name=model_config['model_name'],spatial_name = model_config['parameters']['spatial_block'],usage_model = model_config['parameters']['usage_model'],train_dict=train_dict,logger=logger,model = model,criterion=criterion,wandb=wandb,
                                        train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,**experiment_shared_params,**train_dict)