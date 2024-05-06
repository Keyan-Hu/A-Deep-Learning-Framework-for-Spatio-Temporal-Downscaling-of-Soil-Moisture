import logging
import torch
import argparse
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import sys
sys.path.append(r"C:\Users\Administrator\Desktop\code\LSTM\LSTM")
from model.LSTM import LSTMModel, BiLSTMModel
from log.logging import setup_logger
from utils.utils import Util
from dataset.Dataset import LSTMDataset
from metrics.metrics import CustomMetricCollection, CustomRMSELoss
from train_val import train_val, test

def train(path_dict, Hyperparameter_dict, logger):
    dataset = LSTMDataset("D:\Data_Store\Dataset\exp\data_batch_0.npy")

    # 使用您自定义的split_dataset函数划分数据集
    train_dataset, val_dataset, test_dataset = Util.split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    loader_args = dict(
        num_workers=1,                # 使用的工作进程数量
        prefetch_factor=5,            # 数据预取的因子
        persistent_workers=True       # 是否使用持久化工作进程
    )
        # 使用 DataLoader 加载数据集
    train_loader = DataLoader(train_dataset, batch_size=Hyperparameter_dict['batch_size'], \
                              shuffle=True, drop_last=False, **loader_args)
    val_loader = DataLoader(val_dataset, batch_size=Hyperparameter_dict['batch_size'], \
                            shuffle=False, drop_last=False, **loader_args)
    test_loader = DataLoader(test_dataset, batch_size=Hyperparameter_dict['batch_size'], \
                             shuffle=False, drop_last=False, **loader_args)

    model = LSTMModel(input_dim=Hyperparameter_dict['input_dim'], hidden_size=Hyperparameter_dict['hidden_size'],dropout_rate=Hyperparameter_dict['dropout_rate'])
    
    # 设定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 加载mean和std
    mean = np.load(path_dict['mean'])[0]
    std = np.load(path_dict['std'])[0]
    # 设置优化器
    optimizer = optim.AdamW(model.parameters(), lr=Hyperparameter_dict['learning_rate'], weight_decay=Hyperparameter_dict['weight_decay'])  # 优化器
    warmup_lr = np.arange(1e-7, Hyperparameter_dict['learning_rate'], (Hyperparameter_dict['learning_rate'] - 1e-7) / Hyperparameter_dict['warm_up_step'])
    grad_scaler = torch.cuda.amp.GradScaler()

    # 如果设置了加载预训练模型的标志
    '''
    if path_dict['LSTM_Best_Model']:
        checkpoint = torch.load(path_dict['LSTM_Best_Model'], map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        if 'optimizer' in checkpoint and optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        logging.info(f'Model loaded from {path_dict["LSTM_Best_Model"]}')
    '''
    # 初始化自定义指标收集器
    metric_collection = CustomMetricCollection()
    # 初始化最佳评估指标字典
    best_metrics = dict.fromkeys(['best_RMSE', 'best_loss'], 1)  # 最佳评估指标
    # 初始化非改进的时期计数
    non_improved_epoch = 0  # 当non_improved_epoch等于patience时，调整学习率
    # 损失函数
    criterion = CustomRMSELoss()
    # 定义总步数
    total_step = 0

    # 训练
    for epoch in range(Hyperparameter_dict["epochs"]):
        model, optimizer, grad_scaler, total_step= \
        train_val(mode='train', model=model, dataloader=train_loader, 
                epoch=epoch, non_improved_epoch=non_improved_epoch,device=device, warmup_lr=warmup_lr, grad_scaler=grad_scaler,
                max_grad_norm=20, criterion=criterion, metric_collection=metric_collection,patience=Hyperparameter_dict['patience'],
                optimizer=optimizer, factor=Hyperparameter_dict['factor'],mean = mean, std = std,
                total_step=total_step, warm_up_step=Hyperparameter_dict['warm_up_step'], logger=logger)
        
        # 评估
        if epoch>=Hyperparameter_dict["evaluate_epoch"]:
            with torch.no_grad():
                model, optimizer, best_metrics, non_improved_epoch, epoch_loss,total_step\
                = train_val(mode='val', model=model, dataloader=val_loader, 
                        epoch=epoch, non_improved_epoch=non_improved_epoch, best_metrics=best_metrics,
                        device=device, warmup_lr=warmup_lr,max_grad_norm=20, criterion=criterion, metric_collection=metric_collection,patience=Hyperparameter_dict['patience'],
                        optimizer=optimizer,checkpoint_path=path_dict['LSTM_check_point_Model'], factor=Hyperparameter_dict['factor'],
                        save_interval=Hyperparameter_dict['save_interval'], save_checkpoint=Hyperparameter_dict['save_checkpoint'],
                        total_step=total_step, warm_up_step=Hyperparameter_dict['warm_up_step'],logger=logger,mean = mean, std = std)

    average_metrics = test(model=model, dataloader=test_loader, criterion=criterion,metric_collection=metric_collection, device=device, logger=logger,mean = mean, std = std)

def main(args):
    model_config = Util.load_config(args.model_config_path)
    path_config = Util.load_config(args.path_config_path)
    logger = setup_logger(path_config['log'])
    return train(path_config, model_config, logger)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path', type=str, default='./config/model_config.yaml')
    parser.add_argument('--path_config_path', type=str, default='./config/path_config.yaml')
    args = parser.parse_known_args()[0]
    Hyperparameter_dict = Util.load_config(args.model_config_path)
    # set random seed to make the experiment reproducible
    Util.random_seed(SEED=Hyperparameter_dict['random_seed'])
    main(args)