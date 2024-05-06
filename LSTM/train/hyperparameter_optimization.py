import optuna
import torch
from torch import optim
import numpy as np
import logging
import argparse

from train import train_val
from model.LSTM import LSTMModel, BiLSTMModel
from log.logging import setup_logger
from utils.utils import Util
from dataset.Dataset import LSTMDataset
from metrics.metrics import CustomMetricCollection, CustomRMSELoss
from train_val import train_val
from torch.utils.data import DataLoader

def objective(trial):

    # 定义搜索空间
    dropout_rate = trial.suggest_float('dropout_rate', optimization_dict['dropout_rate']['min'], optimization_dict['dropout_rate']['max'])
    hidden_size = trial.suggest_int('hidden_size', optimization_dict['hidden_size']['min'], optimization_dict['hidden_size']['max'])
    learning_rate = trial.suggest_float('learning_rate', optimization_dict['learning_rate']['min'], optimization_dict['learning_rate']['max'], log=True)

    dataset = LSTMDataset(path_dict['LSTM_Data'])

    # 使用您自定义的split_dataset函数划分数据集
    train_dataset, val_dataset, test_dataset = Util.split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    loader_args = dict(
        num_workers=4,                # 使用的工作进程数量
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

    # 加载模型和其他超参数
    model = LSTMModel(input_dim=41, hidden_size=hidden_size, dropout_rate=dropout_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # 设定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    grad_scaler = torch.cuda.amp.GradScaler()

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

    for epoch in range(optimization_dict["epochs"]):
        model, optimizer, grad_scaler, total_step = \
        train_val(mode='train', model=model, dataloader=train_loader, 
                epoch=epoch, non_improved_epoch=non_improved_epoch,device=device, grad_scaler=grad_scaler,
                max_grad_norm=20, criterion=criterion, metric_collection=metric_collection,patience=optimization_dict['patience'],
                optimizer=optimizer, factor=Hyperparameter_dict['factor'],
                total_step=total_step, logger=logger)
        
        # 评估
        if epoch>=Hyperparameter_dict["evaluate_epoch"]:
            with torch.no_grad():
                model, optimizer, best_metrics, non_improved_epoch, epoch_loss,total_step\
                = train_val(mode='val', model=model, dataloader=val_loader, 
                        epoch=epoch, non_improved_epoch=non_improved_epoch, best_metrics=best_metrics,
                        device=device, max_grad_norm=20, criterion=criterion, metric_collection=metric_collection,patience=optimization_dict['patience'],
                        optimizer=optimizer, factor=Hyperparameter_dict['factor'], total_step=total_step,logger=logger)
    # 返回验证损失
    return best_metrics['best_RMSE']

parser = argparse.ArgumentParser()
parser.add_argument('--model_config_path', type=str, default='../config/model_config.yaml')
parser.add_argument('--path_config_path', type=str, default='../config/path_config.yaml')
parser.add_argument('--hyperparameter_optimization_config_path', type=str, default='../config/hyperparameter_optimization_config.yaml')
args = parser.parse_known_args()[0]
Hyperparameter_dict = Util.load_config(args.model_config_path)
path_dict = Util.load_config(args.path_config_path)
optimization_dict =  Util.load_config(args.hyperparameter_optimization_config_path)
logger = setup_logger(path_dict['hyperparameter_optimization_log'])
# set random seed to make the experiment reproducible
Util.random_seed(SEED=Hyperparameter_dict['random_seed'])
# 创建study对象并运行优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
# 输出最佳结果
best_params_str = str(study.best_params)
best_rmse_str = str(study.best_value)
logger.info(f'最佳参数: {best_params_str}')
logger.info(f'最佳RMSE: {best_rmse_str}')