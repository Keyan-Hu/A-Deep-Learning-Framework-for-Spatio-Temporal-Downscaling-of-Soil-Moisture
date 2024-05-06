import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class CustomMetricCollection:
    def __init__(self):
        self.metrics = {'RMSE': [], 'MSE': [], 'MAE': [], 'R2': [], 'ubRMSE': [], 'PearsonR': []}

    def update(self, output, target):
        self.metrics['RMSE'].append(CustomMetricCollection.calculate_rmse(output, target).item())
        self.metrics['MSE'].append(nn.MSELoss()(output, target).item())
        self.metrics['MAE'].append(nn.L1Loss()(output, target).item())
        self.metrics['R2'].append(CustomMetricCollection.calculate_r2(output, target))
        self.metrics['ubRMSE'].append(CustomMetricCollection.calculate_ubRMSE(output, target))
        self.metrics['PearsonR'].append(CustomMetricCollection.calculate_pearsonr(output, target))

    
    def calculate_averages(self,method=None):
        """计算所有度量值的平均值。"""
        averages = {}
        for key in self.metrics.keys():
            if len(self.metrics[key]) > 0:
                averages[key] = sum(self.metrics[key]) / len(self.metrics[key])
            else:
                averages[key] = None
        if method:
            return averages[method]
        else:
            return averages
    
    def reset(self):
        for key in self.metrics.keys():
            self.metrics[key] = []

    @staticmethod
    def calculate_rmse(output, target):
        mse = nn.MSELoss()(output, target)
        return torch.sqrt(mse)

    @staticmethod
    def calculate_r2(output, target):
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        ss_res = np.sum((output - target) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    @staticmethod
    def calculate_pearsonr(output, target):
        # 将 Tensor 转换为 numpy 数组
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        # 计算均值
        mean_output = np.mean(output)
        mean_target = np.mean(target)

        # 计算与均值之差
        output_diff = output - mean_output
        target_diff = target - mean_target

        # 计算相关性的分子（输出与目标差值的乘积之和）
        numerator = np.sum(output_diff * target_diff)

        # 计算相关性的分母（输出与目标差值平方和的乘积的平方根）
        denominator = np.sqrt(np.sum(output_diff ** 2) * np.sum(target_diff ** 2))

        # 避免除以零
        if denominator == 0:
            return 0

        # 计算并返回皮尔逊相关系数
        pearsonr = numerator / denominator
        return pearsonr if np.isfinite(pearsonr) else 0

    @staticmethod
    def calculate_ubRMSE(output, target):
        # ubRMSE 计算逻辑，仅包含非零目标
        rmse = CustomMetricCollection.calculate_rmse(output, target).item()
        bias = np.mean(output.detach().cpu().numpy() - target.detach().cpu().numpy())
        return np.sqrt(rmse**2 - bias**2)

    
class CustomRMSELoss(nn.Module):
    def __init__(self):
        super(CustomRMSELoss, self).__init__()

    def compute_rmse(self, y_pred, y_true):
        mse_loss = nn.MSELoss()
        mse = mse_loss(y_pred, y_true)
        rmse = torch.sqrt(mse)
        return rmse

    def forward(self, y_pred, y_true):
        return self.compute_rmse(y_pred, y_true)

