import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class SoilMoistureMetrics:
    def __init__(self):
        self.metrics = {'GEFFI': [], 'GPREC': [], 
            'GACCU': [], 'GDOWN': []
        }
    
    def update(self, hr, lr, is_data):
        '''
        hr:降尺度数据
        lr:卫星数据
        is_data:原位数据
        '''
        if hr.size == 0 or lr.size == 0 or is_data.size == 0:
            self.append_nan_to_all_metrics()
            return
        if np.all(np.isnan(hr)) or np.all(np.isnan(lr)) or np.all(np.isnan(is_data)):
            self.append_nan_to_all_metrics()
            return

        lr_clean, is_data_clean_lr = self.clean_and_extract_common_parts(lr, is_data)
        hr_clean, is_data_clean_hr = self.clean_and_extract_common_parts(hr, is_data)

        if lr_clean.size <= 2 or is_data_clean_lr.size <= 2 or hr_clean.size <= 2:
            self.append_nan_to_all_metrics()
            return
        
        if np.all(lr_clean == lr_clean[0]) or np.all(is_data_clean_lr == is_data_clean_lr[0]) or \
           np.all(hr_clean == hr_clean[0]) or np.all(is_data_clean_hr == is_data_clean_hr[0]):
            self.append_nan_to_all_metrics()
            return
        
        slr, rlr = self.calculate_slope_and_pearson(lr_clean, is_data_clean_lr)
        shr, rhr = self.calculate_slope_and_pearson(hr_clean, is_data_clean_hr)
        blr = np.mean(lr_clean - is_data_clean_lr)
        bhr = np.mean(hr_clean - is_data_clean_hr)

        self.metrics['GEFFI'].append(self.calculate_geffi(slr, shr))
        self.metrics['GPREC'].append(self.calculate_gprec(rlr, rhr))
        self.metrics['GACCU'].append(self.calculate_gaccu(blr, bhr))
        self.metrics['GDOWN'].append(self.calculate_gdown(self.metrics['GEFFI'][-1], self.metrics['GPREC'][-1], self.metrics['GACCU'][-1]))
    
    def append_nan_to_all_metrics(self):
        for key in self.metrics:
            self.metrics[key].append(np.nan)
    @staticmethod
    def calculate_slope_and_pearson(x, y):
        """Calculate both slope SXR = RXR * (std_X / std_Y) and Pearson correlation RXR."""
        rxy = SoilMoistureMetrics.calculate_pearsonr(x, y)
        std_x = np.std(x)
        std_y = np.std(y)
        sxr = rxy * (std_x / std_y)
        return sxr, rxy
    
    @staticmethod
    def clean_and_extract_common_parts(data1, data2):
        valid_mask = ~np.isnan(data1) & ~np.isnan(data2)
        return data1[valid_mask], data2[valid_mask]
    
    @staticmethod
    def calculate_pearsonr(output, target):
        """Calculate Pearson correlation coefficient"""
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        correlation_matrix = np.corrcoef(output.flatten(), target.flatten())
        pearsonr = correlation_matrix[0, 1]
        return pearsonr if np.isfinite(pearsonr) else 0 

    @staticmethod
    def calculate_geffi(slr, shr):
        return (abs(1 - slr) - abs(1 - shr)) / (abs(1 - slr) + abs(1 - shr))

    @staticmethod
    def calculate_gprec(rlr, rhr):
        return (abs(1 - rlr) - abs(1 - rhr)) / (abs(1 - rlr) + abs(1 - rhr))

    @staticmethod
    def calculate_gaccu(blr, bhr):
        return (abs(blr) - abs(bhr)) / (abs(blr) + abs(bhr))

    @staticmethod
    def calculate_gdown(geffi, gprec, gaccu):
        return (geffi + gprec + gaccu) / 3

    def reset(self):
        for key in self.metrics:
            self.metrics[key] = []

    def get_metrics(self):
        return self.metrics

class CustomMetricCollection:
    def __init__(self):
        self.metrics = {'RMSE': [], 'MSE': [], 'MAE': [], 'R2': [], 'ubRMSE': [], 'PearsonR': [], 'Bias': []}

    def update(self, output, target):
        mask = target != 0
        self.metrics['RMSE'].append(CustomMetricCollection.calculate_rmse(output[mask], target[mask]))
        self.metrics['MSE'].append(nn.MSELoss()(output[mask], target[mask]))
        self.metrics['MAE'].append(nn.L1Loss()(output[mask], target[mask]))
        self.metrics['R2'].append(CustomMetricCollection.calculate_r2(output[mask], target[mask]))
        self.metrics['ubRMSE'].append(CustomMetricCollection.calculate_ubRMSE(output[mask], target[mask]))
        self.metrics['PearsonR'].append(CustomMetricCollection.calculate_pearsonr(output[mask], target[mask]))
        bias = (output[mask] - target[mask]).mean().item()
        self.metrics['Bias'].append(bias)
        
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
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        correlation_matrix = np.corrcoef(output, target)
        pearsonr = correlation_matrix[0, 1]
        return pearsonr if np.isfinite(pearsonr) else 0 
    @staticmethod
    def calculate_ubRMSE(output, target):
        # ubRMSE 计算逻辑，仅包含非零目标
        rmse = CustomMetricCollection.calculate_rmse(output, target).item()
        bias = np.mean(output.detach().cpu().numpy() - target.detach().cpu().numpy())
        return np.sqrt(rmse**2 - bias**2)

## 损失函数
class CombinedLoss(nn.Module):
    def __init__(self, use_rmse_loss=True, use_ssim_loss=True, ratio=2, **kwargs):
        super(CombinedLoss, self).__init__()
        self.use_rmse_loss = use_rmse_loss
        self.use_ssim_loss = use_ssim_loss
        if use_rmse_loss:
            self.rmse_loss = CustomRMSELoss(ratio=ratio)
        if use_ssim_loss:
            self.ssim_loss = SSIMLoss()

    def forward(self, y_pred, y_true):
        losses = {}
        if self.use_rmse_loss:
            height, width = y_true.shape[2], y_true.shape[3]
            rmse_loss = self.rmse_loss(y_pred, y_true, patch_size=(height, width))
            losses['rmse'] = rmse_loss
        if self.use_ssim_loss:
            ssim_loss = self.ssim_loss(y_pred, y_true)
            losses['ssim'] = ssim_loss
        return losses

# RMSE Loss
class CustomRMSELoss(nn.Module):
    def __init__(self, ratio):
        super(CustomRMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ratio = ratio

    def create_weight_matrix(self, patch_size=16, ratio=1.5):
        N, M = patch_size
        center = (N - 1) / 2.0, (M - 1) / 2.0
        Y, X = torch.meshgrid(torch.arange(N), torch.arange(M))
        distance = torch.sqrt((X - center[1])**2 + (Y - center[0])**2)
        max_distance = torch.sqrt(torch.tensor(N/2)**2 + torch.tensor(M/2)**2)
        weights = 1 + (ratio - 1) * (distance / max_distance)
        weights = weights.to('cuda')
        return weights

    def compute_rmse(self, y_pred, y_true, mask):
        y_pred_weights = y_pred * self.weights
        y_true_weights = y_true * self.weights
        masked_pred = y_pred_weights[mask]
        masked_true = y_true_weights[mask]

        mse = self.mse_loss(masked_pred, masked_true)
        rmse = torch.sqrt(mse)
        return rmse
    
    def forward(self, y_pred, y_true, patch_size):
        mask = y_true != 0
        self.weights = self.create_weight_matrix(patch_size, self.ratio)
        if isinstance(y_pred, tuple):
            total_loss = sum(self.compute_rmse(pred, y_true, mask) for pred in y_pred)
        else:
            total_loss = self.compute_rmse(y_pred, y_true, mask)
        return total_loss

# SSIM Loss
class SSIMLoss(nn.Module):
    def __init__(self, size_average=True):
        super(SSIMLoss, self).__init__()
        self.size_average = size_average

    def forward(self, y_pred, y_true):
        mask = y_true != 0

        if isinstance(y_pred, tuple):
            ssimds = []
            for pred in y_pred:
                pred_masked = pred * mask
                true_masked = y_true * mask

                ssimd = self.compute_ssim(pred_masked, true_masked)
                ssimds.append(ssimd)
            ssimd = sum(ssimds) / len(ssimds)
        else:
            y_pred = y_pred * mask
            y_true = y_true * mask

            ssimd = self.compute_ssim(y_pred, y_true)
        return ssimd

    def compute_ssim(self, y_pred, y_true):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_pred = y_pred.mean([2, 3], keepdim=True)
        mu_true = y_true.mean([2, 3], keepdim=True)
        sigma_pred = ((y_pred - mu_pred) ** 2).mean([2, 3], keepdim=True)
        sigma_true = ((y_true - mu_true) ** 2).mean([2, 3], keepdim=True)
        sigma_pred_true = ((y_pred - mu_pred) * (y_true - mu_true)).mean([2, 3], keepdim=True)

        ssim_map = ((2 * mu_pred * mu_true + C1) * (2 * sigma_pred_true + C2)) / ((mu_pred ** 2 + mu_true ** 2 + C1) * (sigma_pred + sigma_true + C2))
        ssim_value = ssim_map.mean() if self.size_average else ssim_map.mean([1, 2, 3])
        ssimd = 1 - ssim_value
        return ssimd