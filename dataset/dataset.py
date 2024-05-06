import os
import logging
import random
import torch
import numpy as np
from pathlib import Path
from os import listdir
from os.path import splitext
import albumentations as A
from torchvision import transforms
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

class STConvDataset(Dataset):
    """
    加载ST-Conv数据 # time_steps lat lon channels->channels time_steps lat lon
    包含操作： 1. 数据增强、2. 数据标准化、3. 数据转换为张量
    data_dir: 各种数据地址，传入数据维度：T*H*W*C
    mask_dir: 数据随机mask的路径，和SP—MASK数据一致
    train、pre_train: 预训练和训练模式，预训练模式使用SM数据，训练模式屏蔽
    Hyperparameter: 数据加载参数
    """
    def __init__(self, static_data, day_data, half_day_data, hour_data, pre_train=True, train=False, seed=42, indices=None, mode='train', 
                 mask_start=0.3, mask_end=1.0, mask_data_start=0.1, mask_data_end=0.2, **kwargs):
        # 数据
        self.Static_data = static_data
        self.Day_data = day_data
        self.Half_Day_data = half_day_data
        self.Hour_data = hour_data
        # 模式参数
        self.pre_train = pre_train
        self.train = train
        self.seed = seed
        self.mode = mode
        self.kwargs = kwargs
        ## 动态掩码和噪声
        # SM掩码
        self.mask_start = mask_start
        self.mask_end = mask_end
        self.mask_ratio = mask_start
        # data掩码
        self.mask_data_start = mask_data_start
        self.mask_data_end = mask_data_end
        self.mask_data_level = mask_data_start

        self.indices = indices if indices is not None else range(len(self.Half_Day_data))
        self.cumulative_transforms = self._create_cumulative_transforms()

    def update_dataset_parameters(self, epoch, total_epochs):
        self.mask_ratio = self.mask_start + (self.mask_end - self.mask_start) * (epoch / total_epochs)
        self.mask_ratio = 1 if self.mask_ratio >=1 else self.mask_ratio
        self.mask_data_level = self.mask_data_start + (self.mask_data_end - self.mask_data_start) * (epoch / total_epochs)

    def horizontal_flip_np(self, images):
        if images.ndim == 4:  # [time_steps, lat, lon, channels]
            return np.flip(images, axis=2)  # 翻转经度
        elif images.ndim == 3:  # [lat, lon, channels]
            return np.flip(images, axis=1)  

    def vertical_flip_np(self, images):
        if images.ndim == 4:
            return np.flip(images, axis=1)  # 翻转纬度
        elif images.ndim == 3:
            return np.flip(images, axis=0)

    def random_rotate_90_np(self, images):
        k = np.random.choice([0, 1, 2, 3])
        if images.ndim == 4:
            return np.rot90(images, k, axes=(1, 2))  # 纬度和经度上旋转
        elif images.ndim == 3:
            return np.rot90(images, k, axes=(0, 1))

    def transpose_np(self, images):
        if images.ndim == 4:
            return images.transpose(0, 2, 1, 3)  # 交换纬度和经度
        elif images.ndim == 3:
            return images.transpose(1, 0, 2)
    
    #数据掩码处理掩码
    def apply_mask(self, images):
        total_elements = np.prod(images.shape)
        mask_count = int(total_elements * self.mask_data_level)
        mask = np.ones(total_elements, dtype=bool)
        mask[:mask_count] = False
        np.random.shuffle(mask)
        mask = mask.reshape(images.shape)
        return images * mask.reshape(images.shape)
    
    #SM生成掩码函数和
    def generate_mask(self, shape):
        total_elements = np.prod(shape)
        masked_elements = int(total_elements * self.mask_ratio)
        indices = np.arange(total_elements)
        np.random.shuffle(indices)
        mask = np.ones(total_elements, dtype=bool)
        mask[indices[:masked_elements]] = False
        return mask.reshape(shape)

    def _create_cumulative_transforms(self):
        # 预先计算并存储累积概率及其方法
        cum_probabilities = []
        current_prob = 0
        transforms = [
            (self.horizontal_flip_np, 'HorizontalFlip', self.kwargs['HorizontalFlip_p']),
            (self.vertical_flip_np, 'VerticalFlip', self.kwargs['VerticalFlip_p']),
            (self.random_rotate_90_np, 'RandomRotate90', self.kwargs['RandomRotate90_p']),
            (self.transpose_np, 'Transpose', self.kwargs['Transpose_p']),
            (self.apply_mask, 'Mask', self.kwargs['Mask_p'])
        ]
        for transform, name, p in transforms:
            current_prob += p
            cum_probabilities.append((current_prob, (transform, name)))
        return cum_probabilities
    
    def randomly_select_transform(self):
        # 选择增强策略
        random_choice = random.random()
        for prob, (transform, name) in self.cumulative_transforms:
            if random_choice <= prob:
                return name, transform  # 现在返回名称和函数
        return 'none', lambda x: x

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        half_day_sample = self.Half_Day_data[idx]
        static_sample = self.Static_data[idx]
        day_sample = self.Day_data[idx]
        hour_sample = self.Hour_data[idx]
        
        mask_sample = half_day_sample.copy()
        # mode训练、验证、测试模式  pre_train\train预训练、训练模式，全为False则训练模式不使用增强
        if self.mode == 'train' and (self.pre_train or self.train):
            time_length = half_day_sample.shape[0]
            mask_shape = (time_length, half_day_sample.shape[1], half_day_sample.shape[2], 1)  # time_steps lat lon channels
            mask = self.generate_mask(mask_shape)
            mask_sample[..., 0] = mask_sample[..., 0] * mask[..., 0]
            # 选择增强策略
            transform_name, transform = self.randomly_select_transform()
            static_sample = transform(static_sample)
            day_sample = transform(day_sample)
            hour_sample = transform(hour_sample)
            # target数据不用再次mask
            if transform_name != 'Mask':
                half_day_sample = transform(half_day_sample)
                mask_sample = transform(mask_sample)
            else:
                half_day_sample[..., 1:] = transform(half_day_sample[..., 1:])
                mask_sample[..., 1:] = transform(mask_sample[..., 1:])
        else:
            mask_sample[..., 0] = 0
        static_sample = np.nan_to_num(static_sample)
        day_sample = np.nan_to_num(day_sample)
        half_day_sample = np.nan_to_num(half_day_sample)
        hour_sample = np.nan_to_num(hour_sample)
        mask_sample = np.nan_to_num(mask_sample)

        target = half_day_sample[half_day_sample.shape[0]//2, ..., 0]
        static_sample = torch.tensor(static_sample, dtype=torch.float32).permute(2, 0, 1)
        day_sample = torch.tensor(day_sample, dtype=torch.float32).permute(3, 0, 1, 2)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(-1).permute(2, 0, 1)
        hour_sample = torch.tensor(hour_sample, dtype=torch.float32).permute(3, 0, 1, 2)
        mask_sample = torch.tensor(mask_sample, dtype=torch.float32).permute(3, 0, 1, 2)

        return static_sample, day_sample, target, hour_sample, mask_sample
