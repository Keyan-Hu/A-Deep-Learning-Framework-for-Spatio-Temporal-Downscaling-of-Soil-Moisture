import os
import logging
import time
import yaml
import random
import glob
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import random_split
from collections import OrderedDict

class Util:
    
    @staticmethod
    def random_seed(SEED):
        """
        设置随机数生成器的种子，以保证实验的可重复性。
        """
        random.seed(SEED)
        os.environ['PYTHONHASHSEED'] = str(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def calculate_hours_since_year_start(timestamp):
        """
        计算从年初到指定日期的总小时数。
        :param timestamp: datetime对象，表示要计算的日期和时间。
        :return: 自年初以来的总小时数。
        """
        doy = timestamp.timetuple().tm_yday  # 当前日期在一年中的天数
        hour_of_day = timestamp.hour  # 当前小时数
        total_hours = (doy - 1) * 24 + hour_of_day  # 计算总小时数（从0开始计数）
        return total_hours
    
    @staticmethod
    def calculate_time_indices(data_time, strategy):
        time_indices = []
        for single_time in data_time:
            current_time = pd.to_datetime(single_time.values)
            if strategy == "static":
                time_indices.append(single_time)
            elif strategy == "day" and current_time.hour == 12:
                time_indices.append(single_time)
            elif strategy == "half_day" and current_time.hour in [6, 18]:
                time_indices.append(single_time)
            elif strategy == "hour" and current_time.hour % 3 == 0:
                time_indices.append(single_time)
        return time_indices
    
    @staticmethod
    def merge_and_save_temp_data(file_path, temp_files):
        final_data = []
        for temp_file in temp_files:
            with open(temp_file, 'rb') as file:
                final_data.extend(pickle.load(file))
            os.remove(temp_file)
        with open(file_path, 'wb') as file:
            pickle.dump(final_data, file)

    @staticmethod
    def find_closest_time_index(target_time, time_indices):
        # 找到距离目标时间最近的时间索引
        closest_time = min(time_indices, key=lambda x: abs(pd.to_datetime(x.values) - target_time))
        return np.where(time_indices == closest_time)[0][0]
    
    @staticmethod
    def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def load_config(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
    
    @staticmethod
    def load_model_and_optimizer(model, optimizer, device, model_folder,model_name=None, logger= None, model_index=-1):
        """
        加载模型和优化器的状态。
        :param model: 要加载状态的模型对象。
        :param optimizer: 用于模型的优化器对象。
        :param device: 模型和优化器状态加载到的设备（例如 'cuda' 或 'cpu'）。
        :param model_folder: 包含模型文件的文件夹路径。
        :param model_index: 要加载的模型的索引。默认为 -1，表示加载最新的模型。
        :return: None
        """
        # 获取所有模型文件
        model_files = glob.glob(os.path.join(model_folder, "*.pth"))
        if not model_files:
            logger.warning(f"No model files found in {model_folder}")
            return
        # 获取文件夹中所以文件名
        if model_name is not None:
            matching_model_files = [file for file in model_files if re.search(model_name, os.path.basename(file))]
        else:
            matching_model_files = model_files

        # 最新的模型在最后
        matching_model_files.sort(key=os.path.getmtime)
        if model_index == -1:
            model_file = matching_model_files[-1]
        elif model_index is None or (model_index < 0 or model_index >= len(matching_model_files)):
            logger.warning(f"Invalid model index: {model_index}")
            return
        else:
            model_file = matching_model_files[model_index]

        # 加载模型和优化器
        if os.path.isfile(model_file):
            checkpoint = torch.load(model_file, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            if 'optimizer' in checkpoint and optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            logger.info(f"Model loaded from {model_file}")
        else:
            logger.warning(f"Model file not found: {model_file}")
    @staticmethod
    def load_model_and_optimizer1(model, optimizer, device, model_folder, model_name=None, logger=None, model_index=-1):
        """
        加载模型和优化器的状态。
        :param model: 要加载状态的模型对象。
        :param optimizer: 用于模型的优化器对象。
        :param device: 模型和优化器状态加载到的设备（例如 'cuda' 或 'cpu'）。
        :param model_folder: 包含模型文件的文件夹路径。
        :param model_index: 要加载的模型的索引。默认为 -1，表示加载最新的模型。
        :return: None
        """
        # 获取所有模型文件
        model_files = glob.glob(os.path.join(model_folder, "*.pth"))
        if not model_files:
            if logger:
                logger.warning(f"No model files found in {model_folder}")
            return
        # 获取文件夹中所有文件名
        if model_name is not None:
            matching_model_files = [file for file in model_files if re.search(model_name, os.path.basename(file))]
        else:
            matching_model_files = model_files

        # 最新的模型在最后
        matching_model_files.sort(key=os.path.getmtime)
        if model_index == -1:
            model_file = matching_model_files[-1]
        elif model_index is None or (model_index < 0 or model_index >= len(matching_model_files)):
            if logger:
                logger.warning(f"Invalid model index: {model_index}")
            return
        else:
            model_file = matching_model_files[model_index]

        # 加载模型和优化器
        if os.path.isfile(model_file):
            checkpoint = torch.load(model_file, map_location=device)
            state_dict = checkpoint['model_state']
            
            # 调整模型的状态字典以匹配模型的键
            if isinstance(model, torch.nn.DataParallel):
                # 如果模型是用DataParallel包装的，但是状态字典不是
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = 'module.' + k if not k.startswith('module.') else k
                    new_state_dict[name] = v
                state_dict = new_state_dict
            else:
                # 如果模型没有用DataParallel包装，但状态字典是的
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k  # 去掉'module.'前缀
                    new_state_dict[name] = v
                state_dict = new_state_dict
            
            model.load_state_dict(state_dict)
            if 'optimizer' in checkpoint and optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            if logger:
                logger.info(f"Model loaded from {model_file}")
        else:
            if logger:
                logger.warning(f"Model file not found: {model_file}")
class ModelManager:
    @staticmethod
    def save_model(model, path, epoch, mode, optimizer=None, max_checkpoints=2, max_best_models=3):
        # 创建目录
        Path(path).mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())

        # 区分检查点和最佳模型文件名
        file_prefix = "checkpoint" if mode == "checkpoint" else "best"

        # 构建模型保存路径
        model_save_path = f"{path}/{file_prefix}_epoch{epoch}_{timestamp}.pth"
        state_dict = {'model_state': model.state_dict(), 'epoch': epoch}

        if optimizer:
            state_dict['optimizer_state'] = optimizer.state_dict()
        torch.save(state_dict, model_save_path)

        # 日志记录
        logging.info(f"Saved {mode} model for epoch {epoch} at {model_save_path}")

        # 管理保存的模型文件
        ModelManager.manage_saved_models(path, file_prefix, max_checkpoints if mode == 'checkpoint' else max_best_models)

    @staticmethod
    def manage_saved_models(path, file_prefix, max_models):
        # 获取所有相应模式的模型文件
        model_files = glob.glob(f"{path}/{file_prefix}_epoch*.pth")

        # 如果模型数量超过最大值，则删除最旧的模型
        if len(model_files) > max_models:
            # 按修改时间排序
            model_files.sort(key=os.path.getmtime)
            # 保留最新的模型文件，删除其他
            for file in model_files[:-max_models]:
                os.remove(file)
                logging.info(f"Removed old model file: {file}")