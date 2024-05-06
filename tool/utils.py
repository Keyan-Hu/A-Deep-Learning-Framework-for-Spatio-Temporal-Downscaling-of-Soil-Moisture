import os
import logging
import time
import yaml
import re
import random
import glob
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import random_split
from collections import OrderedDict

class Util:
    @staticmethod
    def random_seed(seed):
        """
        设置随机数生成器的种子，以保证实验的可重复性。
        """
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
    @staticmethod
    def split_dataset_indices(dataset_size, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        assert train_ratio + val_ratio + test_ratio == 1, "1"
        indices = np.arange(dataset_size)
        np.random.seed(seed)

        np.random.shuffle(indices)
        train_end = int(train_ratio * dataset_size)
        val_end = train_end + int(val_ratio * dataset_size)
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        return train_indices, val_indices, test_indices
    
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
        with open(file_path, 'wb') as file:
            pickle.dump(final_data, file)

    @staticmethod
    def find_closest_time_index(target_time, time_indices):
        # 找到距离目标时间最近的时间索引
        closest_time = min(time_indices, key=lambda x: abs(pd.to_datetime(x.values) - target_time))
        return np.where(time_indices == closest_time)[0][0]

    @staticmethod
    def load_config(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
    
    @staticmethod
    def load_model_and_optimizer1(model, optimizer, device, model_folder,model_name=None, logger= None, model_index=-1):
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
            matching_model_files = [file for file in model_files if re.search('^' + re.escape(model_name), os.path.basename(file))]
            if len(matching_model_files)==0:
                logger.warning(f"No mathching model")
                return
        else:
            matching_model_files = model_files

        # 最新的模型在最后
        matching_model_files.sort(key=os.path.getmtime)
        if model_index == -1 :
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
    def load_model_and_optimizer(model, optimizer, device, model_folder, model_name=None, logger=None, model_index=-1):
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
            escaped_model_name = re.escape(model_name)  # 转义特殊字符
            matching_model_files = [file for file in model_files if re.search(escaped_model_name, os.path.basename(file))]
            if len(matching_model_files) == 0:
                if logger:
                    logger.warning(f"No matching model")
                return
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

            # 检查模型是否使用了 DataParallel
            if isinstance(model, torch.nn.DataParallel):
                # 如果模型是 DataParallel，但状态字典的键没有 'module.' 前缀，则添加前缀
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = 'module.' + k if not k.startswith('module.') else k
                    new_state_dict[name] = v
                state_dict = new_state_dict
            else:
                # 如果模型不是 DataParallel，但状态字典的键有 'module.' 前缀，则移除前缀
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k  # 移除 'module.' 前缀
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
    current_training_session_timestamp = None
    current_best_model_file = None

    @staticmethod
    def start_new_training_session():
        ModelManager.current_training_session_timestamp = time.strftime("%Y%m%d%H", time.localtime())
        ModelManager.current_best_model_file = None  # 新会话开始时重置

    @staticmethod
    def save_model(file_name:str=None, model=None, checkpoint_path:str=None, best_model_path:str=None, epoch:int=None, mode:str=None, optimizer=None):
        path = best_model_path if mode == "best" else checkpoint_path
        Path(path).mkdir(parents=True, exist_ok=True)
        file_prefix = "checkpoint" if mode == "checkpoint" else "best"
        timestamp = ModelManager.current_training_session_timestamp if mode == "best" else time.strftime("%Y%m%d%H", time.localtime())
        model_save_path = f"{path}/{file_name}_{file_prefix}_epoch{epoch}_{timestamp}.pth"

        # 保存模型状态
        state_dict = {'model_state': model.state_dict(), 'epoch': epoch}
        if optimizer:
            state_dict['optimizer_state'] = optimizer.state_dict()
        torch.save(state_dict, model_save_path)

        # 更新当前会话中保存的最优模型文件路径
        if mode == "best":
            if ModelManager.current_best_model_file is None:
                ModelManager.current_best_model_file = model_save_path
            else:
                os.remove(ModelManager.current_best_model_file)
                ModelManager.current_best_model_file = model_save_path
        elif mode == "checkpoint":
            ModelManager.manage_saved_models(**locals())
        logging.info(f"Saved {mode} model for epoch {epoch} at {model_save_path}")
        
    def manage_saved_models(file_name:str=None,path:str=None, file_prefix:str=None, mode:str=None,**kwargs):
        model_files = glob.glob(f"{path}/{file_name}_{file_prefix}_epoch*.pth")

        if mode == 'checkpoint' and len(model_files) > 3:
            # 检查点模式，保留最新的3个模型
            model_files.sort(key=os.path.getmtime)
            for file in model_files[:-3]:
                os.remove(file)
                logging.info(f"Removed old checkpoint model file: {file}")