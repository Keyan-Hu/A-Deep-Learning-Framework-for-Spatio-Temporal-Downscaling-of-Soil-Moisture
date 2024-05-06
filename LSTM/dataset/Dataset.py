import os
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class LSTMDataset(Dataset):
    def __init__(self, data_file: str):
        """
        初始化数据集
        :param data_file: 数据文件的路径
        """
        self.data_file = data_file

        # 创建 memmap 对象
        self.data_memmap = np.memmap(self.data_file, dtype='float32', mode='r', shape=self._calculate_data_shape())

    def _calculate_data_shape(self):
        with open(self.data_file, 'rb') as f:
            version = np.lib.format.read_magic(f)
            shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
            return shape

    def __len__(self):
        # 返回总的样本数
        return len(self.data_memmap)

    def __getitem__(self, idx):
        # 获取单个样本
        sample = self.data_memmap[idx]
        sample = np.nan_to_num(sample)

        # 分割数据和目标
        data = sample[:-1, 1:]  # [sequence_length, features] 
        target = sample[-1, 0]  # 目标值

        return torch.from_numpy(data).float(), torch.tensor(target).float()

'''
class LSTMDataset(Dataset):
    def __init__(self, data_dir: str):
        """
        初始化数据集
        :param data_dir: 数据文件的目录
        """
        self.data_dir = data_dir
        self.data_files = self._get_data_files()
        self.file_lengths = self._calculate_file_lengths()
        self.total_batches = sum(self.file_lengths)

        # 用于缓存当前文件数据
        self.current_file_index = -1
        self.current_file_data = None

    def _get_data_files(self):
        """
        获取数据目录中的所有数据文件
        """
        return [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir) if file.endswith('.npy')]

    def _calculate_file_lengths(self):
        """
        计算每个文件中的批次数
        """
        lengths = []
        for file in self.data_files:
            data = np.load(file)
            lengths.append(len(data))
        return lengths

    def __len__(self):
        """
        返回数据集中的总批次数
        """
        return self.total_batches

    def __getitem__(self, idx):
        """
        获取指定索引处的数据批次
        :param idx: 批次的索引
        """
        file_index = 0
        while idx >= self.file_lengths[file_index]:
            idx -= self.file_lengths[file_index]
            file_index += 1
            if file_index >= len(self.data_files):
                raise IndexError("索引超出数据范围")

        # 加载新文件或使用缓存
        if file_index != self.current_file_index:
            self.current_file_data = np.load(self.data_files[file_index])
            self.current_file_index = file_index

        batch = self.current_file_data[idx]
        batch = np.nan_to_num(batch)
        data = batch[:-1, 1:]  # 提取除最后一个时间点外的所有时间点作为数据
        target = batch[-1, 0]  # 提取最后一个时间点的第一个变量作为标签

        data = torch.from_numpy(data).float()
        target = torch.tensor(target).float()
        return data, target
'''