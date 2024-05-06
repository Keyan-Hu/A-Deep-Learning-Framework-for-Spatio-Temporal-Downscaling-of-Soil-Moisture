import os
import yaml
import sys
import math
sys.path.append('C:\\Users\\Administrator\\Desktop\\code\\ST-Conv')
import argparse
import pickle
import xarray as xr
import numpy as np
import pandas as pd 
from tool.utils import Util
import random
from datetime import datetime, timedelta
from tool.utils import Util

# 数据预准备函数
def data_preparation(file_path, target_var, var_list, missing_values,mode="train"):
    # 读取nc文件
    data = xr.open_dataset(file_path)
    # 将lat坐标轴从小到大排列并选择特定区域
    data = data.sortby('lat')
    # 提取起始时间和结束时间并转化为datatime对象
    end_time = pd.to_datetime(data.time.values[-1])

    # 创建三个新的变量：Longitude, Latitude, DOY
    longitude = xr.DataArray(np.zeros(data[target_var].shape), 
                            dims=data[target_var].dims, 
                            coords=data[target_var].coords)
    latitude = xr.DataArray(np.zeros(data[target_var].shape), 
                            dims=data[target_var].dims, 
                            coords=data[target_var].coords)
    doy = xr.DataArray(np.zeros(data[target_var].shape), 
                    dims=data[target_var].dims, 
                    coords=data[target_var].coords)

    # 更新 DOY 变量
    for i in range(data.time.shape[0]):
        timestamp = pd.Timestamp(data.time[i].values)
        doy[i, :, :] = Util.calculate_hours_since_year_start(timestamp)

    # 更新 Longitude 和 Latitude 变量
    for i in range(data.lon.shape[0]):
        longitude[:, :, i] = data.lon[i]
    for i in range(data.lat.shape[0]):
        latitude[:, i, :] = data.lat[i]

    # 将新的变量添加到数据集中
    data = data.assign(Longitude=longitude, Latitude=latitude, DOY=doy)

    # 只提取var_list中的变量，如果是重建那么没有SM变量
    if mode == "train":
        data = data[var_list]
    else:
        data = data[var_list[1:]]
    # 质量控制
    # MYD13A1:有效值-2000 to 10000，其他设置为0
    data.NDVI.values = np.where((data.NDVI.values < -2000) | (data.NDVI.values > 10000), np.nan, data.NDVI.values)
    data.EVI.values = np.where((data.EVI.values < -2000) | (data.EVI.values > 10000), np.nan, data.EVI.values)
    # MCD11A1:有效值7500 to 65535 
    data.LST_Day_1km.values = np.where((data.LST_Day_1km.values < 7500) | (data.LST_Day_1km.values > 65535), np.nan, data.LST_Day_1km.values)
    data.LST_Night_1km.values = np.where((data.LST_Night_1km.values < 7500) | (data.LST_Night_1km.values > 65535), np.nan, data.LST_Night_1km.values)
    # MCD43C3：有效值 0 到 32766
    MCD43C3_Var = ['Albedo_BSA_Band1','Albedo_BSA_Band2','Albedo_BSA_Band3','Albedo_BSA_Band4','Albedo_BSA_Band5','Albedo_BSA_Band6','Albedo_BSA_Band7','Albedo_BSA_vis','Albedo_BSA_nir','Albedo_BSA_shortwave'\
                   ,'Albedo_WSA_vis','Albedo_WSA_nir','Albedo_WSA_shortwave','Albedo_WSA_Band1','Albedo_WSA_Band2','Albedo_WSA_Band3','Albedo_WSA_Band4','Albedo_WSA_Band5','Albedo_WSA_Band6','Albedo_WSA_Band7']
    #循环遍历所有变量      
    for var in MCD43C3_Var:
        data[var].values = np.where((data[var].values < 0) | (data[var].values > 32766), np.nan, data[var].values)

    # 使用 missing_values 字典辨别缺失值
    for var, missing_val in missing_values.items():
        if var in data:
            data[var] = data[var].where(data[var] != missing_val, np.nan)
    return data

# 区间数据调用函数，输入经纬度区域序号从1开始,判断是否lon_max和lat_max是否大于序列最大值（取序列最大值）
def select_data(data,lon_num,lat_num,area_size):
    '''
    data:输入数据
    lon_num:经度区域序号
    lat_num:纬度区域序号
    area_size:区域尺寸
    '''
    lon_shape = data.lon.shape[0]
    lat_shape = data.lat.shape[0]
    # 如果最小值大于序列最大值，报错
    if lon_num>lon_shape//area_size or lat_num>lat_shape//area_size:
        print("Error:lon_num or lat_num is too large")
    lon_min = data.lon[(lon_num-1)*area_size]
    lat_min = data.lat[(lat_num-1)*area_size]
    if lon_num*area_size>lon_shape:
        lon_max = data.lon[lon_shape-1]
    elif lon_num*area_size<=lon_shape:
        lon_max = data.lon[lon_num*area_size-1]
    if lat_num*area_size>lat_shape:
        lat_max = data.lat[lat_shape-1]
    elif lat_num*area_size<=lat_shape:
        lat_max = data.lat[lat_num*area_size-1]
    data_subset = data.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    return data_subset

# 点数据调用函数
def select_data_pixel(data, lon, lat, area_size):
    """
    从NetCDF数据集中自适应地提取给定经纬度周围指定像素大小的空间数据。

    :param data: xarray数据集
    :param lon: 目标经度
    :param lat: 目标纬度
    :param area_size: 提取区域的大小（以像素为单位）
    :return: 提取的数据子集
    """
    # 计算最接近给定经纬度的像素点索引
    abs_diff_lon = np.abs(data['lon'] - lon)
    abs_diff_lat = np.abs(data['lat'] - lat)
    min_diff_lon = abs_diff_lon.argmin().item()  # 获取整数索引
    min_diff_lat = abs_diff_lat.argmin().item()

    # 获取经度和纬度的总长度
    total_lon = len(data['lon'])
    total_lat = len(data['lat'])

    # 计算提取区域的索引范围
    half_size = area_size // 2
    start_lon = np.clip(min_diff_lon - half_size, 0, total_lon - area_size)
    end_lon = start_lon + area_size
    start_lat = np.clip(min_diff_lat - half_size, 0, total_lat - area_size)
    end_lat = start_lat + area_size

    # 提取区域数据
    extracted_data = data.isel(lon=slice(int(start_lon), int(end_lon)), lat=slice(int(start_lat), int(end_lat)))

    return extracted_data

# 标准化函数
def vars_normalization(data):
    '''
    对所有数据不同通道标准化，保存每个变量的标准化系数, data以nc格式操作
    排除了0值和NaN值
    '''
    means_list = []
    std_list = []
    normalized_ds = data.copy()

    # 针对每个变量单独计算均值和标准差，并进行标准化
    for var in data.data_vars:
        valid_data = data[var].where(data[var] != 0)  # 排除0值
        mean = valid_data.mean(dim=('time', 'lon', 'lat'))
        std = valid_data.std(dim=('time', 'lon', 'lat'))

        normalized_ds[var] = (data[var] - mean) / std

        means_list.append(mean.values)
        std_list.append(std.values)

    return normalized_ds, means_list, std_list

# 掩码矩阵生成函数
def generate_combined_masks(data, path_file=None, area_size=None, random_time_num=100, time_patch_sel=100,num_random_masks=5000, missing_rate=0.95, method='SP',time_step = None):
    """
    生成结合了从数据中提取的掩码和随机生成掩码的数组。
    :param method: 生成掩码的方法'SP''ST'
    :param data: 输入的数据集，假设是一个四维数组 (time, lon, lat, var)
    :param area_size: 掩码区域大小
    :param num_data_masks: 从数据中提取的掩码数量
    :param num_random_masks: 随机生成的掩码数量
    :param missing_rate: 随机生成掩码的缺失率
    :return: 结合了数据提取掩码和随机掩码的数组
    """
    data = data.to_array()
    data = data.transpose("time", "lon", "lat", "variable")
    data = data.values

    # 从数据中提取掩码
    total_mask = np.isnan(data[..., 0]).astype(int)
    data_masks = []

    time_indices = np.random.choice(data.shape[0], random_time_num, replace=False)  # 选择100个不重复的时间点
    # 数据掩码
    for t in time_indices:
        for _ in range(time_patch_sel):  # 每个时间点提取100个掩码
            i = random.randint(0, total_mask.shape[1] - area_size)
            j = random.randint(0, total_mask.shape[2] - area_size)
            if method == 'SP':
                mask = total_mask[t, i:i+area_size, j:j+area_size]
            else:
                half_T = time_step//2
                mask = total_mask[t-half_T:t+half_T+(time_step % 2), i:i+area_size, j:j+area_size]
            data_masks.append(mask)

    # 随机掩码
    random_masks = [np.random.choice([0, 1], size=(time_step, area_size, area_size) if method != 'SP' else (area_size, area_size), p=[missing_rate, 1-missing_rate])
                    for _ in range(num_random_masks)]
    
    combined_masks = np.array(data_masks + random_masks)
    with open(path_file, 'wb') as file:
        pickle.dump(combined_masks ,file)

    return combined_masks

def generate_ST_data(data, target_var=None, file_path=None, mask_path=None, time_step=None, patch_size=8, stride=3, missing_rate=None, show_info=False):
       # 调整数据格式
    data_value = data.to_array()
    data_value = data_value.transpose("time", "lon", "lat", "variable")
    time, lon, lat, vars = data_value.shape

    temp_files = []
    num_selected_patches = 0
    num_all_patches = 0
    batch_size = 300  # 每批处理300个时间步

    # 计算每天的6:00和18:00对应的时间索引
    time_indices = []
    for single_time in data.time:
        current_time = pd.to_datetime(single_time.values)
        if current_time.hour in [6, 18]:
            time_indices.append(single_time)

    half_T = time_step // 2 if time_step else 0

    for batch_start in range(0, len(time_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(time_indices))
        combined_data = []
        combined_masks = []

        for t_idx in time_indices[batch_start:batch_end]:
            t = np.where(data_value.time == t_idx)[0][0]  # 获取实际的时间索引
            if t > half_T:
                time_start = t - half_T
            else:
                continue
            time_end = min(t + half_T + (time_step % 2), time)

            for i in range(0, lon - patch_size + 1, stride):
                for j in range(0, lat - patch_size + 1, stride):
                    time_block = data_value[time_start:time_end, i:i+patch_size, j:j+patch_size,:]
                    # mask 使用中间和前后尾部的数据叠加
                    mask = np.isnan(time_block[[0, t, -1], ..., 0]).astype(int)
                    patch_missing_rate = mask.sum() / mask.size
                    num_all_patches += 1
                    if patch_missing_rate <= missing_rate:
                        combined_data.append(time_block)
                        combined_masks.append(mask)
                        num_selected_patches += 1

            # 第二个起始点：左上角向右和向下偏移patch_size/2
            start_i = int(patch_size / 2)
            start_j = int(patch_size / 2)
            for i in range(start_i, lon - patch_size + 1, stride):
                for j in range(start_j, lat - patch_size + 1, stride):
                    time_block = data_value[time_start:time_end, i:i+patch_size, j:j+patch_size]
                    mask = np.isnan(time_block[[0, t, -1], ..., 0]).astype(int)
                    patch_missing_rate = mask.sum() / mask.size
                    num_all_patches += 1
                    if patch_missing_rate <= missing_rate:
                        combined_data.append(time_block)
                        combined_masks.append(mask)
                        num_selected_patches += 1

        # 保存当前批次的结果到临时文件
        if file_path and mask_path:
            temp_data_file = f"{file_path}_temp_{batch_start // batch_size}.pickle"
            temp_mask_file = f"{mask_path}_temp_{batch_start // batch_size}.pickle"
            with open(temp_data_file, 'wb') as file:
                pickle.dump(combined_data, file)
            with open(temp_mask_file, 'wb') as file:
                pickle.dump(combined_masks, file)
            temp_files.append((temp_data_file, temp_mask_file))

        if show_info:
            print(f"Batch {batch_start // batch_size}:")
            print(f"Number of selected patches: {num_selected_patches}")
            print(f"Number of all patches: {num_all_patches}")
            print(f"Rate of selected patches: {num_selected_patches/num_all_patches}")

    # 在所有批次处理完成后，合并所有批次的数据
    if file_path and mask_path:
        final_data = []
        final_masks = []
        for data_file, mask_file in temp_files:
            with open(data_file, 'rb') as file:
                final_data.extend(pickle.load(file))
            with open(mask_file, 'rb') as file:
                final_masks.extend(pickle.load(file))
            os.remove(data_file)
            os.remove(mask_file)

        with open(file_path, 'wb') as file:
            pickle.dump(final_data, file)
        with open(mask_path, 'wb') as file:
            pickle.dump(final_masks, file)

def generate_mult_data(vars_day_argu, Day_time_step, vars_half_day_argu, Half_Day_time_step, vars_hour_argu,Hour_time_step, vars_static_argu, \
                        data, Static_file_path=None,Day_file_path=None,Half_Day_file_path=None,Hour_file_path=None,\
                            patch_size=16, stride=3, missing_rate=None, show_info=False):
    # 调整数据格式
    vars_day_list, vars_half_day_list, vars_hour_list, vars_static_list = list(vars_day_argu.keys()), list(vars_half_day_argu.keys()), list(vars_hour_argu.keys()), list(vars_static_argu.keys())
    def data_deal(data,vars_list):
        data = generate_diff_data(data,vars_list)
        print(type(data))
        data_value = data.to_array()
        data_value = data_value.transpose("time", "lon", "lat", "variable")
        return data_value

    Static_data_value,Day_data_value,Half_Day_data_value,Hour_data_value = data_deal(data,vars_static_list),data_deal(data,vars_day_list),data_deal(data,vars_half_day_list),data_deal(data,vars_hour_list)

    data_value = data.to_array()
    data_value = data_value.transpose("time", "lon", "lat", "variable")
    time, lon, lat, vars = data_value.shape

    num_selected_patches = 0
    num_all_patches = 0
    batch_size = 10

    # 计算每天的6:00和18:00对应的时间索引
    time_indices = []
    for single_time in data.time:
        current_time = pd.to_datetime(single_time.values)
        if current_time.hour in [6, 18]:
            time_indices.append(single_time)

    half_T = Half_Day_time_step // 2 if Half_Day_time_step else 0

    for batch_start in range(0, len(time_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(time_indices))
        static_data_list, day_data_list, half_day_data_list, hour_data_list = [], [], [], []

        for t_idx in time_indices[batch_start:batch_end]:
            t = np.where(data_value.time == t_idx)[0][0]  # 获取实际的时间索引
            if t > half_T:
                time_start = t - half_T
            else:
                continue
            time_end = min(t + half_T + (Half_Day_time_step % 2), time)

            # 先计算时间索引并保存
            day_t = Util.find_closest_time_index(pd.to_datetime(t_idx.values), Day_data_value.time)
            half_day_t = Util.find_closest_time_index(pd.to_datetime(t_idx.values), Half_Day_data_value.time)
            hour_t = Util.find_closest_time_index(pd.to_datetime(t_idx.values), Hour_data_value.time)

            # 计算 day, half_day, hour 的前后时间范围
            pre_day = day_t - Day_time_step // 2
            behind_day = day_t + Day_time_step // 2 + (Day_time_step % 2) - len(Day_data_value.time)
            day_time_start = max(pre_day, 0)
            day_time_end = min(day_t + Day_time_step // 2 + (Day_time_step % 2), len(Day_data_value.time))

            pre_half_day = half_day_t - Half_Day_time_step // 2
            behind_half_day = half_day_t + Half_Day_time_step // 2 + (Half_Day_time_step % 2) - len(Half_Day_data_value.time)
            half_day_time_start = max(pre_half_day, 0)
            half_day_time_end = min(half_day_t + Half_Day_time_step // 2 + (Half_Day_time_step % 2), len(Half_Day_data_value.time))

            pre_hour = hour_t - Hour_time_step // 2
            behind_hour = hour_t + Hour_time_step // 2 + (Hour_time_step % 2) - len(Hour_data_value.time)
            hour_time_start = max(pre_hour, 0)
            hour_time_end = min(hour_t + Hour_time_step // 2 + (Hour_time_step % 2), len(Hour_data_value.time))

            for i in range(0, lon - patch_size + 1, stride):
                for j in range(0, lat - patch_size + 1, stride):
                    time_block = data_value[time_start:time_end, i:i+patch_size, j:j+patch_size,:]
                    if time_block.shape[0] < Half_Day_time_step:
                        continue
                    patch_missing_rate = np.isnan(time_block[half_T, ..., 0]).astype(int).sum() / np.isnan(time_block[half_T, ..., 0]).astype(int).size
                    num_all_patches += 1
                    if patch_missing_rate <= missing_rate:
                        ## 时间块
                        # 提取静态数据
                        static_block = Static_data_value[t, i:i+patch_size, j:j+patch_size,:]
                        # 提取日数据
                        day_block = Day_data_value[day_time_start:day_time_end , i:i+patch_size, j:j+patch_size,:]
                        # 提取半日数据
                        half_day_block = Half_Day_data_value[half_day_time_start:half_day_time_end , i:i+patch_size, j:j+patch_size,:]
                        # 提取小时数据
                        hour_block = Hour_data_value[hour_time_start:hour_time_end , i:i+patch_size, j:j+patch_size,:]

                        # Padding if necessary
                        if pre_day < 0 or behind_day > 0:
                            day_block = np.pad(day_block, [(max(-pre_day, 0), max(behind_day, 0)), (0, 0), (0, 0), (0, 0)], mode='constant', constant_values=np.nan)
                        if pre_half_day < 0 or behind_half_day > 0:
                            half_day_block = np.pad(half_day_block, [(max(-pre_half_day, 0), max(behind_half_day, 0)), (0, 0), (0, 0), (0, 0)], mode='constant', constant_values=np.nan)
                        if pre_hour < 0 or behind_hour > 0:
                            hour_block = np.pad(hour_block, [(max(-pre_hour, 0), max(behind_hour, 0)), (0, 0), (0, 0), (0, 0)], mode='constant', constant_values=np.nan)

                        static_data_list.append(static_block)
                        day_data_list.append(day_block)
                        half_day_data_list.append(half_day_block)
                        hour_data_list.append(hour_block)
                        num_selected_patches += 1
            if Static_file_path:
                with open(f"{Static_file_path}_temp_{batch_start // batch_size}.pickle", 'wb') as file:
                    pickle.dump(static_data_list, file)
            if Day_file_path:
                with open(f"{Day_file_path}_temp_{batch_start // batch_size}.pickle", 'wb') as file:
                    pickle.dump(day_data_list, file)
            if Half_Day_file_path:
                with open(f"{Half_Day_file_path}_temp_{batch_start // batch_size}.pickle", 'wb') as file:
                    pickle.dump(half_day_data_list, file)
            if Hour_file_path:
                with open(f"{Hour_file_path}_temp_{batch_start // batch_size}.pickle", 'wb') as file:
                    pickle.dump(hour_data_list, file)

            if show_info:
                print(f"Batch {batch_start // batch_size}:")
                print(f"Number of selected patches: {num_selected_patches}")
                print(f"Number of all patches: {num_all_patches}")

    # 生成静态、日、半日、小时数据和掩码的临时文件列表
    temp_static_files = [f"{Static_file_path}_temp_{i // batch_size}.pickle" for i in range(0, len(time_indices), batch_size) if Static_file_path]
    temp_day_files = [f"{Day_file_path}_temp_{i // batch_size}.pickle" for i in range(0, len(time_indices), batch_size) if Day_file_path]
    temp_half_day_files = [f"{Half_Day_file_path}_temp_{i // batch_size}.pickle" for i in range(0, len(time_indices), batch_size) if Half_Day_file_path]
    temp_hour_files = [f"{Hour_file_path}_temp_{i // batch_size}.pickle" for i in range(0, len(time_indices), batch_size) if Hour_file_path]
    # 合并和保存
    if Static_file_path:
        Util.merge_and_save_temp_data(Static_file_path, temp_static_files)
    if Day_file_path:
        Util.merge_and_save_temp_data(Day_file_path, temp_day_files)
    if Half_Day_file_path:
        Util.merge_and_save_temp_data(Half_Day_file_path, temp_half_day_files)
    if Hour_file_path:
        Util.merge_and_save_temp_data(Hour_file_path, temp_hour_files)

def generate_ST_Cross_data(data, target_var=None, time_file_path=None, time_mask_path=None, area_file_path=None, area_mask_path=None, time_step=None, time_patch_size=8, area_patch_size=16, stride=3, missing_rate=None, show_info=False):

   # 调整数据格式
    data_value = data.to_array()
    data_value = data_value.transpose("time", "lon", "lat", "variable")
    time, lon, lat, vars = data_value.shape

    temp_time_files = []
    temp_area_files = []
    num_selected_patches = 0
    num_all_patches = 0
    batch_size = 100  # 每批处理300个时间步

    # 计算每天的6:00和18:00对应的时间索引
    time_indices = []
    for single_time in data.time:
        current_time = pd.to_datetime(single_time.values)
        if current_time.hour in [6, 18]:
            time_indices.append(single_time)

    half_T = time_step // 2 if time_step else 0

    for batch_start in range(0, len(time_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(time_indices))
        combined_time_data = []
        combined_time_masks = []
        combined_area_data = []
        combined_area_masks = []

        for t_idx in time_indices[batch_start:batch_end]:
            t = np.where(data.time == t_idx)[0][0]  # 获取实际的时间索引
            if t > half_T:
                time_start = t - half_T
            else:
                continue
            time_end = min(t + half_T + (time_step % 2), time)

            for i in range(0, lon - time_patch_size + 1, stride):
                for j in range(0, lat - time_patch_size + 1, stride):
                    time_block = data_value[time_start:time_end, i:i+time_patch_size, j:j+time_patch_size,:]
                    area_start_i = max(i - area_patch_size // 2, 0)
                    area_end_i = min(i + area_patch_size // 2 + (area_patch_size % 2), lon)
                    area_start_j = max(j - area_patch_size // 2, 0)
                    area_end_j = min(j + area_patch_size // 2 + (area_patch_size % 2), lat)
                    area_block = data_value[t, area_start_i:area_end_i, area_start_j:area_end_j, :]
                    if area_block.shape != (area_patch_size, area_patch_size, vars):
                        area_block = np.pad(area_block, [(max(0, area_patch_size // 2 - i), max(0, area_end_i - lon)), (max(0, area_patch_size // 2 - j), max(0, area_end_j - lat)), (0, 0)], mode='constant', constant_values=0)

                    time_block_mask = np.isnan(area_block[t, ..., 0]).astype(int)
                    patch_missing_rate = time_block_mask.sum() /time_block_mask.size
                    num_all_patches += 1
                    if missing_rate is None or patch_missing_rate <= missing_rate:
                        combined_time_data.append(time_block)
                        combined_time_masks.append(time_block_mask)
                        combined_area_data.append(area_block)
                        num_selected_patches += 1

            # 第二个起始点：左上角向右和向下偏移patch_size/2
            start_i = int(time_patch_size / 2)
            start_j = int(time_patch_size / 2)
            for i in range(start_i, lon - time_patch_size + 1, stride):
                for j in range(start_j, lat - time_patch_size + 1, stride):
                    time_block = data_value[time_start:time_end, i:i+time_patch_size, j:j+time_patch_size,:]
                    area_start_i = max(i - area_patch_size // 2, 0)
                    area_end_i = min(i + area_patch_size // 2 + (area_patch_size % 2), lon)
                    area_start_j = max(j - area_patch_size // 2, 0)
                    area_end_j = min(j + area_patch_size // 2 + (area_patch_size % 2), lat)
                    area_block = data_value[t, area_start_i:area_end_i, area_start_j:area_end_j, :]
                    if area_block.shape != (area_patch_size, area_patch_size, vars):
                        area_block = np.pad(area_block, [(max(0, area_patch_size // 2 - i), max(0, area_end_i - lon)), (max(0, area_patch_size // 2 - j), max(0, area_end_j - lat)), (0, 0)], mode='constant', constant_values=np.nan)

                    time_block_mask = np.isnan(area_block[t, ..., 0]).astype(int)
                    patch_missing_rate = time_block_mask.sum() /time_block_mask.size
                    num_all_patches += 1
                    if missing_rate is None or patch_missing_rate <= missing_rate:
                        combined_time_data.append(time_block)
                        combined_time_masks.append(time_block_mask)
                        combined_area_data.append(area_block)
                        num_selected_patches += 1

            # 保存每个时间步的结果到临时文件
            if time_file_path and time_mask_path and area_file_path and area_mask_path:
                temp_time_data_file = f"{time_file_path}_temp_{batch_start // batch_size}.pickle"
                temp_time_mask_file = f"{time_mask_path}_temp_{batch_start // batch_size}.pickle"
                temp_area_data_file = f"{area_file_path}_temp_{batch_start // batch_size}.pickle"
                with open(temp_time_data_file, 'wb') as file:
                    pickle.dump(combined_time_data, file)
                with open(temp_time_mask_file, 'wb') as file:
                    pickle.dump(combined_time_masks, file)
                with open(temp_area_data_file, 'wb') as file:
                    pickle.dump(combined_area_data, file)
                temp_time_files.append((temp_time_data_file, temp_time_mask_file))

        if show_info:
            print(f"Batch {batch_start // batch_size}:")
            print(f"Number of selected patches: {num_selected_patches}")
            print(f"Number of all patches: {num_all_patches}")
            print(f"Rate of selected patches: {num_selected_patches/num_all_patches}")


    # 合并所有临时文件的结果
    final_time_data = []
    final_time_masks = []
    final_area_data = []
    for data_file, mask_file in temp_time_files:
        with open(data_file, 'rb') as file:
            final_time_data.extend(pickle.load(file))
        with open(mask_file, 'rb') as file:
            final_time_masks.extend(pickle.load(file))
        os.remove(data_file)
        os.remove(mask_file)
    for data_file, mask_file in temp_area_files:
        with open(data_file, 'rb') as file:
            final_area_data.extend(pickle.load(file))
        os.remove(data_file)

    # 保存最终的合并文件
    if time_file_path and time_mask_path and area_file_path and area_mask_path:
        with open(time_file_path, 'wb') as file:
            pickle.dump(final_time_data, file)
        with open(time_mask_path, 'wb') as file:
            pickle.dump(final_time_masks, file)
        with open(area_file_path, 'wb') as file:
            pickle.dump(final_area_data, file)

def generate_SP_data(data, target_var=None, file_path=None, mask_path=None, patch_size=8, stride=3, missing_rate=None, show_info=False):
    
    # 调整数据格式
    data_value = data.to_array()
    data_value = data_value.transpose("time", "lon", "lat", "variable")
    time, lon, lat = data.time.shape[0], data.lon.shape[0], data.lat.shape[0]

    temp_files = []
    num_selected_patches = 0
    num_all_patches = 0
    batch_size = 300  # 每批处理300个时间步

    # 计算每天的6:00和18:00对应的时间索引
    time_indices = []
    for single_time in data.time:
        current_time = pd.to_datetime(single_time.values)
        if current_time.hour in [6, 18]:
            time_indices.append(single_time)

    for batch_start in range(0, len(time_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(time_indices))
        combined_data = []
        combined_masks = []

        for t_idx in time_indices[batch_start:batch_end]:
            t = np.where(data.time == t_idx)[0][0]  # 获取实际的时间索引
            # 第一个起始点：左上角
            for i in range(0, lon - patch_size + 1, stride):
                for j in range(0, lat - patch_size + 1, stride):
                    patch = data_value[t, i:i+patch_size, j:j+patch_size, :]
                    mask = np.isnan(patch[..., 0]).astype(int)
                    patch_missing_rate = (mask.sum() / mask.size)
                    num_all_patches += 1
                    if patch_missing_rate <= missing_rate:
                        combined_data.append(patch.values)
                        combined_masks.append(mask)
                        num_selected_patches += 1

        # 保存当前批次的结果到临时文件
        if file_path and mask_path:
            temp_data_file = f"{file_path}_temp_{batch_start // batch_size}.pickle"
            temp_mask_file = f"{mask_path}_temp_{batch_start // batch_size}.pickle"
            with open(temp_data_file, 'wb') as file:
                pickle.dump(combined_data, file)
            with open(temp_mask_file, 'wb') as file:
                pickle.dump(combined_masks, file)
            temp_files.append((temp_data_file, temp_mask_file))

        if show_info:
            print(f"Batch {batch_start // batch_size}:")
            print(f"Number of selected patches: {num_selected_patches}")
            print(f"Number of all patches: {num_all_patches}")
            print(f"Rate of selected patches: {num_selected_patches/num_all_patches}")

    # 在所有批次处理完成后，合并所有批次的数据
    if file_path and mask_path:
        final_data = []
        final_masks = []
        for data_file, mask_file in temp_files:
            with open(data_file, 'rb') as file:
                final_data.extend(pickle.load(file))
            with open(mask_file, 'rb') as file:
                final_masks.extend(pickle.load(file))
            os.remove(data_file)
            os.remove(mask_file)

        with open(file_path, 'wb') as file:
            pickle.dump(final_data, file)
        with open(mask_path, 'wb') as file:
            pickle.dump(final_masks, file)

def generate_diff_data(data,var_list):
    """
    生成不同时间点的时间
    static：静态数据：直接返回一个时间的片段
    day:提取每天的12：00的数据返回
    half_day:提取每天的6：00和18：00的数据返回
    hour：提取每天每三个小时的数据返回
    var_list:第一个变量是以上的类型，后续变量是后面需要提取出来的变量
    """
    temp_data = data
    time_strategy = var_list[0]
    variables = var_list[1:]

    # 定义时间筛选函数
    def filter_time(time, strategy):
        hour = pd.to_datetime(time).hour
        if strategy == "Static":
            return True  # 静态数据，提取所有时间
        elif strategy == "Day":
            return hour == 12
        elif strategy == "Half_Day":
            return hour in [6, 18]
        elif strategy == "Hour":
            return hour % 3 == 0
        else:
            raise ValueError(f"未知的时间策略: {strategy}")

    # 使用列表推导式创建一个布尔型数据数组
    time_cond = [filter_time(t, time_strategy) for t in temp_data.time.values]

    # 将布尔型数组转换为xarray数据数组
    time_cond_xr = xr.DataArray(time_cond, dims=["time"], coords={"time": temp_data.time})

    # 使用这个数据数组作为条件进行筛选
    filtered_times = temp_data.time.where(time_cond_xr, drop=True)

    # 提取特定变量并合并为一个 xarray.Dataset
    extracted_data_arrays = []
    for var in variables:
        if var in temp_data:
            extracted_data_array = temp_data[var].sel(time=filtered_times)
            extracted_data_arrays.append(extracted_data_array)

    # 将提取的数据数组合并为一个 xarray.Dataset
    extracted_dataset = xr.merge(extracted_data_arrays)

    return extracted_dataset

def main(args):
    model_argu = Util.load_config(args.model_config_path)
    missing_values = Util.load_config(args.vars_config_path)
    data_argu = Util.load_config(args.data_process_config_path)
    path_argu = Util.load_config(args.path_config_path)
    vars_day_argu = Util.load_config(args.vars_day_config_path)
    vars_half_day_argu = Util.load_config(args.vars_half_day_config_path)
    vars_hour_argu = Util.load_config(args.vars_hour_config_path)
    vars_static_argu = Util.load_config(args.vars_static_config_path)
    var_list = list(missing_values.keys())
    return model_argu, missing_values, var_list ,data_argu, path_argu, vars_day_argu, vars_half_day_argu, vars_hour_argu, vars_static_argu

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path', type=str, default='./config/model_config.yaml')
    parser.add_argument('--vars_config_path', type=str, default='./config/construct_Data/vars_config.yaml')
    parser.add_argument('--data_process_config_path', type=str, default='./config/data_process_config.yaml')
    parser.add_argument('--path_config_path', type=str, default='./config/path_config.yaml')
    parser.add_argument('--vars_day_config_path', type=str, default='./config/construct_Data/vars_day_config.yaml')
    parser.add_argument('--vars_half_day_config_path', type=str, default='./config/construct_Data/vars_half_day_config.yaml')
    parser.add_argument('--vars_hour_config_path', type=str, default='./config/construct_Data/vars_hour_config.yaml')
    parser.add_argument('--vars_static_config_path', type=str, default='./config/construct_Data/vars_static_config.yaml')

    args = parser.parse_known_args()[0]
    model_argu, missing_values, var_list ,data_argu, path_argu ,vars_day_argu, vars_half_day_argu, vars_hour_argu, vars_static_argu = main(args)

    # 数据提取
    data = data_preparation(path_argu['0.37_Data'],var_list[0],var_list,missing_values,mode="train")

    # 全局Z-score标准化
    data, var_mean, var_std = vars_normalization(data)
    file_path = "D:\Data_Store\Dataset\ST_Conv\std_mean"
    np.save(file_path + "\mean.npy", var_mean[0])
    np.save(file_path + "\std.npy", var_std[0])

    for i in range(len(var_mean)):
        print(var_mean[i],var_std[i])
    '''
    # 生成掩码矩阵(弃用)
    #mask_SP_8 = generate_combined_masks(data,path_argu['mask_SP_8'] ,data_argu['SP_Area'], data_argu['random_time_num'], data_argu['time_patch_sel'], data_argu['num_random_masks'], data_argu['missing_rate'],method='SP')
    #mask_SP_16 = generate_combined_masks(data,path_argu['mask_SP_16'] ,data_argu['ST-Conv_Area'], data_argu['random_time_num'], data_argu['time_patch_sel'], data_argu['num_random_masks'], data_argu['missing_rate'],method='SP')
    #mask_ST_8 = generate_combined_masks(data,path_argu['mask_ST_8'] ,data_argu['SP_Area'], data_argu['random_time_num'], data_argu['time_patch_sel'], data_argu['num_random_masks'], data_argu['missing_rate'],method='ST',time_step=model_argu['Time_step'])

    # RESNET、Unet数据生成
    #generate_SP_data(data,var_list[0], path_argu['SP_Data'],path_argu['SP_Mask'],patch_size=data_argu['SP_Area'],stride=data_argu['stride'],missing_rate=data_argu['SP_Threshold'],show_info=True)
    # 生成不同时间节点数据(弃用)
    generate_mult_data(vars_day_argu, data_argu['ST_Day_T'], vars_half_day_argu, data_argu['ST_Half_Day_T'], vars_hour_argu,data_argu['ST_Hour_T'],
                          vars_static_argu, data, path_argu['ST_Static_Data'],path_argu['ST_Day_Data'],path_argu['ST_Half_Day_Data'],
                          path_argu['ST_Hour_Data'], patch_size=data_argu['ST-Conv_Area'],stride=data_argu['stride'], missing_rate=data_argu['ST-Conv_Threshold'], show_info=True)
    '''