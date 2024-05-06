import os
import yaml
import numpy as np
import pandas as pd

class OriginUtils:
    @staticmethod
    def find_grid_points(mode=None, csv_path=None,find_mode=None):
        """
        给定网络（Shiquanhe、Ali、Maqu、Naqu、CTP），对子表中每个站点数据匹配格点坐标。
        返回一个列表，列表的每个元素是包含站点名、经度、纬度和文件名的列表。
        """
        df = pd.read_excel(csv_path, sheet_name=mode)
        base_path = os.path.dirname(csv_path)
        site_grid_list = []

        for index, row in df.iterrows():
            site_name = row['site_name']
            lon, lat = row['longitude'], row['latitude']
            grid_coord, _ = OriginUtils.find_grid_coordinates(lon, lat,find_mode=find_mode)
            # 构建数据文件名，格式假设为 '站点名_纬度_经度_其他固定字符串.csv'
            filename = f"{site_name}_{lat}_{lon}_3h_clip.csv"
            file_path = os.path.join(base_path, mode, filename)  # 完整的文件路径
            lon, lat = grid_coord
            site_grid_list.append([site_name, lon, lat, file_path])
        
        return site_grid_list
    
    @staticmethod
    def check_time_consistency(data_frames):
        # 检查所有数据框的时间列是否一致
        time_sets = [set(df.iloc[:, 0]) for df in data_frames]
        if not all(t == time_sets[0] for t in time_sets):
            raise ValueError("时间列不匹配")
        return True

    @staticmethod
    def mark_outliers_as_nan(data):
        # 异常检测，不变形状
        mean_of_data = np.mean(data)
        std_dev_data = np.std(data)
        return data.apply(lambda x: x if (mean_of_data - 3 * std_dev_data) < x < (mean_of_data + 3 * std_dev_data) else np.nan)

    @staticmethod
    def cal_avg_grid_point(site_list, lon, lat,find_mode=None):
        # 找到最近的数据点
        closest_grid, _ = OriginUtils.find_grid_coordinates(lon, lat,find_mode=find_mode)
        lon, lat = closest_grid

        data_lists = []
        for site_info in site_list:
            site_name, site_lon, site_lat, file_path = site_info
            if (site_lon, site_lat) == (lon, lat):
                try:
                    site_data = pd.read_csv(file_path)
                except FileNotFoundError:
                    print(f"文件{file_path}未找到。")
                    continue
                site_data['Time'] = pd.to_datetime(site_data.iloc[:, 0])
                data_lists.append(site_data)

        if not data_lists:
            return None
        if not OriginUtils.check_time_consistency(data_lists):
            return None

        aligned_data = pd.concat([df.set_index('Time')['SM'] for df in data_lists], axis=1, join='inner')
        aligned_data = aligned_data.apply(lambda row: OriginUtils.mark_outliers_as_nan(row), axis=1)
        avg_values = aligned_data.mean(axis=1)

        return pd.DataFrame({
            'Time': avg_values.index,
            'SM': avg_values
        })
    
    @staticmethod
    def cal_point_data_specific(site_list, point_name):
        """
        根据提供的点位名称获取指定列的数据。
        :param site_list: 包含站点信息的列表，每个元素是一个包含（站点名称，经度，纬度，文件路径）的元组。
        :param point_name: 指定的点位名称。
        :return: 包含特定点位数据的DataFrame。
        """
        for site_info in site_list:
            site_name, site_lon, site_lat, file_path = site_info
            if site_name == point_name:
                try:
                    site_data = pd.read_csv(file_path)
                    site_data['Time'] = pd.to_datetime(site_data.iloc[:, 0])
                    # 假设SM列包含所需的数据
                    return pd.DataFrame({
                        'Time': site_data['Time'],
                        'SM': site_data['SM']  # 或者使用其他具体的列名
                    })
                except FileNotFoundError:
                    print(f"文件 {file_path} 未找到。")
                    return None
        print(f"点位 {point_name} 的数据未找到。")
        return None
    
    @staticmethod
    def find_grid_coordinates(lon, lat, lon_grid=None, lat_grid=None,find_mode='0.1'):
        """
        输入经纬度，返回最近的格点坐标和索引。
        find_mode:'0.1':0.1_data,'0.37':0.37_data
        """
        if find_mode=='0.1':
            lat_min = 25.550117
            lat_max = 40.450184
            lon_min = 66.95031
            lon_max = 104.95048
            lat_points = 150
            lon_points = 381
        else:
            lat_min = 25.293533
            lat_max = 40.604774
            lon_min = 66.84705
            lon_max = 105.31184
            lat_points = 42
            lon_points = 104
        lat_linspace = np.linspace(lat_min, lat_max, lat_points)
        lon_linspace = np.linspace(lon_min, lon_max, lon_points)
        lon_grid, lat_grid = np.meshgrid(lon_linspace, lat_linspace)
        dist = (lon_grid - lon)**2 + (lat_grid - lat)**2
        lat_ind, lon_ind = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
        return (lon_grid[lat_ind, lon_ind], lat_grid[lat_ind, lon_ind]), (lat_ind, lon_ind)

    @staticmethod
    def load_config(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    @staticmethod
    def align_time_series(point_data, reconstruct_data_point, start_time, freq='3H'):
        """
        将 point_data 和 reconstruct_data_point 对齐，返回对齐后的数据。
        以 point_data 为 end_time 基准，start_time为输入的时间。
        返回一个DataFrame，第一列是Time，第二列是 point_data 的SM值，第三列是 reconstruct_aligned 的SM值。
        """
        # 确定时间范围
        point_start_time = pd.to_datetime(point_data['Time'].iloc[0])
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(point_data['Time'].iloc[-1])

        # 创建新的时间序列
        new_time_range = pd.date_range(start=min(start_time, point_start_time), end=end_time, freq=freq)

        # 调整 reconstruct_data_point 以匹配新的时间序列
        reconstruct_length = len(reconstruct_data_point)
        reconstruct_time_range = pd.date_range(start=start_time, periods=reconstruct_length, freq=freq)

        reconstruct_series = pd.Series(reconstruct_data_point, index=reconstruct_time_range)
        reconstruct_aligned = reconstruct_series.reindex(new_time_range)

        # 拓展 point_data 的时间列
        point_data_aligned = point_data.set_index('Time').reindex(new_time_range)
        point_data_aligned.index.name = 'Time'

        # 合并数据
        merged_data = pd.DataFrame({
            'Time': new_time_range,
            'Point Data SM': point_data_aligned['SM'].values,  # 假设 SM 值在 'SM' 列
            'Reconstructed SM': reconstruct_aligned.values
        })

        return merged_data

    