import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalDilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, dropout_rate=0.):
        super(TemporalDilatedConvBlock, self).__init__()
        self.expand_ratio = 6
        self.conv_time = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size, 1, 1), padding=(padding, 0, 0), dilation=(dilation, 1, 1)),
                                    nn.GELU())
        self.conv1 = nn.Sequential(nn.Conv3d(out_channels, out_channels*self.expand_ratio, kernel_size=(1, 1, 1),bias=False),
                                   nn.GELU())
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0. else nn.Identity()

        self.conv2 = nn.Sequential(nn.Conv3d(out_channels*self.expand_ratio, out_channels, kernel_size=(1, 1, 1),bias=False),
                                   nn.GELU())



    def forward(self, x):
        x = self.conv_time(x)
        x = self.conv1(x)

        x = self.dropout(x)
        x = self.conv2(x)

        return x


class TemporalDilatedConv(nn.Module):
    def __init__(self, hour_in_channels,hour_step,day_in_channels,day_step,half_day_in_channels,half_day_step,static_in_channels, kernel_size, target_time_steps,dropout_rate=None,temporal_usage_max_avg=True,temporal_usage_position_embedding=True):
        super(TemporalDilatedConv, self).__init__()
        # 时间维度的卷积核大小为2，不使用填充
        self.expand_ratio = 6 
        self.temporal_usage_max_avg = temporal_usage_max_avg
        self.temporal_usage_position_embedding = temporal_usage_position_embedding
        self.target_time_steps = target_time_steps
        if temporal_usage_position_embedding == True:
            hour_in_channels -= 1
            day_in_channels -= 1
            half_day_in_channels -= 1
            static_in_channels -= 2

        self.hour_block = self._create_block(hour_in_channels, hour_in_channels, hour_step, kernel_size,dropout_rate)
        self.day_block = self._create_block(day_in_channels, day_in_channels, day_step, kernel_size,dropout_rate)
        self.half_day_block = self._create_block(half_day_in_channels, half_day_in_channels, half_day_step, kernel_size,dropout_rate)  # 数据输入通道只有一个
        self.static_block = nn.Sequential(nn.Conv2d(static_in_channels, static_in_channels, kernel_size=(1, 1)), nn.ReLU())
        self.out_channels = hour_in_channels + day_in_channels + half_day_in_channels 

        self.final_conv = nn.Sequential(nn.Conv3d(self.out_channels, self.out_channels, kernel_size=(2, 1, 1), padding=(0, 0, 0)),
                                        nn.GELU())

        self.out_channels = self.out_channels + static_in_channels
        
        self.conv_2d = nn.Sequential(nn.Conv2d(self.out_channels, self.out_channels * self.expand_ratio, kernel_size=1),
                                 nn.GELU(),
                                 nn.Conv2d(self.out_channels * self.expand_ratio, self.out_channels * self.expand_ratio//2, kernel_size=1),
                                 nn.GELU(),
                                 nn.Conv2d(self.out_channels * self.expand_ratio//2, self.out_channels//2, kernel_size=1))
        
        self.out_channels = self.out_channels//2

    def _create_block(self, in_channels, out_channels, time_steps, kernel_size,dropout_rate):
        layers = []
        current_time_steps = time_steps
        dilation = 1
        while current_time_steps > self.target_time_steps:
            required_reduction = current_time_steps - self.target_time_steps
            adjusted_kernel_size = required_reduction + 1 if required_reduction + 1 < kernel_size else kernel_size
            padding = 0

            layers.append(TemporalDilatedConvBlock(in_channels, out_channels, adjusted_kernel_size, padding, dilation,dropout_rate))
            current_time_steps = current_time_steps - dilation * (adjusted_kernel_size - 1)
            in_channels = out_channels
            dilation *= 2

        return nn.Sequential(*layers)

    def process_temporal_data(self, input_data):
        processed_data = input_data[:, :-1, :, :, :]
        # 将时间通道加到其他所有通道上#time_channel = input_data[:, -1:, :, :, :]
        processed_data += input_data[:, -1:, :, :, :]

        return processed_data

    def process_static_data(self, static_input, hour_input, day_input, half_day_input):
        # 提取经度和纬度通道
        lon = static_input[:, -2:-1, :, :]  # [batch_size, 1, height, width]
        lat = static_input[:, -1:, :, :]    # [batch_size, 1, height, width]

        # 从原始数据中移除经度和纬度通道
        processed_static = static_input[:, :-2, :, :]

        # 扩展经度和纬度
        lon_expanded_hour = lon.unsqueeze(2).expand(-1, -1, hour_input.shape[2], -1, -1)
        lat_expanded_hour = lat.unsqueeze(2).expand(-1, -1, hour_input.shape[2], -1, -1)

        lon_expanded_day = lon.unsqueeze(2).expand(-1, -1, day_input.shape[2], -1, -1)
        lat_expanded_day = lat.unsqueeze(2).expand(-1, -1, day_input.shape[2], -1, -1)

        lon_expanded_half_day = lon.unsqueeze(2).expand(-1, -1, half_day_input.shape[2], -1, -1)
        lat_expanded_half_day = lat.unsqueeze(2).expand(-1, -1, half_day_input.shape[2], -1, -1)

        # 将经度和纬度加到其他所有通道上
        hour_input += lon_expanded_hour + lat_expanded_hour
        day_input += lon_expanded_day + lat_expanded_day
        half_day_input += lon_expanded_half_day + lat_expanded_half_day

        return processed_static, hour_input, day_input, half_day_input

    def get_out_channels(self):
        return self.out_channels

    def forward(self, hour_input, day_input, half_day_input, static_input):
        if self.temporal_usage_position_embedding == True:

            # 绝对时间编码
            hour_input = self.process_temporal_data(hour_input)
            day_input = self.process_temporal_data(day_input)
            half_day_input = self.process_temporal_data(half_day_input)

            # 绝对位置编码
            static_input,hour_input,day_input,half_day_input = self.process_static_data(static_input,hour_input,day_input,half_day_input)

        hour_output = self.hour_block(hour_input)
        day_output = self.day_block(day_input)
        half_day_output = self.half_day_block(half_day_input)
        
        # 静态数据卷积
        static_output = self.static_block(static_input)
        combined_output = torch.cat((hour_output, day_output, half_day_output), dim=1)
        final_output = self.final_conv(combined_output)
        final_output = final_output.squeeze(2)

        final_output = torch.cat((final_output, static_output), dim=1)
        final_outconv = self.conv_2d(final_output)
        return final_outconv
    