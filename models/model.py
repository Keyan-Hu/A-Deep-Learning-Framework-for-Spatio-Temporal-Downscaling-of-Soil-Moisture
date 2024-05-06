import torch
import torch.nn as nn

from models.block import BasicBlock, Bottleneck, Compression_excitation_block, ResBlock, Multi_scale_block
from models.spatial_block import KPNet, ResNet, UNet,ViT
from models.temporal_block import TemporalDilatedConv

class STConvNet(nn.Module):
    def __init__(self, usage_model='ST',temporal_usage_max_avg=False,temporal_usage_position_embedding=False, spatial_block=None, concat_method='serial', output_method='add', # 模型选择
                 hour_in_channels=8, hour_step=17, day_in_channels=23, day_step=5,                            # 时间卷积元素
                 half_day_in_channels=2, half_day_step=9, static_in_channels=11,
                 temporal_kernel_size=2, target_time_steps=2,
                 block=BasicBlock, layers=[3,3,9,3], spatial_out_channels=1,                                  # 空间卷积元素
                 init_mid_channels=128,dropout_rate = 0.1,layer_scale_init_value=1e-6,**kwargs):                       
        """
        初始化类：选择接受空间块和连接模式，之后分别为时间块必要元素和空间块必要元素
        usage_model: 使用模式，可选'ST'（时空模型）, 'temporal'（仅时间模型）, 'spatial'（仅空间模型）
        temporal_usage_max_avg: 是否使用全局池化层
        spatial_block: 空间块类型，可选'resnet', 'unet', 'ernet', 'ernet_lite', 'kpnet', 'kpnet_mul_head', 'kpnet_Encoder_lite', 'vae','Spatial_transformer'
        concat_method: 连接模式，可选'parallel', 'serial'
        output_method: 输出模式，可选'add', 'individual', 'single'
        block: Resnet空间块中的基本块类型，可选'BasicBlock', 'Bottleneck'
        kpnet_mul_head：已经在模型内部实现了concat和残差连接
        """
        super(STConvNet, self).__init__()
        self.concat_method = concat_method
        self.output_method = output_method
        self.spatial_block = spatial_block
        self.usage_model = usage_model
        self.temporal_conv = TemporalDilatedConv(hour_in_channels, hour_step, 
                                                 day_in_channels, day_step, 
                                                 half_day_in_channels, half_day_step, 
                                                 static_in_channels, temporal_kernel_size, 
                                                 target_time_steps,dropout_rate=dropout_rate,
                                                 temporal_usage_max_avg=temporal_usage_max_avg,temporal_usage_position_embedding=temporal_usage_position_embedding)
        
        temporal_out_channels = self.temporal_conv.get_out_channels()
        self.temporal_output = nn.Conv2d(temporal_out_channels, 1, kernel_size=1)
        
        if self.usage_model == 'spatial' or self.concat_method=='parallel':spatial_input_channels = hour_in_channels + day_in_channels + half_day_in_channels + static_in_channels
        elif self.usage_model == 'ST': spatial_input_channels = temporal_out_channels + hour_in_channels + day_in_channels + half_day_in_channels + static_in_channels
        else: spatial_input_channels = temporal_out_channels

        self.final = nn.Conv2d(spatial_out_channels * 2,spatial_out_channels * 1,kernel_size=1)
        if spatial_block is None or spatial_block=='None':
            pass
        elif spatial_block == 'resnet':
            self.spatial_conv = ResNet(eval(block), layers, spatial_input_channels, spatial_out_channels,init_mid_channels=init_mid_channels)
        elif spatial_block == 'unet':
            self.spatial_conv = UNet(spatial_input_channels, spatial_out_channels,init_mid_channels=init_mid_channels)
        elif spatial_block == 'kpnet':
            self.spatial_conv = KPNet(spatial_input_channels, spatial_out_channels,net_depth=layers, init_mid_channels=init_mid_channels, dropout_rate = dropout_rate, layer_scale_init_value=layer_scale_init_value)
        elif spatial_block == 'ViT':
            self.spatial_conv = ViT(input_channels = spatial_input_channels, out_channels = spatial_out_channels, positional_embedding = '1d_absolute_trainable')
        else:
            raise ValueError("Invalid spatial_block type")

    def forward(self, hour_input, day_input, half_day_input, static_input):
        """
        输入：hour_input, day_input, half_day_input, static_input
        输出：spatial_output
        """        
        # 时空卷积
        if self.usage_model == 'ST':
            # 时间卷积处理
            temporal_output = self.temporal_conv(hour_input, day_input, half_day_input, static_input)
            temporal_output_channel = self.temporal_output(temporal_output)

            # 空间卷积处理
            if self.concat_method=='parallel':
                mid_hour = hour_input[:, :, hour_input.size(2) // 2, :, :]
                mid_day = day_input[:, :, day_input.size(2) // 2, :, :]
                mid_half_day = half_day_input[:, :, half_day_input.size(2) // 2, :, :]
                combined_static = torch.cat((mid_hour, mid_day, mid_half_day, static_input), dim=1)
                spatial_output = self.spatial_conv(combined_static)

            elif self.concat_method=='serial':
                mid_hour = hour_input[:, :, hour_input.size(2) // 2, :, :]
                mid_day = day_input[:, :, day_input.size(2) // 2, :, :]
                mid_half_day = half_day_input[:, :, half_day_input.size(2) // 2, :, :]
                combined_static = torch.cat((mid_hour, mid_day, mid_half_day, static_input, temporal_output), dim=1)
                spatial_output = self.spatial_conv(combined_static)


            # 输出模式
            if self.output_method=='add':
                return (spatial_output[0] + temporal_output_channel,) + spatial_output[1:] if isinstance(spatial_output, tuple) else spatial_output + temporal_output_channel
            
            elif self.output_method=='individual':
                if isinstance(spatial_output, tuple):
                    return (*spatial_output, temporal_output_channel)
                else:
                    out = self.final(torch.cat((spatial_output, temporal_output_channel), dim=1))
                    return (out, spatial_output, temporal_output_channel)
                
            elif self.output_method == 'single':
                return spatial_output

        # 时间卷积
        elif self.usage_model == 'temporal':
            temporal_output = self.temporal_conv(hour_input, day_input, half_day_input, static_input)
            temporal_output_channel = self.temporal_output(temporal_output)
            return temporal_output_channel

        # 空间卷积
        elif self.usage_model == 'spatial':
            mid_hour = hour_input[:, :, hour_input.size(2) // 2, :, :] # batchsize channels time_steps lat lon  # channels time_steps lat lon
            mid_day = day_input[:, :, day_input.size(2) // 2, :, :]
            mid_half_day = half_day_input[:, :, half_day_input.size(2) // 2, :, :]
            combined_static = torch.cat((mid_hour, mid_day, mid_half_day, static_input), dim=1)
            spatial_output = self.spatial_conv(combined_static)
            return spatial_output
