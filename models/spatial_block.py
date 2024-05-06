import math
import torch
from typing import List
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from models.block import UpConvBlock,ConvBlock,Compression_excitation_block,ResBlock,FFN
from models.attention import ChannelAttention,SpatialAttention
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

## 小工具
class MyMultiplier:
    """
    自乘器, keep=0时返回当前值不做改变, keep=True返回当前值乘2, keep=False时返回除以2
    """
    def __init__(self, initial_value):
        self.value = initial_value

    def __call__(self, keep=False):
        if keep == -1: 
            pass
        elif keep == True:
            self.value *= 2
        elif keep == False:
            self.value /= 2
        return int(self.value)

class MyRecorder:
    """
    记录器, 输入一个值, 返回上一个值
    """
    def __init__(self, initial_value):
        self.value = initial_value

    def __call__(self):
        return self.value

## ResNet
# V1.0
class ResNet_original(nn.Module):
    def __init__(self, block, layers, num_channels, num_classes,init_mid_channels=128):
        super(ResNet_original, self).__init__()

        self.in_channels = 64
        self.init_mid_channels = init_mid_channels
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, self.in_channels, kernel_size=5, stride=1, padding=2, bias=False), 
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        ) # 16*16*C

        # 下采样层
        # 这里和原版Resnet稍有不同在于这里的第一步可能会在第一步进行1*1卷积
        self.enc1 = self.make_layer(block, self.init_mid_channels, layers[0], stride=1)
        self.enc2 = self.make_layer(block, self.init_mid_channels*2, layers[1], stride=2)
        self.enc3 = self.make_layer(block, self.init_mid_channels*4, layers[2], stride=2)
        self.enc4 = self.make_layer(block, self.init_mid_channels*8, layers[3], stride=2)

        self.dec3 = UpConvBlock(self.init_mid_channels*8*block.expansion, self.init_mid_channels*4, self.init_mid_channels*4*block.expansion)
        self.dec2 = UpConvBlock(self.init_mid_channels*4, self.init_mid_channels*2, self.init_mid_channels*2*block.expansion)
        self.dec1 = UpConvBlock(self.init_mid_channels*2, self.init_mid_channels, self.init_mid_channels*block.expansion)

        self.final = nn.Conv2d(self.init_mid_channels, num_classes, kernel_size=1)
    def forward(self, x):
        # Encoder
        conv1_out = self.conv1(x)
        enc1 = self.enc1(conv1_out)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        # Decoder
        dec3 = self.dec3(enc4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)
        return self.final(dec1)

    def make_layer(self, block, out_channels, blocks, stride=1):
        down_sample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, down_sample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

# V_Whu:https://doi.org/10.1016/j.jhydrol.2022.127570
class ResNet_Whu(nn.Module):
    def __init__(self, in_channels):
        super(ResNet_Whu, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.block_a = ResBlock(in_channels)
        self.block_b = ResBlock(in_channels)
        self.block_c = ResBlock(in_channels)
        
        self.conv_concat = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        f1 = self.relu1(self.conv1(x))
        f2 = self.relu1(self.conv2(f1))
        
        f_a = self.block_a(f2)
        f_b = self.block_b(f_a)
        f_c = self.block_c(f_b)
        
        concat_out = torch.cat((f_a, f_b, f_c), dim=1)
        combined = self.conv_concat(concat_out)
        
        f3 = self.relu1(self.conv3(combined))
        f4 = f1 + f3
        
        output = self.conv3(f4)
        
        return output

# V2.0 
class ResNet_2(nn.Module):
    def __init__(self, block, layers, num_channels, num_classes,init_mid_channels=128):
        super(ResNet_2, self).__init__()
        
        self.in_channels = 64
        self.init_mid_channels = init_mid_channels
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, self.in_channels, kernel_size=5, stride=1, padding=2, bias=False), 
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        ) # 16*16*C

        # 下采样层
        self.enc1 = self.make_layer(block, self.init_mid_channels, layers[0], stride=1)
        self.enc2 = self.make_layer(block, self.init_mid_channels, layers[1], stride=1)
        self.enc3 = self.make_layer(block, self.init_mid_channels, layers[2], stride=1)
        self.enc4 = self.make_layer(block, self.init_mid_channels, layers[3], stride=1)

        self.in_channels = 64

        self.concat_conv = nn.Sequential(nn.Conv2d(self.init_mid_channels * 4, self.in_channels, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(self.in_channels),
                                        nn.ReLU(inplace=True))

        self.final = nn.Conv2d(self.in_channels, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Encoder
        conv1_out = self.conv1(x)
        enc1 = self.enc1(conv1_out)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Decoder
        dec_input = torch.cat((enc1, enc2, enc3, enc4), dim=1)
        dec = self.concat_conv(dec_input)
        output = self.final(dec)
    
        return output

    def make_layer(self, block, out_channels, blocks, stride=1):
        down_sample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, down_sample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
# V3.0 final_version
class ResNet_group(nn.Module):
    def __init__(self, block, layers, num_channels, num_classes,init_mid_channels=128):
        super(ResNet_group, self).__init__()
        
        self.in_channels = 64
        self.init_mid_channels = init_mid_channels
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, self.in_channels, kernel_size=5, stride=1, padding=2, bias=False), 
            nn.GroupNorm(1, self.in_channels),
            nn.ReLU(inplace=True),
        ) # 16*16*C

        # 下采样层
        self.enc1 = self.make_layer(block, self.init_mid_channels, layers[0], stride=1)
        self.enc2 = self.make_layer(block, self.init_mid_channels, layers[1], stride=1)
        self.enc3 = self.make_layer(block, self.init_mid_channels, layers[2], stride=1)
        self.enc4 = self.make_layer(block, self.init_mid_channels, layers[3], stride=1)

        self.in_channels = 64

        self.concat_conv = nn.Sequential(nn.Conv2d(self.init_mid_channels * 4, self.in_channels, kernel_size=1, stride=1, padding=0),
                                        nn.GroupNorm(1, self.in_channels),
                                        nn.ReLU(inplace=True))

        self.final = nn.Conv2d(self.in_channels, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Encoder
        conv1_out = self.conv1(x)
        enc1 = self.enc1(conv1_out)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Decoder
        dec_input = torch.cat((enc1, enc2, enc3, enc4), dim=1)
        dec = self.concat_conv(dec_input)
        output = self.final(dec)
    
        return output
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_channels, num_classes,init_mid_channels=128):
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        self.init_mid_channels = init_mid_channels
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, self.in_channels, kernel_size=5, stride=1, padding=2, bias=False), 
            nn.ReLU(inplace=True),
        ) # 16*16*C

        # 下采样层
        self.enc1 = self.make_layer(block, self.init_mid_channels, layers[0], stride=1)
        self.enc2 = self.make_layer(block, self.init_mid_channels, layers[1], stride=1)
        self.enc3 = self.make_layer(block, self.init_mid_channels, layers[2], stride=1)
        self.enc4 = self.make_layer(block, self.init_mid_channels, layers[3], stride=1)

        self.concat_conv = nn.Sequential(nn.Conv2d(self.init_mid_channels * 4, self.in_channels, kernel_size=1, stride=1, padding=0),
                                        nn.ReLU(inplace=True))

        self.final = nn.Conv2d(self.in_channels, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Encoder
        conv1_out = self.conv1(x)
        enc1 = self.enc1(conv1_out)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Decoder
        dec_input = torch.cat((enc1, enc2, enc3, enc4), dim=1)
        dec = self.concat_conv(dec_input)
        output = self.final(dec)
    
        return output
    def make_layer(self, block, out_channels, blocks, stride=1):
        down_sample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, down_sample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

## UNet
class UNet(nn.Module):
    def __init__(self, num_channels, num_classes,init_mid_channels=128):
        super(UNet, self).__init__()
        multiplier = MyMultiplier(init_mid_channels) 
        self.enc1 = ConvBlock(num_channels, multiplier(keep=-1)) # num_channels, 128
        self.enc2 = ConvBlock(multiplier(keep=-1), multiplier(keep=True)) # 128, 256
        self.enc3 = ConvBlock(multiplier(keep=-1), multiplier(keep=True)) # 256, 512
        self.enc4 = ConvBlock(multiplier(keep=-1), multiplier(keep=True)) # 512, 1024

        self.dec3 = UpConvBlock(multiplier(keep=-1), multiplier(keep=False), multiplier(keep=-1))  # 1024, 512, 512
        self.dec2 = UpConvBlock(multiplier(keep=-1), multiplier(keep=False), multiplier(keep=-1))  # 512, 256, 256
        self.dec1 = UpConvBlock(multiplier(keep=-1), multiplier(keep=False), multiplier(keep=-1))  # 256, 128, 128

        self.final = nn.Conv2d(multiplier(keep=-1), num_classes, kernel_size=1)
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Decoder
        dec3 = self.dec3(enc4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)

        return self.final(dec1)

# KPNet
class KPNet(nn.Module):
    def __init__(self, in_channels, out_channels=1,block = Compression_excitation_block, init_mid_channels=128, dropout_rate = 0.1, layer_scale_init_value=1e-6,net_depth=[2, 2, 2]):
        super(KPNet, self).__init__()

        ## Encoder
        mid_channels=[int(init_mid_channels*(1+i/4)) for i in range(0,3)]
        self.in_channels = mid_channels[0]
        self.conv_first = FFN(in_channels, mid_channels[0], num_layers=4, expansion_ratio=6)
        # 推断模块

        # 压缩激励和多尺度 convolution
        self.Multi_scale_Layers = nn.ModuleList()
        self.Compression_excitation_Layers = nn.ModuleList()
        self.FFN_Layers = nn.ModuleList()
        self.concat_Layers = nn.ModuleList() 

        # 每层dropout_rates一致
        dropout_rates = [x.item() for x in torch.linspace(0, dropout_rate, len(net_depth))]
        for i, out_channel in enumerate(mid_channels):
            Compression_excitation_Layer, in_channels_updated = self.make_layer(self.in_channels, out_channel, block=Compression_excitation_block, layer_config=net_depth[i],drop_rate=dropout_rates[i], layer_scale_init_value=layer_scale_init_value)
            self.FFN_Layers.append(FFN(in_channels=self.in_channels, out_channels = out_channel * Compression_excitation_block.expansion_ratio//2, num_layers=2, expansion_ratio=4))

            step_out_channels = out_channel * Compression_excitation_block.expansion_ratio//2
            concat_Layer = nn.Sequential(nn.Conv2d(out_channel * Compression_excitation_block.expansion_ratio + out_channel * Compression_excitation_block.expansion_ratio//2, step_out_channels, kernel_size=1),
                                      nn.GELU())
            self.concat_Layers.append(concat_Layer)
            self.in_channels = in_channels_updated//2
            self.Compression_excitation_Layers.append(Compression_excitation_Layer)

        # Decoder
        Decoder_mid_channels = (sum(mid_channels) * Compression_excitation_block.expansion_ratio//2 + mid_channels[0])//2
        self.final = nn.Sequential(nn.Conv2d(sum(mid_channels) * Compression_excitation_block.expansion_ratio//2 + mid_channels[0], Decoder_mid_channels, kernel_size=1, bias=False),
                                   nn.GELU(),
                                   nn.Conv2d(Decoder_mid_channels, out_channels, kernel_size=1, bias=False))

    # concat
    def forward(self, x):
        outputs = []
        x1 = self.conv_first(x)
        for i in range(3):
            if i==0:
                Compression_excitation = self.Compression_excitation_Layers[i](x1)
                FFN = self.FFN_Layers[i](x1)
            else:
                Compression_excitation = self.Compression_excitation_Layers[i](concat_out)
                FFN = self.FFN_Layers[i](concat_out)
            step_out = torch.cat([Compression_excitation, FFN], dim=1)
            concat_out = self.concat_Layers[i](step_out)
            outputs.append(concat_out)

        out = torch.cat(outputs, dim=1)
        out = torch.cat([out, x1], dim=1)

        out = self.final(out)
        return out

    def make_layer(self, in_channels, out_channels, block=Compression_excitation_block, layer_config=None, drop_rate=None, layer_scale_init_value=None):
        
        layers = []
        down_sample = None
        if in_channels != out_channels * block.expansion_ratio:
            down_sample = nn.Conv2d(in_channels, out_channels * block.expansion_ratio, kernel_size=1, bias=False)
        
        layers.append(block(in_channels, out_channels, down_sample=down_sample, drop_rate=drop_rate, layer_scale_init_value=layer_scale_init_value))
        in_channels_updated = out_channels * block.expansion_ratio

        for i in range(layer_config-1):
            layers.append(block(in_channels_updated, out_channels, drop_rate=drop_rate, layer_scale_init_value=layer_scale_init_value))

        return nn.Sequential(*layers), in_channels_updated
    
class KPNet1(nn.Module):
    def __init__(self, in_channels, out_channels=1,block = Compression_excitation_block, init_mid_channels=128, dropout_rate = 0.1, layer_scale_init_value=1e-6,net_depth=[2, 2, 2, 2]):
        super(KPNet1, self).__init__()

        ## Encoder
        mid_channels=[int(init_mid_channels*(1+i/4)) for i in range(0,4)]
        self.in_channels = mid_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels[0], kernel_size=1))

        # 压缩激励和多尺度 convolution
        self.Multi_scale_Layers = nn.ModuleList()
        self.Compression_excitation_Layers = nn.ModuleList()

        # 每层dropout_rates一致
        dropout_rates = [x.item() for x in torch.linspace(0, dropout_rate, len(net_depth))]
        
        for i, out_channel in enumerate(mid_channels):
            Compression_excitation_Layer, in_channels_updated = self.make_layer(self.in_channels, out_channel, block=Compression_excitation_block, layer_config=net_depth[i],drop_rate=dropout_rates[i], layer_scale_init_value=layer_scale_init_value,mode=i)
            self.in_channels = in_channels_updated
            self.Compression_excitation_Layers.append(Compression_excitation_Layer)

        # Decoder
        self.concat_conv1 = nn.Sequential(nn.Conv2d((mid_channels[-1]+mid_channels[-2]) * Compression_excitation_block.expansion_ratio, mid_channels[-2] * Compression_excitation_block.expansion_ratio//2, kernel_size=1),
                                        nn.LeakyReLU(inplace=True))
        self.concat_conv2 = nn.Sequential(nn.Conv2d(mid_channels[-3] * Compression_excitation_block.expansion_ratio + mid_channels[-2] * Compression_excitation_block.expansion_ratio//2, mid_channels[-3] * Compression_excitation_block.expansion_ratio//2, kernel_size=1),
                                        nn.LeakyReLU(inplace=True))
        self.concat_conv3 = nn.Sequential(nn.Conv2d(mid_channels[-4] * Compression_excitation_block.expansion_ratio + mid_channels[-3] * Compression_excitation_block.expansion_ratio//2, mid_channels[-4] * Compression_excitation_block.expansion_ratio//2, kernel_size=1),
                                        nn.LeakyReLU(inplace=True))
        self.concat_conv4 = nn.Sequential(nn.Conv2d(mid_channels[0] + mid_channels[-4] * Compression_excitation_block.expansion_ratio//2, mid_channels[0], kernel_size=1),
                                        nn.LeakyReLU(inplace=True))
        self.final = nn.Conv2d((mid_channels[-2] + mid_channels[-3] + mid_channels[-4]) * Compression_excitation_block.expansion_ratio//2 + mid_channels[0], out_channels, kernel_size=1)

    # concat
    def forward(self, x):
        outputs = []
        x = self.conv1(x)

        for i in range(4):
            if i==0:
                Compression_excitation = self.Compression_excitation_Layers[i](x)
            else:
                Compression_excitation = self.Compression_excitation_Layers[i](Compression_excitation)
            outputs.append(Compression_excitation)

        concat_x1 = torch.cat([outputs[-1], outputs[-2]], dim=1)
        concat_x1 = self.concat_conv1(concat_x1)
        
        concat_x2 = torch.cat([outputs[-3], concat_x1], dim=1)
        concat_x2 = self.concat_conv2(concat_x2)

        concat_x3 = torch.cat([outputs[-4], concat_x2], dim=1)
        concat_x3 = self.concat_conv3(concat_x3)

        concat_x4 = torch.cat([x, concat_x3], dim=1)
        concat_x4 = self.concat_conv4(concat_x4)

        out = torch.cat([concat_x4, concat_x3, concat_x2 ,concat_x1], dim=1)
        out = self.final(out)

        return out

    def make_layer(self, in_channels, out_channels, block=Compression_excitation_block, layer_config=None, drop_rate=None, layer_scale_init_value=None,mode=None):
        
        layers = []
        down_sample = None
        if in_channels != out_channels * block.expansion_ratio:
            down_sample = nn.Conv2d(in_channels, out_channels * block.expansion_ratio, kernel_size=1, bias=False)
        
        layers.append(block(in_channels, out_channels, down_sample=down_sample, drop_rate=drop_rate, layer_scale_init_value=layer_scale_init_value,mode=mode))
        in_channels_updated = out_channels * block.expansion_ratio

        for i in range(layer_config-1):
            layers.append(block(in_channels_updated, out_channels, drop_rate=drop_rate, layer_scale_init_value=layer_scale_init_value,mode=mode))

        return nn.Sequential(*layers), in_channels_updated


# ViT
class create_positional_embedding(nn.Module):
    def __init__(self, n_pos_vector=None, dim=None, width=None,height=None,n_head=None):
        super().__init__()
        self.n_pos_vector = n_pos_vector
        self.pos_embedding = nn.Parameter(torch.randn(n_pos_vector, dim))
        self.width = width
        self.height = height
        self.n_head = n_head
        self.dim = dim

    def create_1d_absolute_sincos_embeddings(self, positions, dim):
        assert dim % 2 == 0, 'dim must be divisible by 2 for sinusoidal positional embedding'
        position_embedding = torch.zeros(len(positions), dim, dtype=torch.float)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        div_term = div_term.unsqueeze(0)  # 将div_term改为2D张量以便广播
        position_embedding[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
        position_embedding[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term)
        return position_embedding
    
    def create_2d_relative_trainable_embeddings(self):
        position_embedding = nn.Embedding(self.n_pos_vector, self.dim)
        nn.init.constant_(position_embedding.weight, 0)
        def get_relative_position_index(height, width):
            coords = torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width)))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords_bias = coords_flatten[:, :, None] - coords_flatten[:, None, :]

            relative_coords_bias[0,:,:] += self.height - 1
            relative_coords_bias[1,:,:] += self.width - 1
            relative_coords_bias[0,:,:] *= relative_coords_bias[1,:,:].max()+1

            return relative_coords_bias.sum(0)
        relative_position_bias = get_relative_position_index(self.height, self.width)
        bias_embedding = position_embedding(torch.flatten(relative_position_bias)).reshape(self.height*self.width, self.height*self.width, self.n_head)
        bias_embedding = bias_embedding.permute(2, 0, 1).unsqueeze(0)

    def create_1d_absolute_trainable_embeddings(self):
        position_embedding = nn.Parameter(torch.randn(1, self.n_pos_vector+1, self.dim))
        return position_embedding
    
    def create_2d_absolute_sincos_embeddings(self):
        assert self.dim % 4 == 0, 'dim must be divisible by 4 for sinusoidal positional embedding'
        position_embedding = torch.zeros(self.height * self.width, self.dim)

        height_positions = torch.arange(self.height, dtype=torch.float32)
        width_positions = torch.arange(self.width, dtype=torch.float32)
        height_embedding = self.create_1d_absolute_sincos_embeddings(height_positions, self.dim // 2)
        width_embedding = self.create_1d_absolute_sincos_embeddings(width_positions, self.dim // 2)

        height_index, width_index = torch.meshgrid(torch.arange(self.height), torch.arange(self.width))
        position_embedding[:, :self.dim // 2] = height_embedding[height_index.flatten()]
        position_embedding[:, self.dim // 2:] = width_embedding[width_index.flatten()]

        return position_embedding

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 16, dim_head = 64, dropout = 0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size=32, patch_size=8,input_channels:int=3,mid_channels:int=4096,out_channels:int=1,depth:int=8,heads=16,mlp_dim=2048,dim_head=64, dropout=0.1, emb_dropout=0.1, positional_embedding='1d_absolute_trainable'):
        """:
        depth: block nums
        heads: head nums
        mlp_dim: mlp_hidden_dim
        """
        super(ViT, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = input_channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, mid_channels),
            nn.LayerNorm(mid_channels),
        )

        self.positional_embedding = create_positional_embedding(n_pos_vector=num_patches, dim=mid_channels, width=image_width, height=image_height, n_head=heads)
        
        if positional_embedding == '1d_absolute_trainable':
            self.pos_embedding = self.positional_embedding.create_1d_absolute_trainable_embeddings()
        else:
            raise ValueError(f'positional embedding "{positional_embedding}" not recognized')

        self.cls_token = nn.Parameter(torch.randn(1, 1, mid_channels))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(mid_channels, depth, heads, dim_head, mlp_dim, dropout)
        # decoder
        mid_channels = 4352
        decoder_channels = [mid_channels, mid_channels // 2, mid_channels // 4, mid_channels // 8]
        self.decoder_layers = nn.ModuleList()
        for i in range(len(decoder_channels)-1):
            upsample_layer = nn.Sequential(
                nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(1, decoder_channels[i + 1]),
                nn.LeakyReLU()
            )
            self.decoder_layers.append(upsample_layer)


        self.final_layer = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=3, padding=1)



    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        rec_x = x.view(b, -1, 4, 4)
        for layer in self.decoder_layers:
            rec_x = layer(rec_x)
        return self.final_layer(rec_x)


# 每次修改模型之后都需进行修改
def freeze_decoder(model, usage_model='ST', spatial_name=None, use_freeze_decoder=False, train=False):
    # 如果模型是 DataParallel 实例，使用 model.module 访问原始模型
    original_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    # 函数只在正式训练阶段中使用
    if usage_model == 'temporal' and use_freeze_decoder and train:
        pass
    if usage_model == 'ST' and use_freeze_decoder and train:
        for param in original_model.temporal_conv.parameters():
            param.requires_grad = False
    if usage_model != 'temporal':
        if use_freeze_decoder and train:
            if spatial_name in ['kpnet_Encoder_lite', 'kpnet_mul_head']:
                # 冻结下采样层 (DownConv_Layers)
                for param in original_model.spatial_conv.DownConv_Layers.parameters():
                    param.requires_grad = False
                # 冻结深度可分离卷积层 (stages)
                for param in original_model.spatial_conv.stages.parameters():
                    param.requires_grad = False
            elif spatial_name == 'kpnet':
                for param in original_model.spatial_conv.conv1.parameters():
                    param.requires_grad = False
                for param in original_model.spatial_conv.DilBlock_Layers.parameters():
                    param.requires_grad = False
                for param in original_model.spatial_conv.stages.parameters():        
                    param.requires_grad = False
            elif spatial_name in ['resnet']:
                for param in original_model.spatial_conv.conv1.parameters():
                    param.requires_grad = False
                for layer in [original_model.spatial_conv.enc1, original_model.spatial_conv.enc2, original_model.spatial_conv.enc3, original_model.spatial_conv.enc4]:
                    for param in layer.parameters():
                        param.requires_grad = False
            elif spatial_name in ['unet']:
                # 冻结encoder层
                for param in original_model.spatial_conv.enc1.parameters():
                    param.requires_grad = False
                for param in original_model.spatial_conv.enc2.parameters():
                    param.requires_grad = False
                for param in original_model.spatial_conv.enc3.parameters():
                    param.requires_grad = False
                for param in original_model.spatial_conv.enc4.parameters():
                    param.requires_grad = False
            elif spatial_name == 'ViT':
                for param in original_model.spatial_conv.to_patch_embedding.parameters():
                    param.requires_grad = False
                for param in original_model.spatial_conv.positional_embedding.parameters():
                    param.requires_grad = False
                for param in original_model.spatial_conv.transformer.parameters():
                    param.requires_grad = False
