import math
import torch
import torch.nn as nn
import torch.nn.functional as F

## Unet
# 卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# 上采样块
class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, merge_with):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) # lon->lon*2,in_channels->out_channels
        self.conv = ConvBlock(out_channels + merge_with, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        return x

## ResNet
# 残差块
class BasicBlock_Original(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super(BasicBlock_Original, self).__init__()
        self.first = nn.Sequential(
            # 第一层大小可变
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 第二层大小固定
            nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=1,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        out = self.first(x)
        #将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        if self.down_sample is not None:
            shortcut = self.down_sample(x)
        else:
            shortcut = x
        out += shortcut
        out = nn.ReLU(inplace=True)(out)
        return out
class Bottleneck_Original(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super(Bottleneck_Original, self).__init__()
        self.first = nn.Sequential(
            # 第一层: 1x1 卷积, 减少维度
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 第二层: 3x3 卷积, 可能有下采样
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 第三层: 1x1 卷积, 增加维度
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        shortcut = x
        out = self.first(x)
        if self.down_sample is not None:
            shortcut = self.down_sample(x)
        else:
            shortcut = x
        out += shortcut
        out = nn.ReLU(inplace=True)(out)
        return out
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=1,bias=False),
        )
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        out = self.first(x)
        if self.down_sample is not None:
            shortcut = self.down_sample(x)
        else:
            shortcut = x
        out += shortcut
        out = nn.ReLU(inplace=True)(out)
        return out
    
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super(Bottleneck, self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False),
        )
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        shortcut = x
        out = self.first(x)
        if self.down_sample is not None:
            shortcut = self.down_sample(x)
        else:
            shortcut = x
        out += shortcut
        out = nn.ReLU(inplace=True)(out)
        return out
    
class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        out_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv_concat = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        out1 = self.relu1(self.conv1(x))

        out2 = self.relu2(self.conv2(x + out1))

        out3 = self.relu3(self.conv3(x + out2))

        concat_out = torch.cat((x, out1, out2, out3), dim=1)
        out = self.conv_concat(concat_out)
        final_out = out + x
        return final_out

## KP-Net
class Compression_excitation_block(nn.Module):
    expansion_ratio = 4
    def __init__(self, in_channels, out_channels, down_sample=None, drop_rate=0., layer_scale_init_value=1e-6,mode=None):
        super(Compression_excitation_block, self).__init__()
        self.mode = mode
        self.first = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                   nn.GELU())
        self.conv11 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        if mode in [0] and mode is not None:
            self.conv13 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=out_channels, dilation=1)
            self.conv31 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=out_channels, dilation=1)
        elif mode in [1] and mode is not None:
            self.conv33 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, dilation=1)
        else:
            self.conv331 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, dilation=1)
            self.conv332 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, groups=out_channels, dilation=2)

        self.conv4 = nn.Sequential(nn.Conv2d(out_channels, self.expansion_ratio * out_channels, kernel_size=1),
                                nn.GELU())

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(self.expansion_ratio * out_channels), requires_grad=True) if layer_scale_init_value > 0 else None
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

        self.down_sample = down_sample

    def forward(self, x):
        out_first = self.first(x)

        mid_11 = self.conv11(out_first)
        if self.mode is not None and self.mode in [0]:
            mid_31 = self.conv31(out_first)
            mid_13 = self.conv13(out_first)
            out_mid = out_first + mid_11 + mid_31 + mid_13
        elif self.mode is not None and self.mode in [1]:
            mid_33 = self.conv33(out_first)
            out_mid = out_first + mid_11 + mid_33
        else:
            mid_331 = self.conv331(out_first)
            mid_332 = self.conv332(out_first)
            out_mid = out_first + mid_11 + mid_331 + mid_332

        out = self.conv4(out_mid)

        if self.gamma is not None:
            gamma = self.gamma.view(1, -1, 1, 1)
            out = gamma * out
        out = self.dropout(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)
        else:
            shortcut = x
        out += shortcut
        return out

# FFN
class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, expansion_ratio=4):
        super(FFN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_ratio = expansion_ratio
        self.hidden_channels = out_channels * expansion_ratio
        self.Linear_Layers = nn.ModuleList()
        current_dim = in_channels
        for i in range(num_layers):
            if i < num_layers - 1:
                if i % 2 == 0:
                    next_dim = out_channels * expansion_ratio
                else:
                    next_dim = out_channels
            else:
                next_dim = out_channels
            layer = nn.Sequential(nn.Linear(current_dim, next_dim), nn.GELU())
            self.Linear_Layers.append(layer)
            current_dim = next_dim

    def forward(self, x):
        batch_size, channels, lons, lats = x.size()
        x = x.permute(2, 3, 0, 1).contiguous()
        x = x.view(-1, channels)

        res_hidden = self.Linear_Layers[0](x)
        res_out = self.Linear_Layers[1](res_hidden)
        if len(self.Linear_Layers)>2:
            for i in range(2, len(self.Linear_Layers)):
                if i % 2 == 0:
                    liner_out = self.Linear_Layers[i](res_out)
                    res_hidden = res_hidden + liner_out
                elif i % 2 == 1:
                    liner_out = self.Linear_Layers[i](res_hidden)
                    res_out = res_out + liner_out

            x = liner_out.view(lons, lats, batch_size, self.out_channels)
            x = x.permute(2, 3, 0, 1)
        else:
            x = res_out.view(lons, lats, batch_size, self.out_channels)
            x = x.permute(2, 3, 0, 1)
        return x
    
class Compression_excitation_block1(nn.Module):
    expansion_ratio = 4
    def __init__(self, in_channels, out_channels, down_sample=None, drop_rate=0., layer_scale_init_value=1e-6, mode=None):
        super(Compression_excitation_block1, self).__init__()

        self.mode = mode
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        if mode is not None and mode in [0,2]:
            self.mid_conv = nn.Sequential(nn.Conv2d(out_channels, self.expansion_ratio * out_channels, kernel_size=1),
                                          nn.Conv2d(self.expansion_ratio * out_channels, out_channels, kernel_size=1),
                                          nn.LeakyReLU(inplace=True))
        else:
            self.mid_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                            nn.LeakyReLU(inplace=True))

        self.conv4 = nn.Sequential(nn.Conv2d(out_channels, self.expansion_ratio * out_channels, kernel_size=1),
                                nn.LeakyReLU(inplace=True))

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(self.expansion_ratio * out_channels), requires_grad=True) if layer_scale_init_value > 0 else None
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

        self.down_sample = down_sample

    def forward(self, x):
        out_first = self.first(x)
        out_mid = out_first + self.mid_conv(out_first)
        out = self.conv4(out_mid)

        if self.gamma is not None:
            gamma = self.gamma.view(1, -1, 1, 1)
            out = gamma * out
        out = self.dropout(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)
        else:
            shortcut = x
        out += shortcut
        return out

class Multi_scale_block(nn.Module):
    expansion_ratio = 4
    def __init__(self, in_channels, out_channels, down_sample=None, drop_rate=0., layer_scale_init_value=1e-6):
        super(Multi_scale_block, self).__init__()

        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(self.expansion_ratio * out_channels), requires_grad=True) if layer_scale_init_value > 0 else None

        self.first_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)

        self.conv1x1_final = nn.Conv2d(out_channels, out_channels*self.expansion_ratio, kernel_size=1)
        self.down_sample = down_sample

    def forward(self, x):

        first_conv = self.first_conv(x)
        dw1 = self.LeakyReLU(self.conv1x1(first_conv))
        dw33 = self.LeakyReLU(self.conv3x3(first_conv))
        dw55 = self.LeakyReLU(self.conv5x5(first_conv))
        dw = first_conv + dw1 + dw33+ dw55

        out = self.conv1x1_final(dw)

        if self.gamma is not None:
            gamma = self.gamma.view(1, -1, 1, 1)
            out = gamma * out
        out = self.dropout(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)
        else:
            shortcut = x

        out += shortcut

        return out