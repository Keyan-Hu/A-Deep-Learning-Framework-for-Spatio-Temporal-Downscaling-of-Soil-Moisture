import sys
import torch
import argparse
sys.path.append(r"C:\Users\Administrator\Desktop\code\ST-Conv")
from torchstat import stat
from models.block import BasicBlock,Bottleneck
from models.model import STConvNet
from tool.utils import Util
from torchsummaryX import summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def flops_to_string(flops, units='GFLOPs', precision=2):
    """Convert FLOPs to a human readable string."""
    if units == 'GFLOPs':
        flops_count = flops / 1e9
    elif units == 'MFLOPs':
        flops_count = flops / 1e6
    else:  # Default to FLOPs
        flops_count = flops
    return f"{flops_count:.{precision}f} {units}"

def main(args):
    models_config = Util.load_config(args.models_config_path)
    train_config = models_config['train_shared_parameter'] 
    path_config = Util.load_config(args.path_config_path)
    return train_config, models_config, path_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_config_path', type=str, default='./config/models_config.yaml')
    parser.add_argument('--path_config_path', type=str, default='./config/path_config.yaml')
    args = parser.parse_known_args()[0]

    train_dict, models_config, path_config = main(args)
    shared_parameters = models_config['Shared_parameter']
    experiment_groups = models_config['experiment_groups']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = train_dict['batch_size']

    # 创建模拟输入
    hour_input_shape = (batch_size, shared_parameters['hour_in_channels'], 5, 32, 32)
    day_input_shape = (batch_size, shared_parameters['day_in_channels'], 5, 32, 32)
    half_day_input_shape = (batch_size, shared_parameters['half_day_in_channels'], 5, 32, 32)
    static_input_shape = (batch_size, shared_parameters['static_in_channels'], 32, 32)
    hour_input = torch.randn(hour_input_shape)
    day_input = torch.randn(day_input_shape)
    half_day_input = torch.randn(half_day_input_shape)
    static_input = torch.randn(static_input_shape)
    print("Hour input shape:", hour_input_shape)
    print("Day input shape:", day_input_shape)
    print("Half day input shape:", half_day_input_shape)
    print("Static input shape:", static_input_shape)

    for group in experiment_groups:
        if group['group_name'] == 'Experiment Group 0':
            print(f"Running experiments for group: {group['group_name']}")
            experiment_shared_params = group.get('experiment_shared_parameter', {})
            for model_config in group['models']:
                model = STConvNet(**shared_parameters, **experiment_shared_params, **model_config['parameters'])


    # 模型分析
    model.cpu()
    summary(model, hour_input, day_input, half_day_input, static_input)
    # FLOPs 和 参数 统计
    flop_analyzer = FlopCountAnalysis(model, (hour_input, day_input, half_day_input, static_input))

    flops = flop_analyzer.total()
    print(f"FLOPs: {flops_to_string(flops, 'MFLOPs')}")
    print(parameter_count_table(model))