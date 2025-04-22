import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.hub import load_state_dict_from_url

def shufflenet_v2_x1_0(pretrained,in_channels, classes=4):
    import torchvision
    model = torchvision.models.shufflenet_v2_x1_0(pretrained) 
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth", model_dir="./model_data")
        # model.load_state_dict(state_dict)加载模型权重
        model.load_state_dict(state_dict)
    return model