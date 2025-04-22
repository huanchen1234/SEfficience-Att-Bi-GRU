import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.hub import load_state_dict_from_url

def regnet_x_3_2gf(pretrained,in_channels, classes=4):
    import torchvision
    model = torchvision.models.regnet_x_3_2gf(pretrained) 
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/regnet_x_3_2gf-7071aa85.pth", model_dir="./model_data")
        # model.load_state_dict(state_dict)加载模型权重
        model.load_state_dict(state_dict)
    return model