import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.hub import load_state_dict_from_url

def vgg19(pretrained,in_channels, classes=4):
    import torchvision
    model = torchvision.models.vgg19(pretrained) 
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg19-dcbb9e9d.pth", model_dir="./model_data")
        # model.load_state_dict(state_dict)加载模型权重
        model.load_state_dict(state_dict)
    return model