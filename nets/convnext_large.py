import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.hub import load_state_dict_from_url

def convnext_large(pretrained,in_channels, classes=4):
    import torchvision
    model = torchvision.models.convnext_large(pretrained) 
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/convnext_large-ea097f82.pth", model_dir="./model_data")
        # model.load_state_dict(state_dict)加载模型权重
        model.load_state_dict(state_dict)
    return model