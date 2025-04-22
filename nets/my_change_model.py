import torch
import torch.nn as nn
#这边是我自己因为自己的电脑进行的一定的改动
#from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        #super(x,self).init()含义（单继承，即只有一个父类）在下面的代码中加入super(x,self).init()时调用son的父类farther的属性和方法（方法里对farther数据进行二次操作）
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(

        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #kaiming-normal是正态分布（这边是初始化正态分布）
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    #constant常值填充（tensor（一个N维），val(填充的值））
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                #正态填充(N维，均值，标准差)
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(batch_norm=False, in_channels = 3):
    layers = []
    layers +=[nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)]
    layers +=[nn.BatchNorm2d(64)]
    layers +=[nn.ReLU()]
    layers +=[nn.MaxPool2d(2,2)]
    layers +=[nn.Dropout()]
    layers +=[nn.Conv2d(64, 32, kernel_size=3, padding=1)]
    layers +=[nn.BatchNorm2d(32)]
    layers +=[nn.ReLU()]
    layers +=[nn.MaxPool2d(2, 2)]
    layers +=[nn.Dropout()]
    layers +=[nn.Conv2d(32, 16, kernel_size=3, padding=1)]
    layers +=[nn.BatchNorm2d(16)]
    layers +=[nn.ReLU()]
    layers +=[nn.MaxPool2d(2, 2)]
    layers +=[nn.Dropout()]
    layers +=[nn.Flatten()]
    layers +=[nn.Linear(20000, 128)]
    layers +=[nn.Dropout()]
    layers +=[nn.Linear(128, 64)]
    layers +=[nn.Dropout()]
    layers +=[nn.Linear(64, 4)]
    return nn.Sequential(*layers)

    # layers +=[nn.Conv2d(256, 128, kernel_size=3, padding=1)]
    # layers +=[nn.BatchNorm2d(128)]
    # layers +=[nn.ReLU()]
    # layers +=[nn.MaxPool2d(2, 2)]
    # layers +=[nn.Dropout()]
    # layers +=[nn.Conv2d(128, 64, kernel_size=3, padding=1)]
    # layers +=[nn.BatchNorm2d(64)]
    # layers +=[nn.ReLU()]
    # layers +=[nn.MaxPool2d(2, 2)]
    # layers +=[nn.Dropout()]
    # layers +=[nn.Flatten()]
    # layers +=[nn.Linear(2304,256)]
    # layers +=[nn.Dropout()]
    # layers +=[nn.Linear(256, 128)]
    # layers +=[nn.Dropout()]
    # layers +=[nn.Linear(128, 64)]
    # layers +=[nn.Dropout()]
    # layers +=[nn.Linear(64, 4)]
    # return nn.Sequential(*layers)

def VGG16(pretrained, in_channels, **kwargs):
    model = VGG(make_layers(batch_norm = False, in_channels = in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        # model.load_state_dict(state_dict)
    return model
