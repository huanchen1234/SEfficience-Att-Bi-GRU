import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.hub import load_state_dict_from_url


# # 残差单元
# class Residual(nn.Module):
#     def __init__(self, input_channel, out_channel, use_conv1x1=False, strides=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(input_channel, int(out_channel / 4), kernel_size=1, stride=strides)
#         self.conv2_x = nn.Conv2d(int(out_channel / 4), int(out_channel / 4), kernel_size=3, stride=1, padding=1)
#         self.conv3_x = nn.Conv2d(int(out_channel / 4), out_channel, kernel_size=1)
#         if use_conv1x1:
#             self.conv4_x = nn.Conv2d(input_channel, out_channel, kernel_size=1, stride=strides)
#         else:
#             self.conv4_x = None
#         self.bn1 = nn.BatchNorm2d(int(out_channel / 4))
#         self.bn2 = nn.BatchNorm2d(out_channel)
#         self.relu = nn.ReLU(inplace=True)
 
#     def forward(self, X):
#         Y = self.relu(self.bn1(self.conv1(X)))
#         Y = self.relu(self.bn1(self.conv2(Y)))
#         Y = self.bn2(self.conv3(Y))
#         if self.conv4_x:
#             X = self.conv4_x(X)
#         Y += X
#         return F.relu(Y)


# # 多个残差单元组成的残差块
# def res_block(input_channel, out_channel, num_residuals, first_block=False):
#     blk = []
#     for i in range(num_residuals):
#         if i == 0 and not first_block:
#             blk.append(Residual(input_channel, out_channel, use_conv1x1=True, strides=2))
#             input_channel = out_channel
#         if i == 0 and first_block:
#             blk.append(Residual(input_channel, out_channel, use_conv1x1=True, strides=1))
#             input_channel = out_channel
#         else:
#             blk.append(Residual(input_channel, out_channel))
#     return blk

    
# def resnet152(pretrained,in_channels, classes=4):
#     b1 = nn.Sequential(
#         nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),  # [3,224,224]-->[64,112,112]
#         nn.BatchNorm2d(64),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # [64,112,112]-->[64,56,56]
#     )
#     b2 = nn.Sequential(*res_block(64, 256, 3, first_block=True))
#     b3 = nn.Sequential(*res_block(256, 512, 8))
#     b4 = nn.Sequential(*res_block(512, 1024, 36))
#     b5 = nn.Sequential(*res_block(1024, 2048, 3))
#     model = nn.Sequential(b1, b2, b3, b4, b5,
#                           nn.AdaptiveAvgPool2d((1, 1)),
#                           nn.Flatten(),
#                           nn.Linear(2048, classes)
#                           )
#     if pretrained:
#         state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet152-394f9c45.pth", model_dir="./model_data")
#         # model.load_state_dict(state_dict)加载模型权重
#         model.load_state_dict(state_dict)
#     return model


# class DownSample(nn.Module):
#     def __init__(self, in_channel, out_channel, stride):
#         super(DownSample, self).__init__()
#         self.down = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True)
#         )
        
#     def forward(self, x):
#         out = self.down(x)
#         return out
def resnet152(pretrained,in_channels, classes=4):
    import torchvision
    model = torchvision.models.resnet152(pretrained) 
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet152-394f9c45.pth", model_dir="./model_data")
        # model.load_state_dict(state_dict)加载模型权重
        model.load_state_dict(state_dict)
    return model



#————————————————————————
#注意力机制
#——————————————————————————————————
# import torch
# import torch.nn as nn
 
 
 
#__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
#           'wide_resnet50_2', 'wide_resnet101_2']
#
#
#model_urls = {
#    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
#    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
#    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
#    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
#}
 
 
'''****************CF的注意力机制结构---SimAM--无参数结构*******************'''
class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()
 
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
 
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s
 
    @staticmethod
    def get_module_name():
        return "simam"
 
    def forward(self, x):
 
        b, c, h, w = x.size()
        
        n = w * h - 1
 
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
 
        return x * self.activaton(y)
 
 
'''****************CF的注意力机制结构---SE******************* '''
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
 
 
'''****************CF的注意力机制结构---CA******************* '''
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
 
    def forward(self, x):
        return self.relu(x + 3) / 6
 
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
 
    def forward(self, x):
        return x * self.sigmoid(x)
 
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
 
        mip = max(8, inp // reduction)
 
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
 
    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
 
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
 
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
 
        out = identity * a_w * a_h
 
        return out
'''*******************************************************************************'''
 
 #将注意力机制放入了残差块中
 
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
 
 
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
 
 
class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, attention_module=None):
        super(BasicBlock, self).__init__()
        self.attention_module = attention_module
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = SELayer(planes, 16)
        self.ca = CoordAtt(planes,planes)
        self.simam = simam_module(planes)
        
        if attention_module == "simam":
            self.conv2 = nn.Sequential(
                    self.conv2,
                    self.simam
                    )
        elif attention_module == "se":
            self.bn2 = nn.Sequential(
                    self.bn2, 
                    self.se
                    )
        elif attention_module == "ca":
            self.bn2 = nn.Sequential(
                    self.bn2, 
                    self.ca
                    )        
        
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
#        if self.attention_module == "simam":
#            out = self.simam(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
 
        return out
 
 
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
 
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, attention_module=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.se = SELayer(planes, 16)
        self.ca = CoordAtt(planes,planes)
        self.simam = simam_module(planes)
        
        if attention_module == "simam":
            self.conv2 = nn.Sequential(
                    self.conv2,
                    self.simam
                    )
        elif attention_module == "se":
            self.bn3 = nn.Sequential(
                    self.bn3, 
                    self.se
                    )
        elif attention_module == "ca":
            self.bn3 = nn.Sequential(
                    self.bn3, 
                    self.ca
                    )  
 
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
 
        return out

# class ResNet152(nn.Module):
#     def __init__(self, classes_num):            # 指定了分类数目
#         super(ResNet152, self).__init__()
#         self.pre = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )
#         # -----------------------------------------------------------------------
#         self.layer1_first = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(256)
#         )
#         self.layer1_next = nn.Sequential(
#             nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(256)
#         )
#         # -----------------------------------------------------------------------
#         self.layer2_first = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(512)
#         )
#         self.layer2_next = nn.Sequential(
#             nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(512)
#         )
#         # -----------------------------------------------------------------------
#         self.layer3_first = nn.Sequential(
#             nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(1024)
#         )
#         self.layer3_next = nn.Sequential(
#             nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(1024)
#         )
#         # -----------------------------------------------------------------------
#         self.layer4_first = nn.Sequential(
#             nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(2048)
#         )
#         self.layer4_next = nn.Sequential(
#             nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(2048)
#         )
#         # -----------------------------------------------------------------------
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(2048 * 1 * 1, 1000),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(1000, classes_num)
#         )

#     def forward(self, x):
#         out = self.pre(x)
#         # -----------------------------------------------------------------------
#         layer1_shortcut = DownSample(64, 256, 1)
#         # layer1_shortcut.to('cuda:0')
#         layer1_identity = layer1_shortcut(out)
#         out = self.layer1_first(out)
#         out = F.relu(out + layer1_identity, inplace=True)

#         for i in range(2):
#             identity = out
#             out = self.layer1_next(out)
#             out = F.relu(out + identity, inplace=True)
#         # -----------------------------------------------------------------------
#         layer2_shortcut = DownSample(256, 512, 2)
#         # layer2_shortcut.to('cuda:0')
#         layer2_identity = layer2_shortcut(out)
#         out = self.layer2_first(out)
#         out = F.relu(out + layer2_identity, inplace=True)

#         for i in range(7):
#             identity = out
#             out = self.layer2_next(out)
#             out = F.relu(out + identity, inplace=True)
#         # -----------------------------------------------------------------------
#         layer3_shortcut = DownSample(512, 1024, 2)
#         # layer3_shortcut.to('cuda:0')
#         layer3_identity = layer3_shortcut(out)
#         out = self.layer3_first(out)
#         out = F.relu(out + layer3_identity, inplace=True)

#         for i in range(35):
#             identity = out
#             out = self.layer3_next(out)
#             out = F.relu(out + identity, inplace=True)
#         # -----------------------------------------------------------------------
#         layer4_shortcut = DownSample(1024, 2048, 2)
#         # layer4_shortcut.to('cuda:0')
#         layer4_identity = layer4_shortcut(out)
#         out = self.layer4_first(out)
#         out = F.relu(out + layer4_identity, inplace=True)

#         for i in range(2):
#             identity = out
#             out = self.layer4_next(out)
#             out = F.relu(out + identity, inplace=True)
#         # -----------------------------------------------------------------------
#         out = self.avg_pool(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc(out)

#         return out
# def resnet(pretrained, in_channels, **kwargs):
#     model = ResNet152(ResNet152(3))
#     if pretrained:
#         state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet152-394f9c45.pth", model_dir="./model_data")
#         # model.load_state_dict(state_dict)加载模型权重
#         model.load_state_dict(state_dict)
#     return model