import torch
import torch.nn as nn
import torch.nn.functional as F

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups

        # 将特征图分成若干组
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        
        # 交换通道顺序
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        # 将特征图合并为原始形状
        x = x.view(batch_size, num_channels, height, width)
        return x

class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super(ShuffleNetUnit, self).__init__()
        mid_channels = out_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, groups=groups)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.shuffle1 = ShuffleBlock(groups)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, groups=groups)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shuffle2 = ShuffleBlock(groups)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.shuffle1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.shuffle2(x)

        # 残差连接
        # x += identity
        return x

class ShuffleNet(nn.Module):
    def __init__(self, groups=3, num_classes=1000):
        super(ShuffleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self._make_stage(24, 144, 4, groups)
        self.stage3 = self._make_stage(144, 288, 8, groups)
        self.stage4 = self._make_stage(288, 576, 4, groups)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(576, num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, groups):
        stage = []
        stage.append(ShuffleNetUnit(in_channels, out_channels, groups))
        for _ in range(1, num_blocks):
            stage.append(ShuffleNetUnit(out_channels, out_channels, groups))
        return nn.Sequential(*stage)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# # 创建ShuffleNet模型实例
# shufflenet_model = ShuffleNet(groups=3, num_classes=1000)

# # 打印模型结构
# print(shufflenet_model)
def shufflenet_test(groups, num_classes):
    model = ShuffleNet(groups=groups, num_classes=num_classes)
    return model