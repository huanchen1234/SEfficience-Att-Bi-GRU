import torch
import torch.nn as nn
import numpy as np
from nets.shuffleNetv2 import shufflenet_v2_x1_0
from nets.CNNLSTMModel import CNNLSTMModel
from nets.lstmGRU import bidirectional_rnn
from nets.shufflenet import shufflenet_test
from nets.wideshufflenet import wide_shufflenet_test
from nets.EfficienceNet import efficientnet_b0
from nets.EfficienceNet import efficientnet_b1
from nets.EfficienceNet import efficientnet_b2
from nets.EfficienceNet import efficientnet_b3
from nets.EfficienceNet import efficientnet_b4
from nets.EfficienceNet import efficientnet_b5
from nets.EfficienceNet import efficientnet_b6
from nets.EfficienceNet import efficientnet_b7
from nets.EfficienceNetV2 import efficientnetv2_l
from nets.EfficienceNetV2 import efficientnetv2_m
from nets.EfficienceNetV2 import efficientnetv2_s
from nets.visiontransformer import vit_base_patch16_224
from nets.visiontransformer import vit_base_patch32_224
from nets.visiontransformer import vit_large_patch16_224

def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width) * get_output_length(height) 

# ------------------------------------------
# 普通版本
# ------------------------------------------
class Siamese(nn.Module):
    def __init__(self, input_shape, pretrained=True):
        super(Siamese, self).__init__()
        # self.vgg = inception_resent_V2(1000, pretrained='imagenet')
        # self.vgg = shufflenet_ v2_x1_0(pretrained,4)
        # self.vgg = efficientnet_b7(4)
        # self.vgg = efficientnet_b1(5)
        self.vgg = efficientnet_b0(5)
        # self.vgg = efficientnet_b1(5)
        # self.vgg = efficientnetv2_s(4)
        # self.vgg = vit_base_patch16_224(5)
        # self.vgg = vit_base_patch32_224(4)
        # self.vgg = vit_large_patch16_224(4)
        # del self.vgg.avgpool
        # del self.vgg.classifier
        
        flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        self.fully_connect1 = torch.nn.Linear(1000, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)
        self.lstm = torch.nn.LSTM(1000,1,batch_first=True,num_layers=3, bidirectional=True)

    def forward(self, x):
        x1, x2 = x
        #------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        #------------------------------------------#
        x1 = self.vgg(x1)
        x2 = self.vgg(x2)   
        #-------------------------#
        #   相减取绝对值，取l1距离
        #-------------------------#     
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)
        #-------------------------#
        #   进行两次全连接
        #-------------------------#
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        x = CNNLSTMModel(x)
        return x

# # ------------------------------------------
# # 普通版本+att
# # ------------------------------------------
# class Siamese(nn.Module):
#     def __init__(self, input_shape, pretrained=False):
#         super(Siamese, self).__init__()
#         # self.vgg = inception_resent_V2(1000, pretrained='imagenet')
#         # self.vgg = shufflenet_test(4,1000)
#         # self.vgg = wide_shufflenet_test(4,1000)
#         # self.vgg = shufflenet_v2_x1_0(pretrained,4)
#         self.vgg = efficientnet_b1(5)
#         # self.vgg = efficientnet_b1(4)
#         # self.vgg = efficientnet_b0(5)
#         # self.vgg = vit_base_patch16_224(5)
#         # self.vgg = efficientnet_b0(1000)
#         # del self.vgg.avgpool
#         # del self.vgg.classifier
        
#         flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
#         self.fully_connect1 = torch.nn.Linear(1000, 512)
#         self.lstm = torch.nn.LSTM(1000,512,batch_first=True,num_layers=3, bidirectional=True)
#         self.attn = nn.Linear(512, 512)
#         self.act3 = nn.Sigmoid()
#         self.fully_connect2 = torch.nn.Linear(512, 1)

#     def forward(self, x):
#         x1, x2 = x
#         #------------------------------------------#
#         #   我们将两个输入传入到主干特征提取网络
#         #------------------------------------------#
#         x1 = self.vgg(x1)
#         x2 = self.vgg(x2)   
#         #-------------------------#
#         #   相减取绝对值，取l1距离
#         #-------------------------#     
#         x1 = torch.flatten(x1, 1)
#         x2 = torch.flatten(x2, 1)
#         x = torch.abs(x1 - x2)
#         #-------------------------#
#         #   进行两次全连接
#         #-------------------------#
#         x = self.fully_connect1(x)
#         # x = self.lstm(x)
#         attn = self.attn(x)  # bs, 2*lstm_units
#         attn = self.act3(attn)
#         x = x * attn
#         x = self.fully_connect2(x)
#         # x = CNNLSTMModel(x)
#         # x = bidirectional_rnn(x)
#         return x

# # --------------------------------------------------
# # 加入conv1与lstm
# # --------------------------------------------------
# class Siamese(nn.Module):
#     def __init__(self, input_shape, pretrained=False,window=5, dim=4, lstm_units=16, num_layers=2):
#         super(Siamese, self).__init__()
#         # self.vgg = resnet152(pretrained,4)
#         self.vgg = efficientnet_b2(4)
#         flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
#         self.conv1d = nn.Conv1d(1000, 1, 1)
#         self.act1 = nn.Sigmoid()
#         self.lstm = nn.LSTM(1, lstm_units, batch_first=True, num_layers=1, bidirectional=True)
#         self.cls = nn.Linear(lstm_units * 2, 1)


#     def forward(self, x):
#         x1, x2 = x
#         #------------------------------------------#
#         #   我们将两个输入传入到主干特征提取网络
#         #------------------------------------------#
#         x1 = self.vgg(x1)
#         x2 = self.vgg(x2)   
#         #-------------------------#
#         #   相减取绝对值，取l1距离
#         #-------------------------#     
#         x1 = torch.flatten(x1, 1)
#         x2 = torch.flatten(x2, 1)
#         x = torch.abs(x1 - x2)
#         #-------------------------#
#         #   进行cnn lstm的组合
#         #-------------------------#
#         x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
#         x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
#         x = self.act1(x)
#         x = x.transpose(-1, -2)  # bs, 1, lstm_units
#         x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
#         x = x.squeeze(dim=1)  # bs, 2*lstm_units
#         x = self.cls(x)
#         return x
    

# # --------------------------------------------------
# # 加入conv1与lstm 加入注意力机制ECA
# # --------------------------------------------------
# class Siamese(nn.Module):
#     def __init__(self, input_shape, pretrained=False,window=5, dim=4, lstm_units=32, num_layers=2):
#         super(Siamese, self).__init__()
#         self.vgg = convnext_large(pretrained,4)
#         # self.conv1d = nn.Conv1d(1000,1,1)
#         # self.act1 = nn.Sigmoid()
#         # self.maxPool = nn.MaxPool1d(kernel_size=window)
#         # self.drop = nn.Dropout(p=0.01)
#         self.lstm = nn.LSTM(1000, lstm_units, batch_first=True, num_layers=1, bidirectional=True)
#         # self.act2 = nn.Tanh()
#         self.attn = nn.Linear(lstm_units * 2, lstm_units * 2)
#         self.act3 = nn.Sigmoid()
#         self.cls = nn.Linear(lstm_units * 2, 1)
#         # self.act4 = nn.Tanh()


#     def forward(self, x):
#         x1, x2 = x
#         #------------------------------------------#
#         #   我们将两个输入传入到主干特征提取网络
#         #------------------------------------------#
#         x1 = self.vgg(x1)
#         x2 = self.vgg(x2)   
#         #-------------------------#
#         #   相减取绝对值，取l1距离
#         #-------------------------#     
#         x1 = torch.flatten(x1, 1)
#         x2 = torch.flatten(x2, 1)
#         x = torch.abs(x1 - x2)
#         #-------------------------#
#         #   进行注意力机制 cnn lstm的组合
#         #-------------------------#
#         # x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
#         # x = self.conv1d(x)# in： bs, dim, window out: bs, lstm_units, window
#         # # x = self.act1(x)
#         # # x = self.maxPool(x)  # bs, lstm_units, 1
#         # # x = self.drop(x)
#         # x = x.transpose(-1, -2)  # bs, 1, lstm_units
#         x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
#         # x = self.act2(x)
#         x = x.squeeze(dim=1)  # bs, 2*lstm_units
#         attn = self.attn(x)  # bs, 2*lstm_units
#         attn = self.act3(attn)
#         x = x * attn
#         x = self.cls(x)
#         # x = self.act4(x)
#         return x
    
# # --------------------------------------------------
# # 加入conv1与lstm 加入注意力机制ECA（正反不同）
# # --------------------------------------------------
# class Siamese(nn.Module):
#     def __init__(self, input_shape, pretrained=False,window=5, dim=4, lstm_units=32, num_layers=2):
#         super(Siamese, self).__init__()
#         self.vgg = shufflenet_v2_x1_0(pretrained,4)
#         # self.conv1d = nn.Conv1d(1000,1,1)
#         # self.act1 = nn.Sigmoid()
#         # self.maxPool = nn.MaxPool1d(kernel_size=window)
#         # self.drop = nn.Dropout(p=0.01)
#         self.lstm = nn.LSTM(1000, lstm_units, batch_first=True, num_layers=1, bidirectional=True)
#         # self.act2 = nn.Tanh()
#         # self.attn = nn.Linear(lstm_units * 2, lstm_units * 2)
#         # self.attn = nn.Linear(lstm_units * 2, lstm_units * 2)
#         self.attn = nn.Linear(40 , 40)
#         self.act3 = nn.Sigmoid()
#         self.cls = nn.Linear(40, 1)
#         # self.cls = nn.Linear(lstm_units * 2, 1)
#         # self.act4 = nn.Tanh()


#     def forward(self, x):
#         x1, x2 = x

#         #------------------------------------------#
#         #   我们将两个输入传入到主干特征提取网络
#         #------------------------------------------#
#         x1 = self.vgg(x1)
#         x2 = self.vgg(x2)   
#         #-------------------------#
#         #   相减取绝对值，取l1距离
#         #-------------------------#     
#         x1 = torch.flatten(x1, 1)
#         x2 = torch.flatten(x2, 1)
#         x = torch.abs(x1 - x2)
#         #-------------------------#
#         #   进行注意力机制 cnn lstm的组合
#         #-------------------------#
#         # x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
#         # x = self.conv1d(x)# in： bs, dim, window out: bs, lstm_units, window
#         # # x = self.act1(x)
#         # # x = self.maxPool(x)  # bs, lstm_units, 1
#         # # x = self.drop(x)
#         # x = x.transpose(-1, -2)  # bs, 1, lstm_units
#         # x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
#         x = bidirectional_rnn(x,forward_rnn_type='GRU', backward_rnn_type='LSTM', hidden_size=20)
#         # x, _ = self.forward_rnn(x)
#         # x = self.act2(x)
#         x = x.squeeze(dim=1)  # bs, 2*lstm_units
#         # attn = self.attn(x)  # bs, 2*lstm_units
#         attn = self.attn(x)  # bs, 2*lstm_units
#         attn = self.act3(attn)
#         # x = x * attn
#         x = self.act3(x)
#         x = self.cls(x)
#         # x = self.act4(x)
#         return x
    
# # --------------------------------------------------
# # 加入conv1与lstm 注意力机制SE
# # --------------------------------------------------
# class Siamese(nn.Module):
#     def __init__(self, input_shape, pretrained=False,window=5, dim=4, lstm_units=16, num_layers=2):
#         super(Siamese, self).__init__()
#         self.vgg = resnet152(pretrained,4)
#         self.conv1d = nn.Conv1d(1000, 1, 1)
#         self.act1 = nn.Sigmoid()
#         self.maxPool = nn.MaxPool1d(kernel_size=window)
#         self.drop = nn.Dropout(p=0.01)
#         self.lstm = nn.LSTM(1, lstm_units, batch_first=True, num_layers=1, bidirectional=True)
#         self.act2 = nn.Tanh()
#         self.cls = nn.Linear(lstm_units * 2, 1)
#         self.act4 = nn.Tanh()

#         self.se_fc = nn.Linear(1, 1)


#     def forward(self, x):
#         x1, x2 = x
#         #------------------------------------------#
#         #   我们将两个输入传入到主干特征提取网络
#         #------------------------------------------#
#         x1 = self.vgg(x1)
#         x2 = self.vgg(x2)   
#         #-------------------------#
#         #   相减取绝对值，取l1距离
#         #-------------------------#     
#         x1 = torch.flatten(x1, 1)
#         x2 = torch.flatten(x2, 1)
#         x = torch.abs(x1 - x2)
#         #-------------------------#
#         #   进行cnn lstm的组合
#         #-------------------------#
#         x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
#         x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window 
#         # x = self.act1(x)

#         # se
#         avg = x.mean(dim=1)  # bs, window
#         se_attn = self.se_fc(avg).softmax(dim=-1)  # bs, window
#         x = torch.einsum("bnd,bd->bnd", x, se_attn)

#         # x = self.maxPool(x)  # bs, lstm_units, 1
#         # x = self.drop(x)
#         x = x.transpose(-1, -2)  # bs, 1, lstm_units
#         x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
#         # x = self.act2(x)
#         x = x.squeeze(dim=1)  # bs, 2*lstm_units
#         x = self.cls(x)
#         # x = self.act4(x)
#         x = self.se_fc(x)
#         return x


# # --------------------------------------------------
# # 加入conv1与lstm 注意力机制CBAM
# # --------------------------------------------------
# class Siamese(nn.Module):
#     def __init__(self, input_shape, pretrained=False,window=5, dim=4, lstm_units=16, num_layers=2):
#         super(Siamese, self).__init__()
#         self.vgg = resnet152(pretrained,4)
#         self.conv1d = nn.Conv1d(dim, lstm_units, 1)
#         self.act1 = nn.Sigmoid()
#         self.maxPool = nn.MaxPool1d(kernel_size=window)
#         self.drop = nn.Dropout(p=0.01)
#         self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=1, bidirectional=True)
#         self.act2 = nn.Tanh()
#         self.cls = nn.Linear(lstm_units * 2, 1)
#         self.act4 = nn.Tanh()

#         self.se_fc = nn.Linear(window, window)
#         self.hw_fc = nn.Linear(lstm_units, lstm_units)


#     def forward(self, x):
#         x1, x2 = x
#         #------------------------------------------#
#         #   我们将两个输入传入到主干特征提取网络
#         #------------------------------------------#
#         x1 = self.vgg(x1)
#         x2 = self.vgg(x2)   
#         #-------------------------#
#         #   相减取绝对值，取l1距离
#         #-------------------------#     
#         x1 = torch.flatten(x1, 1)
#         x2 = torch.flatten(x2, 1)
#         x = torch.abs(x1 - x2)
#         #-------------------------#
#         #   进行cnn lstm的组合
#         #-------------------------#
#         x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
#         x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
#         x = self.act1(x)

#         # chanal
#         avg = x.mean(dim=1)  # bs, window
#         se_attn = self.se_fc(avg).softmax(dim=-1)  # bs, window
#         x = torch.einsum("bnd,bd->bnd", x, se_attn)

#         # wh
#         avg = x.mean(dim=2)  # bs, lstm_units
#         hw_attn = self.hw_fc(avg).softmax(dim=-1)  # bs, lstm_units
#         x = torch.einsum("bnd,bn->bnd", x, hw_attn)

#         x = self.maxPool(x)  # bs, lstm_units, 1
#         x = self.drop(x)
#         x = x.transpose(-1, -2)  # bs, 1, lstm_units
#         x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
#         x = self.act2(x)
#         x = x.squeeze(dim=1)  # bs, 2*lstm_units
#         x = self.cls(x)
#         x = self.act4(x)
#         return x



# # --------------------------------------------------
# # 加入conv1与lstm 注意力机制HW
# # --------------------------------------------------
# class Siamese(nn.Module):
#     def __init__(self, input_shape, pretrained=False,window=5, dim=4, lstm_units=16, num_layers=2):
#         super(Siamese, self).__init__()
#         self.conv1d = nn.Conv1d(dim, lstm_units, 1)
#         self.act1 = nn.Sigmoid()
#         self.maxPool = nn.MaxPool1d(kernel_size=window)
#         self.drop = nn.Dropout(p=0.01)
#         self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=1, bidirectional=True)
#         self.act2 = nn.Tanh()
#         self.cls = nn.Linear(lstm_units * 2, 1)
#         self.act4 = nn.Tanh()

#         self.hw_fc = nn.Linear(lstm_units, lstm_units)


#     def forward(self, x):
#         x1, x2 = x
#         #------------------------------------------#
#         #   我们将两个输入传入到主干特征提取网络
#         #------------------------------------------#
#         x1 = self.vgg(x1)
#         x2 = self.vgg(x2)   
#         #-------------------------#
#         #   相减取绝对值，取l1距离
#         #-------------------------#     
#         x1 = torch.flatten(x1, 1)
#         x2 = torch.flatten(x2, 1)
#         x = torch.abs(x1 - x2)
#         #-------------------------#
#         #   进行cnn lstm的组合
#         #-------------------------#
#         x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
#         x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
#         x = self.act1(x)

#         # wh
#         avg = x.mean(dim=2)  # bs, lstm_units
#         hw_attn = self.hw_fc(avg).softmax(dim=-1)  # bs, lstm_units
#         x = torch.einsum("bnd,bn->bnd", x, hw_attn)

#         x = self.maxPool(x)  # bs, lstm_units, 1
#         x = self.drop(x)
#         x = x.transpose(-1, -2)  # bs, 1, lstm_units
#         x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
#         x = self.act2(x)
#         x = x.squeeze(dim=1)  # bs, 2*lstm_units
#         x = self.cls(x)
#         x = self.act4(x)
#         return x