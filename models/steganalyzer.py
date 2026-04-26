import torch
import torch.nn as nn
import torch.nn.functional as F


class Steganalyzer(nn.Module):
    #隐写分析器，输入通道数为3，使用谱归一化有助于训练稳定性
    def __init__(self, n_channels=3, use_spectral_norm=True):
        super(Steganalyzer, self).__init__()
        self.n_channels = n_channels
        
        conv = nn.Conv2d
        # 如果使用谱归一化，定义一个lambda函数，用于在卷积层后应用谱归一化
        if use_spectral_norm:
            conv = lambda c_in, c_out, **kwargs: nn.utils.spectral_norm(nn.Conv2d(c_in, c_out, **kwargs))
        
        #卷积层架构
        # 每个卷积层后跟一个批量归一化层和一个LeakyReLU激活函数和一个池化层
        # 最后一个卷积层后跟一个全局平均池化层
        self.conv_layers = nn.Sequential(
            conv(n_channels, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            conv(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            
            conv(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            conv(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            
            conv(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            conv(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            
            conv(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            conv(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        #全连接层架构
        # 第一个全连接层后跟一个批量归一化层和一个LeakyReLU激活函数
        # 第二个全连接层sigmoid激活函数
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, image):
        # 前向传播
        # 先通过卷积层，提取消息特征图
        x = self.conv_layers(image)
        #将特征图展平
        x = x.view(x.size(0), -1)
        # 通过全连接层，输出判断结果
        output = self.fc_layers(x)
        return output
