import torch
import torch.nn as nn
import torch.nn.functional as F

# SSIM损失函数，使编码后的图像与原图的SSIM值最大，SSMI值越大，说明两者越相似
class SSIMLoss(nn.Module):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    def __init__(self, window_size=11, size_average=True):
        #局部窗口大小，用于计算SSIM值，将高斯窗口注册为模型参数
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.register_buffer('_window', self.create_window(window_size, self.channel))

    @property
    def window(self):
        return self._window
    
    #高斯窗口生成，生成一维高斯窗口，用于计算SSIM值
    #sigma为高斯窗口的标准差，用于控制窗口的宽度
    #返回归一化后的高斯窗口
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    #将一维高斯窗口扩展为2维窗口，用于计算SSIM值
    #window_size为高斯窗口的大小，channel为图像通道数
    #返回归一化后的2D高斯窗口
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    #SSIM损失计算
    def ssim(self, img1, img2):
        window = self._window
        if window.device != img1.device:
            window = window.to(img1.device)
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        return 1 - self.ssim(img1, img2)
