# 模块：SSIM 结构相似性损失
# 该文件实现 SSIMLoss，将结构相似性（SSIM）作为损失函数返回 1 - SSIM。
# SSIM 值越大表示两图像越相似，因此损失值越小代表编码图像与载体图像越接近。
# 主要用于训练阶段约束编码图像的视觉质量，避免引入明显失真。

# 导入 PyTorch 主模块，用于张量运算
import torch
# 导入 torch.nn，用于构建可训练的损失模块
import torch.nn as nn
# 导入 torch.nn.functional，提供 conv2d 等函数式 API
import torch.nn.functional as F


# SSIM损失函数，使编码后的图像与原图的SSIM值最大，SSIM值越大，说明两者越相似
class SSIMLoss(nn.Module):
    # SSIM 计算中用于稳定分母的常数 C1 = (k1 * L)^2，k1=0.01，L=1（归一化像素范围）
    C1 = 0.01 ** 2
    # SSIM 计算中用于稳定分母的常数 C2 = (k2 * L)^2，k2=0.03，L=1（归一化像素范围）
    C2 = 0.03 ** 2

    # 构造函数：初始化 SSIM 计算所需的高斯窗口
    # 参数：
    #   window_size：局部滑动窗口大小，默认 11
    #   size_average：是否对所有像素取平均，True 返回标量，False 返回每张图的均值
    def __init__(self, window_size=11, size_average=True):
        # 调用父类 nn.Module 的构造函数
        super(SSIMLoss, self).__init__()
        # 保存局部窗口大小
        self.window_size = window_size
        # 保存是否对所有像素取平均的标志
        self.size_average = size_average
        # 默认通道数为 3（RGB 图像）
        self.channel = 3
        # 创建高斯窗口并注册为 buffer（不参与梯度更新，但随模型一起迁移设备）
        self.register_buffer('_window', self.create_window(window_size, self.channel))

    #高斯窗口生成，生成一维高斯窗口，用于计算SSIM值
    #sigma为高斯窗口的标准差，用于控制窗口的宽度
    #返回归一化后的高斯窗口
    def gaussian(self, window_size, sigma):
        # 构造一维高斯核，对每个位置 x 计算高斯函数值
        gauss = torch.Tensor([
            # 以窗口中心为均值，sigma 为标准差计算高斯函数值
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
            for x in range(window_size)
        ])
        # 对高斯核进行归一化，使其和为 1
        return gauss / gauss.sum()

    #将一维高斯窗口扩展为2维窗口，用于计算SSIM值
    #window_size为高斯窗口的大小，channel为图像通道数
    #返回归一化后的2D高斯窗口
    def create_window(self, window_size, channel):
        # 生成一维高斯窗口并增加一个维度，shape 为 [window_size, 1]
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        # 通过外积构造二维高斯窗口，shape 为 [1, 1, window_size, window_size]
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        # 将二维窗口扩展到指定通道数，shape 为 [channel, 1, window_size, window_size]
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        # 返回可用的多通道高斯卷积核
        return window

    #SSIM损失计算
    # 计算 img1 与 img2 之间的 SSIM 值
    # 参数：
    #   img1：第一张图像张量，shape 为 [B, C, H, W]
    #   img2：第二张图像张量，shape 为 [B, C, H, W]
    # 返回：SSIM 值（标量或每张图的均值，取决于 size_average）
    def ssim(self, img1, img2):
        # 取出预注册的高斯窗口
        window = self._window
        # 如果窗口所在设备与输入图像不一致，则将窗口迁移到输入图像所在设备
        if window.device != img1.device:
            window = window.to(img1.device)
        # 使用高斯窗口对 img1 做分组卷积，计算局部均值 mu1
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        # 使用高斯窗口对 img2 做分组卷积，计算局部均值 mu2
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        # 计算 mu1 的平方
        mu1_sq = mu1.pow(2)
        # 计算 mu2 的平方
        mu2_sq = mu2.pow(2)
        # 计算 mu1 与 mu2 的乘积
        mu1_mu2 = mu1 * mu2

        # 计算 img1 的局部方差 sigma1_sq = E[X^2] - (E[X])^2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        # 计算 img2 的局部方差 sigma2_sq = E[Y^2] - (E[Y])^2
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        # 计算 img1 与 img2 的局部协方差 sigma12 = E[XY] - E[X]E[Y]
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        # 按 SSIM 公式计算每个像素位置的 SSIM 值：
        # SSIM = ((2*mu1*mu2 + C1)(2*sigma12 + C2)) / ((mu1^2 + mu2^2 + C1)(sigma1^2 + sigma2^2 + C2))
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        # 若 size_average 为 True，则返回所有像素 SSIM 的整体均值
        if self.size_average:
            return ssim_map.mean()
        # 否则按 batch 维度分别求均值，返回每张图的 SSIM 值
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    # 前向计算：返回 1 - SSIM 作为损失
    # 参数：
    #   img1：第一张图像张量，shape 为 [B, C, H, W]
    #   img2：第二张图像张量，shape 为 [B, C, H, W]
    # 返回：标量损失张量，值越小表示两图越相似
    def forward(self, img1, img2):
        # SSIM 越大越相似，损失取 1 - SSIM 以最小化
        return 1 - self.ssim(img1, img2)
