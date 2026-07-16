import numpy as np
import torch
import torch.nn as nn

# 本模块实现可微量化噪声层
# 主要功能是：使用基于傅里叶级数的可微近似来实现像素值的量化（round）操作
# 传统的 round 函数不可导，无法用于反向传播；本模块用三角函数级数近似 round 的导数
# 量化范围映射到 [0, 255]，模拟 8 位图像的量化失真
# 该噪声层由 noise_argparser.py 中的 quant 配置项控制（在 Noiser 中以 'QuantizationPlaceholder' 占位符传入）

def transform(tensor, target_range):
    # 将张量线性映射到 target_range 指定的数值范围
    # 先归一化到 [0, 1]，再缩放平移到 [target_range[0], target_range[1]]
    # 参数:
    #   tensor: 输入张量
    #   target_range: 目标范围 (min, max)
    # 返回:
    #   映射后的张量
    source_min = tensor.min()  # 输入张量的最小值
    source_max = tensor.max()  # 输入张量的最大值

    # 第一步：归一化到 [0, 1]
    tensor_target = (tensor - source_min)/(source_max - source_min)
    # 第二步：缩放平移到目标范围 [target_range[0], target_range[1]]
    tensor_target = tensor_target * (target_range[1] - target_range[0]) + target_range[0]
    return tensor_target


class Quantization(nn.Module):
    # 可微量化噪声层
    # 通过傅里叶级数近似 round 函数，使量化操作可导，便于反向传播训练
    # 量化过程：将像素值映射到 [0, 255] -> 用傅里叶级数近似 round -> 映射回原始范围
    def __init__(self, device=None):
        # 初始化量化噪声层
        # 参数:
        #   device: 计算设备，若为 None 则自动选择（有 CUDA 用 cuda，否则用 cpu）
        super(Quantization, self).__init__()
        # 若未指定设备，则根据 CUDA 可用性自动选择
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self._device = device  # 保存计算设备
        self.min_value = 0.0   # 量化范围下界（对应 8 位图像 0）
        self.max_value = 255.0  # 量化范围上界（对应 8 位图像 255）
        self.N = 10  # 傅里叶级数展开的项数，项数越多近似越精确

        # 计算傅里叶级数的权重: ((-1)^(n+1)) / (pi * (n+1))，n = 0..N-1
        weights_data = torch.tensor([((-1) ** (n + 1)) / (np.pi * (n + 1)) for n in range(self.N)])
        # 计算傅里叶级数的尺度因子: 2 * pi * (n+1)，n = 0..N-1
        scales_data = torch.tensor([2 * np.pi * (n + 1) for n in range(self.N)])
        # 连续 4 次 unsqueeze，将权重和尺度因子扩展为 4 维张量（便于后续广播）
        # 形状: (N,) -> (N, 1) -> (N, 1, 1) -> (N, 1, 1, 1) -> (N, 1, 1, 1, 1)
        for _ in range(4):
            weights_data.unsqueeze_(-1)
            scales_data.unsqueeze_(-1)

        # 将权重和尺度因子注册为 buffer（随模型一起移动设备但不参与梯度更新）
        self.register_buffer('weights', weights_data)
        self.register_buffer('scales', scales_data)

    @property
    def device(self):
        # 设备属性：返回当前模块所在的计算设备
        return self._device

    def to(self, *args, **kwargs):
        # 重写 to 方法：在调用父类 to 移动参数的同时，同步更新 self._device
        # 这样可以保证后续 forward 时使用正确的设备信息
        # 参数:
        #   *args: 位置参数，第一个通常是目标设备
        #   **kwargs: 关键字参数，可能包含 device
        # 返回:
        #   移动设备后的模块本身
        module = super().to(*args, **kwargs)
        # 处理位置参数中的设备
        if len(args) > 0:
            self._device = args[0] if isinstance(args[0], torch.device) else torch.device(args[0])
        # 处理关键字参数中的设备
        elif 'device' in kwargs:
            self._device = kwargs['device'] if isinstance(kwargs['device'], torch.device) else torch.device(kwargs['device'])
        return module

    def fourier_rounding(self, tensor):
        # 使用傅里叶级数近似 round 函数
        # 近似公式: round(x) ≈ x + sum_n [((-1)^(n+1)) / (pi*(n+1)) * sin(2*pi*(n+1)*x)]
        # 参数:
        #   tensor: 输入张量，值域应为 [0, 255]
        # 返回:
        #   近似量化后的张量
        # 计算每一项: weights[n] * sin(tensor * scales[n])，利用广播在所有维度上同时计算
        z = torch.mul(self.weights, torch.sin(torch.mul(tensor, self.scales)))
        # 在第 0 维（傅里叶项数维度）上求和
        z = torch.sum(z, dim=0)
        # 返回原张量加上傅里叶修正项，近似 round 操作
        return tensor + z

    def forward(self, noised_and_cover):
        # 前向传播：执行可微量化
        # 流程: 映射到 [0, 255] -> clamp -> 傅里叶近似 round -> 映射回原始范围
        # 参数:
        #   noised_and_cover: 包含 [noised_image, cover_image] 的列表/元组
        # 返回:
        #   元组 (量化后的噪声图像, cover_image)
        noised_image = noised_and_cover[0]  # 取出噪声图像
        # 将图像像素值线性映射到 [0, 255] 范围（模拟 8 位图像）
        noised_image = transform(noised_image, (0, 255))
        # 先 clamp 到 [0, 255]，再用傅里叶级数近似 round 进行量化
        noised_image = self.fourier_rounding(noised_image.clamp(self.min_value, self.max_value))
        # 将量化后的图像映射回原始输入的数值范围（保持数据分布一致）
        noised_image = transform(noised_image, (noised_and_cover[0].min(), noised_and_cover[0].max()))
        return (noised_image, noised_and_cover[1])


# def main():
#     import numpy as np
#     qq = Quantization()
#     min_value = -1.4566678979
#     max_value = 2.334567325678
#     data = torch.rand((12, 3, 64, 64))
#     data = (max_value - min_value) * data + min_value
#     print(f'data.min(): {data.min()}')
#     print(f'data.max(): {data.max()}')
#     print(f'data.avg(): {data.mean()}')
#     data_tf, _ = qq.forward([data, None])
#     print('perform quantization')
#     print(f'data.min(): {data_tf.min()}')
#     print(f'data.max(): {data_tf.max()}')
#     print(f'data.avg(): {data_tf.mean()}')
#
#     print(f'mse diff:  {np.mean((data.numpy()-data_tf.numpy())**2)}')
#     print(f'mabs diff: {np.mean(np.abs(data.numpy()-data_tf.numpy()))}')
#
# if __name__ == '__main__':
#     main()
