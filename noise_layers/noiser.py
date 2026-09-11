import numpy as np
import torch.nn as nn
from noise_layers.identity import Identity


# 本模块实现噪声层组合器 Noiser
# 主要功能是：将多个噪声层组合成一个序列噪声模块，每次前向传播时随机选择一个噪声层执行
# 这种随机选择策略可以增强隐写模型对多种噪声扰动的鲁棒性
# trainer.py 通过 '+' 拆分 noise_config 字符串，构造噪声层实例列表后传入 Noiser
# 支持的噪声层: crop、cropout、dropout、resize、jpeg、quant

class Noiser(nn.Module):
    # Noiser 噪声层组合器：将多个噪声层组合，每次 forward 随机选取一个执行
    # 配置和噪声层序列由 noise_layers 参数控制
    # 若 noise_layers 为空，则默认使用 Identity（不施加任何噪声）
    def __init__(self, noise_layers: list, device):
        # 初始化 Noiser 噪声层组合器
        # 参数:
        #   noise_layers: 已实例化的 nn.Module 噪声层列表
        #   device: 计算设备（用于实例化需要设备的噪声层，如 JpegCompression）
        super(Noiser, self).__init__()
        self.noise_layers = nn.ModuleList()  # 使用 ModuleList 以便注册子模块参数
        self._layer_names = []  # 保存每个噪声层的名称，便于日志和调试

        # 若噪声层列表为空，则默认添加 Identity（不施加噪声）
        if len(noise_layers) == 0:
            self.noise_layers.append(Identity())
            self._layer_names.append('identity')
        else:
            # 遍历噪声层列表，逐个添加
            for layer in noise_layers:
                self.noise_layers.append(layer)
                self._layer_names.append(layer.__class__.__name__)

    def forward(self, encoded_and_cover):
        # 前向传播：从噪声层列表中随机选择一个执行
        # 参数:
        #   encoded_and_cover: 包含 [encoded_image, cover_image] 的列表/元组
        # 返回:
        #   所选噪声层的输出
        # 随机选取一个噪声层的索引
        idx = np.random.randint(0, len(self.noise_layers))
        # 转为 list：部分噪声层（如 Cropout）会对输入做原地赋值 noised_and_cover[0] = ...，
        # 元组不可变会导致 TypeError，因此统一转为可变 list 传入
        if isinstance(encoded_and_cover, tuple):
            encoded_and_cover = list(encoded_and_cover)
        # 调用所选噪声层的 forward 方法
        return self.noise_layers[idx](encoded_and_cover)
