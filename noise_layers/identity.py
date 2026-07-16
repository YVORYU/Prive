import torch.nn as nn

# 本模块实现恒等映射噪声层
# 主要功能是：对输入图像不做任何改变，直接原样返回
# 当用户未配置任何噪声层时，Noiser 会默认使用 Identity 作为占位噪声层
# 这样可以保证训练流程在无噪声配置时仍然能够正常运行
class Identity(nn.Module):
    # 恒等映射噪声层：不改变输入图像，直接返回
    # 用于无噪声场景下的占位，确保 Noiser 至少有一个噪声层
    def __init__(self):
        # 初始化恒等映射噪声层，无需任何参数
        super(Identity, self).__init__()

    def forward(self, noised_and_cover):
        # 前向传播：直接返回输入，不做任何修改
        # 参数:
        #   noised_and_cover: 包含 [noised_image, cover_image] 的列表/元组
        # 返回:
        #   原样返回 noised_and_cover
        return noised_and_cover
