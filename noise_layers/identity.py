import torch.nn as nn

"""
实现"映射"噪声层，不改变图像
"""
class Identity(nn.Module):
    """
    Identity-mapping noise layer. Does not change the image
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, noised_and_cover):
        return noised_and_cover
