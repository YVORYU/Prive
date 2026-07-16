# 模块：losses 包初始化文件
# 该文件负责聚合 losses 包中对外暴露的损失函数类，方便上层模块统一导入。
# 主要导出两类损失：
#   1. SSIMLoss：结构相似性损失，用于约束编码图像的视觉质量
#   2. FeatureRedirectionLoss：特征偏移隐私损失，用于规避 CBIR 图像检索关联

# 从 ssim_loss 子模块导入 SSIMLoss 结构相似性损失类
from .ssim_loss import SSIMLoss
# 从 advanced_privacy_loss 子模块导入 FeatureRedirectionLoss 特征偏移损失类
from .advanced_privacy_loss import FeatureRedirectionLoss

# 定义本包对外公开的类列表，使用 from losses import * 时只会导入这些名称
__all__ = [
    'SSIMLoss',                  # SSIM 结构相似性损失
    'FeatureRedirectionLoss'     # 特征偏移（隐私重定向）损失
]
