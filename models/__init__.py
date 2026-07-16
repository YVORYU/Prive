# ==============================================================================
# 模块：models 包初始化文件
# 职责：统一导出 PrivSteg 项目中所有神经网络模型组件，包括编码器、解码器、
#       隐写分析器（判别器）和内容识别器（CLIP 特征提取器）。
#       通过该 __init__.py，外部代码可直接使用 `from models import Encoder` 等方式
#       引入所需模型，简化模块路径管理。
# ==============================================================================

# 导入编码器：负责在载体图像中嵌入二进制消息和隐私扰动
from .encoder import Encoder
# 导入解码器：从编码图像中预测消息 logits
from .decoder import Decoder
# 导入隐写分析器：判别器式二分类器，用于对抗训练
from .steganalyzer import Steganalyzer
# 导入内容识别器：封装冻结的 CLIP 模型，用于图像/文本特征提取
from .content_recognizer import ContentRecognizer

# 模块公开接口列表，约束 `from models import *` 的导出范围
__all__ = [
    'Encoder',           # 编码器类
    'Decoder',           # 解码器类
    'Steganalyzer',      # 隐写分析器类
    'ContentRecognizer'  # 内容识别器类
]
