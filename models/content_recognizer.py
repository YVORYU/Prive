# ==============================================================================
# 模块：内容识别器（ContentRecognizer）
# 职责：封装冻结的 CLIP ViT-B/32 模型，用于提取图像和文本的特征向量，
#       并对输出做 L2 归一化以便后续计算 cosine similarity。
#       在 PrivSteg 中，主要用于：
#         1) 计算 FeatureRedirectionLoss 中的 CLIP escape loss，引导编码图像的 CLIP
#            特征远离原始载体图像的特征，降低 CBIR 检索关联风险；
#         2) 在验证脚本中计算编码图像与原始载体图像的 CLIP 相似度，评估隐私保护效果。
# 注意：CLIP 模型参数完全冻结，不参与反向传播。
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class ContentRecognizer(nn.Module):
    # 内容识别器主类
    # 职责：基于冻结的 CLIP 模型提取图像/文本特征，并做归一化以用于 cosine 相似度计算

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', model_name='ViT-B/32'):
        # 构造函数：初始化内容识别器
        # 参数：
        #   device (str or torch.device): 模型所在设备，默认自动选择 GPU/CPU
        #   model_name (str): CLIP 模型名称，默认 'ViT-B/32'
        super(ContentRecognizer, self).__init__()
        # 保存 CLIP 模型名称
        self.model_name = model_name
        # 将字符串形式的 device 转换为 torch.device 对象
        self._device = torch.device(device) if isinstance(device, str) else device
        # 加载 CLIP 模型，用于特征提取；忽略 preprocess（此处使用自定义归一化流程）
        self.clip_model, _ = clip.load(model_name, device=self._device)
        # 冻结 CLIP 模型的所有参数，只训练我们的全连接层（当前类无新增可训练层）
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # 将 CLIP 模型设为评估模式，关闭 dropout 和 BN 更新
        self.clip_model.eval()

        # 注册 CLIP 模型的均值张量，形状 (1,3,1,1)，作为 buffer 随模型迁移设备
        self.register_buffer('clip_mean', torch.tensor([0.481, 0.458, 0.408]).view(1, 3, 1, 1))
        # 注册 CLIP 模型的标准差张量，形状 (1,3,1,1)，作为 buffer 随模型迁移设备
        self.register_buffer('clip_std', torch.tensor([0.269, 0.261, 0.275]).view(1, 3, 1, 1))

    # 设备属性，用于获取当前模型所在设备
    @property
    def device(self):
        # 返回当前模型所在的 torch.device
        return self._device

    # 重写 to 方法，将模型和 CLIP 模型移动到指定设备
    def to(self, *args, **kwargs):
        # 调用父类 to 完成本模块参数的设备迁移
        module = super().to(*args, **kwargs)
        # 从位置参数中解析目标设备
        if len(args) > 0:
            self._device = args[0] if isinstance(args[0], torch.device) else torch.device(args[0])
        # 否则从关键字参数中解析 device
        elif 'device' in kwargs:
            self._device = kwargs['device'] if isinstance(kwargs['device'], torch.device) else torch.device(kwargs['device'])
        # 单独将 CLIP 模型迁移到目标设备（因为它不是 nn.Module 的子模块，需手动迁移）
        self.clip_model = self.clip_model.to(self._device)
        return module

    def forward(self, image):
        # 前向传播：提取图像的 CLIP 特征向量（已归一化）
        # 参数：
        #   image (Tensor): 输入图像，形状 (B, 3, H, W)，取值范围 [-1, 1]
        # 返回：
        #   Tensor: 归一化后的图像特征，形状 (B, feat_dim)，可直接用于 cosine 相似度

        # 将图像双线性插值到 CLIP 期望的 224x224 输入尺寸
        image_resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        # 将像素值从 [-1, 1] 映射到 [0, 1]，匹配 CLIP 的输入约定
        image_resized = (image_resized + 1) / 2
        # 用 CLIP 的均值和标准差做标准化，确保输入分布与 CLIP 预训练一致
        image_resized = (image_resized - self.clip_mean.to(image.device)) / self.clip_std.to(image.device)
        # 通过 CLIP 图像编码器提取特征向量
        image_features = self.clip_model.encode_image(image_resized)
        # L2 归一化特征向量，便于后续直接计算 cosine 相似度
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, text):
        # 文本编码方法：将文本字符串编码为归一化的特征向量
        # 参数：
        #   text (str or List[str]): 待编码的文本或文本列表
        # 返回：
        #   Tensor: 归一化后的文本特征，形状 (B, feat_dim)
        # 使用 CLIP 分词器对文本进行 tokenize，超长文本自动截断
        tokens = clip.tokenize(text, truncate=True).to(self._device)
        # 在无梯度上下文中编码文本，节省显存（CLIP 文本编码器也是冻结的）
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
        # L2 归一化文本特征向量
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
