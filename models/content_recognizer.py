import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class ContentRecognizer(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', model_name='ViT-B/32'):
        super(ContentRecognizer, self).__init__()
        self.model_name = model_name
        self._device = torch.device(device) if isinstance(device, str) else device
        #加载CLIP模型，用于特征提取
        self.clip_model, _ = clip.load(model_name, device=self._device)
        # 冻结CLIP模型的参数，只训练我们的全连接层
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        # 注册CLIP模型的均值和标准差，用于归一化输入图像
        self.register_buffer('clip_mean', torch.tensor([0.481, 0.458, 0.408]).view(1, 3, 1, 1))
        self.register_buffer('clip_std', torch.tensor([0.269, 0.261, 0.275]).view(1, 3, 1, 1))

    # 设备属性，用于获取当前模型所在设备
    @property
    def device(self):
        return self._device

    #重写to方法，将模型和CLIP模型移动到指定设备
    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        if len(args) > 0:
            self._device = args[0] if isinstance(args[0], torch.device) else torch.device(args[0])
        elif 'device' in kwargs:
            self._device = kwargs['device'] if isinstance(kwargs['device'], torch.device) else torch.device(kwargs['device'])
        self.clip_model = self.clip_model.to(self._device)
        return module

    def forward(self, image):
        image_resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        image_resized = (image_resized + 1) / 2
        image_resized = (image_resized - self.clip_mean.to(image.device)) / self.clip_std.to(image.device)
        image_features = self.clip_model.encode_image(image_resized)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features