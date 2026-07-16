# 模块：隐写数据集（SteganographyDataset）
# 该文件实现 PyTorch Dataset，递归加载指定目录下的图像文件，
# 返回包含 cover_image（载体图像）、message（随机二进制消息）、image_path（图像路径）的字典。
# 训练时数据应放在数据集根目录下的 train/ 和 val/ 子目录中，
# 支持 .png、.jpg、.jpeg、.bmp 格式，图像会被 resize 到 Config.image_size 并归一化到 [-1, 1]。

# 导入 os 模块，用于目录遍历与路径拼接
import os
# 导入 random 模块（保留导入，便于后续扩展随机数据增强）
import random
# 导入 PyTorch 主模块，用于张量操作（生成随机 message 等）
import torch
# 从 torch.utils.data 导入 Dataset 基类，用于自定义数据集
from torch.utils.data import Dataset
# 从 PIL 导入 Image，用于图像读取
from PIL import Image
# 导入 torchvision.transforms，用于构建图像预处理流水线
import torchvision.transforms as transforms

#定义数据集
# SteganographyDataset：隐写训练数据集
# 递归遍历 root_dir 下所有支持格式的图像，按索引返回载体图像、随机消息与路径
class SteganographyDataset(Dataset):
    # 构造函数：初始化数据集
    # 参数：
    #   root_dir：数据集根目录，递归扫描其下所有图像
    #   image_size：图像 resize 后的边长（正方形）
    #   message_length：随机二进制消息的长度
    #   transform：可选的自定义图像预处理，若为 None 则使用默认流水线
    def __init__(self, root_dir, image_size=256, message_length=30, transform=None):
        # 保存数据集根目录
        self.root_dir = root_dir
        # 保存目标图像边长
        self.image_size = image_size
        # 保存消息长度
        self.message_length = message_length
        #默认的图像处理操作，包括调整大小、转换为张量、归一化
        #如果提供了自定义的transform，就使用自定义的transform
        # 若未提供自定义 transform，则构建默认预处理流水线
        if transform is None:
            # 组合 resize、转张量、归一化三个变换
            self.transform = transforms.Compose([
                # 将图像 resize 到 image_size x image_size
                transforms.Resize((image_size, image_size)),
                # 将 PIL 图像转换为张量，像素值范围 [0, 1]
                transforms.ToTensor(),
                # 按均值 0.5、方差 0.5 归一化，将像素映射到 [-1, 1]
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            # 使用调用方提供的自定义 transform
            self.transform = transform
        #获取所有图像的路径
        #递归地遍历目录，找到所有图像文件
        #将图像文件的路径添加到列表中
        # 初始化图像路径列表
        self.image_paths = []
        # 使用 os.walk 递归遍历 root_dir 下的所有子目录
        for root, _, files in os.walk(root_dir):
            # 遍历当前目录下的所有文件
            for file in files:
                # 仅保留扩展名为 png/jpg/jpeg/bmp 的图像文件（大小写不敏感）
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    # 将匹配的图像完整路径加入列表
                    self.image_paths.append(os.path.join(root, file))

        # 打印找到的图像数量与所在目录，便于调试与确认
        print(f"Found {len(self.image_paths)} images in {root_dir}")

    # 返回数据集中图像样本的总数
    def __len__(self):
        # 返回图像路径列表的长度
        return len(self.image_paths)

    # 按索引获取一个样本
    # 参数：idx - 样本索引
    # 返回：包含 cover_image、message、image_path 的字典
    def __getitem__(self, idx):
        #根据索引获取图像路径
        #打开图像文件，将其转换为RGB模式
        #如果提供了transform，就应用transform
        #生成随机消息
        #返回包含图像、消息和路径的字典
        # 取出指定索引对应的图像路径
        image_path = self.image_paths[idx]
        # 使用 PIL 打开图像并转换为 RGB 三通道模式（避免 RGBA/灰度带来的通道不一致）
        image = Image.open(image_path).convert('RGB')

        # 若存在 transform（默认或自定义），则对图像应用预处理
        if self.transform:
            # 应用预处理流水线，输出张量
            image = self.transform(image)

        # 随机生成长度为 message_length 的 0/1 二进制消息，并转为 float 张量
        message = torch.randint(0, 2, (self.message_length,)).float()

        # 返回样本字典：载体图像、随机消息、图像路径
        return {
            'cover_image': image,         # 载体图像张量，shape 为 [3, H, W]，范围 [-1, 1]
            'message': message,           # 随机二进制消息张量，shape 为 [message_length]
            'image_path': image_path      # 图像在磁盘上的路径，便于调试与可视化
        }
