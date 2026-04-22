import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

#定义数据集
class SteganographyDataset(Dataset):
    def __init__(self, root_dir, image_size=256, message_length=30, transform=None):
        self.root_dir = root_dir
        self.image_size = image_size
        self.message_length = message_length
        #默认的图像处理操作，包括调整大小、转换为张量、归一化
        #如果提供了自定义的transform，就使用自定义的transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
        #获取所有图像的路径
        #递归地遍历目录，找到所有图像文件
        #将图像文件的路径添加到列表中
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        #根据索引获取图像路径
        #打开图像文件，将其转换为RGB模式
        #如果提供了transform，就应用transform
        #生成随机消息
        #返回包含图像、消息和路径的字典
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        message = torch.randint(0, 2, (self.message_length,)).float()
        
        return {
            'cover_image': image,
            'message': message,
            'image_path': image_path
        }
