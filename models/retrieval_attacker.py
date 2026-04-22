import torch
import torch.nn as nn
import torch.nn.functional as F


"""
检索攻击模型
"""
#基础检索攻击模型架构
#4个卷积层，每个卷积层后跟一个批量归一化层和一个LeakyReLU激活函数和一个池化层
class RetrievalAttacker(nn.Module):
    def __init__(self, feature_dim=512, use_attention=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_attention = use_attention
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=1),
                nn.Sigmoid()
            )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, feature_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        feat = self.feature_extractor(x)
        
        if self.use_attention:
            attn = self.attention(feat)
            feat = feat * attn
        
        feat = self.pool(feat)
        feat = feat.view(feat.size(0), -1)
        feat = self.fc(feat)
        feat = F.normalize(feat, p=2, dim=1)
        
        return feat


class RetrievalAttackerWithMemory(nn.Module):
    def __init__(self, feature_dim=512, memory_size=1000, temperature=0.07):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.temperature = temperature
        
        self.attacker = RetrievalAttacker(feature_dim=feature_dim)
        
        self.register_buffer('memory_bank', torch.zeros(memory_size, feature_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('memory_count', torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def update_memory(self, features):
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        ptr = int(self.memory_ptr)
        
        if ptr + batch_size > self.memory_size:
            remaining = self.memory_size - ptr
            self.memory_bank[ptr:] = features[:remaining]
            self.memory_bank[:batch_size - remaining] = features[remaining:]
            self.memory_ptr[0] = batch_size - remaining
        else:
            self.memory_bank[ptr:ptr + batch_size] = features
            self.memory_ptr[0] = (ptr + batch_size) % self.memory_size
        
        self.memory_count[0] = min(self.memory_count[0] + batch_size, self.memory_size)
    
    def forward(self, x, return_similarities=False):
        features = self.attacker(x)
        
        if return_similarities:
            similarities = torch.mm(features, self.memory_bank.t()) / self.temperature
            return features, similarities
        
        return features
    
    def get_hard_negatives(self, features, k=5):
        valid_count = int(self.memory_count[0])
        if valid_count == 0:
            return None
        
        valid_memory = self.memory_bank[:valid_count]
        similarities = torch.mm(features, valid_memory.t())
        actual_k = min(k, valid_count)
        _, indices = similarities.topk(actual_k, dim=1)
        hard_negatives = valid_memory[indices]
        return hard_negatives


class AdversarialRetrievalModule(nn.Module):
    def __init__(self, feature_dim=512, num_attack_heads=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_attack_heads = num_attack_heads
        
        self.attack_heads = nn.ModuleList([
            RetrievalAttacker(feature_dim=feature_dim) 
            for _ in range(num_attack_heads)
        ])
        
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * num_attack_heads, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, feature_dim)
        )
    
    def forward(self, x, return_all_features=False):
        all_features = []
        for head in self.attack_heads:
            feat = head(x)
            all_features.append(feat)
        
        fused = torch.cat(all_features, dim=1)
        fused = self.fusion(fused)
        fused = F.normalize(fused, p=2, dim=1)
        
        if return_all_features:
            return fused, all_features
        return fused
