# 模块：高级隐私损失（Feature Redirection Loss）
# 该文件实现 FeatureRedirectionLoss，用于在 Phase 2 阶段对编码图像的 CLIP/DINOv2 特征
# 进行偏移，降低其与载体图像特征之间的相似度，从而规避基于内容的图像检索（CBIR）关联。
# 损失由两部分组成：
#   1. escape_loss：二次 Hinge + Softplus 预热，将编码特征推离载体特征
#   2. redirect_loss：可选的 memory bank hard-negative redirection，将编码特征重定向至最难负样本
# 通过加权组合 CLIP 与 DINOv2 两路特征损失，得到最终 total_privacy_loss。

# 导入 PyTorch 主模块，用于张量计算
import torch
# 导入 torch.nn，用于构建可训练的损失模块
import torch.nn as nn
# 导入 torch.nn.functional，提供函数式 API（cosine_similarity、relu、softplus 等）
import torch.nn.functional as F


# FeatureRedirectionLoss：特征偏移隐私损失
# 通过对 CLIP / DINOv2 特征施加 escape + redirect 损失，降低编码图像与载体图像的特征相似度
class FeatureRedirectionLoss(nn.Module):
    # 构造函数：初始化各项损失权重与 margin 参数
    # 参数：
    #   clip_weight：CLIP escape loss 的全局权重
    #   dino_weight：DINOv2 escape loss 的全局权重（设为 0 可禁用 DINO 分支）
    #   escape_margin：escape loss 的目标 margin，控制特征被推离的距离
    #   clip_redirect_weight：CLIP hard-negative redirect loss 的权重（设为 0 可禁用）
    #   dino_redirect_weight：DINOv2 hard-negative redirect loss 的权重（设为 0 可禁用）
    #   clip_redirect_margin：CLIP redirect 的目标 margin
    #   dino_redirect_margin：DINOv2 redirect 的目标 margin
    #   clip_max_target_sim：CLIP redirect 控制项的相似度上限，超过则施加惩罚
    #   dino_max_target_sim：DINOv2 redirect 控制项的相似度上限，超过则施加惩罚
    def __init__(self, clip_weight=1.0, dino_weight=0.5, escape_margin=0.5,
                 clip_redirect_weight=0.3, dino_redirect_weight=0.2,
                 clip_redirect_margin=0.3, dino_redirect_margin=0.3,
                 clip_max_target_sim=0.88, dino_max_target_sim=0.65,
                 clip_redirect_target_sim=0.80, dino_redirect_target_sim=0.55):
        # 调用父类 nn.Module 的构造函数，完成模块初始化
        super().__init__()
        # 保存 CLIP escape loss 权重
        self.clip_weight = clip_weight
        # 保存 DINOv2 escape loss 权重
        self.dino_weight = dino_weight
        # 保存 escape loss 的目标 margin
        self.escape_margin = escape_margin
        # 保存 CLIP redirect loss 权重
        self.clip_redirect_weight = clip_redirect_weight
        # 保存 DINOv2 redirect loss 权重
        self.dino_redirect_weight = dino_redirect_weight
        # 保存 CLIP redirect 的目标 margin（遗留参数，目标相似度现由 redirect_target_sim 控制）
        self.clip_redirect_margin = clip_redirect_margin
        # 保存 DINOv2 redirect 的目标 margin（遗留参数）
        self.dino_redirect_margin = dino_redirect_margin
        # 保存 CLIP redirect 控制项的相似度上限（必须大于 clip_redirect_target_sim）
        self.clip_max_target_sim = clip_max_target_sim
        # 保存 DINOv2 redirect 控制项的相似度上限
        self.dino_max_target_sim = dino_max_target_sim
        # 重定向目标相似度（按特征空间分别标定）：将编码特征拉近 hard negative 直至达到该值。
        # 取值必须高于该空间典型 hard negative 相似度（CLIP≈0.73 / DINO≈0.47），
        # 否则 relu(target - sim) 恒为 0，redirect 名存实亡
        self.clip_redirect_target_sim = clip_redirect_target_sim
        self.dino_redirect_target_sim = dino_redirect_target_sim

    # 计算 escape loss：将编码特征推离载体特征
    # 参数：
    #   feat_cover：载体图像的特征张量，shape 为 [B, D]
    #   feat_encoded：编码图像的特征张量，shape 为 [B, D]
    #   margin：目标 margin，希望余弦相似度下降到 (1 - margin) 以下
    # 返回：
    #   loss：标量损失张量（二次 Hinge 主项 + Softplus 预热项）
    #   sim.mean().item()：当前 batch 内编码特征与载体特征的平均余弦相似度（用于监控）
    def escape_loss(self, feat_cover, feat_encoded, margin):
        # 沿特征维度计算编码特征与载体特征的余弦相似度，shape 为 [B]
        sim = F.cosine_similarity(feat_encoded, feat_cover, dim=1)
        # 计算目标相似度阈值，相似度高于此值才施加惩罚
        target = 1 - margin
        # 计算当前相似度超出目标的部分（正表示违反 margin 约束）
        diff = sim - target
        # 二次 Hinge 主项：对超出部分取 ReLU 后平方，乘以放大系数 25.0 加速收敛
        hinge_sq = torch.mean(F.relu(diff) ** 2) * 25.0
        # Softplus 预热项：在 diff 较小时提供平滑梯度，避免训练初期不稳定
        warmup = torch.mean(F.softplus(diff)) * 0.1
        # 总 escape loss 为主项与预热项之和
        loss = hinge_sq + warmup
        # 返回损失张量与平均相似度（用于日志/监控）
        return loss, sim.mean().item()

    # 计算 redirect loss：将编码特征重定向至 memory bank 中的最难负样本
    # 参数：
    #   feat_encoded：编码图像特征，shape 为 [B, D]
    #   hard_negatives：memory bank 中的难负样本特征，shape 为 [B, K, D]，K 为负样本数
    #   target_sim：重定向目标相似度，将编码特征拉近负样本直至达到该值
    #   max_target_sim：相似度控制上限（必须大于 target_sim），超过则施加惩罚
    # 返回：
    #   redirect_loss：标量重定向损失张量
    #   control_loss：标量控制损失张量，用于约束相似度不超过 max_target_sim
    def redirect_loss(self, feat_encoded, hard_negatives, target_sim, max_target_sim):
        # 计算每个编码特征与所有难负样本的余弦相似度，shape 为 [B, K]
        sim_to_neg = F.cosine_similarity(feat_encoded.unsqueeze(1), hard_negatives, dim=2)
        # 在 K 维度上找到相似度最大的负样本索引，shape 为 [B]
        # （调用方已在候选阶段排除自身/近重复，argmax 得到的是"最相似的他人"）
        hardest_idx = sim_to_neg.argmax(dim=1)
        # 根据索引取出每个样本对应的最难负样本特征，shape 为 [B, D]
        hardest_neg = hard_negatives[torch.arange(feat_encoded.size(0)), hardest_idx]
        # 计算编码特征与最难负样本的余弦相似度，shape 为 [B]
        sim_to_hardest = F.cosine_similarity(feat_encoded, hardest_neg, dim=1)
        # 主重定向损失：线性 Hinge，把相似度拉升到 target_sim 即停止（避免过度推动）
        redirect_loss = torch.mean(F.relu(target_sim - sim_to_hardest))
        # 控制损失：当相似度超过 max_target_sim 时施加 Hinge 惩罚，防止过度靠近负样本
        control_loss = torch.mean(F.relu(sim_to_hardest - max_target_sim))
        # 返回重定向损失与控制损失
        return redirect_loss, control_loss

    # 前向计算：组合 CLIP 与 DINOv2 两路 escape + redirect 损失
    # 参数：
    #   clip_cover：CLIP 载体特征，shape 为 [B, D_c]
    #   clip_encoded：CLIP 编码特征，shape 为 [B, D_c]
    #   dino_cover：DINOv2 载体特征，可选，shape 为 [B, D_d]
    #   dino_encoded：DINOv2 编码特征，可选，shape 为 [B, D_d]
    #   clip_hard_negatives：CLIP 难负样本，可选，shape 为 [B, K, D_c]
    #   dino_hard_negatives：DINOv2 难负样本，可选，shape 为 [B, K, D_d]
    #   escape_margin：可选，覆盖默认 CLIP escape margin（用于渐进调度）
    #   dino_escape_margin：可选，DINOv2 独立的 escape margin（DINO 目标通常更宽松）；
    #     缺省时回退到 escape_margin，保持旧行为
    # 返回：
    #   total_loss：标量总损失张量
    #   loss_dict：包含各项子损失数值的字典，用于日志记录
    def forward(self, clip_cover, clip_encoded, dino_cover=None, dino_encoded=None,
                clip_hard_negatives=None, dino_hard_negatives=None, escape_margin=None,
                dino_escape_margin=None):
        # 若调用方未提供 escape_margin，则使用构造时设置的默认值
        if escape_margin is None:
            escape_margin = self.escape_margin
        # DINOv2 缺省时与 CLIP 共用同一 margin（向后兼容）
        if dino_escape_margin is None:
            dino_escape_margin = escape_margin

        # 总损失初始化为 0.0（标量），后续累加各路加权损失
        total_loss = 0.0
        # 子损失字典，记录各分支损失值用于日志与监控
        loss_dict = {}

        # 计算 CLIP escape loss 及当前 CLIP 相似度
        clip_escape, clip_sim = self.escape_loss(clip_cover, clip_encoded, escape_margin)
        # 累加加权后的 CLIP escape loss 至总损失
        total_loss += self.clip_weight * clip_escape
        # 记录 CLIP escape loss 数值
        loss_dict['clip_escape'] = clip_escape.item()
        # 记录 CLIP 平均相似度数值
        loss_dict['clip_sim'] = clip_sim

        # 如果提供了 CLIP 难负样本，且 redirect 权重大于 0，则计算 CLIP redirect loss
        if clip_hard_negatives is not None and self.clip_redirect_weight > 0:
            # 计算 CLIP 重定向损失与控制损失（目标/上限均使用相似度直值，避免 margin 反推造成的目标冲突）
            clip_redir, clip_ctrl = self.redirect_loss(
                clip_encoded, clip_hard_negatives,
                self.clip_redirect_target_sim, self.clip_max_target_sim
            )
            # 累加加权后的 CLIP redirect loss 与 control loss 至总损失
            # （control loss 必须参与优化，否则相似度上限约束不生效）
            total_loss += self.clip_redirect_weight * (clip_redir + clip_ctrl)
            # 记录 CLIP redirect loss 数值
            loss_dict['clip_redirect'] = clip_redir.item()
            # 记录 CLIP control loss 数值
            loss_dict['clip_control'] = clip_ctrl.item()

        # 如果同时提供了 DINOv2 的载体与编码特征，则计算 DINOv2 分支损失
        if dino_cover is not None and dino_encoded is not None:
            # 计算 DINOv2 escape loss（使用独立的 dino_escape_margin）及当前 DINOv2 相似度
            dino_escape, dino_sim = self.escape_loss(dino_cover, dino_encoded, dino_escape_margin)
            # 累加加权后的 DINOv2 escape loss 至总损失
            total_loss += self.dino_weight * dino_escape
            # 记录 DINOv2 escape loss 数值
            loss_dict['dino_escape'] = dino_escape.item()
            # 记录 DINOv2 平均相似度数值
            loss_dict['dino_sim'] = dino_sim

            # 如果提供了 DINOv2 难负样本，且 redirect 权重大于 0，则计算 DINOv2 redirect loss
            if dino_hard_negatives is not None and self.dino_redirect_weight > 0:
                # 计算 DINOv2 重定向损失与控制损失
                dino_redir, dino_ctrl = self.redirect_loss(
                    dino_encoded, dino_hard_negatives,
                    self.dino_redirect_target_sim, self.dino_max_target_sim
                )
                # 累加加权后的 DINOv2 redirect loss 与 control loss 至总损失
                total_loss += self.dino_redirect_weight * (dino_redir + dino_ctrl)
                # 记录 DINOv2 redirect loss 数值
                loss_dict['dino_redirect'] = dino_redir.item()
                # 记录 DINOv2 control loss 数值
                loss_dict['dino_control'] = dino_ctrl.item()

        # 记录最终的总隐私损失数值
        loss_dict['total_privacy_loss'] = total_loss.item()
        # 返回总损失张量与子损失字典
        return total_loss, loss_dict
