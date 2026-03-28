import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss(feature1, feature2, feature3, temperature=0):
    """
    修复版对比损失：支持任意batch_size，无警告、无报错
    1. 对齐损失：批量内三路特征两两MSE
    2. 均匀性损失：兼容批量输入，计算时间步特征的分布均匀性
    """
    # L2归一化（沿通道维度，支持批量）
    f1_norm = F.normalize(feature1, dim=1)
    f2_norm = F.normalize(feature2, dim=1)
    f3_norm = F.normalize(feature3, dim=1)

    # 1. 对齐损失（完全兼容批量输入）
    align_loss = (F.mse_loss(f1_norm, f2_norm) + F.mse_loss(f1_norm, f3_norm) + F.mse_loss(f2_norm, f3_norm)) / 3

    # 2. 均匀性损失（修复：支持批量 + 消除.T警告 + 适配pdist的2D输入要求）
    # 输入形状: [B, C, L] → 调整为 [B*L, C]（把所有时间步展平为独立样本）
    B, C, L = f1_norm.shape
    # 维度置换: [B, C, L] → [B, L, C]
    seq_features = f1_norm.permute(0, 2, 1)
    # 展平批量+时间步: [B*L, C]（2D张量，完美适配pdist）
    seq_features = seq_features.reshape(B * L, C)

    # 计算两两欧式距离
    pairwise_dist = torch.pdist(seq_features, p=2)
    uniform_loss = torch.mean(torch.exp(-2 * pairwise_dist))

    return align_loss + temperature * uniform_loss