import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import contrastive_loss

class SharedFeatureEncoder(nn.Module):
    """共享权重的特征编码器，保持输入输出形状不变"""

    def __init__(self, input_channels=256, kernel_size=3, padding=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(input_channels),
            nn.PReLU(num_parameters=input_channels),
            nn.Conv1d(input_channels, input_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(input_channels)
        )
        self.residual_prelu = nn.PReLU(num_parameters=input_channels)

    def forward(self, x):
        # x: (1, 256, 280) -> output: (1, 256, 280)
        out = self.encoder(x)
        out = out + x  # 残差连接
        out = self.residual_prelu(out)
        return out


class ContrastiveFusionModule(nn.Module):
    """对比学习融合模块：提取三路共有特征"""

    def __init__(self, input_channels=256):
        super().__init__()
        self.shared_encoder = SharedFeatureEncoder(input_channels)
        # 跨路注意力机制：学习三路特征的自适应融合权重
        self.cross_attention = nn.Sequential(
            nn.Conv1d(input_channels * 3, input_channels, kernel_size=1),  # 压缩通道
            nn.Sigmoid()  # 生成0-1的注意力权重
        )

    def forward(self, x1, x2, x3):
        # 1. 共享编码：三路输入映射到同一特征空间
        feature_1 = self.shared_encoder(x1)  # (1, 256, 280)
        feature_2 = self.shared_encoder(x2)
        feature_3 = self.shared_encoder(x3)

        # 2. 跨路注意力融合：计算共有特征
        concat_features = torch.cat([feature_1, feature_2, feature_3], dim=1)  # (1, 768, 280)
        attention_weights = self.cross_attention(concat_features)  # (1, 256, 280)

        # 3. 加权融合：保留三路一致的共有特征
        fused_mean = (feature_1 + feature_2 + feature_3) / 3  # 基础融合
        commonfeature = fused_mean * attention_weights  # 注意力增强

        return commonfeature, feature_1, feature_2, feature_3

if __name__ == "__main__":
    # 1. 配置测试超参数（与模型定义的通道/序列长度一致）
    batch_size = 2  # 批量大小
    channels = 256  # 特征通道数
    seq_len = 280  # 序列长度

    # 2. 构造测试输入（Conv1d输入格式：[batch, channels, sequence_length]）
    test_x1 = torch.randn(batch_size, channels, seq_len)
    test_x2 = torch.randn(batch_size, channels, seq_len)
    test_x3 = torch.randn(batch_size, channels, seq_len)

    # 3. 初始化模型
    model = ContrastiveFusionModule(input_channels=channels)

    # 4. 切换到训练模式 (BN层训练/推理行为不同)
    model.train()

    # 5. 前向传播
    common_feature, f1, f2, f3 = model(test_x1, test_x2, test_x3)

    # 6. 打印输出形状
    print(f"融合共有特征形状: {common_feature.shape}")
    print(f"单路编码特征形状: {f1.shape}")
    print(f"单路编码特征形状: {f2.shape}")
    print(f"单路编码特征形状: {f3.shape}")

    # 7. 形状校验（断言验证，不符合会直接报错）
    expected_shape = torch.Size([batch_size, channels, seq_len])
    assert common_feature.shape == expected_shape, "融合特征形状不符合预期！"
    assert f1.shape == expected_shape, "单路特征f1形状错误！"
    assert f2.shape == expected_shape, "单路特征f2形状错误！"
    assert f3.shape == expected_shape, "单路特征f3形状错误！"
    print("✅ 所有输出形状校验通过！")

    # 8. 测试对比损失函数（验证损失计算正常）
    loss = contrastive_loss(f1, f2, f3)
    print(f"✅ 对比损失计算完成，损失值: {loss.item():.4f}")

    # 9. 额外验证：输出无NaN/Inf（模型训练必备）
    assert not torch.isnan(common_feature).any(), "输出存在NaN值！"
    assert not torch.isinf(common_feature).any(), "输出存在Inf值！"
    print("✅ 输出数值合法性校验通过！")
