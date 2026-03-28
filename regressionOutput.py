import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.prelu1 = nn.PReLU(num_parameters=out_channels)
        self.prelu2 = nn.PReLU(num_parameters=out_channels)

        # 快捷连接（Shortcut）：若通道数/步长变化，用1×1卷积对齐
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # 主路径：卷积 → BN → PReLU → 卷积 → BN
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 残差连接：主路径输出 + 快捷连接输入 → PReLU
        out += self.shortcut(x)
        out = self.prelu2(out)
        return out

# --------------------------
# 带残差连接的回归头
# --------------------------
class SignalRegressionHeadWithResidual(nn.Module):
    def __init__(self, in_channels=256, mid_channels=128):
        super().__init__()
        # 1. 初始降维卷积（256→mid_channels，保持时间维度280）
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.PReLU(num_parameters=mid_channels)
        )

        # 2. 共享残差特征提取（堆叠2个残差块，保持时间维度280）
        self.shared_residual = nn.Sequential(
            ResidualBlock1D(mid_channels, mid_channels),  # 残差块1：保持通道
            nn.Conv1d(mid_channels, mid_channels // 2, kernel_size=1),  # 降维：128→64
            nn.BatchNorm1d(mid_channels // 2),
            nn.PReLU(num_parameters=mid_channels // 2),
            ResidualBlock1D(mid_channels // 2, mid_channels // 2)  # 残差块2：保持通道
        )

        self.branch_1 = nn.Sequential(
            nn.Conv1d(mid_channels // 2, mid_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels // 4),
            nn.PReLU(num_parameters=mid_channels // 4),
            nn.Conv1d(mid_channels // 4, 1, kernel_size=1), # 最终压缩到1通道
            nn.Linear(210,210)
        )

        self.branch_2 = nn.Sequential(
            nn.Conv1d(mid_channels // 2, mid_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels // 4),
            nn.PReLU(num_parameters=mid_channels // 4),
            nn.Conv1d(mid_channels // 4, 1, kernel_size=1),  # 最终压缩到1通道
            nn.Linear(210, 210)
        )

    def forward(self, x):
        # 输入: (1, 256, 280)
        # 1. 初始降维
        x = self.init_conv(x)  # (1, 128, 280)

        # 2. 共享残差特征提取
        x = self.shared_residual(x)  # (1, 64, 280)

        # 3. 时间维度对齐：280→210（线性插值）
        x = F.interpolate(x, size=210, mode='linear', align_corners=False)  # (1, 64, 210)

        # 4. 双分支输出
        out_1 = self.branch_1(x)  # (1, 1, 210)
        out_2 = self.branch_2(x)  # (1, 1, 210)
        return out_1, out_2


if __name__ == "__main__":
    # 构造测试输入
    batch_size = 2
    # 模型输入要求：(batch_size, in_channels=256, 时间维度=280)
    test_input = torch.randn(batch_size, 256, 280)

    # 初始化模型
    model = SignalRegressionHeadWithResidual()

    # 切换到训练模式 (BN在训练和推理时行为不同)
    model.train()

    # 前向传播
    out1, out2 = model(test_input)

    # 打印输出形状，预期: 两个输出均为 torch.Size([2, 1, 210])
    print(f"分支1输出形状: {out1.shape}")
    print(f"分支2输出形状: {out2.shape}")

    # 校验形状是否正确
    expected_shape = torch.Size([batch_size, 1, 210])
    assert out1.shape == expected_shape, "分支1输出形状不符合预期！"
    assert out2.shape == expected_shape, "分支2输出形状不符合预期！"
