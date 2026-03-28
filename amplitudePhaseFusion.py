import torch
import torch.nn as nn
import torch.nn.functional as F


class AmpPhaseFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # ===================== concat后卷积 (输入: 512*280) =====================
        self.amp_phase_conv = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.PReLU(num_parameters=256),


            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.PReLU(num_parameters=256),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.PReLU(num_parameters=256),
        )

    def forward(self, x1, x2):
        """
        前向传播
        :param x1: 输入张量, shape=(batch_size, 256, 280)
        :param x2: 输入张量, shape=(batch_size, 256, 280)
        :return: 输出张量, shape=(batch_size, 256, 280)
        """
        # 通道拼接 + 卷积
        concat1 = torch.cat([x1, x2], dim=1)
        final_out = self.amp_phase_conv(concat1)

        return final_out


# 维度测试代码
if __name__ == "__main__":
    # 构造测试输入
    batch_size = 2
    test_x1 = torch.randn(batch_size, 256, 280)
    test_x2 = torch.randn(batch_size, 256, 280)


    # 初始化模型
    model = AmpPhaseFusion()

    # 切换到训练模式 (BN在训练和推理时行为不同)
    model.train()

    # 前向传播
    output = model(test_x1, test_x2)

    # 打印输出形状，预期: torch.Size([2, 256, 280])
    print(f"模型输出形状: {output.shape}")