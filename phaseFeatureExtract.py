import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseFeatureExtract(nn.Module):
    def __init__(self):
        super().__init__()
        # ===================== 左侧2D卷积分支1 (输入: 1*33*9) =====================
        # 第一层卷积：1→4通道
        self.conv2d_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=4),
            nn.PReLU(num_parameters=4),
        )
        # 主路：4→16→64通道（最后一层不加激活，残差相加后再激活）
        self.branch_2d_1_main = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU(num_parameters=16),

            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
        )
        # 残差路：4→64通道 1x1卷积，匹配通道数
        self.branch_2d_1_res = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64)
        )
        # 残差相加后的激活
        self.prelu_2d_1 = nn.PReLU(num_parameters=64)

        # ===================== 左侧2D卷积分支2 (输入: 1*17*19) =====================
        # 主路：1→16→64通道（最后一层不加激活，残差相加后再激活）
        self.branch_2d_2_main = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU(num_parameters=16),

            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
        )
        # 残差路：1→64通道 1x1卷积，匹配通道数
        self.branch_2d_2_res = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64)
        )
        # 残差相加后的激活
        self.prelu_2d_2 = nn.PReLU(num_parameters=64)

        # ===================== 第一次concat后的2D卷积 (128→64通道) =====================
        self.conv_after_concat1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(num_parameters=64)
        )

        # ===================== 左侧2D卷积分支3 (输入: 1*9*36) =====================
        self.branch_2d_3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(num_parameters=64)
        )

        # ===================== 第二次concat后的2D卷积 (128→64→256通道) =====================
        self.conv_after_concat2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(num_parameters=64),

            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.PReLU(num_parameters=256)
        )

        # ===================== 展平后的线性层 (256*323 → 256*280) =====================
        self.linear_head = nn.Sequential(
            nn.Linear(in_features=323, out_features=280),
            nn.BatchNorm1d(num_features=256),
            nn.PReLU(num_parameters=256)
        )

        # ===================== 右侧1D卷积分支 (输入: 1*280) =====================
        # 第一层卷积：1→4通道
        self.conv1d_1_1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=4),
            nn.PReLU(num_parameters=4),
        )
        # 1D第一残差块主路：4→16→64通道（最后一层不加激活）
        self.branch_1d_1_main = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=16),
            nn.PReLU(num_parameters=16),

            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=64),
        )
        # 1D第一残差块残差路：4→64通道 1x1卷积
        self.branch_1d_1_res = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(num_features=64)
        )
        # 第一残差块相加后的激活
        self.prelu_1d_1 = nn.PReLU(num_parameters=64)

        # 1D第二残差块主路：64→128→256通道（最后一层不加激活）
        self.branch_1d_2_main = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=128),
            nn.PReLU(num_parameters=128),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=256),
        )
        # 1D第二残差块残差路：64→256通道 1x1卷积
        self.branch_1d_2_res = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(num_features=256)
        )
        # 第二残差块相加后的激活
        self.prelu_1d_2 = nn.PReLU(num_parameters=256)

        # ===================== 最终输出的1D卷积 (512→256通道) =====================
        self.final_conv = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.PReLU(num_parameters=256)
        )

    def forward(self, x1, x2, x3, x4):
        """
        前向传播（严格对齐架构图数据流）
        :param x1: 输入张量, shape=(batch_size, 1, 33, 9)
        :param x2: 输入张量, shape=(batch_size, 1, 17, 19)
        :param x3: 输入张量, shape=(batch_size, 1, 9, 36)
        :param x4: 输入张量, shape=(batch_size, 1, 280)
        :return: 输出张量, shape=(batch_size, 256, 280)
        """
        # --------------------- 2D分支1处理 + 残差连接 ---------------------
        out1_1 = self.conv2d_1_1(x1)  # 1*33*9 → 4*33*9
        out1_main = self.branch_2d_1_main(out1_1)  # 4*33*9 → 64*33*9
        out1_res = self.branch_2d_1_res(out1_1)  # 4*33*9 → 64*33*9 残差
        out1 = self.prelu_2d_1(out1_main + out1_res)  # 残差相加+激活
        out1 = F.interpolate(out1, size=(17, 19), mode='bilinear', align_corners=False)  # 64*33*9 → 64*17*19

        # --------------------- 2D分支2处理 + 残差连接 ---------------------
        out2_main = self.branch_2d_2_main(x2)  # 1*17*19 → 64*17*19
        out2_res = self.branch_2d_2_res(x2)  # 1*17*19 → 64*17*19 残差
        out2 = self.prelu_2d_2(out2_main + out2_res)  # 残差相加+激活 → 64*17*19

        # --------------------- 第一次通道拼接 + 卷积 ---------------------
        concat1 = torch.cat([out1, out2], dim=1)  # 64+64=128通道 → 128*17*19
        out_concat1 = self.conv_after_concat1(concat1)  # 128*17*19 → 64*17*19

        # --------------------- 2D分支3处理 + 尺寸对齐 ---------------------
        out3 = self.branch_2d_3(x3)  # 1*9*36 → 64*9*36
        out3 = F.interpolate(out3, size=(17, 19), mode='bilinear', align_corners=False)  # 64*9*36 → 64*17*19

        # --------------------- 第二次通道拼接 + 卷积 ---------------------
        concat2 = torch.cat([out_concat1, out3], dim=1)  # 64+64=128通道 → 128*17*19
        out_concat2 = self.conv_after_concat2(concat2)  # 128*17*19 → 256*17*19

        # --------------------- 展平空间维度 + 线性映射 ---------------------
        flatten_feat = torch.flatten(out_concat2, start_dim=2)  # 256*17*19 → 256*323
        linear_out = self.linear_head(flatten_feat)  # 256*323 → 256*280

        # --------------------- 1D分支处理 + 两级残差连接 ---------------------
        out_1d_1 = self.conv1d_1_1(x4)  # 1*280 → 4*280
        # 第一级残差
        out_1d_main1 = self.branch_1d_1_main(out_1d_1)  # 4*280 → 64*280
        out_1d_res1 = self.branch_1d_1_res(out_1d_1)  # 4*280 → 64*280 残差
        out_1d_2 = self.prelu_1d_1(out_1d_main1 + out_1d_res1)  # 残差相加+激活 → 64*280
        # 第二级残差
        out_1d_main2 = self.branch_1d_2_main(out_1d_2)  # 64*280 → 256*280
        out_1d_res2 = self.branch_1d_2_res(out_1d_2)  # 64*280 → 256*280 残差
        out_1d_final = self.prelu_1d_2(out_1d_main2 + out_1d_res2)  # 残差相加+激活 → 256*280

        # --------------------- 最终双分支拼接 + 输出卷积 ---------------------
        final_concat = torch.cat([linear_out, out_1d_final], dim=1)  # 256+256=512通道 → 512*280
        final_out = self.final_conv(final_concat)  # 512*280 → 256*280

        return final_out


# 维度测试代码
if __name__ == "__main__":
    # 构造测试输入
    batch_size = 2
    test_x1 = torch.randn(batch_size, 1, 33, 10)
    test_x2 = torch.randn(batch_size, 1, 17, 19)
    test_x3 = torch.randn(batch_size, 1, 9, 36)
    test_x4 = torch.randn(batch_size, 1, 280)

    # 初始化模型
    model = PhaseFeatureExtract()

    # 切换到训练模式 (BN在训练和推理时行为不同)
    model.train()

    # 前向传播
    output = model(test_x1, test_x2, test_x3, test_x4)

    # 打印输出形状，预期: torch.Size([2, 256, 280])
    print(f"模型输出形状: {output.shape}")
    # 校验形状是否正确
    assert output.shape == torch.Size([batch_size, 256, 280]), "输出形状不符合预期！"
    print("模型维度校验通过！")