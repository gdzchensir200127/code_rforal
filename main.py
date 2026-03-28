import torch
import torch.nn as nn
import torch.nn.functional as F

from amplitudeFeatureExtract import AmpFeatureExtract
from phaseFeatureExtract import PhaseFeatureExtract
from amplitudePhaseFusion import AmpPhaseFusion
from multiBinFusion import ContrastiveFusionModule
from regressionOutput import SignalRegressionHeadWithResidual


class OneNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 相位特征提取器
        self.phaseFeatureExtractBinOne = PhaseFeatureExtract()
        self.phaseFeatureExtractBinTwo = PhaseFeatureExtract()
        self.phaseFeatureExtractBinThree = PhaseFeatureExtract()

        # 幅度特征提取器
        self.ampFeatureExtractBinOne = AmpFeatureExtract()
        self.ampFeatureExtractBinTwo = AmpFeatureExtract()
        self.ampFeatureExtractBinThree = AmpFeatureExtract()

        # 幅相融合模块
        self.amplitudePhaseFusionBinOne = AmpPhaseFusion()
        self.amplitudePhaseFusionBinTwo = AmpPhaseFusion()
        self.amplitudePhaseFusionBinThree = AmpPhaseFusion()

        # 多频点融合模块
        self.multiBinFusion = ContrastiveFusionModule()

        # 回归输出头
        self.regressionOutput = SignalRegressionHeadWithResidual()

    def forward(self,
                # Bin1 相位输入 (4个)
                phaseBinOneSTFT1,phaseBinOneSTFT2,phaseBinOneSTFT3,phaseBinOne,
                # Bin1 幅度输入 (4个)
                ampBinOneSTFT1,ampBinOneSTFT2,ampBinOneSTFT3,ampBinOne,
                # Bin2 相位输入 (4个)
                phaseBinTwoSTFT1,phaseBinTwoSTFT2,phaseBinTwoSTFT3,phaseBinTwo,
                # Bin2 幅度输入 (4个)
                ampBinTwoSTFT1,ampBinTwoSTFT2,ampBinTwoSTFT3,ampBinTwo,
                # Bin3 相位输入 (4个)
                phaseBinThreeSTFT1,phaseBinThreeSTFT2,phaseBinThreeSTFT3,phaseBinThree,
                # Bin3 幅度输入 (4个)
                ampBinThreeSTFT1,ampBinThreeSTFT2,ampBinThreeSTFT3,ampBinThree):

        # Bin1 特征提取 + 幅相融合
        phaseBinOneOutput = self.phaseFeatureExtractBinOne(phaseBinOneSTFT1,phaseBinOneSTFT2,phaseBinOneSTFT3,phaseBinOne)
        ampBinOneOutput = self.ampFeatureExtractBinOne(ampBinOneSTFT1,ampBinOneSTFT2,ampBinOneSTFT3,ampBinOne)
        binOneOutput = self.amplitudePhaseFusionBinOne(ampBinOneOutput, phaseBinOneOutput)

        # Bin2 特征提取 + 幅相融合
        phaseBinTwoOutput = self.phaseFeatureExtractBinTwo(phaseBinTwoSTFT1,phaseBinTwoSTFT2,phaseBinTwoSTFT3,phaseBinTwo)
        ampBinTwoOutput = self.ampFeatureExtractBinTwo(ampBinTwoSTFT1,ampBinTwoSTFT2,ampBinTwoSTFT3,ampBinTwo)
        binTwoOutput = self.amplitudePhaseFusionBinTwo(ampBinTwoOutput, phaseBinTwoOutput)

        # Bin3 特征提取 + 幅相融合
        phaseBinThreeOutput = self.phaseFeatureExtractBinThree(phaseBinThreeSTFT1,phaseBinThreeSTFT2,phaseBinThreeSTFT3,phaseBinThree)
        ampBinThreeOutput = self.ampFeatureExtractBinThree(ampBinThreeSTFT1,ampBinThreeSTFT2,ampBinThreeSTFT3,ampBinThree)
        binThreeOutput = self.amplitudePhaseFusionBinThree(ampBinThreeOutput, phaseBinThreeOutput)

        # 多频点对比融合
        binFusionOutput, binOneFeature, binTwoFeature, binThreeFeature = self.multiBinFusion(binOneOutput, binTwoOutput, binThreeOutput)

        # 最终回归输出
        finalOut1, finalOut2 = self.regressionOutput(binFusionOutput)
        return finalOut1, finalOut2, binOneFeature, binTwoFeature, binThreeFeature

def get_model_parameters(model: nn.Module) -> tuple:
    """
    计算模型总参数量和可训练参数量
    :param model: PyTorch模型
    :return: (总参数量, 可训练参数量)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

if __name__ == "__main__":
    # 构造测试输入
    batch_size = 4
    # 统一构造24个测试输入（对应forward的24个参数）
    test_inputs = [
        torch.randn(batch_size, 1, 33, 10),
        torch.randn(batch_size, 1, 17, 19),
        torch.randn(batch_size, 1, 9, 36),
        torch.randn(batch_size, 1, 280),
    ] * 6  # 3个bin × 相位4+幅度4 = 24个输入

    # 初始化模型
    model = OneNet()

    # ===================== 打印模型参数量 =====================
    total_params, trainable_params = get_model_parameters(model)
    print("="*50)
    print(f"模型总参数量: {total_params:,}")
    print(f"模型可训练参数量: {trainable_params:,}")
    print("="*50)

    # 切换到训练模式
    model.train()

    # 前向传播
    out1, out2, feature1, feature2, feature3 = model(*test_inputs)

    # 打印输出形状
    print(f"分支1输出形状: {out1.shape}")
    print(f"分支2输出形状: {out2.shape}")
    print(f"特征1输出形状: {feature1.shape}")
    print(f"特征2输出形状: {feature2.shape}")
    print(f"特征3输出形状: {feature3.shape}")

    # 形状校验
    expected_shape_1 = torch.Size([batch_size, 1, 210])
    expected_shape_2 = torch.Size([batch_size, 256, 280])
    assert out1.shape == expected_shape_1, "分支1输出形状不符合预期！"
    assert out2.shape == expected_shape_1, "分支2输出形状不符合预期！"
    assert feature1.shape == expected_shape_2, "特征1输出形状不符合预期！"
    assert feature2.shape == expected_shape_2, "特征2输出形状不符合预期！"
    assert feature3.shape == expected_shape_2, "特征3输出形状不符合预期！"

    print("\n✅ 所有输出形状校验通过！")