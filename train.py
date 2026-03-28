from main import OneNet
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from loss import contrastive_loss
from collections import defaultdict

# ==================== 配置参数 ====================
baseline_type='avg'
GT_ROOT = r"/home/chenjz/rforal_result/gt"
INPUT_ROOT = rf"/home/chenjz/rforal_result/{baseline_type}_input"
BATCH_SIZE = 4  # 请根据显存调整
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
LR_DECAY_EPOCH = 25
LR_DECAY_GAMMA = 0.75

# ==================== 新增/修改的GPU配置 ====================
VISIBLE_GPUS = "0"  #修改这里指定GPU，例如单卡"0"，多卡"0,1,2"
# 设置环境变量，必须在第一次调用torch.cuda之前设置
os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPUS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================================================

LOG_DIR = f"./runs/{baseline_type}_steplr_1e-4_weightdecay_1e-5_epoch_100/log"
MODEL_SAVE_PATH = f"./runs/{baseline_type}_steplr_1e-4_weightdecay_1e-5_epoch_100/model/best_model.pth"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# ==================== 数据路径收集 ====================
def get_all_sample_pairs():
    """
    遍历目录，收集所有匹配的输入-GT样本对
    返回: 样本列表，每个元素是 (input_dir, gt_dir)
    """
    sample_pairs = []

    # 遍历GT目录结构
    for e_dir in os.listdir(GT_ROOT):
        e_path_gt = os.path.join(GT_ROOT, e_dir)
        e_path_input = os.path.join(INPUT_ROOT, e_dir)
        if not os.path.isdir(e_path_gt) or not os.path.isdir(e_path_input):
            continue

        for u_dir in os.listdir(e_path_gt):
            u_path_gt = os.path.join(e_path_gt, u_dir)
            u_path_input = os.path.join(e_path_input, u_dir)
            if not os.path.isdir(u_path_gt) or not os.path.isdir(u_path_input):
                continue

            # 遍历四级目录
            for l4_dir in os.listdir(u_path_gt):
                l4_path_gt = os.path.join(u_path_gt, l4_dir)
                l4_path_input = os.path.join(u_path_input, l4_dir)
                if not os.path.isdir(l4_path_gt) or not os.path.isdir(l4_path_input):
                    continue

                # 遍历五级目录（样本目录）
                for l5_dir in os.listdir(l4_path_gt):
                    l5_path_gt = os.path.join(l4_path_gt, l5_dir)
                    l5_path_input = os.path.join(l4_path_input, l5_dir)

                    # 检查GT文件是否存在
                    gt_height = os.path.join(l5_path_gt, "change_lip_height.csv")
                    gt_width = os.path.join(l5_path_gt, "change_lip_width.csv")
                    if not os.path.exists(gt_height) or not os.path.exists(gt_width):
                        continue

                    # 检查输入文件是否存在（简化检查，实际可检查全部24个文件）
                    if not os.path.isdir(l5_path_input):
                        continue

                    sample_pairs.append((l5_path_input, l5_path_gt))

    return sample_pairs


# ==================== 自定义Dataset ====================
class RforalDataset(Dataset):
    def __init__(self, sample_pairs):
        self.samples = sample_pairs

        # 定义需要加载的输入文件名列表
        self.input_files = {
            '280': [
                f'abs_0_{baseline_type}_diff.csv', f'abs_1_{baseline_type}_diff.csv', f'abs_2_{baseline_type}_diff.csv',
                f'angle_0_{baseline_type}_diff.csv', f'angle_1_{baseline_type}_diff.csv', f'angle_2_{baseline_type}_diff.csv'
            ],
            '9x36': [
                f'stft_magnitude_abs_0_{baseline_type}_diff_16.csv', f'stft_magnitude_abs_1_{baseline_type}_diff_16.csv',
                f'stft_magnitude_abs_2_{baseline_type}_diff_16.csv', f'stft_magnitude_angle_0_{baseline_type}_diff_16.csv',
                f'stft_magnitude_angle_1_{baseline_type}_diff_16.csv', f'stft_magnitude_angle_2_{baseline_type}_diff_16.csv'
            ],
            '17x19': [
                f'stft_magnitude_abs_0_{baseline_type}_diff_32.csv', f'stft_magnitude_abs_1_{baseline_type}_diff_32.csv',
                f'stft_magnitude_abs_2_{baseline_type}_diff_32.csv', f'stft_magnitude_angle_0_{baseline_type}_diff_32.csv',
                f'stft_magnitude_angle_1_{baseline_type}_diff_32.csv', f'stft_magnitude_angle_2_{baseline_type}_diff_32.csv'
            ],
            '33x10': [
                f'stft_magnitude_abs_0_{baseline_type}_diff_64.csv', f'stft_magnitude_abs_1_{baseline_type}_diff_64.csv',
                f'stft_magnitude_abs_2_{baseline_type}_diff_64.csv', f'stft_magnitude_angle_0_{baseline_type}_diff_64.csv',
                f'stft_magnitude_angle_1_{baseline_type}_diff_64.csv', f'stft_magnitude_angle_2_{baseline_type}_diff_64.csv'
            ]
        }

        self.gt_files = ['change_lip_height.csv', 'change_lip_width.csv']

    def __len__(self):
        return len(self.samples)

    def _load_csv(self, filepath, expected_shape):
        """加载CSV文件，跳过第一行，返回numpy数组"""
        df = pd.read_csv(filepath, skiprows=1, header=None)
        data = df.values.astype(np.float32)
        # 确保形状正确
        if data.shape != expected_shape:
            raise ValueError(f"文件 {filepath} 形状错误，期望 {expected_shape}，实际 {data.shape}")
        return data

    def __getitem__(self, idx):
        input_dir, gt_dir = self.samples[idx]

        # 1. 加载输入数据
        inputs = []
        inputs_new = []
        # 加载6个 (280,) -> reshape为 (1, 280)
        for fname in self.input_files['280']:
            fpath = os.path.join(input_dir, fname)
            data = self._load_csv(fpath, (280, 1))  # CSV是281行1列，skiprows=1后是280行1列
            data = data.flatten()  # 变成(280,)
            data = torch.tensor(data).unsqueeze(0)  # (1, 280)
            inputs.append(data)

        # 加载6个 (9, 36) -> reshape为 (1, 9, 36)
        for fname in self.input_files['9x36']:
            fpath = os.path.join(input_dir, fname)
            data = self._load_csv(fpath, (9, 36))  # 10行36列 -> 9行36列
            data = torch.tensor(data).unsqueeze(0)  # (1, 9, 36)
            inputs.append(data)

        # 加载6个 (17, 19) -> reshape为 (1, 17, 19)
        for fname in self.input_files['17x19']:
            fpath = os.path.join(input_dir, fname)
            data = self._load_csv(fpath, (17, 19))  # 18行19列 -> 17行19列
            data = torch.tensor(data).unsqueeze(0)  # (1, 17, 19)
            inputs.append(data)

        # 加载6个 (33, 10) -> reshape为 (1, 33, 10)
        for fname in self.input_files['33x10']:
            fpath = os.path.join(input_dir, fname)
            data = self._load_csv(fpath, (33, 10))  # 34行10列 -> 33行10列
            data = torch.tensor(data).unsqueeze(0)  # (1, 33, 10)
            inputs.append(data)

        inputs_new.append(inputs[21])
        inputs_new.append(inputs[15])
        inputs_new.append(inputs[9])
        inputs_new.append(inputs[3])
        inputs_new.append(inputs[18])
        inputs_new.append(inputs[12])
        inputs_new.append(inputs[6])
        inputs_new.append(inputs[0])
        inputs_new.append(inputs[22])
        inputs_new.append(inputs[16])
        inputs_new.append(inputs[10])
        inputs_new.append(inputs[4])
        inputs_new.append(inputs[19])
        inputs_new.append(inputs[13])
        inputs_new.append(inputs[7])
        inputs_new.append(inputs[1])
        inputs_new.append(inputs[23])
        inputs_new.append(inputs[17])
        inputs_new.append(inputs[11])
        inputs_new.append(inputs[5])
        inputs_new.append(inputs[20])
        inputs_new.append(inputs[14])
        inputs_new.append(inputs[8])
        inputs_new.append(inputs[2])

        # 2. 加载GT数据
        gts = []
        for fname in self.gt_files:
            fpath = os.path.join(gt_dir, fname)
            data = self._load_csv(fpath, (210, 1))  # 211行1列 -> 210行1列
            data = data.flatten()  # (210,)
            data = torch.tensor(data).unsqueeze(0)  # (1, 210)
            gts.append(data)

        # 返回: (inputs_list, gts_list)
        # inputs_list长度为24，gts_list长度为2
        return inputs_new, gts


# ==================== 训练/验证/测试函数 ====================
# ==================== 【修改】训练函数：新增分项loss记录 ====================
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0.0
    # 新增：分项损失累加
    total_loss1 = 0.0  # 高度损失
    total_loss2 = 0.0  # 宽度损失
    total_loss_feat = 0.0  # 对比损失

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
    for inputs, gts in pbar:
        batch_size = inputs[0].size(0)
        # 将数据移动到设备
        inputs = [x.to(device) for x in inputs]
        gts = [x.to(device) for x in gts]

        # 前向传播
        output1, output2, feature1, feature2, feature3 = model(*inputs)
        loss_feature = contrastive_loss(feature1, feature2, feature3)
        loss1 = criterion(output1, gts[0])
        loss2 = criterion(output2, gts[1])
        loss = (loss1 + loss2) * 0.5 + loss_feature * 0.5

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累加总损失和分项损失
        total_loss += loss.item() * batch_size
        total_loss1 += loss1.item() * batch_size
        total_loss2 += loss2.item() * batch_size
        total_loss_feat += loss_feature.item() * batch_size

        pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'L1':f"{loss1.item():.4f}", 'L2':f"{loss2.item():.4f}", 'Feat':f"{loss_feature.item():.4f}"})

    # 计算平均损失
    num_samples = len(dataloader.dataset)
    avg_loss = total_loss / num_samples
    avg_loss1 = total_loss1 / num_samples
    avg_loss2 = total_loss2 / num_samples
    avg_loss_feat = total_loss_feat / num_samples

    # 写入TensorBoard
    writer.add_scalar('Loss/Train/total_loss', avg_loss, epoch)
    writer.add_scalar('Loss/Train/loss1_height', avg_loss1, epoch)
    writer.add_scalar('Loss/Train/loss2_width', avg_loss2, epoch)
    writer.add_scalar('Loss/Train/loss_feature', avg_loss_feat, epoch)
    return avg_loss, avg_loss1, avg_loss2, avg_loss_feat


# ==================== 【修改】验证函数：新增分项loss记录 ====================
def validate(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0.0
    # 新增：分项损失累加
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_loss_feat = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]")
        for inputs, gts in pbar:
            batch_size = inputs[0].size(0)
            inputs = [x.to(device) for x in inputs]
            gts = [x.to(device) for x in gts]

            output1, output2, feature1, feature2, feature3 = model(*inputs)
            loss_feature = contrastive_loss(feature1, feature2, feature3)
            loss1 = criterion(output1, gts[0])
            loss2 = criterion(output2, gts[1])
            loss = (loss1 + loss2) * 0.5 + loss_feature * 0.5

            # 累加损失
            total_loss += loss.item() * batch_size
            total_loss1 += loss1.item() * batch_size
            total_loss2 += loss2.item() * batch_size
            total_loss_feat += loss_feature.item() * batch_size

            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'L1':f"{loss1.item():.4f}", 'L2':f"{loss2.item():.4f}", 'Feat':f"{loss_feature.item():.4f}"})

    # 计算平均损失
    num_samples = len(dataloader.dataset)
    avg_loss = total_loss / num_samples
    avg_loss1 = total_loss1 / num_samples
    avg_loss2 = total_loss2 / num_samples
    avg_loss_feat = total_loss_feat / num_samples

    # 写入TensorBoard（核心需求：验证阶段分项loss）
    writer.add_scalar('Loss/Val/total_loss', avg_loss, epoch)
    writer.add_scalar('Loss/Val/loss1_height', avg_loss1, epoch)
    writer.add_scalar('Loss/Val/loss2_width', avg_loss2, epoch)
    writer.add_scalar('Loss/Val/loss_feature', avg_loss_feat, epoch)
    return avg_loss, avg_loss1, avg_loss2, avg_loss_feat


# ==================== 【修改】测试函数：新增分项loss记录+打印 ====================
def test(model, dataloader, criterion, device, writer):
    model.eval()
    total_loss = 0.0
    # 新增：分项损失累加
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_loss_feat = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="[Test]")
        for inputs, gts in pbar:
            batch_size = inputs[0].size(0)
            inputs = [x.to(device) for x in inputs]
            gts = [x.to(device) for x in gts]

            output1, output2, feature1, feature2, feature3 = model(*inputs)
            loss_feature = contrastive_loss(feature1, feature2, feature3)
            loss1 = criterion(output1, gts[0])
            loss2 = criterion(output2, gts[1])
            loss = (loss1 + loss2) * 0.5 + loss_feature * 0.5

            # 累加损失
            total_loss += loss.item() * batch_size
            total_loss1 += loss1.item() * batch_size
            total_loss2 += loss2.item() * batch_size
            total_loss_feat += loss_feature.item() * batch_size

            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'L1':f"{loss1.item():.4f}", 'L2':f"{loss2.item():.4f}", 'Feat':f"{loss_feature.item():.4f}"})

    # 计算平均损失
    num_samples = len(dataloader.dataset)
    avg_loss = total_loss / num_samples
    avg_loss1 = total_loss1 / num_samples
    avg_loss2 = total_loss2 / num_samples
    avg_loss_feat = total_loss_feat / num_samples

    # 写入TensorBoard（核心需求：测试阶段分项loss）
    writer.add_scalar('Loss/Test/total_loss', avg_loss)
    writer.add_scalar('Loss/Test/loss1_height', avg_loss1)
    writer.add_scalar('Loss/Test/loss2_width', avg_loss2)
    writer.add_scalar('Loss/Test/loss_feature', avg_loss_feat)

    # 打印分项损失
    print(f"\n【测试结果】")
    print(f"总损失: {avg_loss:.6f}")
    print(f"高度损失(loss1): {avg_loss1:.6f}")
    print(f"宽度损失(loss2): {avg_loss2:.6f}")
    print(f"对比损失(loss_feature): {avg_loss_feat:.6f}")
    return avg_loss, avg_loss1, avg_loss2, avg_loss_feat


# ==================== 主函数 ====================
def main():
    # ==================== 新增：硬件环境检查 ====================
    print("=" * 60)
    print("系统环境检查")
    print("=" * 60)

    if torch.cuda.is_available():
        available_gpu_count = torch.cuda.device_count()
        print(f"[成功] CUDA 可用。")
        print(f"[信息] 您指定的物理GPU ID: {VISIBLE_GPUS}")
        print(f"[信息] 当前程序可见的GPU数量: {available_gpu_count}")

        # 列出所有可见GPU的详细信息
        for i in range(available_gpu_count):
            print(f"  -> [GPU {i}] {torch.cuda.get_device_name(i)}")

    else:
        print(f"[警告] CUDA不可用，将使用CPU进行训练，速度可能较慢。")

    print(f"[信息] 最终计算设备: {DEVICE}")
    print("=" * 60 + "\n")
    # ==========================================================

    # 1. 收集所有样本
    print("正在收集样本路径...")
    all_samples = get_all_sample_pairs()
    print(f"共找到 {len(all_samples)} 个样本对")

    # 2. 划分数据集 (7:1:2)
    train_val_samples, test_samples = train_test_split(all_samples, test_size=0.2, random_state=42)
    train_samples, val_samples = train_test_split(train_val_samples, test_size=1 / 8,
                                                  random_state=42)  # 0.8 * 0.125 = 0.1

    print(f"训练集: {len(train_samples)}, 验证集: {len(val_samples)}, 测试集: {len(test_samples)}")

    # 3. 创建Dataset和DataLoader
    train_dataset = RforalDataset(train_samples)
    val_dataset = RforalDataset(val_samples)
    test_dataset = RforalDataset(test_samples)

    # 注意：由于每个样本返回的是列表，需要使用默认的collate_fn或者确保能正确batch
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 4. 初始化模型
    print("初始化模型...")
    model = OneNet()

    # 5. 多卡并行 (修改了此处的日志输出)
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        print(f"[信息] 检测到 {gpu_count} 张可见GPU，启动DataParallel并行模式。")
        model = nn.DataParallel(model)
    elif gpu_count == 1:
        print(f"[信息] 使用单张GPU进行训练。")

    model = model.to(DEVICE)

    # 6. 定义损失函数、优化器、学习率调度器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_EPOCH, gamma=LR_DECAY_GAMMA)

    # 7. 初始化TensorBoard和早停
    writer = SummaryWriter(LOG_DIR)

    # 8. 训练循环
    best_val_loss = float('inf')

    print("开始训练...")
    for epoch in range(EPOCHS):
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # 训练和验证（接收分项loss）
        train_loss, train_l1, train_l2, train_feat = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch, writer)
        val_loss, val_l1, val_l2, val_feat = validate(model, val_loader, criterion, DEVICE, epoch, writer)

        # 更新学习率
        scheduler.step()

        # 打印分项损失
        print(
            f"Epoch {epoch + 1} Summary: \n"
            f"Train | Total:{train_loss:.6f} | L1:{train_l1:.6f} | L2:{train_l2:.6f} | Feat:{train_feat:.6f}\n"
            f"Val   | Total:{val_loss:.6f} | L1:{val_l1:.6f} | L2:{val_l2:.6f} | Feat:{val_feat:.6f}\n"
            f"LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, MODEL_SAVE_PATH)
            print(f"  -> 保存最佳模型 (Val Total Loss: {val_loss:.6f})")

    # 9. 测试
    print("\n训练完成，开始测试...")
    # 加载最佳模型
    checkpoint = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载 Epoch {checkpoint['epoch'] + 1} 的最佳模型 (Val Loss: {checkpoint['val_loss']:.6f})")

    test(model, test_loader, criterion, DEVICE, writer)

    writer.close()
    print("全部完成！")


if __name__ == "__main__":
    main()