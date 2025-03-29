import os
import argparse
import logging

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets.dataset import setup_data
from models.lenet5 import LeNet5, weight_init

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Train LeNet5 on MNIST")
    parser.add_argument("--resume", action="store_true", help="Resume training from a checkpoint")
    parser.add_argument("--ckpt", type=str, default="save_checkpoints/LeNet5.pth", help="Path to checkpoint file")
    parser.add_argument("--start_epoch", type=int, default=0, help="Epoch to resume training from (0-indexed)")
    parser.add_argument("--total_epochs", type=int, default=100, help="Total number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id to use (default: 0)")
    parser.add_argument("--data_dir", type=str, default="/data1/nliu/2025_homework/Pro_001_MNIST/data/MNIST/", help="Path to MNIST data directory")
    return parser.parse_args()

# 设置日志，格式只显示级别和消息
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

def train_epoch(model, train_loader, criterion, optimizer, epoch, use_gpu):
    """
    单个 epoch 的训练过程
    """
    model.train()
    train_loss_total = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for data, target in train_bar:
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss_total += loss.item() * data.size(0)
        lr_current = optimizer.param_groups[0]['lr']
        train_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_current:.2e}")
    avg_train_loss = train_loss_total / len(train_loader.dataset)
    return avg_train_loss

def test_epoch(model, test_loader, criterion, epoch, use_gpu):
    """
    单个 epoch 的验证过程
    """
    model.eval()
    test_loss_total = 0
    correct = 0
    test_bar = tqdm(test_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
    with torch.no_grad():
        for data, target in test_bar:
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            data = Variable(data)
            target = Variable(target.long())
            output = model(data)
            loss = criterion(output, target)
            test_loss_total += loss.item() * data.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            test_bar.set_postfix(loss=f"{loss.item():.4f}")
    avg_test_loss = test_loss_total / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return avg_test_loss, accuracy

def main():
    args = parse_args()
    total_epochs = args.total_epochs
    start_epoch = args.start_epoch
    batch_size = args.batch_size
    lr = args.lr

    # 判断 GPU 可用性，并设置指定 GPU
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.cuda.set_device(args.gpu)
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_gpu else {}

    # 数据加载（通过 data_dir 参数指定数据路径）
    train_loader, test_loader = setup_data(batch_size, kwargs, args.data_dir)

    # 数据验证：检查样本数量与 batch 数是否匹配预期
    train_samples = len(train_loader.dataset)
    test_samples = len(test_loader.dataset)
    train_batches = len(train_loader)
    test_batches = len(test_loader)
    logger.info("Data Verification:")
    logger.info(f"Train samples: {train_samples} (Batches: {train_batches}, Expected: {batch_size}*{train_batches} = {batch_size * train_batches})")
    logger.info(f"Test samples: {test_samples} (Batches: {test_batches}, Expected: {batch_size}*{test_batches} = {batch_size * test_batches})")

    # 模型定义与初始化
    model = LeNet5()
    if use_gpu:
        model = model.cuda()
    model.apply(weight_init)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction='sum')

    # 断点续训处理
    if args.resume:
        if os.path.exists(args.ckpt):
            model.load_state_dict(torch.load(args.ckpt))
            logger.info(f"Resuming training from checkpoint {args.ckpt} at epoch {start_epoch}")
        else:
            logger.error(f"Checkpoint file {args.ckpt} not found. Exiting.")
            exit(1)
    else:
        logger.info("Starting training from scratch.")

    # 用于记录训练过程中的指标
    train_loss_history = []
    test_loss_history = []
    accuracy_history = []

    for epoch in range(start_epoch, total_epochs):
        logger.info(f"Epoch {epoch+1}/{total_epochs}")
        logger.info("Start Train")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, use_gpu)
        logger.info("Finish Train")
        logger.info(f"Train Loss: {train_loss:.6f}")
        train_loss_history.append(train_loss)

        logger.info("Start Validation")
        test_loss, accuracy = test_epoch(model, test_loader, criterion, epoch, use_gpu)
        logger.info("Finish Validation")
        logger.info(f"Test Loss: {test_loss:.6f}, Accuracy: {accuracy:.2f}%")
        test_loss_history.append(test_loss)
        accuracy_history.append(accuracy)

        # 保存当前 epoch 的模型至 save_checkpoints 目录下
        ckpt_name = os.path.join("save_checkpoints", f"LeNet5_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_name)
        logger.info(f"Checkpoint saved: {ckpt_name}")

    # 加载最终模型并测试最终准确率
    final_ckpt = os.path.join("save_checkpoints", f"LeNet5_epoch_{total_epochs}.pth")
    model.load_state_dict(torch.load(final_ckpt))
    final_test_loss, final_accuracy = test_epoch(model, test_loader, criterion, total_epochs, use_gpu)
    logger.info(f"Final Test Accuracy: {final_accuracy:.2f}%")

    # 绘制并保存 loss 曲线到 logs 目录下
    plt.figure(figsize=(10, 6))
    epochs = list(range(start_epoch+1, total_epochs+1))
    plt.plot(epochs, train_loss_history, label="Train Loss", marker='o')
    plt.plot(epochs, test_loss_history, label="Test Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    loss_fig_path = os.path.join("logs", "loss_curve.png")
    plt.savefig(loss_fig_path)
    logger.info(f"Loss curve saved: {loss_fig_path}")

if __name__ == '__main__':
    main()
