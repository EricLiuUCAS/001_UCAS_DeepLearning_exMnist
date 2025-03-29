# Project Name

## 项目简介

本项目基于 PyTorch 实现了 MNIST 数据集的训练框架，包含数据加载、模型定义、训练、测试、日志记录以及模型 checkpoint 保存等功能。

## 项目结构

project_name
│
├── data
│   └── MNIST
│       ├── train-images-idx3-ubyte.gz
│       ├── train-labels-idx1-ubyte.gz
│       ├── t10k-images-idx3-ubyte.gz
│       └── t10k-labels-idx1-ubyte.gz
│
├── logs
│   └── (日志文件、loss 曲线图片等)
│
├── save_checkpoints
│   └── (模型检查点文件，如 LeNet5_epoch_X.pth)
│
├── src
│   ├── datasets
│   │   └── dataset.py         # 数据加载与预处理代码
│   │
│   ├── models
│   │   └── lenet5.py          # 模型结构定义，如 LeNet5
│   │
│   └── train.py               # 主训练脚本（包含参数解析、训练、测试、保存日志等）
│
├── requirements.txt           # Python 依赖列表
└── README.md                  # 项目说明及使用文档


## 使用方法

- 安装依赖：
pip install -r requirements.txt
- 从头训练：
python src/train.py --total_epochs 100 --batch_size 32 --lr 0.001 --gpu 0
- 断点续训（例如从第 50 个 epoch 续训）：
python src/train.py --resume --ckpt save_checkpoints/LeNet5_epoch_50.pth --start_epoch 50 --total_epochs 100

## 说明

训练过程中生成的日志文件保存在 `logs/` 目录下，模型检查点保存在 `save_checkpoints/` 目录下，最终的 loss 曲线图保存在 `logs/loss_curve.png`。
