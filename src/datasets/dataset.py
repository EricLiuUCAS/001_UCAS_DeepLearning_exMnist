import os
import struct
import gzip
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def dataread(image, label, data_dir):
    """
    读取 MNIST 数据（gzip 格式）。
    :param image: 图像文件名
    :param label: 标签文件名
    :param data_dir: 数据存放目录
    :return: (imgs, labels)
    """
    with gzip.open(os.path.join(data_dir, label), 'rb') as f_label:
        magic, num = struct.unpack('>II', f_label.read(8))
        labels = np.frombuffer(f_label.read(), dtype=np.uint8)
    with gzip.open(os.path.join(data_dir, image), 'rb') as f_img:
        magic, num, rows, cols = struct.unpack('>IIII', f_img.read(16))
        imgs = np.frombuffer(f_img.read(), dtype=np.uint8).reshape(len(labels), rows, cols)
    return imgs, labels

def get_data(data_dir):
    """
    获取训练和测试数据
    :param data_dir: 数据存放目录（例如 './data/MNIST/'）
    :return: (train_img, train_label, test_img, test_label)
    """
    train_img, train_label = dataread('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', data_dir)
    test_img, test_label = dataread('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz', data_dir)
    return train_img, train_label, test_img, test_label

def setup_data(batch_size, kwargs, data_dir):
    """
    构建 DataLoader，转换 NumPy 数组时调用 .copy() 避免只读警告。
    :param batch_size: 批次大小
    :param kwargs: DataLoader 参数
    :param data_dir: 数据存放目录
    :return: (train_loader, test_loader)
    """
    train_img, train_label, test_img, test_label = get_data(data_dir)
    train_x = torch.from_numpy(train_img.copy().reshape(-1, 1, 28, 28)).float()
    train_y = torch.from_numpy(train_label.astype(int))
    test_x = torch.from_numpy(test_img.copy().reshape(-1, 1, 28, 28)).float()
    test_y = torch.from_numpy(test_label.astype(int))
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader
