# -*- coding: utf-8 -*-
"""
@Time ： 2025/3/29 17:01
@Auth ： nliu
@File ：setup.py
@IDE ：PyCharm
@Motto：冲顶会！！！！
from setuptools import setup, find_packages

setup(
    name="mnist_training",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
        "matplotlib"
    ],
)
