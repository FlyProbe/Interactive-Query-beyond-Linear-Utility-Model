import numpy as np
import time
import scipy
import os
import logging
from datetime import datetime
from scipy.optimize import linprog
import warnings
import torch  # 引入 PyTorch

from tqdm import tqdm
import time


def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"experiment_{timestamp}.log")

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    stream_handler = logging.StreamHandler()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            file_handler,
            stream_handler
        ],
        force=True
    )
    logging.info(f"日志文件已创建: {log_filename}")

def get_weighted_distance_sq(D, e, u):
    """计算数据集 D 中所有点到期望点 e 的加权距离平方 (支持多维广播)"""
    return torch.sum(u * (D - e) ** 2, dim=-1)


def simulate_user_oracle(p_idx, q_idx, D, true_e, true_u):
    """模拟用户的真实选择 (Oracle)"""
    dist_p = get_weighted_distance_sq(D[p_idx], true_e, true_u)
    dist_q = get_weighted_distance_sq(D[q_idx], true_e, true_u)
    return (dist_p < dist_q).item()

def load_dataset(filepath):
    """解析带有 [n d] 开头的文本数据集"""
    try:
        with open(filepath, 'r') as f:
            raw_data = f.read().split()

        n, d = int(raw_data[0]), int(raw_data[1])

        D = np.array(raw_data[2:2 + n * d], dtype=float).reshape(n, d)
        col_min = D.min(axis=0)
        col_max = D.max(axis=0)

        # 计算极差，并加入 1e-8 的微小偏移量防止除以 0 的灾难（当某一整列数值全相同时）
        range_val = np.maximum(col_max - col_min, 1e-8)

        # 归一化公式： (X - X_min) / (X_max - X_min)
        D_normalized = (D - col_min) / range_val

        return D_normalized
    except Exception as e:
        print(f"数据加载失败，请检查文件路径和格式。错误: {e}")
        return None