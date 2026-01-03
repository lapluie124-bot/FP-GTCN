"""
模型评估模块
"""
import torch
import numpy as np
from torch_geometric.data import Data


def evaluate_model(model, loader, edge_index, device):
    """
    评估模型并返回预测结果

    Args:
        model: 模型
        loader: 数据加载器
        edge_index: 图边索引
        device: 设备

    Returns:
        tuple: (真实值数组, 预测值数组)
    """
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for batch in loader:
            sequences, labels = zip(*batch)
            labels = torch.cat(labels, dim=0).to(device).squeeze(-1)
            seqs = [[Data(x=g.x.to(device), edge_index=g.edge_index.to(device)) for g in s] for s in sequences]
            preds = model(seqs, edge_index.to(device))
            predictions.extend(preds.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    actuals = np.array(actuals)
    predictions = np.array(predictions)

    return actuals, predictions


def denormalize_predictions(actuals_norm, predictions_norm, target_min, target_max):
    """
    将归一化的预测结果还原到原始尺度

    Args:
        actuals_norm: 归一化的真实值
        predictions_norm: 归一化的预测值
        target_min: 目标变量最小值
        target_max: 目标变量最大值

    Returns:
        tuple: (还原后的真实值, 还原后的预测值)
    """
    actuals = actuals_norm * (target_max - target_min) + target_min
    predictions = predictions_norm * (target_max - target_min) + target_min
    return actuals, predictions