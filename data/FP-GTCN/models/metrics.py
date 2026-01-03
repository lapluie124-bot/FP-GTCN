"""
评估指标模块
计算各类回归指标
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


def calculate_metrics(actuals, predictions):
    """
    计算回归评估指标

    Args:
        actuals: 真实值数组
        predictions: 预测值数组

    Returns:
        dict: 包含各项指标的字典
    """
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    # 计算Pearson相关系数（如果需要）
    try:
        pearson_corr, _ = pearsonr(actuals, predictions)
    except:
        pearson_corr = np.nan

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson_corr
    }

    return metrics


def print_metrics(metrics, prefix=""):
    """
    打印评估指标

    Args:
        metrics: 指标字典
        prefix: 打印前缀（如"Train", "Val", "Test"）
    """
    prefix_str = f"{prefix} " if prefix else ""
    print(f"{prefix_str}RMSE: {metrics['rmse']:.4f}, "
          f"MAE: {metrics['mae']:.4f}, "
          f"R²: {metrics['r2']:.4f}")
    if not np.isnan(metrics.get('pearson', np.nan)):
        print(f"{prefix_str}Pearson: {metrics['pearson']:.4f}")