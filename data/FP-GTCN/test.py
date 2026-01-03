import os
import torch
import random
import yaml
import numpy as np
from torch.utils.data import DataLoader, Dataset
import warnings

from models.model import GCNTCN
from train import train_model
from evaluate import evaluate_model, denormalize_predictions
from models.metrics import calculate_metrics, print_metrics

warnings.filterwarnings("ignore")

# === 加载配置参数 ===
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config["output_dir"]
MODEL_SAVE_PATH = config["model_save_path"]
HIDDEN_DIM = config["hidden_dim"]
WINDOW_SIZE = config["window_size"]
BATCH_SIZE = config["batch_size"]
NUM_EPOCHS = config["num_epochs"]
LEARNING_RATE = config["learning_rate"]
DROPOUT = config["dropout"]
KERNEL_SIZE = config["kernel_size"]
VISUALIZE_SAMPLES = config["visualize_samples"]
USE_CUDA = torch.cuda.is_available()
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(RANDOM_SEED)


class GCNTCNDataSet(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)


def load_and_split_data():
    """
    加载预处理好的数据

    Returns:
        tuple: (训练加载器, 验证加载器, 测试加载器, 边索引, 特征维度, 设备, target_min, target_max)
    """
    device = torch.device('cuda' if USE_CUDA else 'cpu')
    train_data = torch.load(f"{OUTPUT_DIR}/train_dataset.pt", map_location=device)
    test_data = torch.load(f"{OUTPUT_DIR}/test_dataset.pt", map_location=device)
    val_data = torch.load(f"{OUTPUT_DIR}/val_dataset.pt", map_location=device)
    edge_index = torch.load(f"{OUTPUT_DIR}/edge_index.pt", map_location=device)
    target_range = torch.load(f"{OUTPUT_DIR}/target_range.pt")
    target_min, target_max = target_range['min'], target_range['max']

    def get_loader(data):
        return DataLoader(GCNTCNDataSet(data), batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=lambda x: x)

    feature_dim = GCNTCNDataSet(train_data)[0][0][0].x.size(1)

    return (get_loader(train_data), get_loader(val_data), get_loader(test_data),
            edge_index, feature_dim, device, target_min, target_max)


def main():
    """主函数：完整的训练和测试流程"""
    # 1. 加载数据
    train_loader, val_loader, test_loader, edge_index, feature_dim, device, target_min, target_max = load_and_split_data()

    # 2. 创建模型
    model = GCNTCN(feature_dim, HIDDEN_DIM, WINDOW_SIZE, KERNEL_SIZE, dropout=DROPOUT).to(device)

    # 3. 训练模型
    print("开始训练...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, edge_index, device,
        NUM_EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH
    )

    # 4. 加载最佳模型
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # 5. 在测试集上评估
    print("\n在测试集上评估...")
    actuals_norm, predictions_norm = evaluate_model(model, test_loader, edge_index, device)
    actuals, predictions = denormalize_predictions(actuals_norm, predictions_norm,
                                                   target_min, target_max)

    # 6. 计算并打印指标
    metrics = calculate_metrics(actuals, predictions)
    print_metrics(metrics, prefix="Test")

    # 7. 保存输出结果
    MODEL_TYPE = "gcntcn"
    os.makedirs("optuna_best_output", exist_ok=True)
    np.save(f"optuna_best_output/train_losses_{MODEL_TYPE}.npy", np.array(train_losses))
    np.save(f"optuna_best_output/actuals_{MODEL_TYPE}.npy", actuals)
    np.save(f"optuna_best_output/predictions_{MODEL_TYPE}.npy", predictions)

    print(f"\n结果已保存至 optuna_best_output/ 目录")


if __name__ == "__main__":
    main()