"""
模型训练模块
"""
import torch
from torch_geometric.data import Data
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_one_epoch(model, train_loader, criterion, optimizer, edge_index, device):
    """
    训练一个epoch

    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        edge_index: 图边索引
        device: 设备

    Returns:
        float: 平均训练损失
    """
    model.train()
    epoch_train_loss = 0

    for batch in train_loader:
        sequences, labels = zip(*batch)
        labels = torch.cat(labels, dim=0).to(device).squeeze(-1)
        seqs = [[Data(x=g.x.to(device), edge_index=g.edge_index.to(device)) for g in s] for s in sequences]

        loss = criterion(model(seqs, edge_index.to(device)), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    return epoch_train_loss / len(train_loader)


def validate(model, val_loader, criterion, edge_index, device):
    """
    验证模型

    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        edge_index: 图边索引
        device: 设备

    Returns:
        float: 平均验证损失
    """
    model.eval()
    epoch_val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            sequences, labels = zip(*batch)
            labels = torch.cat(labels, dim=0).to(device).squeeze(-1)
            seqs = [[Data(x=g.x.to(device), edge_index=g.edge_index.to(device)) for g in s] for s in sequences]
            loss = criterion(model(seqs, edge_index.to(device)), labels)
            epoch_val_loss += loss.item()

    return epoch_val_loss / len(val_loader)


def train_model(model, train_loader, val_loader, edge_index, device,
                num_epochs, learning_rate, model_save_path,
                early_stop_patience=20):
    """
    完整训练流程

    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        edge_index: 图边索引
        device: 设备
        num_epochs: 训练轮数
        learning_rate: 学习率
        model_save_path: 模型保存路径
        early_stop_patience: 早停耐心值

    Returns:
        tuple: (训练损失列表, 验证损失列表)
    """
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    best_loss = float('inf')
    train_losses, val_losses = [], []
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     edge_index, device)
        val_loss = validate(model, val_loader, criterion, edge_index, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
            print(f"Epoch {epoch}: 保存最佳模型（验证损失: {val_loss:.4f}）")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"早停触发：验证损失连续 {early_stop_patience} 轮未提升")
                break

        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return train_losses, val_losses