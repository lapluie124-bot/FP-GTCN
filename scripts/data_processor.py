
import os
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


class ChemicalDataProcessor:
    """负责图结构构建、滑动窗口切分、数据集创建"""

    def __init__(self, edge_list, node_features, time_col):
        self.edge_list = edge_list
        self.scaler = StandardScaler()
        self.node_to_idx = {name: i for i, name in enumerate(node_features) if name != time_col}
        self.target_min = None
        self.target_max = None

    def create_sliding_windows(self, sequence, window_size, step_size=1, target_offset=1):
        X, y = [], []
        seq_len = sequence.shape[0]
        for i in range(0, seq_len - window_size - target_offset + 1, step_size):
            X.append(sequence[i:i + window_size])
            y.append(sequence[i + window_size + target_offset - 1])
        return np.array(X), np.array(y)

    def build_graph_structure(self):
        edge_index = []
        for src, dst in self.edge_list:
            if src in self.node_to_idx and dst in self.node_to_idx:
                edge_index.append([self.node_to_idx[src], self.node_to_idx[dst]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        print(f"图结构构建完成：{len(self.node_to_idx)} 节点, {edge_index.size(1)} 条边")
        return edge_index

    def create_data(self, processed_data, target_series, raw_data, time_col, target_node,
                    window_size, step_size, target_offset, normalize, train_ratio, val_ratio, target_csv):
        features_df = processed_data.copy()
        if time_col in features_df.columns:
            features_df = features_df.drop(columns=[time_col])

        target_data = target_series.values.reshape(-1, 1)
        self.target_min = target_data.min()
        self.target_max = target_data.max()
        target_data_norm = (target_data - self.target_min) / (self.target_max - self.target_min)

        features = features_df.values
        if normalize:
            features = self.scaler.fit_transform(features)

        X, _ = self.create_sliding_windows(features, window_size, step_size, target_offset)
        y, _ = self.create_sliding_windows(target_data_norm, window_size, step_size, target_offset)
        y = y[:, -1, 0].reshape(-1, 1)

        meta_df = raw_data[[time_col, 'time_delta_hours', 'time_diff_hours', target_node]].copy()
        meta_df = meta_df.reset_index(drop=True)
        meta_df = meta_df.iloc[window_size + target_offset - 1:]
        meta_df = meta_df.iloc[::step_size].reset_index(drop=True)
        meta_df = meta_df.iloc[:len(y)]
        meta_df.to_csv(target_csv, index=False)

        total_len = len(X)
        train_end = int(total_len * train_ratio)
        val_end = train_end + int(total_len * val_ratio)

        X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
        y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

        edge_index = self.build_graph_structure()

        def create_sequence_data(X_data, y_data):
            data_list = []
            for i in range(len(X_data)):
                sequence = []
                for t in range(window_size):
                    x = torch.tensor(X_data[i, t], dtype=torch.float).view(-1, 1)
                    seq_data = Data(x=x, edge_index=edge_index)
                    sequence.append(seq_data)
                y_tensor = torch.tensor(y_data[i], dtype=torch.float)
                data_list.append((sequence, y_tensor))
            return data_list

        train_data = create_sequence_data(X_train, y_train)
        val_data = create_sequence_data(X_val, y_val)
        test_data = create_sequence_data(X_test, y_test)

        print(f"创建训练样本 {len(train_data)} 条，验证样本 {len(val_data)} 条，测试样本 {len(test_data)} 条")
        return train_data, val_data, test_data, edge_index

    def save_preprocessed_data(self, train_data, val_data, test_data, edge_index,
                               processed_data, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(train_data, os.path.join(output_dir, "train_dataset.pt"))
        torch.save(val_data, os.path.join(output_dir, "val_dataset.pt"))
        torch.save(test_data, os.path.join(output_dir, "test_dataset.pt"))
        torch.save(edge_index, os.path.join(output_dir, "edge_index.pt"))
        torch.save(self.scaler, os.path.join(output_dir, "scaler.pkl"))
        torch.save({'min': self.target_min, 'max': self.target_max},
                   os.path.join(output_dir, "target_range.pt"))
        if processed_data is not None:
            processed_data.to_csv(os.path.join(output_dir, "processed_features.csv"), index=False)
        print("预处理数据已保存至", output_dir)