# === Global parameter configuration ===
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

STEP_SIZE = 1
TARGET_OFFSET = 1
NORMALIZE = True
SHUFFLE = False
BATCH_SIZE = 32

OUTPUT_DIR = './processed_data'
TARGET_CSV = f"{OUTPUT_DIR}/target.csv"

import os
import re
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from config import load_config

config = load_config()
WINDOW_SIZE = config["window_size"]
print(f"[预处理] 使用window_size={WINDOW_SIZE}")

FILE_PATHS = [
    '三氟乙醇精馏4.25-6.29.xlsx',
    '三氟乙醇精馏7.01-8.13.xlsx',
    '三氟乙醇精馏8.13-9.26.xlsx'
]
NODE_FEATURES = [
    '时间', '蒸汽开度/%', '釜温', '中温', '顶温', '回流比',
    '精馏釜液位/cm', 'V02121A液位/cm', 'V02122A液位/cm',
    'V02123A液位/cm', 'V02123B液位/cm'
]

EDGE_LIST = [
    ("精馏釜液位/cm", "中温"), ("精馏釜液位/cm", "顶温"),("精馏釜液位/cm", "釜温"),
    ("釜温", "中温"), ("釜温", "顶温"),
    ("蒸汽开度/%", "中温"), ("蒸汽开度/%", "顶温"),("蒸汽开度/%", "釜温"), ("蒸汽开度/%", "回流比"),
    ("中温", "顶温"), ("中温", "回流比"), ("顶温", "回流比"),
    ("回流比", "V02121A液位/cm"), ("回流比", "V02122A液位/cm"), ("回流比", "V02123A液位/cm"), ("回流比", "V02123B液位/cm"),
    ("中温", "V02121A液位/cm"), ("中温", "V02122A液位/cm"), ("中温", "V02123A液位/cm"), ("中温", "V02123B液位/cm"),
    ("顶温", "V02121A液位/cm"), ("顶温", "V02122A液位/cm"), ("顶温", "V02123A液位/cm"), ("顶温", "V02123B液位/cm"),
    ("V02122A液位/cm", "中温"), ("V02122A液位/cm", "顶温")
]

TARGET_NODE = '取样结果_TFE'
TIME_COL = '时间'


def extract_hour_from_time_str(val):
    try:
        t = pd.to_datetime(val, format="%H:%M:%S", errors="coerce")
        if pd.isnull(t):
            return np.nan
        return t.hour
    except:
        return np.nan


class ChemicalDataProcessor:
    def __init__(self, file_paths, node_features, edge_list, target_node, time_col):
        self.file_paths = file_paths
        self.node_features = node_features
        self.edge_list = edge_list
        self.target_node = target_node
        self.time_col = time_col
        self.raw_data = None
        self.processed_data = None
        self.scaler = StandardScaler()
        self.node_to_idx = {name: i for i, name in enumerate(node_features) if name != time_col}

    def load_data(self):
        all_data = []
        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                print(f"警告：文件不存在，已跳过 -> {file_path}")
                continue
            excel_file = pd.ExcelFile(file_path)
            file_name = os.path.basename(file_path)
            for sheet_name in excel_file.sheet_names:
                try:
                    df = excel_file.parse(sheet_name)
                    df.columns = [col.replace(' ', '').strip() for col in df.columns]
                except Exception as e:
                    print(f"读取工作表 {sheet_name} 失败，错误：{str(e)}")
                    continue

                if self.time_col not in df.columns:
                    print(f"警告：{sheet_name} 缺少时间列 '{self.time_col}'，已跳过")
                    continue

                try:
                    df[self.time_col] = df[self.time_col].astype(str).apply(lambda x: self._parse_time(x.strip(), sheet_name))
                except Exception as e:
                    print(f"时间解析失败：{sheet_name} 错误：{str(e)}")
                    continue

                prev_time = None
                valid_indices = []
                for i, row in df.iterrows():
                    current_time = row[self.time_col]
                    if pd.isna(current_time):
                        continue
                    if prev_time is not None and current_time < prev_time:
                        current_time += timedelta(days=1)
                        df.at[i, self.time_col] = current_time
                    prev_time = current_time
                    valid_indices.append(i)
                df = df.loc[valid_indices].reset_index(drop=True)
                if len(df) < 5:
                    continue
                df['source_file'] = file_name
                df['sheet_name'] = sheet_name
                all_data.append(df)
        if not all_data:
            raise ValueError("没有成功加载任何数据！")
        self.raw_data = pd.concat(all_data, ignore_index=True)
        print(f"数据加载完成，共 {len(self.raw_data)} 条记录")
        return self.raw_data

    def _parse_time(self, time_str, sheet_name):
        if not time_str:
            return None
        match = re.match(r'(\d{2})(\d{3})', sheet_name[:5])
        if not match:
            return None
        year = int(match.group(1)) + 2000
        day_of_year = int(match.group(2))
        sheet_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        try:
            t = datetime.strptime(time_str, '%H:%M:%S').time()
        except ValueError:
            t = datetime.strptime(time_str, '%H:%M').time()
        return datetime.combine(sheet_date, t)

    def preprocess_data(self, fill_na=0.0):
        if self.raw_data is None:
            self.load_data()
        df = self.raw_data.copy()
        df['time_delta_hours'] = np.arange(len(df)) * 1.0
        df['time_diff_hours'] = 1.0

        if '回流比' in df.columns:
            print("[预处理] 正在将 '回流比' 字段从时间格式转换为小时整数")
            df['回流比'] = df['回流比'].astype(str).apply(extract_hour_from_time_str)

        for col in df.columns:
            if col != self.time_col:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.fillna(fill_na, inplace=True)

        if self.target_node not in df.columns:
            raise ValueError(f"目标节点 {self.target_node} 不存在")

        self.raw_data = df.copy()  
        self.target_series = df[self.target_node]
        feature_cols = [col for col in self.node_features + ['time_delta_hours', 'time_diff_hours']
                        if col in df.columns and col != self.target_node]
        self.processed_data = df[feature_cols]
        print(f"数据预处理完成：保留 {len(feature_cols)} 个数值型特征列")
        print(f"最终特征列: {feature_cols}")
        return self.processed_data

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

    def create_data(self, window_size=10):
        if self.processed_data is None:
            self.preprocess_data()

        features_df = self.processed_data.copy()
        if self.time_col in features_df.columns:
            features_df = features_df.drop(columns=[self.time_col])

        target_data = self.target_series.values.reshape(-1, 1)
        self.target_min = target_data.min()
        self.target_max = target_data.max()
        target_data_norm = (target_data - self.target_min) / (self.target_max - self.target_min)

        features = features_df.values
        if NORMALIZE:
            features = self.scaler.fit_transform(features)

        X, _ = self.create_sliding_windows(features, window_size, STEP_SIZE, TARGET_OFFSET)
        y, _ = self.create_sliding_windows(target_data_norm, window_size, STEP_SIZE, TARGET_OFFSET)
        y = y[:, -1, 0].reshape(-1, 1)

        meta_df = self.raw_data[[self.time_col, 'time_delta_hours', 'time_diff_hours', self.target_node]].copy()
        meta_df = meta_df.reset_index(drop=True)
        meta_df = meta_df.iloc[window_size + TARGET_OFFSET - 1:]
        meta_df = meta_df.iloc[::STEP_SIZE].reset_index(drop=True)
        meta_df = meta_df.iloc[:len(y)]
        meta_df.to_csv(TARGET_CSV, index=False)

        total_len = len(X)
        train_end = int(total_len * TRAIN_RATIO)
        val_end = train_end + int(total_len * VAL_RATIO)

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
        return train_data, val_data, test_data, edge_index, self.scaler, self.target_min, self.target_max


    def save_preprocessed_data(self, train_data, val_data, test_data, edge_index, scaler, target_min, target_max):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        torch.save(train_data, os.path.join(OUTPUT_DIR, "train_dataset.pt"))
        torch.save(val_data, os.path.join(OUTPUT_DIR, "val_dataset.pt"))
        torch.save(test_data, os.path.join(OUTPUT_DIR, "test_dataset.pt"))
        torch.save(edge_index, os.path.join(OUTPUT_DIR, "edge_index.pt"))
        torch.save(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
        torch.save({'min': target_min, 'max': target_max}, os.path.join(OUTPUT_DIR, "target_range.pt"))
        if self.processed_data is not None:
            self.processed_data.to_csv(os.path.join(OUTPUT_DIR, "processed_features.csv"), index=False)
        print("预处理数据已保存至", OUTPUT_DIR)


# ========== train_main.py ==========
if __name__ == "__main__":
    processor = ChemicalDataProcessor(
        file_paths=FILE_PATHS,
        node_features=NODE_FEATURES,
        edge_list=EDGE_LIST,
        target_node=TARGET_NODE,
        time_col=TIME_COL
    )
    train_data, val_data, test_data, edge_index, scaler, target_min, target_max = processor.create_data(
        window_size=WINDOW_SIZE
    )
    processor.save_preprocessed_data(train_data, val_data, test_data, edge_index, scaler, target_min, target_max)
    print("[主程序] 数据预处理和保存完成")

