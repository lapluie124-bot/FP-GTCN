import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def extract_hour_from_time_str(val):
    try:
        t = pd.to_datetime(val, format="%H:%M:%S", errors="coerce")
        if pd.isnull(t):
            return np.nan
        return t.hour
    except:
        return np.nan


class ChemicalDataLoader:
    """负责原始数据的加载和基础预处理"""

    def __init__(self, file_paths, node_features, target_node, time_col):
        self.file_paths = file_paths
        self.node_features = node_features
        self.target_node = target_node
        self.time_col = time_col
        self.raw_data = None
        self.processed_data = None
        self.target_series = None

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
                    df[self.time_col] = df[self.time_col].astype(str).apply(
                        lambda x: self._parse_time(x.strip(), sheet_name))
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