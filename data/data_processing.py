
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.config import load_config
from utils.data_loader import ChemicalDataLoader
from scripts.data_processor import ChemicalDataProcessor

# === 全局参数配置 ===
config = load_config()
WINDOW_SIZE = config["window_size"]
print(f"[预处理] 使用window_size={WINDOW_SIZE}")

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
    ("精馏釜液位/cm", "中温"), ("精馏釜液位/cm", "顶温"), ("精馏釜液位/cm", "釜温"),
    ("釜温", "中温"), ("釜温", "顶温"),
    ("蒸汽开度/%", "中温"), ("蒸汽开度/%", "顶温"), ("蒸汽开度/%", "釜温"), ("蒸汽开度/%", "回流比"),
    ("中温", "顶温"), ("中温", "回流比"), ("顶温", "回流比"),
    ("回流比", "V02121A液位/cm"), ("回流比", "V02122A液位/cm"), ("回流比", "V02123A液位/cm"),
    ("回流比", "V02123B液位/cm"),
    ("中温", "V02121A液位/cm"), ("中温", "V02122A液位/cm"), ("中温", "V02123A液位/cm"), ("中温", "V02123B液位/cm"),
    ("顶温", "V02121A液位/cm"), ("顶温", "V02122A液位/cm"), ("顶温", "V02123A液位/cm"), ("顶温", "V02123B液位/cm"),
    ("V02122A液位/cm", "中温"), ("V02122A液位/cm", "顶温")
]

TARGET_NODE = '取样结果_TFE'
TIME_COL = '时间'

# ========== 主程序 ==========
if __name__ == "__main__":
    # 1. 数据加载
    loader = ChemicalDataLoader(
        file_paths=FILE_PATHS,
        node_features=NODE_FEATURES,
        target_node=TARGET_NODE,
        time_col=TIME_COL
    )
    loader.preprocess_data()

    # 2. 数据处理
    processor = ChemicalDataProcessor(
        edge_list=EDGE_LIST,
        node_features=NODE_FEATURES,
        time_col=TIME_COL
    )

    train_data, val_data, test_data, edge_index = processor.create_data(
        processed_data=loader.processed_data,
        target_series=loader.target_series,
        raw_data=loader.raw_data,
        time_col=TIME_COL,
        target_node=TARGET_NODE,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        target_offset=TARGET_OFFSET,
        normalize=NORMALIZE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        target_csv=TARGET_CSV
    )

    # 3. 保存数据
    processor.save_preprocessed_data(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        edge_index=edge_index,
        processed_data=loader.processed_data,
        output_dir=OUTPUT_DIR
    )

    print("[主程序] 数据预处理和保存完成")