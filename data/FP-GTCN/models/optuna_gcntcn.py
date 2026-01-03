import optuna
import yaml
import subprocess
import re
import os
import pandas as pd
import logging
from optuna.pruners import MedianPruner

# === 日志配置 ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optuna_gcntcn.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("optuna_gcntcn")

CONFIG_PATH = "config.yaml"
RESULTS_CSV = "optuna_gcntcn_results.csv"
results_list = []

def update_config(params):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config.update(params)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(config, f)
        logger.info(f"配置更新成功: {params}")
    except Exception as e:
        logger.error(f"配置更新失败: {e}")
        raise

def run_training():
    logger.info("开始训练过程")
    proc = subprocess.Popen(
        ["python", "gcntcn.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding="utf-8",
        errors="replace"
    )
    logs = []
    for line in proc.stdout:
        print(line, end="")
        logs.append(line)
    proc.wait()
    exit_code = proc.returncode
    if exit_code != 0:
        logger.warning(f"训练过程异常退出，退出码: {exit_code}")
    else:
        logger.info("训练过程正常结束")
    return "".join(logs)

def parse_rmse(log_text):
    match = re.search(r"RMSE:\s*([\d\.]+)", log_text)
    if match:
        rmse = float(match.group(1))
        logger.info(f"成功解析RMSE: {rmse}")
        return rmse
    else:
        logger.warning("无法从日志中解析RMSE值")
        return None


# def objective(trial):
#     hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
#     dropout = trial.suggest_float("dropout", 0.1, 0.5)
#     learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
#     kernel_size = trial.suggest_int("kernel_size", 2, 5)
#
#     trial_params = {
#         "hidden_dim": hidden_dim,
#         "dropout": dropout,
#         "learning_rate": learning_rate,
#         "kernel_size": kernel_size
#     }

def objective(trial):
    # === 搜索空间 ===
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    # kernel_size（单值）在 Inception-TCN 内部会展开为 (3, k, k+2)
    kernel_size = trial.suggest_int("kernel_size", 3, 9, step=2)  # 直接限制为奇数更稳
    # 也可以用下面这句（若你想保留原 2..5 的范围则自动转奇数）：
    # kernel_size    = _to_odd(trial.suggest_int("kernel_size", 2, 5))

    # Inception 并行相关的开关（兼容 gcntcn_inception 或适配器）
    inception_depthwise = trial.suggest_categorical("inception_depthwise", [False, True])
    inception_use_pool_branch = trial.suggest_categorical("inception_use_pool_branch", [False, True])

    # 扩张率方案（和你原有 [1,2,4] 风格兼容）
    dilation_choice = trial.suggest_categorical(
        "dilation_str",
        ["1,2", "1,2,4", "1,2,4,8"]
    )
    dilation_rates = [int(x) for x in dilation_choice.split(",")]

    trial_params = {
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "kernel_size": kernel_size,
        "dilation_rates": dilation_rates,
        # Inception-TCN 相关（你在适配器/模型里读取这些键即可）
        "inception_depthwise": inception_depthwise,
        "inception_use_pool_branch": inception_use_pool_branch,

    }
    logger.info(f"[Optuna Trial {trial.number}] 尝试参数: {trial_params}")
    update_config(trial_params)

    log_text = run_training()
    rmse = parse_rmse(log_text)

    if rmse is None:
        logger.error(f"[Optuna Trial {trial.number}] 无法解析RMSE，本次trial失败")
        rmse = float("inf")

    # === Optuna 剪枝反馈 ===
    trial.report(rmse, step=0)
    if trial.should_prune():
        logger.warning(f"[Trial {trial.number}] 被剪枝，RMSE = {rmse:.4f}")
        raise optuna.TrialPruned()

    logger.info(f"[Optuna Trial {trial.number}] 结果: RMSE={rmse:.4f}")
    results_list.append({
        "trial_id": trial.number,
        **trial_params,
        "rmse": rmse
    })
    return rmse

if __name__ == "__main__":
    logger.info("======= Optuna 超参数优化开始 =======")
    try:
        study = optuna.create_study(
            direction="minimize",
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=0, interval_steps=1)
        )
        study.optimize(objective, n_trials=200)

        df = pd.DataFrame(results_list)
        df.to_csv(RESULTS_CSV, index=False)
        logger.info(f"所有搜索结果已保存至 {RESULTS_CSV}")
        logger.info(f"所有日志已保存至 {os.path.abspath('optuna_gcntcn.log')}")

        logger.info("========== 搜索完成 ==========")
        logger.info(f"最优验证RMSE: {study.best_value:.4f}")
        logger.info("最优超参数配置：")
        for k, v in study.best_params.items():
            logger.info(f"  {k}: {v}")
    except Exception as e:
        logger.critical(f"程序异常终止: {e}", exc_info=True)
    finally:
        logger.info("======= Optuna 超参数优化结束 =======")
