from pathlib import Path

RL_SEEDS = [0, 1, 2, 3, 4]

# =========================================================
# Project paths
# =========================================================
# 请按你的 HCP / VM / 本地实际绝对路径修改
PROJECT_ROOT = Path.home() / "AD_main_experiment"
# PROJECT_ROOT = Path("/Users/daxiaoniqiu/anaconda3/AI4H/pythonProject/myvenv/thesis/thesis/AD_main_experiment")

DATA_DIR = PROJECT_ROOT / "connectivity_dataset_out"

# input files
X_PATH = DATA_DIR / "X.npy"
Y_PATH = DATA_DIR / "y.npy"
FEATURE_NAMES_PATH = DATA_DIR / "feature_names.npy"
SUBJECT_IDS_PATH = DATA_DIR / "subject_ids.npy"

# output dirs
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENT_DIR / "results"
LOG_DIR = EXPERIMENT_DIR / "logs"
FOLD_DIR = RESULTS_DIR / "folds"
BASELINE_DIR = RESULTS_DIR / "baseline"

# create dirs if missing
for p in [RESULTS_DIR, LOG_DIR, FOLD_DIR, BASELINE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# =========================================================
# Data / task settings
# =========================================================
LABEL_POSITIVE = 1
LABEL_NEGATIVE = 0

# =========================================================
# Cross-validation settings
# =========================================================
OUTER_N_SPLITS = 5
INNER_N_SPLITS = 3
GLOBAL_RANDOM_STATE = 42
BASELINE_SEEDS = list(RL_SEEDS) if len(RL_SEEDS) == 5 else [0, 1, 2, 3, 4]

# =========================================================
# Resource settings for VM / HPC
# =========================================================
# GridSearchCV 内部保持单线程，避免和外层 task 并行发生过度抢占。
N_JOBS_GRIDSEARCH = 1
# 对 16 CPU VM，先用 8 比较稳；若机器空闲且内存充足，可试 10~12。
N_JOBS_PARALLEL = 12

# =========================================================
# Baseline hyperparameter grids
# 说明：
# sklearn 的 LogisticRegression 用 C 表示正则强度的倒数
# C 越小，正则越强
# 对你这个 p >> n 的 FC 任务，baseline 保持经典、稀疏、不过宽即可
# =========================================================
LASSO_PARAM_GRID = {
    "clf__C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
}

ELASTIC_NET_PARAM_GRID = {
    "clf__C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    "clf__l1_ratio": [0.5, 0.7, 0.9]
}

# =========================================================
# Model settings
# =========================================================
MAX_ITER = 5000
TOL = 1e-4

# =========================================================
# Save paths
# =========================================================
OUTER_FOLD_JSON = PROJECT_ROOT / "experiments" / "outer_folds.json"
DATA_SUMMARY_JSON = RESULTS_DIR / "data_summary.json"
