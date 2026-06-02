import pandas as pd
import numpy as np

# ========= 1. 路径 =========
x_csv_path = "/Users/daxiaoniqiu/anaconda3/AI4H/pythonProject/myvenv/thesis/thesis/AD_main_experiment/connectivity_dataset_out/X_connectivity.csv"
label_csv_path = "/Users/daxiaoniqiu/anaconda3/AI4H/pythonProject/myvenv/thesis/thesis/AD_main_experiment/main160_selected.csv"
out_dir = "/Users/daxiaoniqiu/anaconda3/AI4H/pythonProject/myvenv/thesis/thesis/AD_main_experiment/connectivity_dataset_out"

# ========= 2. 读取 =========
df = pd.read_csv(x_csv_path)
label_df = pd.read_csv(label_csv_path)

# ========= 3. metadata 列 =========
meta_cols = [
    "sub", "ses", "task", "run",
    "bold_file", "confounds_file", "original_decision",
    "remaining_frames", "remaining_minutes",
    "n_scrubbed", "pct_scrubbed",
    "fd_mean", "fd_max", "dvars_std_mean", "dvars_std_max"
]
meta_cols = [c for c in meta_cols if c in df.columns]

# ========= 4. 特征列 =========
feature_cols = [c for c in df.columns if c not in meta_cols]

# ========= 5. subject 对齐 =========
df["sub"] = df["sub"].astype(str)
label_df["id"] = label_df["id"].astype(str)
label_df["id_clean"] = label_df["id"].str.replace("_", "")

merged = df.merge(
    label_df[["id_clean", "AD_label"]],
    left_on="sub",
    right_on="id_clean",
    how="inner"
)

# ========= 6. 提取 X 和 y =========
X = merged[feature_cols].to_numpy(dtype=np.float32)
y = merged["AD_label"].to_numpy(dtype=np.int64)
subject_ids = merged["sub"].astype(str).to_numpy()
feature_names = np.array(feature_cols, dtype=object)

print("Merged shape:", merged.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)

# ========= 7. 保存 =========
np.save(f"{out_dir}/X.npy", X)
np.save(f"{out_dir}/y.npy", y)
np.save(f"{out_dir}/subject_ids.npy", subject_ids)
np.save(f"{out_dir}/feature_names.npy", feature_names)

print("Saved:")
print(f"{out_dir}/X.npy")
print(f"{out_dir}/y.npy")
print(f"{out_dir}/subject_ids.npy")
print(f"{out_dir}/feature_names.npy")

print(X.shape)
print(y.shape)
print(feature_names.shape)