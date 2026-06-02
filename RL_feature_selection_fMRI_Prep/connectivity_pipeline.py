import os
import re
import numpy as np
import pandas as pd

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

# ========= Inputs =========
BOLD_LIST = "mvp12_bold.txt"         # 已经生成的 12 行文件列表（每行一个 preproc bold.nii.gz）
OUT_DIR = "/Users/daxiaoniqiu/anaconda3/AI4H/pythonProject/myvenv/thesis/thesis/AD_MVP/mvp_connectivity_out"     # 输出目录
ATLAS = "schaefer"                  # "schaefer" or "aal"
N_ROIS = 200                        # Schaefer 100/200/400... 先用 200 跑通
TR = 3.0                            # TR=3.0
STANDARDIZE = "zscore_sample"       # nilearn 的标准化
CONNECTIVITY_KIND = "correlation"   # correlation / partial correlation (MVP 先 correlation)

# confounds：先用最常见、最稳的几个（MVP跑通优先）
CONFOUND_COLS = [
    "trans_x","trans_y","trans_z", #头在xyz方向移动多少mm
    "rot_x","rot_y","rot_z",  #头旋转多少(弧度)
    "csf","white_matter"
]

# ========= Helpers =========
def parse_sub_ses(path: str):
    # path like: .../sub-002S6009/ses-20170417/func/...
    m = re.search(r"(sub-[^/]+)/((ses-[^/]+))", path)
    if not m:
        return None, None, None
    sub = m.group(1)
    ses = m.group(3)
    sub_ses = f"{sub}_{ses}"
    return sub, ses, sub_ses

def confounds_path_from_bold(bold_path: str):
    # same folder, swap desc-preproc_bold -> desc-confounds_timeseries
    # and keep suffix .tsv
    # e.g. ..._desc-preproc_bold.nii.gz -> ..._desc-confounds_timeseries.tsv
    d = os.path.dirname(bold_path)
    fname = os.path.basename(bold_path)

    # 1) 去掉 _space-XXXX 这一段（因为 confounds 文件名一般不含 space）
    fname = re.sub(r"_space-[^_]+", "", fname)
    fname = fname.replace("_desc-preproc_bold.nii.gz", "_desc-confounds_timeseries.tsv")

    return os.path.join(d, fname)

def load_confounds(tsv_path: str):
    df = pd.read_csv(tsv_path, sep="\t")
    # 有些列可能不存在，做一个交集
    cols = [c for c in CONFOUND_COLS if c in df.columns]
    if len(cols) == 0:
        return None, []
    conf = df[cols].copy()
    # 简单缺失处理：线性插值 + 两端填充（MVP够用）
    conf = conf.interpolate(limit_direction="both").fillna(method="bfill").fillna(method="ffill")
    return conf, cols

# ========= Main =========
os.makedirs(OUT_DIR, exist_ok=True)

bold_df = pd.read_csv(BOLD_LIST, header=None, names=["bold_path"])
bold_df["bold_path"] = bold_df["bold_path"].astype(str).str.strip()
bold_df = bold_df[bold_df["bold_path"] != ""].copy()

# atlas
if ATLAS == "schaefer":
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=N_ROIS, yeo_networks=7, resolution_mm=2)
    labels_img = atlas.maps
    labels = atlas.labels
elif ATLAS == "aal":
    atlas = datasets.fetch_atlas_aal(version="SPM12")
    labels_img = atlas.maps
    labels = atlas.labels
else:
    raise ValueError("ATLAS must be 'schaefer' or 'aal'")

masker = NiftiLabelsMasker(
    labels_img=labels_img,
    standardize=STANDARDIZE,
    t_r=TR,
    detrend=True  #去线性漂移，把随时间缓慢上升/下降的线性趋势去掉，scanner drift让信号慢慢变大或变小，影响相关性计算。
)

conn = ConnectivityMeasure(kind=CONNECTIVITY_KIND)

rows = []
all_features = []

for i, row in bold_df.iterrows():
    bold_path = row["bold_path"]
    sub, ses, sub_ses = parse_sub_ses(bold_path)

    if sub_ses is None:
        rows.append({"bold_path": bold_path, "status": "FAIL", "reason": "cannot_parse_sub_ses"})
        continue

    conf_path = confounds_path_from_bold(bold_path)
    if not os.path.exists(bold_path):
        rows.append({"sub": sub, "ses": ses, "status": "FAIL", "reason": "missing_bold", "bold_path": bold_path})
        continue
    if not os.path.exists(conf_path):
        rows.append({"sub": sub, "ses": ses, "status": "FAIL", "reason": "missing_confounds", "conf_path": conf_path})
        continue

    confounds, used_cols = load_confounds(conf_path)
    if confounds is None:
        rows.append({"sub": sub, "ses": ses, "status": "FAIL", "reason": "no_confounds_cols_found", "conf_path": conf_path})
        continue

    try:
        # timeseries: (n_timepoints, n_rois)
        ts = masker.fit_transform(bold_path, confounds=confounds.values)

        # connectivity matrix: (n_rois, n_rois)
        mat = conn.fit_transform([ts])[0]

        # 保存矩阵（npy）
        out_mat = os.path.join(OUT_DIR, f"{sub_ses}_{ATLAS}{N_ROIS}_{CONNECTIVITY_KIND}.npy")
        np.save(out_mat, mat)

        # 展平成 feature（取上三角，去掉对角线）
        triu_idx = np.triu_indices_from(mat, k=1)
        feats = mat[triu_idx]

        all_features.append(pd.Series(feats, name=sub_ses))
        rows.append({
            "sub": sub, "ses": ses, "id": sub_ses,
            "status": "OK",
            "n_tp": ts.shape[0],
            "n_rois": ts.shape[1],
            "conf_cols": ",".join(used_cols),
            "out_mat": out_mat
        })

    except Exception as e:
        rows.append({"sub": sub, "ses": ses, "id": sub_ses, "status": "FAIL", "reason": repr(e)})

# 汇总表
summary_df = pd.DataFrame(rows)
summary_csv = os.path.join(OUT_DIR, "summary.csv")
summary_df.to_csv(summary_csv, index=False)

# features 表（行=subject_session，列=edges）
if len(all_features) > 0:
    feat_df = pd.DataFrame(all_features)
    feat_csv = os.path.join(OUT_DIR, f"features_{ATLAS}{N_ROIS}_{CONNECTIVITY_KIND}.csv")
    feat_df.to_csv(feat_csv, index=True)
else:
    feat_csv = None

print("✅ Done")
print(f"- summary: {summary_csv}")
if feat_csv:
    print(f"- features: {feat_csv}")
print(summary_df["status"].value_counts(dropna=False))
