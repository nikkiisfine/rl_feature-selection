#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd



# ---------------------- ID normalization & label selection ----------------------
def normalize_subject_id(x: str) -> str:
    """Normalize subject identifiers across sources.

    Examples:
      - 'sub-002S6009_ses-20170417' -> '002S6009'
      - 'sub-002S6009' -> '002S6009'
      - '002_S_6009' -> '002S6009'
      - '002S6009' -> '002S6009'
    """
    if x is None:
        return ""
    s = str(x).strip()
    if "sub-" in s:
        s = s.split("sub-", 1)[1]
    s = re.split(r"_ses-|_run-|\s", s, maxsplit=1)[0]
    s = s.replace("_", "").replace("-", "")
    m = re.search(r"(\d{3})S(\d{4})", s)
    if m:
        return f"{m.group(1)}S{m.group(2)}"
    return s


def build_latest_labels_per_subject(
    manifest_df: pd.DataFrame,
    id_col: str = "id",
    clinical_date_col: str = "clinical_date",
    label_col: str = "AD_label",
) -> pd.DataFrame:
    """Pick ONE label per subject: the label at the most recent clinical date."""
    df = manifest_df.copy()
    if id_col not in df.columns:
        raise KeyError(f"id_col '{id_col}' not found in manifest columns: {list(df.columns)}")
    if label_col not in df.columns:
        raise KeyError(f"label_col '{label_col}' not found in manifest columns: {list(df.columns)}")
    if clinical_date_col not in df.columns:
        raise KeyError(f"clinical_date_col '{clinical_date_col}' not found in manifest columns: {list(df.columns)}")

    df["_subj"] = df[id_col].apply(normalize_subject_id)
    df["_dt"] = pd.to_datetime(df[clinical_date_col], errors="coerce")
    df = df[df["_subj"].astype(bool) & df["_dt"].notna()].copy()
    df = df.sort_values(["_subj", "_dt"]).groupby("_subj", as_index=False).tail(1)
    out = df[["_subj", label_col]].rename(columns={"_subj": "subj", label_col: "label"})
    return out.reset_index(drop=True)

def infer_n_roi(n_features: int) -> int | None:
    """
    Solve n*(n-1)/2 = n_features for integer n.
    """
    disc = 1 + 8 * n_features
    n = (1 + int(math.isqrt(disc))) / 2
    if n.is_integer():
        n = int(n)
        if n * (n - 1) // 2 == n_features:
            return n
    return None


def make_edge_mapping(n_roi: int, prefix: str = "roi") -> pd.DataFrame:
    """
    Create mapping for upper-triangular edges: (0,1), (0,2), ..., (n-2, n-1)
    """
    rows = []
    idx = 0
    for i in range(n_roi):
        for j in range(i + 1, n_roi):
            idx += 1
            rows.append(
                {
                    "feature_idx": idx,  # 1-based index (more human-friendly)
                    "roi_i": i + 1,      # 1-based ROI id
                    "roi_j": j + 1,
                    "feature_name": f"edge_{idx:05d}__{prefix}{i+1:03d}__{prefix}{j+1:03d}",
                }
            )
    return pd.DataFrame(rows)


def build_bids_key_from_manifest(df: pd.DataFrame,
                                 id_col="id",
                                 date_col="rs_fmri_Date",
                                 label_col="AD_label") -> pd.DataFrame:
    """
    Convert manifest rows to:
    key = sub-XXX_ses-YYYYMMDD, label
    Example:
      id: 002_S_6009 -> sub-002S6009
      rs_fmri_Date: 2017-04-17 -> ses-20170417
      key: sub-002S6009_ses-20170417
    """
    out = df[[id_col, date_col, label_col]].copy()

    # drop rows missing required fields
    out = out.dropna(subset=[id_col, date_col, label_col])

    # normalize id
    out[id_col] = out[id_col].astype(str)
    sub = out[id_col].str.replace("_", "", regex=False)  # 002_S_6009 -> 002S6009
    out["sub"] = "sub-" + sub

    # normalize date -> yyyymmdd
    # accept string like 2017-04-17 or Timestamp
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    ses = out[date_col].dt.strftime("%Y%m%d")
    out["ses"] = "ses-" + ses

    out["key"] = out["sub"] + "_" + out["ses"]
    out = out.rename(columns={label_col: "label"})

    # if duplicates exist (should not, but just in case), keep the first
    out = out.drop_duplicates(subset=["key"], keep="first")[["key", "label"]]

    # enforce int labels (0/1)
    out["label"] = out["label"].astype(int)

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--features_csv", required=True, help="Feature CSV (first col is sample id/key)")
    ap.add_argument("--manifest_xlsx", required=True, help="Excel: T1 cohort manifest.xlsx")
    ap.add_argument("--sheet", default="joined_output", help="Sheet name, default: joined_output")
    ap.add_argument("--id_col", default="id", help="Manifest column for subject id")
    ap.add_argument("--date_col", default="rs_fmri_Date", help="Manifest column for session date")
    ap.add_argument("--label_col", default="AD_label", help="Manifest column for label")
    ap.add_argument("--clinical_date_col", default="clinical_date",
                    help="Column in the manifest used to pick the *latest* label per subject (e.g., clinical visit date)")

    # missing value handling
    ap.add_argument("--impute", choices=["mean", "median", "zero", "drop_rows", "drop_cols"], default="mean",
                    help="Missing value handling (imputation)")
    ap.add_argument("--drop_constant_cols", action="store_true", help="Drop columns with zero variance")

    # scaling
    ap.add_argument("--standardize", action="store_true",
                    help='Apply "z-score standardization" to features (recommended)')

    # output
    ap.add_argument("--out_dir", default="step4_preprocessed", help="Output directory")

    # feature naming
    ap.add_argument("--rename_features", action="store_true",
                    help="Rename numeric features to ROI-pair names, and write edge_index.csv")
    ap.add_argument("--roi_prefix", default="roi", help="ROI name prefix for renaming")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- load features ----------
    feat_df = pd.read_csv(args.features_csv)
    if feat_df.shape[1] < 3:
        raise ValueError("Feature CSV seems too small. Expect: [id/key] + many features.")

    key_col = feat_df.columns[0]
    feat_df[key_col] = feat_df[key_col].astype(str)

    # numeric feature matrix
    X = feat_df.drop(columns=[key_col])
    # convert all to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # ---------- load manifest & build labels ----------
    manifest = pd.read_excel(args.manifest_xlsx, sheet_name=args.sheet)
    labels_df = build_latest_labels_per_subject(
        manifest,
        id_col=args.id_col,
        clinical_date_col=args.clinical_date_col,
        label_col=args.label_col,
    )
    labels_df.to_csv(out_dir / "labels_latest_per_subject.csv", index=False)

    # ---------- merge (by subject; label = most recent clinical label) ----------
    feat_df["subj"] = feat_df[key_col].apply(normalize_subject_id)
    merged = feat_df[[key_col, "subj"]].merge(labels_df, on="subj", how="left")
    n_missing_label = merged["label"].isna().sum()
    if n_missing_label > 0:
        print(f"[WARN] {n_missing_label} rows in features have NO label match in manifest (will be dropped).")

    keep_mask = merged["label"].notna()
    feat_df = feat_df.loc[keep_mask].reset_index(drop=True)
    merged = merged.loc[keep_mask].reset_index(drop=True)

    y = merged["label"].astype(int).values
    X = feat_df.drop(columns=[key_col, "subj"]).apply(pd.to_numeric, errors="coerce")

    # ---------- missing values in X ----------
    # Replace inf/-inf -> NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    n_nan = int(X.isna().sum().sum())
    if n_nan > 0:
        print(f"[INFO] Found {n_nan} missing values (NaN) in X. Handling via: {args.impute}")

    if args.impute == "drop_rows":
        row_nan = X.isna().any(axis=1)
        dropped = int(row_nan.sum())
        X = X.loc[~row_nan].reset_index(drop=True)
        y = y[~row_nan.values]
        feat_df = feat_df.loc[~row_nan].reset_index(drop=True)
        print(f"[INFO] Dropped {dropped} rows due to NaNs.")
    elif args.impute == "drop_cols":
        col_nan = X.isna().any(axis=0)
        dropped = int(col_nan.sum())
        X = X.loc[:, ~col_nan]
        print(f"[INFO] Dropped {dropped} columns due to NaNs.")
    else:
        if args.impute == "mean":
            fill_vals = X.mean(axis=0)
        elif args.impute == "median":
            fill_vals = X.median(axis=0)
        elif args.impute == "zero":
            fill_vals = 0.0
        X = X.fillna(fill_vals)

    # ---------- drop constant columns ----------
    if args.drop_constant_cols:
        std = X.std(axis=0)
        constant_cols = std[std == 0].index
        if len(constant_cols) > 0:
            X = X.drop(columns=list(constant_cols))
            print(f"[INFO] Dropped {len(constant_cols)} constant columns (zero variance).")

    # ---------- class balance check ----------
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"[INFO] Class distribution (label -> count): {dist}")
    if len(dist) == 2:
        maj = max(dist.values())
        min_ = min(dist.values())
        ratio = maj / max(min_, 1)
        print(f"[INFO] Imbalance ratio (major/minor) = {ratio:.2f}")

    # ---------- standardization ----------
    # Why: make features comparable scale; helps LR/SVM and stabilizes RL reward optimization.
    if args.standardize:
        mu = X.mean(axis=0)
        sigma = X.std(axis=0).replace(0, 1.0)
        X = (X - mu) / sigma
        print('[INFO] Applied "z-score standardization" to X.')

    # ---------- feature naming ----------
    edge_index_path = out_dir / "edge_index.csv"
    if args.rename_features:
        # If columns are like 0..N-1 or strings of ints, rename them
        n_features = X.shape[1]
        n_roi = infer_n_roi(n_features)
        if n_roi is None:
            print(f"[WARN] Cannot infer n_roi from n_features={n_features}. Skip renaming.")
        else:
            mapping = make_edge_mapping(n_roi, prefix=args.roi_prefix)
            mapping.to_csv(edge_index_path, index=False)

            # rename columns in current order
            new_cols = mapping["feature_name"].tolist()
            X.columns = new_cols
            print(f"[INFO] Renamed features using ROI-pair names. Saved mapping to: {edge_index_path}")

    # ---------- write outputs ----------
    out_features = out_dir / "X_features.csv"
    out_labels = out_dir / "y_labels.csv"
    out_ids = out_dir / "sample_ids.txt"

    # Save X with key column
    X_out = pd.concat([feat_df[[key_col]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    X_out.to_csv(out_features, index=False)

    pd.DataFrame({key_col: feat_df[key_col], "label": y}).to_csv(out_labels, index=False)
    feat_df[key_col].to_csv(out_ids, index=False, header=False)

    print("[DONE] Saved:")
    print(f"  - {out_features}")
    print(f"  - {out_labels}")
    print(f"  - {out_ids}")
    if args.rename_features and edge_index_path.exists():
        print(f"  - {edge_index_path}")


if __name__ == "__main__":
    main()
