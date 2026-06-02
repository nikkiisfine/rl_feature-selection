#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in CSV: {sorted(missing)}")


def _parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_datetime(out[c], errors="coerce")
        if out[c].isna().any():
            bad = out[out[c].isna()][[c]].head(10)
            raise ValueError(f"Some values in '{c}' could not be parsed as dates. Examples:\n{bad}")
    return out


def choose_best_pair_per_subject(
    df: pd.DataFrame,
    *,
    id_col: str,
    clinical_date_col: str,
    label_col: str,
    t1w_image_id_col: str,
    t1w_date_col: str,
    rsfmri_image_id_col: str,
    rsfmri_date_col: str,
    window_days: int = 180,
    w_clin: float = 1.0,
    w_pair: float = 0.25,
) -> pd.DataFrame:
    """
    For each subject (id), there may be multiple candidate rows (pairs of T1w and rs-fMRI).
    Keep only rows where BOTH modalities are within +/-window_days of the clinical_date,
    then choose the row with minimal score:
        score = w_clin*(|t1w_date-clinical| + |rs_date-clinical|) + w_pair*|t1w_date-rs_date|
    Tie-breakers:
        1) minimal score
        2) most recent clinical_date
    """

    needed = [
        id_col, label_col, clinical_date_col,
        t1w_image_id_col, t1w_date_col,
        rsfmri_image_id_col, rsfmri_date_col
    ]
    _require_cols(df, needed)
    df = _parse_dates(df, [clinical_date_col, t1w_date_col, rsfmri_date_col]).copy()

    # Drop rows with missing IDs for either modality (can't form a pair)
    df = df.dropna(subset=[t1w_image_id_col, rsfmri_image_id_col]).copy()

    # Compute day distances
    df["dt_t1w"] = (df[t1w_date_col] - df[clinical_date_col]).abs().dt.days
    df["dt_rs"] = (df[rsfmri_date_col] - df[clinical_date_col]).abs().dt.days
    df["dt_pair"] = (df[t1w_date_col] - df[rsfmri_date_col]).abs().dt.days

    # Window filter: both must be within +/-window_days
    df = df[(df["dt_t1w"] <= window_days) & (df["dt_rs"] <= window_days)].copy()

    if df.empty:
        raise ValueError(
            f"No rows left after applying the +/-{window_days} day window for BOTH T1w and rs-fMRI."
        )

    # Score
    df["pair_score"] = w_clin * (df["dt_t1w"] + df["dt_rs"]) + w_pair * df["dt_pair"]

    # Choose best per subject id
    df = df.sort_values(
        [id_col, "pair_score", clinical_date_col],
        ascending=[True, True, False]
    )
    best = df.drop_duplicates(subset=[id_col], keep="first").copy()

    # Cleanup helper columns (keep if you want to debug)
    best = best.drop(columns=["dt_t1w", "dt_rs", "dt_pair", "pair_score"], errors="ignore")

    return best


def select_balanced_subjects(
    input_csv: str,
    output_csv: str,
    *,
    n_subjects: int = 20,
    seed: int = 42,
    window_days: int = 180,
    w_clin: float = 1.0,
    w_pair: float = 0.25,
    id_col: str = "id",
    label_col: str = "AD_label",
    clinical_date_col: str = "clinical_date",
    t1w_image_id_col: str = "T1w_image_id",
    t1w_date_col: str = "T1w_date",
    rsfmri_image_id_col: str = "rs_fmri_Image_ID",
    rsfmri_date_col: str = "rs_fmri_Date",
) -> pd.DataFrame:

    if n_subjects <= 0:
        raise ValueError("n_subjects must be positive.")
    if n_subjects % 2 != 0:
        raise ValueError("n_subjects must be even (so it can be split evenly between AD_label=0/1).")

    df = pd.read_csv(input_csv)

    # Normalize label
    _require_cols(df, [id_col, label_col])
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    if df[label_col].isna().any():
        bad = df[df[label_col].isna()][[id_col, label_col]].head(10)
        raise ValueError(
            "Some AD_label values could not be parsed as numeric. Examples:\n"
            f"{bad.to_string(index=False)}"
        )
    df[label_col] = df[label_col].astype(int)
    df = df[df[label_col].isin([0, 1])].copy()

    # NEW: choose best pair per subject within window
    df_best = choose_best_pair_per_subject(
        df,
        id_col=id_col,
        clinical_date_col=clinical_date_col,
        label_col=label_col,
        t1w_image_id_col=t1w_image_id_col,
        t1w_date_col=t1w_date_col,
        rsfmri_image_id_col=rsfmri_image_id_col,
        rsfmri_date_col=rsfmri_date_col,
        window_days=window_days,
        w_clin=w_clin,
        w_pair=w_pair,
    )

    # Stratified sample (分层抽样)
    half = n_subjects // 2
    df0 = df_best[df_best[label_col] == 0]
    df1 = df_best[df_best[label_col] == 1]

    if len(df0) < half or len(df1) < half:
        raise ValueError(
            "Not enough subjects to sample evenly after pair selection.\n"
            f"Need at least {half} with AD_label=0 and {half} with AD_label=1.\n"
            f"Available: label0={len(df0)}, label1={len(df1)}"
        )

    sample0 = df0.sample(n=half, random_state=seed)
    sample1 = df1.sample(n=half, random_state=seed)
    selected = pd.concat([sample0, sample1], ignore_index=True)

    # Shuffle final order
    selected = selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Save
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_path, index=False)

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Select balanced subjects (50% label=1, 50% label=0) and choose best T1w+rs-fMRI pair "
                    "within +/-window_days of clinical_date."
    )
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="selected_subjects.csv")
    parser.add_argument("--n_subjects", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    # Pair selection params
    parser.add_argument("--window_days", type=int, default=180, help="+/- window around clinical_date")
    parser.add_argument("--w_clin", type=float, default=1.0, help="weight for closeness to clinical_date")
    parser.add_argument("--w_pair", type=float, default=0.25, help="weight for T1w-rs pairing closeness")

    # Column names
    parser.add_argument("--id_col", type=str, default="id")
    parser.add_argument("--label_col", type=str, default="AD_label")
    parser.add_argument("--clinical_date_col", type=str, default="clinical_date")

    parser.add_argument("--t1w_image_id_col", type=str, default="T1w_image_id")
    parser.add_argument("--t1w_date_col", type=str, default="T1w_date")

    parser.add_argument("--rsfmri_image_id_col", type=str, default="rs_fmri_Image_ID")
    parser.add_argument("--rsfmri_date_col", type=str, default="rs_fmri_Date")

    args = parser.parse_args()

    selected = select_balanced_subjects(
        args.input_csv,
        args.output_csv,
        n_subjects=args.n_subjects,
        seed=args.seed,
        window_days=args.window_days,
        w_clin=args.w_clin,
        w_pair=args.w_pair,
        id_col=args.id_col,
        label_col=args.label_col,
        clinical_date_col=args.clinical_date_col,
        t1w_image_id_col=args.t1w_image_id_col,
        t1w_date_col=args.t1w_date_col,
        rsfmri_image_id_col=args.rsfmri_image_id_col,
        rsfmri_date_col=args.rsfmri_date_col,
    )

    print("Done. Saved:", args.output_csv)
    print("Counts by AD_label:")
    print(selected[args.label_col].value_counts().sort_index().to_string())

    print("\nSubject IDs:")
    print(", ".join(selected[args.id_col].astype(str).tolist()))
    print("\nT1w_image_id list:")
    print(", ".join(selected[args.t1w_image_id_col].astype(str).tolist()))
    print("\nrs_fmri_Image_ID list:")
    print(", ".join(selected[args.rsfmri_image_id_col].astype(str).tolist()))


if __name__ == "__main__":
    main()
