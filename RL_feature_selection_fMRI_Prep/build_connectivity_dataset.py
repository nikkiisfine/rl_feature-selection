#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build connectivity dataset from fMRIPrep derivatives with scrubbing.

Pipeline:
1. Read scrubbing QC summary (KEEP / REVIEW / DROP)
2. For KEEP + REVIEW runs:
   - locate preproc BOLD and confounds TSV
   - compute scrub mask from FD + std_dvars
   - regress confounds + filter + standardize using Nilearn
   - extract ROI time series with atlas
   - compute Pearson connectivity
3. Save:
   - X_connectivity.csv
   - run_metadata.csv
   - feature_names.csv
   - dropped_runs.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from nilearn.maskers import NiftiLabelsMasker


# ----------------------------
# Argument parsing
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build connectivity dataset from fMRIPrep outputs with scrubbing."
    )
    parser.add_argument(
        "--derivatives-dir",
        type=str,
        required=True,
        help="Path to fMRIPrep derivatives root, e.g. ./fmriprep_func/derivatives",
    )
    parser.add_argument(
        "--qc-csv",
        type=str,
        required=True,
        help="Path to scrubbing QC summary CSV",
    )
    parser.add_argument(
        "--atlas-img",
        type=str,
        required=True,
        help="Path to atlas labels NIfTI image, e.g. Schaefer/AAL labels image",
    )
    parser.add_argument(
        "--atlas-labels",
        type=str,
        default=None,
        help="Optional text/csv/tsv file with ROI names",
    )
    parser.add_argument(
        "--label-csv",
        type=str,
        default=None,
        help="Optional phenotype/label CSV containing subject labels",
    )
    parser.add_argument(
        "--label-sub-col",
        type=str,
        default="subject_id",
        help="Column name in label CSV for subject ID",
    )
    parser.add_argument(
        "--label-y-col",
        type=str,
        default="label",
        help="Column name in label CSV for target label",
    )
    parser.add_argument(
        "--tr",
        type=float,
        default=3.0,
        help="TR in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--fd-thresh",
        type=float,
        default=0.7,
        help="FD threshold for scrubbing (default: 0.7)",
    )
    parser.add_argument(
        "--dvars-z-thresh",
        type=float,
        default=2.0,
        help="standardized DVARS threshold for scrubbing (default: 2.0)",
    )
    parser.add_argument(
        "--expand",
        type=int,
        default=1,
        help="Expand scrubbed frames by +/- n frames (default: 1)",
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=4,
        help="Drop first n frames (default: 4)",
    )
    parser.add_argument(
        "--min-minutes",
        type=float,
        default=3.0,
        help="Minimum remaining usable minutes after scrubbing (default: 3.0)",
    )
    parser.add_argument(
        "--high-pass",
        type=float,
        default=0.01,
        help="High-pass filter in Hz (default: 0.01)",
    )
    parser.add_argument(
        "--low-pass",
        type=float,
        default=0.08,
        help="Low-pass filter in Hz (default: 0.08)",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize ROI time series",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./connectivity_dataset_out",
        help="Output directory",
    )
    return parser.parse_args()


# ----------------------------
# Utilities
# ----------------------------
def resolve_atlas_img_and_labels(atlas_img_path: str | None, atlas_labels_path: str | None):
    """
    Resolve atlas image and ROI labels.

    Priority:
    1) If local atlas image exists, use it. Labels are read from file if available.
    2) If local atlas image does not exist but looks like a Schaefer atlas request,
       auto-fetch Schaefer atlas maps + labels from nilearn.
    """

    # ---------- helper: read local label names ----------
    def _read_local_labels(p: Path):
        if not p.exists():
            return []
        try:
            suffix = p.suffix.lower()
            if suffix == ".txt":
                return [
                    line.strip()
                    for line in p.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            elif suffix in [".csv", ".tsv"]:
                sep = "\t" if suffix == ".tsv" else ","
                df = pd.read_csv(p, sep=sep)
                if df.shape[1] >= 1:
                    vals = df.iloc[:, 0].astype(str).tolist()
                    return [x.strip() for x in vals if x.strip()]
        except Exception as e:
            print(f"[WARN] Failed to read local atlas labels: {e}")
        return []

    # ---------- case 1: local atlas image exists ----------
    if atlas_img_path is not None:
        p = Path(atlas_img_path).expanduser().resolve()
        if p.exists():
            print(f"[INFO] Using local atlas image: {p}")
            labels = []
            if atlas_labels_path is not None:
                labels = _read_local_labels(Path(atlas_labels_path).expanduser().resolve())
                if labels:
                    print(f"[INFO] Loaded {len(labels)} ROI labels from local label file.")
            return str(p), labels

    # ---------- case 2: auto-fetch Schaefer ----------
    atlas_name = Path(atlas_img_path).name if atlas_img_path is not None else "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"

    if "Schaefer2018" in atlas_name:
        from nilearn.datasets import fetch_atlas_schaefer_2018

        parcels_match = re.search(r"(\d+)Parcels", atlas_name)
        networks_match = re.search(r"(\d+)Networks", atlas_name)
        resolution_match = re.search(r"_(\d+)mm", atlas_name)

        n_rois = int(parcels_match.group(1)) if parcels_match else 200
        yeo_networks = int(networks_match.group(1)) if networks_match else 7
        resolution_mm = int(resolution_match.group(1)) if resolution_match else 2

        print(
            f"[INFO] Auto-fetching Schaefer atlas via nilearn "
            f"(n_rois={n_rois}, yeo_networks={yeo_networks}, resolution_mm={resolution_mm})"
        )

        atlas = fetch_atlas_schaefer_2018(
            n_rois=n_rois,
            yeo_networks=yeo_networks,
            resolution_mm=resolution_mm,
        )

        atlas_img = atlas["maps"]
        labels = atlas["labels"]

        cleaned_labels = []
        for x in labels:
            if isinstance(x, bytes):
                cleaned_labels.append(x.decode("utf-8"))
            else:
                cleaned_labels.append(str(x))

        if len(cleaned_labels) == n_rois + 1:
            print(
                f"[INFO] Auto-loaded {len(cleaned_labels)} Schaefer labels; "
                f"dropping first background label to match {n_rois} parcels."
            )
            cleaned_labels = cleaned_labels[1:]

        print(f"[INFO] Final Schaefer ROI labels count: {len(cleaned_labels)}")
        print(f"[INFO] Using fetched atlas image: {atlas_img}")

        return str(atlas_img), cleaned_labels

    raise FileNotFoundError(
        f"Atlas image not found locally and automatic atlas fetching could not be applied: {atlas_img_path}"
    )

def load_roi_names(atlas_labels_path: Optional[str], atlas_img_path: str) -> List[str]:
    """
    Load ROI names.

    Priority:
    1) If atlas_labels_path exists, read from txt/csv/tsv
    2) Otherwise, try auto-fetching Schaefer labels from nilearn
    3) If that fails, return [] and later fallback to ROI_1, ROI_2, ...
    """
    atlas_img_path = Path(atlas_img_path)

    # ---- 1) user-provided atlas labels ----
    if atlas_labels_path is not None:
        p = Path(atlas_labels_path)
        if p.exists():
            try:
                suffix = p.suffix.lower()
                if suffix == ".txt":
                    names = [
                        line.strip()
                        for line in p.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    if len(names) > 0:
                        print(f"[INFO] Loaded ROI labels from TXT: {p}")
                        return names

                elif suffix in [".csv", ".tsv"]:
                    sep = "\t" if suffix == ".tsv" else ","
                    df = pd.read_csv(p, sep=sep)
                    if df.shape[1] >= 1:
                        names = df.iloc[:, 0].astype(str).tolist()
                        names = [x.strip() for x in names if str(x).strip()]
                        if len(names) > 0:
                            print(f"[INFO] Loaded ROI labels from table: {p}")
                            return names

            except Exception as e:
                print(f"[WARN] Failed to read atlas labels from {p}: {e}")
        else:
            print(f"[WARN] atlas label file not found: {p}. Will try auto-loading.")

    # ---- 2) try Schaefer auto-fetch ----
    atlas_name = atlas_img_path.name
    if "Schaefer2018" in atlas_name:
        try:
            from nilearn.datasets import fetch_atlas_schaefer_2018

            parcels_match = re.search(r"(\d+)Parcels", atlas_name)
            networks_match = re.search(r"(\d+)Networks", atlas_name)
            resolution_match = re.search(r"_(\d+)mm", atlas_name)

            n_rois = int(parcels_match.group(1)) if parcels_match else 200
            yeo_networks = int(networks_match.group(1)) if networks_match else 7
            resolution_mm = int(resolution_match.group(1)) if resolution_match else 2

            print(
                f"[INFO] Auto-fetching Schaefer labels via nilearn "
                f"(n_rois={n_rois}, yeo_networks={yeo_networks}, resolution_mm={resolution_mm})"
            )

            atlas = fetch_atlas_schaefer_2018(
                n_rois=n_rois,
                yeo_networks=yeo_networks,
                resolution_mm=resolution_mm,
            )

            labels = atlas["labels"]

            # nilearn may return bytes in some versions
            cleaned_labels = []
            for x in labels:
                if isinstance(x, bytes):
                    cleaned_labels.append(x.decode("utf-8"))
                else:
                    cleaned_labels.append(str(x))

    # Some nilearn Schaefer label sets include a background label
            if len(cleaned_labels) == n_rois + 1:
                print(
                    f"[INFO] Auto-loaded {len(cleaned_labels)} Schaefer labels; "
                    f"dropping first background label to match {n_rois} parcels."
                )
                cleaned_labels = cleaned_labels[1:]

            print(f"[INFO] Final Schaefer ROI labels count: {len(cleaned_labels)}")
            return cleaned_labels

        except Exception as e:
            print(f"[WARN] Failed to auto-fetch Schaefer labels: {e}")

    # ---- 3) fallback ----
    print("[WARN] No valid atlas labels loaded. Will fallback to generic ROI_1, ROI_2, ...")
    return []


def robust_standardize(series: pd.Series) -> pd.Series:
    """
    Robust z-score: 0.6745 * (x - median) / MAD
    Used only when std_dvars is unavailable.
    """
    x = pd.to_numeric(series, errors="coerce")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return pd.Series(np.zeros(len(x)), index=series.index)
    return 0.6745 * (x - med) / mad


def expand_mask(mask: np.ndarray, n_expand: int) -> np.ndarray:
    """
    Expand True values by +/- n_expand frames.
    """
    if n_expand <= 0:
        return mask.copy()

    expanded = mask.copy()
    idx = np.where(mask)[0]
    for i in idx:
        left = max(0, i - n_expand)
        right = min(len(mask), i + n_expand + 1)
        expanded[left:right] = True
    return expanded


def make_subject_key(sub_val: str) -> str:
    """
    Normalize subject ID to no-prefix form, e.g. sub-002S1280 -> 002S1280
    """
    s = str(sub_val)
    return s.replace("sub-", "")


def build_feature_names(roi_names: List[str]) -> List[str]:
    """
    Upper-triangle edge names:
    ROI1__ROI2
    """
    feats = []
    n = len(roi_names)
    for i in range(n):
        for j in range(i + 1, n):
            feats.append(f"{roi_names[i]}__{roi_names[j]}")
    return feats


def vectorize_upper_triangle(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(mat, k=1)
    return mat[iu]


def find_bold_and_confounds(
    derivatives_dir: Path,
    sub: str,
    ses: str,
    task: Optional[str] = None,
    run: Optional[str] = None
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find matching preproc BOLD and confounds file for one run.
    """
    func_dir = derivatives_dir / f"sub-{sub}" / f"ses-{ses}" / "func"
    if not func_dir.exists():
        return None, None

    bold_candidates = sorted(func_dir.glob("*desc-preproc_bold.nii.gz"))
    conf_candidates = sorted(func_dir.glob("*desc-confounds_timeseries.tsv"))

    def _match(cands: List[Path]) -> List[Path]:
        out = []
        for p in cands:
            name = p.name
            if task is not None and f"_task-{task}" not in name:
                continue
            if run is not None and f"_run-{run}" not in name:
                continue
            out.append(p)
        return out

    bold_matched = _match(bold_candidates)
    conf_matched = _match(conf_candidates)

    if len(bold_matched) == 0 or len(conf_matched) == 0:
        return None, None

    # Usually one rest run per subject/session in your dataset
    return bold_matched[0], conf_matched[0]


def choose_confound_columns(conf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build nuisance regressors for denoising.

    Practical default:
    - 24 motion regressors:
      6 motion params
      6 derivatives
      6 squared
      6 derivative squared
    - first 5 aCompCor if available
    """
    motion_cols = [
        "trans_x", "trans_y", "trans_z",
        "rot_x", "rot_y", "rot_z",
        "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
        "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
        "trans_x_power2", "trans_y_power2", "trans_z_power2",
        "rot_x_power2", "rot_y_power2", "rot_z_power2",
        "trans_x_derivative1_power2", "trans_y_derivative1_power2", "trans_z_derivative1_power2",
        "rot_x_derivative1_power2", "rot_y_derivative1_power2", "rot_z_derivative1_power2",
    ]

    existing_motion = [c for c in motion_cols if c in conf_df.columns]

    acomp_cols = sorted([c for c in conf_df.columns if re.fullmatch(r"a_comp_cor_\d+", c)])
    acomp_cols = acomp_cols[:5]

    cols = existing_motion + acomp_cols

    if len(cols) == 0:
        raise ValueError("No confound columns found for nuisance regression.")

    X = conf_df[cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X


def compute_scrub_mask(
    conf_df: pd.DataFrame,
    fd_thresh: float,
    dvars_z_thresh: float,
    expand: int,
    skip_first: int
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute frame-level scrub mask.
    True = scrub / censor
    """
    fd = pd.to_numeric(conf_df["framewise_displacement"], errors="coerce").fillna(0.0)

    if "std_dvars" in conf_df.columns:
        dvars_std = pd.to_numeric(conf_df["std_dvars"], errors="coerce").fillna(0.0)
    elif "dvars" in conf_df.columns:
        dvars_std = robust_standardize(conf_df["dvars"]).fillna(0.0)
    else:
        raise ValueError("Confounds file has neither 'std_dvars' nor 'dvars'.")

    n = len(conf_df)
    scrub = np.zeros(n, dtype=bool)

    # Drop first N frames
    scrub[:skip_first] = True

    bad_fd = fd.to_numpy() > fd_thresh
    bad_dvars = dvars_std.to_numpy() > dvars_z_thresh
    union_bad = bad_fd | bad_dvars

    union_bad = expand_mask(union_bad, expand)
    scrub = scrub | union_bad

    stats = {
        "n_total_frames": n,
        "n_fd_outliers": int(bad_fd[skip_first:].sum()),
        "n_dvars_outliers": int((dvars_std.to_numpy()[skip_first:] > dvars_z_thresh).sum()),
        "fd_mean": float(np.nanmean(fd.to_numpy()[skip_first:])),
        "fd_max": float(np.nanmax(fd.to_numpy()[skip_first:])),
        "dvars_std_mean": float(np.nanmean(dvars_std.to_numpy()[skip_first:])),
        "dvars_std_max": float(np.nanmax(dvars_std.to_numpy()[skip_first:])),
    }
    return scrub, stats


def main() -> None:
    args = parse_args()

    derivatives_dir = Path(args.derivatives_dir)
    qc_csv = Path(args.qc_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not derivatives_dir.exists():
        raise FileNotFoundError(f"derivatives dir not found: {derivatives_dir}")
    if not qc_csv.exists():
        raise FileNotFoundError(f"qc csv not found: {qc_csv}")

    qc_df = pd.read_csv(qc_csv)

    required_cols = {"sub", "ses", "decision"}
    missing = required_cols - set(qc_df.columns)
    if missing:
        raise ValueError(f"QC CSV missing columns: {missing}")

    # Keep KEEP + REVIEW, exclude DROP/ERROR
    qc_df = qc_df[qc_df["decision"].isin(["KEEP", "REVIEW"])].copy()

    qc_df["sub"] = qc_df["sub"].astype(str)
    qc_df["ses"] = qc_df["ses"].astype(str)

    # If "sub" column accidentally contains a full confounds filename,
    # recover subject ID like 002S1280 from it.
    qc_df["sub"] = qc_df["sub"].str.extract(r"(\d{3}S\d{4})", expand=False).fillna(qc_df["sub"])

    # Normalize ses to plain date if needed
    qc_df["ses"] = qc_df["ses"].str.replace("ses-", "", regex=False)

    # Optional labels
    label_map = None
    if args.label_csv is not None:
        label_df = pd.read_csv(args.label_csv)
        if args.label_sub_col not in label_df.columns or args.label_y_col not in label_df.columns:
            raise ValueError("label csv does not contain specified subject/label columns")
        label_df = label_df[[args.label_sub_col, args.label_y_col]].copy()
        label_df[args.label_sub_col] = label_df[args.label_sub_col].astype(str).map(make_subject_key)
        label_map = dict(zip(label_df[args.label_sub_col], label_df[args.label_y_col]))

    atlas_img_resolved, atlas_roi_names = resolve_atlas_img_and_labels(
    args.atlas_img,
    args.atlas_labels,
)

    masker = NiftiLabelsMasker(
        labels_img=atlas_img_resolved,
        standardize=args.standardize,
        detrend=True,
        low_pass=args.low_pass,
        high_pass=args.high_pass,
        t_r=args.tr,
    )

    X_rows: List[np.ndarray] = []
    metadata_rows: List[Dict[str, object]] = []
    dropped_rows: List[Dict[str, object]] = []

    final_feature_names: Optional[List[str]] = None

    for _, row in qc_df.iterrows():
        sub = str(row["sub"])
        ses = str(row["ses"])
        task = str(row["task"]) if "task" in row and pd.notna(row["task"]) else None
        run = str(row["run"]) if "run" in row and pd.notna(row["run"]) else None
        original_decision = str(row["decision"])

        bold_path, conf_path = find_bold_and_confounds(
            derivatives_dir=derivatives_dir,
            sub=sub,
            ses=ses,
            task=task,
            run=run,
        )

        if bold_path is None or conf_path is None:
            dropped_rows.append({
                "sub": sub,
                "ses": ses,
                "task": task,
                "run": run,
                "reason": "missing bold/confounds file",
                "original_decision": original_decision,
            })
            continue

        try:
            conf_df = pd.read_csv(conf_path, sep="\t")

            if "framewise_displacement" not in conf_df.columns:
                raise ValueError("missing framewise_displacement")
            if "std_dvars" not in conf_df.columns and "dvars" not in conf_df.columns:
                raise ValueError("missing std_dvars/dvars")

            scrub_mask, scrub_stats = compute_scrub_mask(
                conf_df=conf_df,
                fd_thresh=args.fd_thresh,
                dvars_z_thresh=args.dvars_z_thresh,
                expand=args.expand,
                skip_first=args.skip_first,
            )

            # sample_mask keeps only non-scrubbed frames
            sample_mask = np.where(~scrub_mask)[0]
            remaining_minutes = len(sample_mask) * args.tr / 60.0

            if remaining_minutes < args.min_minutes:
                dropped_rows.append({
                    "sub": sub,
                    "ses": ses,
                    "task": task,
                    "run": run,
                    "reason": f"remaining_minutes={remaining_minutes:.2f} < {args.min_minutes:.2f}",
                    "original_decision": original_decision,
                })
                continue

            confounds_for_clean = choose_confound_columns(conf_df)

            # This is the key denoising step:
            # - confounds: nuisance regression
            # - sample_mask: scrubbing
            # - masker also applies detrending / filtering / optional standardization
            timeseries = masker.fit_transform(
                str(bold_path),
                confounds=confounds_for_clean,
                sample_mask=sample_mask,
            )

            if timeseries.ndim != 2 or timeseries.shape[0] < 5:
                raise ValueError(f"insufficient cleaned timeseries shape={timeseries.shape}")

            # Infer ROI names after first successful extraction
            if final_feature_names is None:
                n_rois = timeseries.shape[1]

                if len(atlas_roi_names) == n_rois:
                    roi_names = atlas_roi_names
                elif len(atlas_roi_names) > 0 and len(atlas_roi_names) != n_rois:
                    raise ValueError(
                        f"atlas labels count ({len(atlas_roi_names)}) != extracted ROI count ({n_rois})"
                    )
                else:
                    roi_names = [f"ROI_{i+1}" for i in range(n_rois)]

                final_feature_names = build_feature_names(roi_names)

            corr = np.corrcoef(timeseries.T)
            feat_vec = vectorize_upper_triangle(corr)

            if len(feat_vec) != len(final_feature_names):
                raise ValueError("feature vector length mismatch")

            X_rows.append(feat_vec)

            meta = {
                "sub": sub,
                "ses": ses,
                "task": task,
                "run": run,
                "bold_file": str(bold_path),
                "confounds_file": str(conf_path),
                "original_decision": original_decision,
                "remaining_frames": len(sample_mask),
                "remaining_minutes": round(remaining_minutes, 3),
                "n_scrubbed": int(scrub_mask.sum()),
                "pct_scrubbed": round(float(scrub_mask.mean()), 4),
                "fd_mean": round(scrub_stats["fd_mean"], 4),
                "fd_max": round(scrub_stats["fd_max"], 4),
                "dvars_std_mean": round(scrub_stats["dvars_std_mean"], 4),
                "dvars_std_max": round(scrub_stats["dvars_std_max"], 4),
            }

            if label_map is not None:
                meta["y"] = label_map.get(make_subject_key(sub), np.nan)

            metadata_rows.append(meta)

        except Exception as e:
            print(f"[ERROR] sub-{sub} ses-{ses} task-{task} run-{run}: {e}")
            dropped_rows.append({
                "sub": sub,
                "ses": ses,
                "task": task,
                "run": run,
                "reason": str(e),
                "original_decision": original_decision,
            })

    if len(X_rows) == 0:
        raise RuntimeError("No valid runs processed. Check paths / atlas / QC settings.")

    X = np.vstack(X_rows)
    meta_df = pd.DataFrame(metadata_rows)
    feat_df = pd.DataFrame({"feature_name": final_feature_names})
    X_df = pd.DataFrame(X, columns=final_feature_names)

    # Put metadata in front for easier inspection
    X_out = pd.concat([meta_df.reset_index(drop=True), X_df.reset_index(drop=True)], axis=1)

    dropped_df = pd.DataFrame(dropped_rows)

    x_csv = outdir / "X_connectivity.csv"
    meta_csv = outdir / "run_metadata.csv"
    feat_csv = outdir / "feature_names.csv"
    dropped_csv = outdir / "dropped_runs.csv"

    X_out.to_csv(x_csv, index=False)
    meta_df.to_csv(meta_csv, index=False)
    feat_df.to_csv(feat_csv, index=False)
    dropped_df.to_csv(dropped_csv, index=False)

    print(f"Processed runs: {len(meta_df)}")
    print(f"Dropped runs during connectivity build: {len(dropped_df)}")
    print(f"Saved: {x_csv}")
    print(f"Saved: {meta_csv}")
    print(f"Saved: {feat_csv}")
    print(f"Saved: {dropped_csv}")


if __name__ == "__main__":
    main()