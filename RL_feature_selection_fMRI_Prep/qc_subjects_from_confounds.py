#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class QCResult:
    sub: str
    ses: str
    run: str
    path: str

    n_vol: int
    tr: Optional[float]

    # FD  framewise_Displancement 相邻两个TR之间头动变化量
    mean_fd: float
    max_fd: float
    pct_fd_gt: float

    # DVARS (raw) 全脑信号在相邻TR之间变化多少
    mean_dvars: float
    max_dvars: float

    # DVARS (z)
    mean_dvars_z: float
    max_dvars_z: float
    pct_dvars_z_gt: float

    # Decision
    decision: str
    reason: str
    remaining_minutes: Optional[float]


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "n/a":
            return None
        return float(s)
    except Exception:
        return None


def parse_entities(tsv_path: str) -> Tuple[str, str, str]:
    """Extract sub/ses/run from BIDS-like filename if possible."""
    base = os.path.basename(tsv_path)
    sub = "NA"
    ses = "NA"
    run = "NA"
    for part in base.split("_"):
        if part.startswith("sub-"):
            sub = part.replace("sub-", "")
        elif part.startswith("ses-"):
            ses = part.replace("ses-", "")
        elif part.startswith("run-"):
            run = part.replace("run-", "")
    return sub, ses, run


def read_column(tsv_path: str, colname: str) -> List[float]:
    vals: List[float] = []
    with open(tsv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or colname not in reader.fieldnames:
            return []
        for row in reader:
            v = safe_float(row.get(colname))
            if v is None:
                continue
            vals.append(v)
    return vals


def compute_stats(x: List[float], skip_first: int = 0) -> Tuple[float, float]:
    if skip_first > 0:
        x = x[skip_first:]
    if not x:
        return (float("nan"), float("nan"))
    return (sum(x) / len(x), max(x))


def pct_gt(x: List[float], thr: float, skip_first: int = 0) -> float:
    if skip_first > 0:
        x = x[skip_first:]
    if not x:
        return float("nan")
    cnt = sum(1 for v in x if v > thr)
    return cnt / len(x)


def remaining_minutes(n_vol: int, tr: Optional[float], n_scrub: int, skip_first: int = 0) -> Optional[float]:
    if tr is None:
        return None
    usable = max(0, (n_vol - skip_first) - n_scrub)
    return usable * tr / 60.0


def decide(
    *,
    mean_fd: float,
    pct_fd_gt: float,
    max_fd: float,
    mean_dvars_z: float,
    pct_dvars_z_gt: float,
    max_dvars_z: float,
    rem_min: Optional[float],
    # thresholds
    fd_thr: float,
    fd_pct_soft: float,
    fd_pct_hard: float,
    mean_fd_soft: float,
    mean_fd_hard: float,
    dvars_z_thr: float,
    dvars_pct_soft: float,
    dvars_pct_hard: float,
    dvars_max_soft: float,
    dvars_max_hard: float,
    min_minutes_soft: Optional[float],
    min_minutes_hard: Optional[float],
) -> Tuple[str, str]:
    reasons: List[str] = []

    # Hard drop conditions
    hard = False
    if pct_fd_gt == pct_fd_gt and pct_fd_gt >= fd_pct_hard:
        hard = True
        reasons.append(f"FD%>thr={pct_fd_gt:.3f} >= {fd_pct_hard:.3f}")
    if mean_fd == mean_fd and mean_fd >= mean_fd_hard:
        hard = True
        reasons.append(f"meanFD={mean_fd:.3f} >= {mean_fd_hard:.3f}")
    if max_dvars_z == max_dvars_z and max_dvars_z >= dvars_max_hard:
        hard = True
        reasons.append(f"maxDVARSz={max_dvars_z:.3f} >= {dvars_max_hard:.3f}")
    if pct_dvars_z_gt == pct_dvars_z_gt and pct_dvars_z_gt >= dvars_pct_hard:
        hard = True
        reasons.append(f"DVARSz%>thr={pct_dvars_z_gt:.3f} >= {dvars_pct_hard:.3f}")
    if min_minutes_hard is not None and rem_min is not None and rem_min < min_minutes_hard:
        hard = True
        reasons.append(f"remaining={rem_min:.2f}min < {min_minutes_hard:.2f}min")

    if hard:
        return "DROP", "; ".join(reasons) if reasons else "Hard criteria met"

    # Soft review conditions
    soft = False
    if pct_fd_gt == pct_fd_gt and pct_fd_gt >= fd_pct_soft:
        soft = True
        reasons.append(f"FD%>thr={pct_fd_gt:.3f} >= {fd_pct_soft:.3f}")
    if mean_fd == mean_fd and mean_fd >= mean_fd_soft:
        soft = True
        reasons.append(f"meanFD={mean_fd:.3f} >= {mean_fd_soft:.3f}")
    if max_dvars_z == max_dvars_z and max_dvars_z >= dvars_max_soft:
        soft = True
        reasons.append(f"maxDVARSz={max_dvars_z:.3f} >= {dvars_max_soft:.3f}")
    if pct_dvars_z_gt == pct_dvars_z_gt and pct_dvars_z_gt >= dvars_pct_soft:
        soft = True
        reasons.append(f"DVARSz%>thr={pct_dvars_z_gt:.3f} >= {dvars_pct_soft:.3f}")
    if min_minutes_soft is not None and rem_min is not None and rem_min < min_minutes_soft:
        soft = True
        reasons.append(f"remaining={rem_min:.2f}min < {min_minutes_soft:.2f}min")

    if soft:
        return "REVIEW", "; ".join(reasons) if reasons else "Soft criteria met"

    return "KEEP", "OK"


def compute_qc_one(
    tsv_path: str,
    *,
    fd_thr: float,
    dvars_z_thr: float,
    skip_first: int,
    tr: Optional[float],
    min_minutes_soft: Optional[float],
    min_minutes_hard: Optional[float],
    fd_pct_soft: float,
    fd_pct_hard: float,
    mean_fd_soft: float,
    mean_fd_hard: float,
    dvars_pct_soft: float,
    dvars_pct_hard: float,
    dvars_max_soft: float,
    dvars_max_hard: float,
) -> QCResult:
    sub, ses, run = parse_entities(tsv_path)

    fd = read_column(tsv_path, "framewise_displacement")
    dvars = read_column(tsv_path, "dvars")
    dvars_z = read_column(tsv_path, "dvars_std")  # some confounds use dvars_std
    if not dvars_z:
        dvars_z = read_column(tsv_path, "std_dvars")  # alternative name
    if not dvars_z:
        dvars_z = read_column(tsv_path, "dvars_z")    # rare, but try

    # Compute basic n_vol from FD length if possible, else from any available series
    series_for_len = fd or dvars or dvars_z
    n_vol = len(series_for_len)

    mean_fd, max_fd = compute_stats(fd, skip_first=skip_first)
    pct_fd = pct_gt(fd, fd_thr, skip_first=skip_first)

    mean_dvars, max_dvars = compute_stats(dvars, skip_first=skip_first)

    mean_dz, max_dz = compute_stats(dvars_z, skip_first=skip_first) if dvars_z else (float("nan"), float("nan"))
    pct_dz = pct_gt(dvars_z, dvars_z_thr, skip_first=skip_first) if dvars_z else float("nan")

    # Scrub count (FD-based)
    n_scrub = 0
    if skip_first > 0:
        fd_for_scrub = fd[skip_first:]
    else:
        fd_for_scrub = fd
    if fd_for_scrub:
        n_scrub = sum(1 for v in fd_for_scrub if v > fd_thr)

    rem_min = remaining_minutes(n_vol=n_vol, tr=tr, n_scrub=n_scrub, skip_first=skip_first)

    decision, reason = decide(
        mean_fd=mean_fd, pct_fd_gt=pct_fd, max_fd=max_fd,
        mean_dvars_z=mean_dz, pct_dvars_z_gt=pct_dz, max_dvars_z=max_dz,
        rem_min=rem_min,
        fd_thr=fd_thr,
        fd_pct_soft=fd_pct_soft, fd_pct_hard=fd_pct_hard,
        mean_fd_soft=mean_fd_soft, mean_fd_hard=mean_fd_hard,
        dvars_z_thr=dvars_z_thr,
        dvars_pct_soft=dvars_pct_soft, dvars_pct_hard=dvars_pct_hard,
        dvars_max_soft=dvars_max_soft, dvars_max_hard=dvars_max_hard,
        min_minutes_soft=min_minutes_soft, min_minutes_hard=min_minutes_hard,
    )

    return QCResult(
        sub=sub, ses=ses, run=run, path=tsv_path,
        n_vol=n_vol, tr=tr,
        mean_fd=mean_fd, max_fd=max_fd, pct_fd_gt=pct_fd,
        mean_dvars=mean_dvars, max_dvars=max_dvars,
        mean_dvars_z=mean_dz, max_dvars_z=max_dz, pct_dvars_z_gt=pct_dz,
        decision=decision, reason=reason,
        remaining_minutes=rem_min,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmriprep_dir", required=True, help="Path to fMRIPrep output directory (contains sub-*/)")
    ap.add_argument(
        "--pattern",
        default="sub-*/ses-*/func/*desc-confounds_timeseries.tsv",
        help="Glob pattern inside fMRIPrep dir"
    )

    # FD settings
    ap.add_argument("--fd", type=float, default=0.5, help="FD threshold (mm) for scrubbing")
    ap.add_argument("--fd-pct-soft", type=float, default=0.20, help="REVIEW if pct(FD>thr) >= this")
    ap.add_argument("--fd-pct-hard", type=float, default=0.50, help="DROP if pct(FD>thr) >= this")
    ap.add_argument("--mean-fd-soft", type=float, default=0.30, help="REVIEW if mean FD >= this")
    ap.add_argument("--mean-fd-hard", type=float, default=0.60, help="DROP if mean FD >= this")

    # DVARS settings (z-score type)
    ap.add_argument("--dvars-z", type=float, default=1.5, help="DVARS z threshold")
    ap.add_argument("--dvars-pct-soft", type=float, default=0.10, help="REVIEW if pct(DVARSz>thr) >= this")
    ap.add_argument("--dvars-pct-hard", type=float, default=0.25, help="DROP if pct(DVARSz>thr) >= this")
    ap.add_argument("--dvars-max-soft", type=float, default=4.0, help="REVIEW if max DVARSz >= this")
    ap.add_argument("--dvars-max-hard", type=float, default=6.0, help="DROP if max DVARSz >= this")

    # Timing / scrubbing
    ap.add_argument("--tr", type=float, default=None, help="TR in seconds (optional, used to compute remaining minutes)")
    ap.add_argument("--skip-first", type=int, default=0, help="Ignore first N volumes when computing QC")
    ap.add_argument("--min-minutes-soft", type=float, default=None, help="REVIEW if remaining minutes < this (requires --tr)")
    ap.add_argument("--min-minutes-hard", type=float, default=None, help="DROP if remaining minutes < this (requires --tr)")

    ap.add_argument("--out", default="qc_summary.csv", help="Output CSV file")

    args = ap.parse_args()

    search_glob = os.path.join(args.fmriprep_dir, args.pattern)
    paths = sorted(glob.glob(search_glob))
    if not paths:
        raise SystemExit(f"No files matched: {search_glob}")

    results: List[QCResult] = []
    for p in paths:
        results.append(
            compute_qc_one(
                p,
                fd_thr=args.fd,
                dvars_z_thr=args.dvars_z,
                skip_first=args.skip_first,
                tr=args.tr,
                min_minutes_soft=args.min_minutes_soft,
                min_minutes_hard=args.min_minutes_hard,
                fd_pct_soft=args.fd_pct_soft,
                fd_pct_hard=args.fd_pct_hard,
                mean_fd_soft=args.mean_fd_soft,
                mean_fd_hard=args.mean_fd_hard,
                dvars_pct_soft=args.dvars_pct_soft,
                dvars_pct_hard=args.dvars_pct_hard,
                dvars_max_soft=args.dvars_max_soft,
                dvars_max_hard=args.dvars_max_hard,
            )
        )

    # Write CSV
    fieldnames = [
        "sub", "ses", "run", "path",
        "n_vol", "tr",
        "mean_fd", "max_fd", "pct_fd_gt",
        "mean_dvars", "max_dvars",
        "mean_dvars_z", "max_dvars_z", "pct_dvars_z_gt",
        "remaining_minutes",
        "decision", "reason",
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({
                "sub": r.sub, "ses": r.ses, "run": r.run, "path": r.path,
                "n_vol": r.n_vol, "tr": r.tr,
                "mean_fd": r.mean_fd, "max_fd": r.max_fd, "pct_fd_gt": r.pct_fd_gt,
                "mean_dvars": r.mean_dvars, "max_dvars": r.max_dvars,
                "mean_dvars_z": r.mean_dvars_z, "max_dvars_z": r.max_dvars_z, "pct_dvars_z_gt": r.pct_dvars_z_gt,
                "remaining_minutes": r.remaining_minutes,
                "decision": r.decision, "reason": r.reason,
            })

    # Console summary
    n_keep = sum(1 for r in results if r.decision == "KEEP")
    n_review = sum(1 for r in results if r.decision == "REVIEW")
    n_drop = sum(1 for r in results if r.decision == "DROP")
    print(f"Matched {len(results)} runs")
    print(f"KEEP={n_keep}  REVIEW={n_review}  DROP={n_drop}")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
