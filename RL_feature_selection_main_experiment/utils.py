import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    BASE_PRIOR_JOINT,
    BASE_PRIOR_WARMUP,
    CLASSIFIER_MAX_ITER,
    CLASS_WEIGHT,
    GLOBAL_SEED,
    INNER_N_SPLITS,
    MOVING_AVG_WINDOW,
    N_FOLDS_FOR_ARRAY,
    NETWORK_PAIR_PRIOR_JOINT,
    NETWORK_PAIR_PRIOR_WARMUP,
    PLATEAU_PATIENCE,
    PLATEAU_STD_TOL,
    RL_SEEDS,
    SAFETY_STOP_HOURS,
    SUBNETWORK_PAIR_PRIOR_JOINT,
    SUBNETWORK_PAIR_PRIOR_WARMUP,
)

# ---------------------------------------------------------------------------
# Seed / randomness
# ---------------------------------------------------------------------------

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(x_path: Path, y_path: Path, feature_names_path: Path):
    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(int)
    feature_names = np.load(feature_names_path, allow_pickle=True)

    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got shape={X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected y to be 1D, got shape={y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Sample size mismatch: X={X.shape[0]}, y={y.shape[0]}")
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature dimension mismatch: X has {X.shape[1]} columns, "
            f"but feature_names has {len(feature_names)} entries"
        )
    return X, y, feature_names


def load_outer_folds(path: Path):
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# SLURM task → (seed, fold) mapping
# ---------------------------------------------------------------------------

def get_task_mapping(task_id: int = None) -> Tuple[int, int, int]:
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n_folds = N_FOLDS_FOR_ARRAY
    seed_idx = task_id // n_folds
    fold_id = task_id % n_folds
    if seed_idx >= len(RL_SEEDS):
        raise ValueError(
            f"task_id={task_id} maps to seed_idx={seed_idx}, "
            f"but only {len(RL_SEEDS)} seeds are configured."
        )
    seed = RL_SEEDS[seed_idx]
    return task_id, seed, fold_id


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------

def make_inner_cv(y_train: np.ndarray, seed: int):
    return StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=seed)


# ---------------------------------------------------------------------------
# Downstream classifiers
# ---------------------------------------------------------------------------

def make_downstream_classifier(random_state: int = GLOBAL_SEED):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            C=0.3,
            solver="lbfgs",
            class_weight=CLASS_WEIGHT,
            max_iter=CLASSIFIER_MAX_ITER,
            random_state=random_state,
        )),
    ])


def refit_make_downstream_classifier(random_state: int = GLOBAL_SEED):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            C=0.3,
            solver="lbfgs",
            class_weight=CLASS_WEIGHT,
            max_iter=CLASSIFIER_MAX_ITER,
            random_state=random_state,
        )),
    ])


def make_warmup_classifier():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            C=0.5,
            solver="liblinear",
            max_iter=1000,
            random_state=0,
        )),
    ])


# ---------------------------------------------------------------------------
# Subset validation helpers
# ---------------------------------------------------------------------------

def sanitize_subset(subset, n_features: int) -> np.ndarray:
    if subset is None:
        return np.asarray([], dtype=int)
    subset = np.asarray(subset, dtype=int).reshape(-1)
    subset = np.unique(subset)
    subset = subset[(subset >= 0) & (subset < n_features)]
    return subset.astype(int)


# ---------------------------------------------------------------------------
# AUC evaluation
# ---------------------------------------------------------------------------

def evaluate_subset_auc_cv(
    X: np.ndarray,
    y: np.ndarray,
    subset: np.ndarray,
    inner_cv,
) -> float:
    subset = sanitize_subset(subset, X.shape[1])
    if len(subset) == 0:
        return 0.5

    X_sub = X[:, subset]
    aucs = []
    for tr_idx, va_idx in inner_cv.split(X_sub, y):
        y_va = y[va_idx]
        if len(np.unique(y_va)) < 2:
            continue
        clf = make_downstream_classifier()
        clf.fit(X_sub[tr_idx], y[tr_idx])
        y_prob = clf.predict_proba(X_sub[va_idx])[:, 1]
        aucs.append(roc_auc_score(y_va, y_prob))

    return float(np.mean(aucs)) if aucs else 0.5


def evaluate_subset_auc_cv_warmup(
    X: np.ndarray,
    y: np.ndarray,
    subset: np.ndarray,
    inner_cv,
) -> float:
    subset = sanitize_subset(subset, X.shape[1])
    if len(subset) == 0:
        return 0.5

    X_sub = X[:, subset]
    aucs = []
    for tr_idx, va_idx in inner_cv.split(X_sub, y):
        y_va = y[va_idx]
        if len(np.unique(y_va)) < 2:
            continue
        clf = make_warmup_classifier()
        clf.fit(X_sub[tr_idx], y[tr_idx])
        y_prob = clf.predict_proba(X_sub[va_idx])[:, 1]
        aucs.append(roc_auc_score(y_va, y_prob))

    if not aucs:
        return 0.5
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    # Warmup prioritises stability over occasional high scores
    return mean_auc - 0.05 * std_auc


def evaluate_subset_holdout_auc(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    subset: np.ndarray,
    random_state: int = GLOBAL_SEED,
) -> float:
    subset = sanitize_subset(subset, X_fit.shape[1])
    if len(subset) == 0:
        return 0.5
    if len(np.unique(y_fit)) < 2:
        return 0.5
    if len(np.unique(y_eval)) < 2:
        return 0.5

    clf = refit_make_downstream_classifier(random_state=random_state)
    clf.fit(X_fit[:, subset], y_fit)
    y_prob = clf.predict_proba(X_eval[:, subset])[:, 1]
    return float(roc_auc_score(y_eval, y_prob))


def evaluate_subset_outer_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subset: np.ndarray,
) -> float:
    return evaluate_subset_holdout_auc(
        X_fit=X_train,
        y_fit=y_train,
        X_eval=X_test,
        y_eval=y_test,
        subset=subset,
        random_state=GLOBAL_SEED,
    )


# ---------------------------------------------------------------------------
# Training signal utilities
# ---------------------------------------------------------------------------

def moving_average(values: List[float], window: int = MOVING_AVG_WINDOW) -> List[float]:
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(values[start: i + 1])))
    return out


def plateau_detected(ma_hist: List[float]) -> bool:
    if len(ma_hist) < PLATEAU_PATIENCE:
        return False
    recent = ma_hist[-PLATEAU_PATIENCE:]
    return float(np.std(recent)) < PLATEAU_STD_TOL


# ---------------------------------------------------------------------------
# Feature / network name parsing
# ---------------------------------------------------------------------------

_NETWORK_KEYWORD_LIST = ["Default", "Limbic", "SalVentAttn", "Cont", "DorsAttn", "Vis", "SomMot"]


def parse_network_from_roi(roi_name: str) -> str:
    roi_name = str(roi_name)
    tokens = roi_name.split("_")
    for key in _NETWORK_KEYWORD_LIST:
        if key in tokens or key in roi_name:
            return key
    return "UNKNOWN"


def parse_roi_label(roi_name: str):
    roi_name = str(roi_name)
    tokens = roi_name.split("_")
    network = parse_network_from_roi(roi_name)
    if network == "UNKNOWN":
        return "UNKNOWN", "UNKNOWN"
    try:
        net_idx = tokens.index(network)
    except ValueError:
        return network, network
    suffix = tokens[net_idx + 1:]
    if suffix and suffix[-1].isdigit():
        suffix = suffix[:-1]
    if not suffix:
        return network, network
    if network == "Cont" and suffix and suffix[0].startswith("PFC"):
        subnetwork = f"{network}_{suffix[0]}"
    else:
        subnetwork = f"{network}_{'_'.join(suffix)}"
    return network, subnetwork


def parse_edge_networks(feature_name: str) -> Tuple[str, str]:
    parts = str(feature_name).split("__")
    if len(parts) != 2:
        return "UNKNOWN", "UNKNOWN"
    return parse_network_from_roi(parts[0]), parse_network_from_roi(parts[1])


def get_selected_subnetwork_pairs(selected_features, feature_names):
    pairs = set()
    for idx in selected_features:
        if int(idx) < 0 or int(idx) >= len(feature_names):
            continue
        fname = str(feature_names[int(idx)])
        parts = fname.split("__")
        if len(parts) != 2:
            continue
        left, right = parts
        try:
            _, left_sub = parse_roi_label(left)
            _, right_sub = parse_roi_label(right)
        except Exception:
            continue
        if left_sub is None or right_sub is None:
            continue
        pairs.add(tuple(sorted((left_sub, right_sub))))
    return pairs


def summarize_selected_networks(selected_feature_names: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for fname in selected_feature_names:
        n1, n2 = parse_edge_networks(str(fname))
        key = f"{n1}__{n2}"
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Network prior construction
# ---------------------------------------------------------------------------

def _normalize_pair_dict(d):
    return {tuple(sorted(k)): float(v) for k, v in d.items()}


_NETWORK_PAIR_PRIOR_WARMUP = _normalize_pair_dict(NETWORK_PAIR_PRIOR_WARMUP)
_NETWORK_PAIR_PRIOR_JOINT = _normalize_pair_dict(NETWORK_PAIR_PRIOR_JOINT)
_SUBNETWORK_PAIR_PRIOR_WARMUP = _normalize_pair_dict(SUBNETWORK_PAIR_PRIOR_WARMUP)
_SUBNETWORK_PAIR_PRIOR_JOINT = _normalize_pair_dict(SUBNETWORK_PAIR_PRIOR_JOINT)


def build_network_prior(feature_names, stage: str = "warmup") -> np.ndarray:
    priors = np.zeros(len(feature_names), dtype=np.float32)
    if stage == "warmup":
        base_prior = BASE_PRIOR_WARMUP
        subnet_pair_prior = _SUBNETWORK_PAIR_PRIOR_WARMUP
        net_pair_prior = _NETWORK_PAIR_PRIOR_WARMUP
    else:
        base_prior = BASE_PRIOR_JOINT
        subnet_pair_prior = _SUBNETWORK_PAIR_PRIOR_JOINT
        net_pair_prior = _NETWORK_PAIR_PRIOR_JOINT

    for i, feat in enumerate(feature_names):
        feat = str(feat)
        parts = feat.split("__")
        if len(parts) != 2:
            priors[i] = float(base_prior)
            continue
        left, right = parts
        left_net, left_sub = parse_roi_label(left)
        right_net, right_sub = parse_roi_label(right)
        subnet_pair = tuple(sorted((left_sub, right_sub)))
        net_pair = tuple(sorted((left_net, right_net)))
        p_sub = float(subnet_pair_prior.get(subnet_pair, 0.0))
        p_net = float(net_pair_prior.get(net_pair, 0.0))
        priors[i] = float(base_prior + p_sub + 0.6 * p_net)
    return priors


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

def jaccard_similarity(a: List[int], b: List[int]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_json(obj, path: Path):
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def setup_logger(log_file: Path, logger_name: str = "rl_main"):
    ensure_dir(log_file.parent)
    logger = logging.getLogger(logger_name + str(log_file))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def save_checkpoint(checkpoint: dict, path: Path):
    ensure_dir(path.parent)
    torch.save(checkpoint, path)


def load_checkpoint(path: Path):
    return torch.load(path, map_location="cpu")


def should_stop_for_walltime(start_time: float) -> bool:
    elapsed_hours = (time.time() - start_time) / 3600.0
    return elapsed_hours >= SAFETY_STOP_HOURS
