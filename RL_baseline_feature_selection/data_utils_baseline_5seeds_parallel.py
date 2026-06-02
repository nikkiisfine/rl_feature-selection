import json
import numpy as np
from sklearn.model_selection import StratifiedKFold

from config_baseline_5seeds_parallel import (
    X_PATH,
    Y_PATH,
    FEATURE_NAMES_PATH,
    SUBJECT_IDS_PATH,
    OUTER_N_SPLITS,
    INNER_N_SPLITS,
    GLOBAL_RANDOM_STATE,
    BASELINE_SEEDS,
    OUTER_FOLD_JSON,
    DATA_SUMMARY_JSON,
)


def load_data():
    """
    Load X, y, feature_names, subject_ids from .npy files.
    """
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    feature_names = np.load(FEATURE_NAMES_PATH, allow_pickle=True)
    subject_ids = np.load(SUBJECT_IDS_PATH, allow_pickle=True)

    if X.shape[0] != len(y):
        raise ValueError(f"Mismatch: X has {X.shape[0]} rows, but y has {len(y)} labels.")
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"Mismatch: X has {X.shape[1]} features, but feature_names has {len(feature_names)} entries."
        )
    if X.shape[0] != len(subject_ids):
        raise ValueError(
            f"Mismatch: X has {X.shape[0]} rows, but subject_ids has {len(subject_ids)} entries."
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int)
    feature_names = np.asarray(feature_names)
    subject_ids = np.asarray(subject_ids)
    return X, y, feature_names, subject_ids


def summarize_data(X, y, feature_names, subject_ids):
    """
    Save dataset summary for reproducibility.
    """
    unique, counts = np.unique(y, return_counts=True)
    label_dist = {int(k): int(v) for k, v in zip(unique, counts)}

    summary = {
        "n_subjects": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "label_distribution": label_dist,
        "first_5_subject_ids": subject_ids[:5].tolist(),
        "first_5_feature_names": feature_names[:5].tolist(),
    }

    with open(DATA_SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)


def create_outer_folds(X, y):
    """
    Create fixed outer CV folds shared across all baselines and RL.
    Outer folds should NOT vary across seeds.
    """
    skf = StratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        shuffle=True,
        random_state=GLOBAL_RANDOM_STATE,
    )

    folds = []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        folds.append(
            {
                "fold_id": fold_id,
                "train_idx": train_idx.tolist(),
                "test_idx": test_idx.tolist(),
            }
        )
    return folds


def save_outer_folds(folds):
    with open(OUTER_FOLD_JSON, "w") as f:
        json.dump(folds, f, indent=2)



def load_outer_folds():
    with open(OUTER_FOLD_JSON, "r") as f:
        return json.load(f)



def get_or_create_outer_folds(X, y):
    """
    Load outer folds if they already exist; otherwise create and save them.
    """
    if OUTER_FOLD_JSON.exists():
        return load_outer_folds()
    folds = create_outer_folds(X, y)
    save_outer_folds(folds)
    return folds



def make_inner_cv(y_train, random_state):
    """
    Inner CV inside each outer fold.
    Seed-dependent, so baseline seeds capture some training randomness.
    """
    return StratifiedKFold(
        n_splits=INNER_N_SPLITS,
        shuffle=True,
        random_state=random_state,
    )



def get_baseline_seeds():
    return list(BASELINE_SEEDS)
