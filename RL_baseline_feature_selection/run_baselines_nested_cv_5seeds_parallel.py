import json
import argparse
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from config_baseline_5seeds_parallel import (
    BASELINE_DIR,
    LASSO_PARAM_GRID,
    ELASTIC_NET_PARAM_GRID,
    N_JOBS_GRIDSEARCH,
    N_JOBS_PARALLEL,
    MAX_ITER,
    TOL,
    BASELINE_SEEDS,
)
from data_utils_baseline_5seeds_parallel import (
    load_data,
    summarize_data,
    get_or_create_outer_folds,
    make_inner_cv,
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def make_lasso_pipeline(seed):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1",
            solver="saga",
            class_weight="balanced",
            max_iter=MAX_ITER,
            tol=TOL,
            random_state=seed,
        )),
    ])



def make_elastic_net_pipeline(seed):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=MAX_ITER,
            tol=TOL,
            random_state=seed,
        )),
    ])



def extract_selected_features_from_lr(pipeline, threshold=1e-8):
    coef = pipeline.named_steps["clf"].coef_.ravel()
    selected_idx = np.where(np.abs(coef) > threshold)[0]
    return selected_idx.tolist(), coef



def run_single_fold(method_name, seed, X, y, feature_names, subject_ids, fold_info):
    """
    Run one outer fold for one baseline method and one seed.
    Outer folds remain fixed; seed controls model / inner-CV randomness.
    """
    fold_id = fold_info["fold_id"]
    train_idx = np.asarray(fold_info["train_idx"])
    test_idx = np.asarray(fold_info["test_idx"])

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if method_name == "lasso":
        pipeline = make_lasso_pipeline(seed)
        param_grid = LASSO_PARAM_GRID
    elif method_name == "elastic_net":
        pipeline = make_elastic_net_pipeline(seed)
        param_grid = ELASTIC_NET_PARAM_GRID
    else:
        raise ValueError(f"Unsupported method: {method_name}")

    inner_cv = make_inner_cv(y_train, random_state=seed * 100 + fold_id)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=N_JOBS_GRIDSEARCH,
        refit=True,
        verbose=0,
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_prob = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)

    selected_idx, coef = extract_selected_features_from_lr(best_model)
    selected_feature_names = feature_names[selected_idx].tolist()

    return {
        "method": method_name,
        "seed": int(seed),
        "fold_id": int(fold_id),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_subject_ids": subject_ids[train_idx].tolist(),
        "test_subject_ids": subject_ids[test_idx].tolist(),
        "best_params": grid.best_params_,
        "best_inner_cv_auc": float(grid.best_score_),
        "outer_test_auc": float(test_auc),
        "n_selected_features": int(len(selected_idx)),
        "selected_feature_indices": selected_idx,
        "selected_feature_names": selected_feature_names,
        "coef": coef.tolist(),
    }



def run_one_task(method_name, seed, fold_info, X, y, feature_names, subject_ids, method_dir):
    fold_id = fold_info["fold_id"]
    seed_dir = method_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    fold_outfile = seed_dir / f"{method_name}_seed{seed}_fold{fold_id}.json"
    if fold_outfile.exists():
        print(f"[SKIP] {fold_outfile}")
        with open(fold_outfile, "r") as f:
            return json.load(f)

    print(f"[RUN] method={method_name}, seed={seed}, outer_fold={fold_id}")
    fold_result = run_single_fold(
        method_name=method_name,
        seed=seed,
        X=X,
        y=y,
        feature_names=feature_names,
        subject_ids=subject_ids,
        fold_info=fold_info,
    )

    with open(fold_outfile, "w") as f:
        json.dump(fold_result, f, indent=2)

    return fold_result



def build_seed_summaries(method_name, all_results, method_dir):
    grouped = defaultdict(list)
    for result in all_results:
        grouped[int(result["seed"])].append(result)

    seed_summaries = []
    for seed in sorted(grouped.keys()):
        seed_results = sorted(grouped[seed], key=lambda r: r["fold_id"])
        seed_outer_aucs = [r["outer_test_auc"] for r in seed_results]
        seed_n_selected = [r["n_selected_features"] for r in seed_results]

        seed_summary = {
            "method": method_name,
            "seed": int(seed),
            "n_outer_folds": len(seed_results),
            "mean_outer_auc": float(np.mean(seed_outer_aucs)),
            "std_outer_auc": float(np.std(seed_outer_aucs)),
            "outer_aucs": seed_outer_aucs,
            "mean_n_selected_features": float(np.mean(seed_n_selected)),
            "std_n_selected_features": float(np.std(seed_n_selected)),
        }

        seed_dir = method_dir / f"seed_{seed}"
        with open(seed_dir / f"{method_name}_seed{seed}_summary.json", "w") as f:
            json.dump(seed_summary, f, indent=2)

        seed_summaries.append(seed_summary)

    return seed_summaries



def run_method(method_name):
    X, y, feature_names, subject_ids = load_data()
    summarize_data(X, y, feature_names, subject_ids)
    outer_folds = get_or_create_outer_folds(X, y)

    method_dir = BASELINE_DIR / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for seed in BASELINE_SEEDS:
        for fold_info in outer_folds:
            tasks.append((seed, fold_info))

    all_results = Parallel(n_jobs=N_JOBS_PARALLEL, backend="loky")(
        delayed(run_one_task)(
            method_name,
            seed,
            fold_info,
            X,
            y,
            feature_names,
            subject_ids,
            method_dir,
        )
        for seed, fold_info in tasks
    )

    all_results = sorted(all_results, key=lambda r: (r["seed"], r["fold_id"]))
    seed_summaries = build_seed_summaries(method_name, all_results, method_dir)

    grand_outer_aucs = [r["outer_test_auc"] for r in all_results]
    grand_n_selected = [r["n_selected_features"] for r in all_results]

    summary = {
        "method": method_name,
        "baseline_seeds": list(BASELINE_SEEDS),
        "n_seeds": len(BASELINE_SEEDS),
        "n_total_runs": len(all_results),
        "parallel_tasks": int(N_JOBS_PARALLEL),
        "gridsearch_n_jobs": int(N_JOBS_GRIDSEARCH),
        "mean_outer_auc_across_all_seed_fold_runs": float(np.mean(grand_outer_aucs)),
        "std_outer_auc_across_all_seed_fold_runs": float(np.std(grand_outer_aucs)),
        "mean_of_seed_mean_outer_auc": float(np.mean([s["mean_outer_auc"] for s in seed_summaries])),
        "std_of_seed_mean_outer_auc": float(np.std([s["mean_outer_auc"] for s in seed_summaries])),
        "mean_n_selected_features_across_all_seed_fold_runs": float(np.mean(grand_n_selected)),
        "std_n_selected_features_across_all_seed_fold_runs": float(np.std(grand_n_selected)),
        "seed_summaries": seed_summaries,
    }

    with open(method_dir / f"{method_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["lasso", "elastic_net"],
        help="Which baseline method to run.",
    )
    args = parser.parse_args()
    run_method(args.method)



if __name__ == "__main__":
    main()
