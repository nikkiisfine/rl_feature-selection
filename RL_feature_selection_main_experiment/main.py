import argparse
import traceback
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
torch.autograd.set_detect_anomaly(True)

from .config import (
    FEATURE_NAMES_PATH,
    GLOBAL_SEED,
    LOG_ROOT,
    MAX_EPISODES,
    OUTER_FOLDS_PATH,
    PARAM_SELECTION_METRIC,
    REFIT_SUBSET_SOURCE,
    RESULT_FILENAME,
    RESULTS_ROOT,
    STATUS_FILENAME,
    X_PATH,
    Y_PATH,
)
from .experiment import nested_select_best_params, run_single_trial
from .utils import (
    ensure_dir,
    evaluate_subset_outer_test,
    get_task_mapping,
    load_dataset,
    load_outer_folds,
    sanitize_subset,
    save_json,
    set_global_seed,
    setup_logger,
    summarize_selected_networks,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=None, help="Optional manual task id for local testing.")
    parser.add_argument("--episodes", type=int, default=MAX_EPISODES)
    parser.add_argument("--profile", action="store_true", help="Use reduced search space.")
    args = parser.parse_args()

    task_id, seed, fold_id = get_task_mapping(args.task_id)

    run_dir = RESULTS_ROOT / f"seed_{seed}" / f"fold_{fold_id}"
    ensure_dir(run_dir)
    logger = setup_logger(LOG_ROOT / f"seed_{seed}_fold_{fold_id}.log")

    logger.info(f"Starting RL main experiment | task_id={task_id} | seed={seed} | fold={fold_id}")
    logger.info("Target pipeline: prior init -> trajectory sampling -> classifier reward -> policy update -> nested CV -> refit -> outer test")

    try:
        set_global_seed(GLOBAL_SEED + seed)
        X, y, feature_names = load_dataset(X_PATH, Y_PATH, FEATURE_NAMES_PATH)
        outer_folds = load_outer_folds(OUTER_FOLDS_PATH)

        final_result_path = run_dir / RESULT_FILENAME
        if final_result_path.exists():
            logger.info("Final result already exists. Skipping completed job.")
            return

        fold_info = outer_folds[fold_id]
        tr_idx = np.array(fold_info["train_idx"])
        te_idx = np.array(fold_info["test_idx"])

        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        logger.info(
            f"Data loaded | X shape={X.shape} | y shape={y.shape} | "
            f"train={X_train.shape[0]} | test={X_test.shape[0]} | features={X.shape[1]}"
        )

        best_entry, nested_results = nested_select_best_params(
            X_outer_train=X_train,
            y_outer_train=y_train,
            feature_names=feature_names,
            seed=seed,
            logger=logger,
            outer_run_dir=run_dir,
            max_episodes=args.episodes,
            profile=args.profile,
        )
        logger.info(f"Best params selected by nested CV: {best_entry['params']}")

        refit_dir = run_dir / "refit_best_params"
        ensure_dir(refit_dir)
        refit_seed = GLOBAL_SEED + seed * 10000 + fold_id
        refit_summary = run_single_trial(
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            params=best_entry["params"],
            seed=refit_seed,
            logger=logger,
            trial_dir=refit_dir,
            max_episodes=args.episodes,
            trial_idx=99999,
        )

        selected_features = sanitize_subset(refit_summary["chosen_subset"], X_train.shape[1])
        if len(selected_features) == 0:
            raise RuntimeError("Refit with best params did not produce a valid subset.")

        outer_test_auc = evaluate_subset_outer_test(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            subset=selected_features,
        )

        selected_feature_names = [str(feature_names[i]) for i in selected_features]
        selected_network_counts = summarize_selected_networks(selected_feature_names)

        final_result = {
            "task_id": int(task_id),
            "seed": int(seed),
            "fold_id": int(fold_id),
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
            "n_features_total": int(X.shape[1]),
            "pipeline_steps": [
                "initialize_policy_with_prior",
                "sample_subset_trajectory",
                "train_classifier_and_compute_auc_reward",
                "update_policy",
                "nested_cv_select_best_params",
                "refit_rl_on_outer_train",
                "train_classifier_on_selected_subset",
                "evaluate_outer_test_auc",
            ],
            "param_selection_metric": PARAM_SELECTION_METRIC,
            "best_params": best_entry["params"],
            "nested_cv_best_mean_val_auc": float(best_entry["mean_val_auc"]),
            "nested_cv_best_std_val_auc": float(best_entry["std_val_auc"]),
            "refit_subset_source": refit_summary.get(
                "refit_subset_source",
                best_entry["params"].get("refit_subset_source", REFIT_SUBSET_SOURCE),
            ),
            "refit_top_k": int(refit_summary.get("chosen_subset_size", len(selected_features))),
            "refit_selection_summary": refit_summary.get("refit_selection_summary", {}),
            "refit_reward": float(refit_summary["final_reward"]),
            "refit_inner_cv_auc": float(refit_summary["final_inner_auc"]),
            "outer_test_auc": float(outer_test_auc),
            "selected_k": int(len(selected_features)),
            "selected_features": selected_features.tolist(),
            "selected_feature_names": selected_feature_names,
            "selected_network_counts": selected_network_counts,
            "best_episode": int(refit_summary["best_episode"]),
            "warmup_completed": bool(refit_summary.get("warmup_completed", False)),
            "warmup_end_episode": refit_summary.get("warmup_end_episode", None),
            "warmup_end_reason": refit_summary.get("warmup_end_reason", None),
            "joint_episode_count": int(refit_summary.get("joint_episode_count", 0)),
            "best_stop_reason": refit_summary.get("best_stop_reason", None),
            "best_selection_score": refit_summary.get("best_selection_score", None),
            "reward_history": refit_summary["reward_history"],
            "moving_avg_history": refit_summary["moving_avg_history"],
            "inner_auc_history": refit_summary["inner_auc_history"],
            "selected_size_history": refit_summary["selected_size_history"],
            "stop_reason_history": refit_summary["stop_reason_history"],
            "nested_cv_results": nested_results,
            "sparsity_penalty_history": refit_summary["sparsity_penalty_history"],
            "cap_penalty_history": refit_summary["cap_penalty_history"],
            "continue_logit_history": refit_summary["continue_logit_history"],
            "stop_logit_history": refit_summary["stop_logit_history"],
            "stop_count_scale_history": refit_summary["stop_count_scale_history"],
            "stop_quality_scale_history": refit_summary["stop_quality_scale_history"],
            "raw_stop_gain_scale_history": refit_summary["raw_stop_gain_scale_history"],
            "stop_gain_scale_history": refit_summary["stop_gain_scale_history"],
            "stop_advantage_scale_history": refit_summary["stop_advantage_scale_history"],
            "phase_history": refit_summary["phase_history"],
            "warmup_transition_history": refit_summary.get("warmup_transition_history", []),
            "warmup_softmax_temperature": float(refit_summary["warmup_softmax_temperature"]),
            "joint_softmax_temperature": float(refit_summary["joint_softmax_temperature"]),
            "logits_std_history": refit_summary["logits_std_history"],
            "top25_logits_mean_history": refit_summary["top25_logits_mean_history"],
            "top25_logits_std_history": refit_summary["top25_logits_std_history"],
            "top50_logits_mean_history": refit_summary["top50_logits_mean_history"],
            "top50_logits_std_history": refit_summary["top50_logits_std_history"],
            "top25_minus_top50_gap_history": refit_summary["top25_minus_top50_gap_history"],
            "max_minus_median_logit_history": refit_summary["max_minus_median_logit_history"],
            "feature_snapshot_episodes": refit_summary["feature_snapshot_episodes"],
            "final_stop_probability_history": refit_summary["final_stop_probability_history"],
            "final_stop_selected_count_history": refit_summary["final_stop_selected_count_history"],
            "final_subset_rank_summary": refit_summary["final_subset_rank_summary"],
            "stop_prob_by_selected_count_summary": refit_summary["stop_prob_by_selected_count_summary"],
            "joint_checkpoint_history": refit_summary.get("joint_checkpoint_history", []),
            "ppo_update_history": refit_summary.get("ppo_update_history", []),
            "late_length_penalty_history": refit_summary["late_length_penalty_history"],
            "stop_bonus_history": refit_summary["stop_bonus_history"],
            "compact_bonus_history": refit_summary["compact_bonus_history"],
            "feature_signal_history": refit_summary["feature_signal_history"],
            "feature_bonus_history": refit_summary["feature_bonus_history"],
            "step_cost_history": refit_summary["step_cost_history"],
            "refit_summary": {
                "final_reward": float(refit_summary["final_reward"]),
                "final_inner_auc": float(refit_summary["final_inner_auc"]),
                "final_subset_size": int(refit_summary["final_subset_size"]),
                "chosen_subset_size": int(refit_summary["chosen_subset_size"]),
                "best_episode": int(refit_summary["best_episode"]),
                "warmup_completed": bool(refit_summary.get("warmup_completed", False)),
                "warmup_end_episode": refit_summary.get("warmup_end_episode", None),
                "warmup_end_reason": refit_summary.get("warmup_end_reason", None),
                "joint_episode_count": int(refit_summary.get("joint_episode_count", 0)),
                "best_episode_reward": float(refit_summary["best_episode_reward"]),
                "best_episode_inner_auc": float(refit_summary["best_episode_inner_auc"]),
                "best_selection_score": refit_summary.get("best_selection_score", None),
                "best_stop_reason": refit_summary.get("best_stop_reason", None),
                "refit_selection_summary": refit_summary.get("refit_selection_summary", {}),
            },
        }
        save_json(final_result, final_result_path)
        save_json(
            refit_summary.get("final_subset_logit_trajectories", {}),
            run_dir / "final_subset_logit_trajectories.json",
        )
        save_json(
            refit_summary.get("final_subset_rank_trajectories", {}),
            run_dir / "final_subset_rank_trajectories.json",
        )
        save_json(
            refit_summary.get("final_subset_rank_summary", {}),
            run_dir / "final_subset_rank_summary.json",
        )
        save_json(
            {
                "stop_prob_by_selected_count_summary": refit_summary.get("stop_prob_by_selected_count_summary", {}),
                "joint_stop_step_records": refit_summary.get("joint_stop_step_records", []),
                "final_stop_probability_history": refit_summary.get("final_stop_probability_history", []),
                "final_stop_selected_count_history": refit_summary.get("final_stop_selected_count_history", []),
            },
            run_dir / "joint_stop_analysis.json",
        )
        save_json(
            {
                "seed": int(seed),
                "fold_id": int(fold_id),
                "is_completed": True,
                "stop_reason": "main_job_finished",
                "outer_test_auc": float(outer_test_auc),
            },
            run_dir / STATUS_FILENAME,
        )
        logger.info(f"Finished successfully. Outer test AUC = {outer_test_auc:.4f}")

    except Exception as e:
        logger.error(f"Job failed with exception: {e}")
        logger.error(traceback.format_exc())
        save_json(
            {
                "seed": int(seed),
                "fold_id": int(fold_id),
                "is_completed": False,
                "stop_reason": "exception",
                "error": str(e),
            },
            run_dir / STATUS_FILENAME,
        )
        raise


if __name__ == "__main__":
    main()
