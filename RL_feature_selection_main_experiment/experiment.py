import itertools

import numpy as np

from .config import (
    GLOBAL_SEED,
    MAX_EPISODES,
    PARAM_SELECTION_METRIC,
    PROFILE_SEARCH_SPACE,
    REFIT_SUBSET_SOURCE,
    SEARCH_SPACE,
    TRIALS_FILENAME,
    WARMUP_SOFTMAX_TEMPERATURE,
    JOINT_SOFTMAX_TEMPERATURE,
)
from .trainer import RLSelectorTrainer
from .utils import (
    ensure_dir,
    evaluate_subset_holdout_auc,
    make_inner_cv,
    sanitize_subset,
    save_json,
)


def run_single_trial(X_train, y_train, feature_names, params, seed, logger, trial_dir, max_episodes, trial_idx=0):
    trainer = RLSelectorTrainer(
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        params=params,
        seed=seed,
        logger=logger,
        run_dir=trial_dir,
        trial_idx=trial_idx,
    )
    finished = trainer.train(max_episodes=max_episodes)
    chosen_subset = trainer.get_refit_subset()

    if len(chosen_subset) > 0:
        trainer.final_subset_logit_trajectories = trainer.build_final_subset_logit_trajectories(chosen_subset)
        trainer.final_subset_rank_trajectories = trainer.build_final_subset_rank_trajectories(chosen_subset)
        trainer.final_subset_rank_summary = trainer.build_final_subset_rank_summary(chosen_subset)
    else:
        trainer.final_subset_logit_trajectories = {}
        trainer.final_subset_rank_trajectories = {}
        trainer.final_subset_rank_summary = {}

    trainer.stop_prob_by_selected_count_summary = trainer.summarize_stop_prob_by_selected_count()

    trial_summary = {
        "params": params,
        "refit_subset_source": params.get("refit_subset_source", REFIT_SUBSET_SOURCE),
        "seed": seed,
        "trial_idx": int(trial_idx),
        "finished": finished,
        "best_episode_reward": float(trainer.best_reward),
        "best_episode_inner_auc": float(trainer.best_inner_auc),
        "best_episode_subset": trainer.best_subset,
        "best_episode_subset_size": len(trainer.best_subset) if trainer.best_subset is not None else 0,
        "best_episode": int(trainer.best_episode),
        "best_selection_score": list(trainer.best_selection_score),
        "best_stop_reason": trainer.best_stop_reason,
        "top_episode_candidates": trainer.top_episode_candidates,
        "refit_selection_summary": trainer.refit_selection_summary,
        "refit_subset_source": trainer.refit_subset_source,
        "refit_top_k": int(len(chosen_subset)),
        "final_reward": float(trainer.final_reward),
        "final_inner_auc": float(trainer.final_inner_auc),
        "final_subset": trainer.final_subset,
        "final_subset_size": len(trainer.final_subset) if trainer.final_subset is not None else 0,
        "chosen_subset": chosen_subset.tolist(),
        "chosen_subset_size": int(len(chosen_subset)),
        "policy_logits": trainer.final_policy_logits,
        "warmup_completed": trainer.warmup_completed,
        "warmup_end_episode": trainer.warmup_end_episode,
        "warmup_end_reason": trainer.warmup_end_reason,
        "joint_episode_count": trainer.joint_episode_count,
        "joint_feature_temperature_history": trainer.joint_feature_temperature_history,
        "joint_stop_temperature_history": trainer.joint_stop_temperature_history,
        "joint_stop_barrier_history": trainer.joint_stop_barrier_history,
        "joint_stop_applied_barrier_history": trainer.joint_stop_applied_barrier_history,
        "ppo_stop_entropy_coef_history": trainer.ppo_stop_entropy_coef_history,
        "ppo_feat_entropy_coef_history": trainer.ppo_feat_entropy_coef_history,
        "joint_refit_episode": trainer.joint_refit_episode,
        "joint_refit_reason": trainer.joint_refit_reason,
        "joint_refit_readiness_history": trainer.joint_refit_readiness_history,
        "warmup_transition_history": trainer.warmup_transition_history,
        "initial_prior_summary": trainer.initial_prior_summary,
        "latest_prior_summary": trainer.latest_prior_summary,
        "prior_summary_history": trainer.prior_summary_history,
        "reward_history": trainer.reward_history,
        "moving_avg_history": trainer.moving_avg_history,
        "inner_auc_history": trainer.inner_auc_history,
        "inner_auc_warmup_history": trainer.inner_auc_warmup_history,
        "selected_size_history": trainer.selected_size_history,
        "stop_reason_history": trainer.stop_reason_history,
        "sparsity_penalty_history": trainer.sparsity_penalty_history,
        "cap_penalty_history": trainer.cap_penalty_history,
        "continue_logit_history": trainer.continue_logit_history,
        "stop_logit_history": trainer.stop_logit_history,
        "stop_count_scale_history": trainer.stop_count_scale_history,
        "stop_quality_scale_history": trainer.stop_quality_scale_history,
        "raw_stop_gain_scale_history": trainer.raw_stop_gain_scale_history,
        "stop_gain_scale_history": trainer.stop_gain_scale_history,
        "stop_advantage_scale_history": trainer.stop_advantage_scale_history,
        "phase_history": trainer.phase_history,
        "warmup_softmax_temperature": float(WARMUP_SOFTMAX_TEMPERATURE),
        "joint_softmax_temperature": float(JOINT_SOFTMAX_TEMPERATURE),
        "late_length_penalty_history": trainer.late_length_penalty_history,
        "stop_bonus_history": trainer.stop_bonus_history,
        "compact_bonus_history": trainer.compact_bonus_history,
        "feature_signal_history": trainer.feature_signal_history,
        "feature_bonus_history": trainer.feature_bonus_history,
        "high_reward_feature_boost_history": trainer.high_reward_feature_boost_history,
        "step_cost_history": trainer.step_cost_history,
        "logits_std_history": trainer.logits_std_history,
        "top25_logits_mean_history": trainer.top25_logits_mean_history,
        "top25_logits_std_history": trainer.top25_logits_std_history,
        "top50_logits_mean_history": trainer.top50_logits_mean_history,
        "top50_logits_std_history": trainer.top50_logits_std_history,
        "top25_minus_top50_gap_history": trainer.top25_minus_top50_gap_history,
        "top50_minus_top100_gap_history": trainer.top50_minus_top100_gap_history,
        "max_minus_median_logit_history": trainer.max_minus_median_logit_history,
        "max_median_gap_history": trainer.max_median_gap_history,
        "top50_overlap_recent_history": trainer.top50_overlap_recent_history,
        "top100_overlap_recent_history": trainer.top100_overlap_recent_history,
        "shaping_score_history": trainer.shaping_score_history,
        "pair_pos_ratio_history": trainer.pair_pos_ratio_history,
        "pair_neg_ratio_history": trainer.pair_neg_ratio_history,
        "pair_count_history": trainer.pair_count_history,
        "pair_term_history": trainer.pair_term_history,
        "feature_snapshot_episodes": trainer.feature_snapshot_episodes,
        "joint_stop_step_records": trainer.joint_stop_step_records,
        "final_stop_probability_history": trainer.final_stop_probability_history,
        "final_stop_selected_count_history": trainer.final_stop_selected_count_history,
        "final_subset_logit_trajectories": trainer.final_subset_logit_trajectories,
        "final_subset_rank_trajectories": trainer.final_subset_rank_trajectories,
        "final_subset_rank_summary": trainer.final_subset_rank_summary,
        "stop_prob_by_selected_count_summary": trainer.stop_prob_by_selected_count_summary,
        "joint_checkpoint_history": trainer.joint_checkpoint_history,
        "ppo_update_history": trainer.ppo_update_history,
        "step_structure_bonus_history": trainer.step_structure_bonus_history,
    }
    return trial_summary


def build_search_grid(profile: bool = False):
    active_space = PROFILE_SEARCH_SPACE if profile else SEARCH_SPACE
    return list(
        itertools.product(
            active_space["k_stop"],
            active_space["lambda_reg"],
            active_space["learning_rate"],
            active_space["refit_subset_source"],
        )
    )


def nested_select_best_params(X_outer_train, y_outer_train, feature_names, seed, logger, outer_run_dir, max_episodes, profile=False):
    inner_cv = make_inner_cv(y_outer_train, seed=GLOBAL_SEED + seed)
    search_grid = build_search_grid(profile=profile)
    logger.info(f"Total hyperparameter trials for nested selection: {len(search_grid)}")

    nested_results = []
    best_entry = None
    best_metric = -np.inf

    for trial_idx, (k_stop, lambda_reg, learning_rate, refit_subset_source) in enumerate(search_grid):
        params = {
            "k_stop": int(k_stop),
            "lambda_reg": float(lambda_reg),
            "learning_rate": float(learning_rate),
            "refit_subset_source": refit_subset_source,
        }
        logger.info(f"Nested selection trial {trial_idx + 1}/{len(search_grid)} | params={params}")

        fold_metrics = []
        split_summaries = []

        for inner_fold_id, (tr_idx, va_idx) in enumerate(inner_cv.split(X_outer_train, y_outer_train)):
            X_inner_tr, X_inner_va = X_outer_train[tr_idx], X_outer_train[va_idx]
            y_inner_tr, y_inner_va = y_outer_train[tr_idx], y_outer_train[va_idx]

            split_dir = outer_run_dir / "nested_model_selection" / f"trial_{trial_idx:03d}" / f"inner_fold_{inner_fold_id:02d}"
            ensure_dir(split_dir)

            split_seed = GLOBAL_SEED + seed * 1000 + trial_idx * 100 + inner_fold_id
            trial_summary = run_single_trial(
                X_train=X_inner_tr,
                y_train=y_inner_tr,
                feature_names=feature_names,
                params=params,
                seed=split_seed,
                logger=logger,
                trial_dir=split_dir,
                max_episodes=max_episodes,
                trial_idx=trial_idx * 100 + inner_fold_id,
            )

            subset_for_validation = sanitize_subset(trial_summary["chosen_subset"], X_inner_tr.shape[1])
            val_auc = evaluate_subset_holdout_auc(
                X_fit=X_inner_tr,
                y_fit=y_inner_tr,
                X_eval=X_inner_va,
                y_eval=y_inner_va,
                subset=subset_for_validation,
                random_state=split_seed,
            )
            fold_metrics.append(float(val_auc))
            split_summaries.append({
                "inner_fold_id": int(inner_fold_id),
                "seed": int(split_seed),
                "val_auc": float(val_auc),
                "subset_size": int(len(subset_for_validation)),
                "subset": subset_for_validation.tolist(),
                "training_final_reward": float(trial_summary["final_reward"]),
                "training_final_inner_auc": float(trial_summary["final_inner_auc"]),
            })
            logger.info(
                f"Nested val | trial={trial_idx:03d} | inner_fold={inner_fold_id} | "
                f"val_auc={val_auc:.4f} | subset_size={len(subset_for_validation)}"
            )

        mean_val_auc = float(np.mean(fold_metrics)) if fold_metrics else 0.5
        std_val_auc = float(np.std(fold_metrics)) if fold_metrics else 0.0
        entry = {
            "trial_idx": int(trial_idx),
            "params": params,
            "selection_metric": float(mean_val_auc),
            "metric_name": PARAM_SELECTION_METRIC,
            "mean_val_auc": mean_val_auc,
            "std_val_auc": std_val_auc,
            "inner_fold_results": split_summaries,
        }
        nested_results.append(entry)

        if mean_val_auc > best_metric:
            best_metric = mean_val_auc
            best_entry = entry

        save_json(nested_results, outer_run_dir / TRIALS_FILENAME)

    if best_entry is None:
        raise RuntimeError("Nested CV failed to select a valid hyperparameter setting.")

    return best_entry, nested_results
