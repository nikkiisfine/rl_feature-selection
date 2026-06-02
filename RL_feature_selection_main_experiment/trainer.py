import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import (
    ADAPTIVE_WARMUP_MIN_EPISODES,
    BARRIER_SCALE,
    CAP_BASE_PENALTY,
    CAP_LATE_SCALE,
    CAP_LATE_THRESHOLD,
    CHECKPOINT_EVERY,
    CHECKPOINT_FILENAME,
    GRAD_CLIP_NORM,
    HIGH_REWARD_FEATURE_CREDIT_EMA,
    HIGH_REWARD_FEATURE_CREDIT_JOINT_SCALE,
    HIGH_REWARD_FEATURE_CREDIT_WARMUP_SCALE,
    HIGH_REWARD_RECENT_WINDOW,
    HIGH_REWARD_STABLE_MIN_OCCURRENCES,
    HIGH_REWARD_STABLE_TOP_K,
    HYBRID_SCORE_ALPHA,
    JOINT_CHECKPOINT_EVAL_EVERY,
    JOINT_CHECKPOINT_REFIT_AUC_DELTA_MAX,
    JOINT_CHECKPOINT_SIZE_STD_MAX,
    JOINT_CHECKPOINT_STABILITY_WINDOW,
    JOINT_FEATURE_BONUS_SCALE,
    JOINT_FEATURE_TEMP_EARLY,
    JOINT_FEATURE_TEMP_LATE,
    JOINT_FEATURE_TEMP_MID,
    JOINT_GREEDY_OVERLAP_THRESHOLD,
    JOINT_LENGTH_STD_MAX,
    JOINT_POSITIVE_SHAPING_NEG_COEF,
    JOINT_PRIOR_DELTA_SCALE,
    JOINT_PRIOR_STAGE,
    JOINT_READY_MIN_CHECKPOINTS,
    JOINT_REFIT_MIN_EPISODES,
    JOINT_REWARD_PLATEAU_DELTA,
    JOINT_REWARD_PLATEAU_WINDOW,
    JOINT_SAMPLED_OVERLAP_THRESHOLD,
    JOINT_STOP_BARRIER_EARLY,
    JOINT_STOP_BARRIER_LATE,
    JOINT_STOP_BARRIER_MID,
    JOINT_STOP_TEMP_EARLY,
    JOINT_STOP_TEMP_LATE,
    JOINT_STOP_TEMP_MID,
    LATE_LENGTH_PENALTY_SCALE,
    LOG_EVERY,
    MAX_SELECTION_STEPS_BUFFER,
    MIN_FEATURES_BEFORE_STOP,
    MOVING_AVG_WINDOW,
    NETWORK_PAIR_PRIOR_JOINT,
    NETWORK_PAIR_PRIOR_WARMUP,
    NETWORK_PAIR_SHAPING_COEF,
    NON_TARGET_NETWORKS,
    PHASE2_FEATURE_LR,
    PHASE2_STOP_LR,
    PPO_CHECKPOINT_SAMPLED_ROLLOUTS,
    PPO_CLIP_EPS,
    PPO_CONSENSUS_ROLLOUTS,
    PPO_EPOCHS,
    PPO_FEAT_ENTROPY_COEF_EARLY,
    PPO_FEAT_ENTROPY_COEF_LATE,
    PPO_FEAT_ENTROPY_COEF_MID,
    PPO_LENGTH_NORMALIZE_ADVANTAGE,
    PPO_LR,
    PPO_MAX_GRAD_NORM,
    PPO_MINI_BATCH_SIZE,
    PPO_STEP_STRUCTURE_BONUS_SCALE,
    PPO_STEP_STRUCTURE_SUBNETWORK_MIX,
    PPO_STOP_ENTROPY_COEF_EARLY,
    PPO_STOP_ENTROPY_COEF_LATE,
    PPO_STOP_ENTROPY_COEF_MID,
    PPO_UPDATE_EVERY,
    PPO_VALUE_COEF,
    REFIT_SUBSET_SOURCE,
    SPARSITY_LINEAR_WEIGHT,
    STATUS_FILENAME,
    STEP_COST,
    STOP_REMAINING_TOP_M,
    SUBNETWORK_PAIR_PRIOR_JOINT,
    INIT_STOP_LOGIT,
    INIT_STOP_COUNT_SCALE,
    INIT_CONTINUE_LOGIT,
    TOP_EPISODES_FOR_REFIT,
    WARMUP_ABOVE_MEDIAN_FRACTION_THRESHOLD,
    WARMUP_COARSE_PAIR_SHAPING_SCALE,
    WARMUP_FALLBACK_MAX_MEDIAN_GAP_THRESHOLD,
    WARMUP_FALLBACK_TOP25_TOP50_GAP_THRESHOLD,
    WARMUP_FALLBACK_TOP50_OVERLAP_THRESHOLD,
    WARMUP_MAX_MEDIAN_GAP_THRESHOLD,
    WARMUP_MIN_STEPS,
    WARMUP_MAX_STEPS,
    WARMUP_NO_IMPROVEMENT_DELTA_THRESHOLD,
    WARMUP_NO_IMPROVEMENT_WINDOW,
    WARMUP_PLATEAU_WINDOW,
    WARMUP_PRIOR_STAGE,
    WARMUP_REWARD_WINDOW,
    WARMUP_SOFTMAX_TEMPERATURE,
    WARMUP_TOP25_TOP50_GAP_THRESHOLD,
    WARMUP_TOP50_OVERLAP_THRESHOLD,
)
from .model import SequentialPolicy
from .utils import (
    build_network_prior,
    ensure_dir,
    evaluate_subset_auc_cv,
    evaluate_subset_auc_cv_warmup,
    get_selected_subnetwork_pairs,
    jaccard_similarity,
    load_checkpoint,
    make_inner_cv,
    moving_average,
    parse_edge_networks,
    sanitize_subset,
    save_checkpoint,
    save_json,
    set_global_seed,
    should_stop_for_walltime,
)


class RLSelectorTrainer:
    def __init__(self, X_train, y_train, feature_names, params, seed, logger, run_dir, trial_idx=0):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.params = params
        self.seed = seed
        self.logger = logger
        self.run_dir = run_dir
        self.trial_idx = trial_idx

        self.n_features = X_train.shape[1]
        self.k_stop = int(params["k_stop"])
        self.k_safety_cap = int(params["k_stop"])
        self.lambda_reg = float(params["lambda_reg"])
        self.learning_rate = float(params["learning_rate"])
        self.prior_beta = 1.0
        self.k_stop_ref = int(params["k_stop"])
        self.refit_subset_source = str(params.get("refit_subset_source", REFIT_SUBSET_SOURCE))
        self.logger.info(f"[Refit] Using subset source: {self.refit_subset_source}")
        self.cap_violation_penalty = float(self.lambda_reg)

        set_global_seed(seed)
        self.current_prior_stage = WARMUP_PRIOR_STAGE
        prior = build_network_prior(feature_names, stage=self.current_prior_stage)
        self.model = SequentialPolicy(
            n_features=self.n_features,
            prior_logits=prior,
            k_stop_ref=self.k_stop_ref,
            init_stop_logit=INIT_STOP_LOGIT,
            init_stop_count_scale=INIT_STOP_COUNT_SCALE,
            init_continue_logit=INIT_CONTINUE_LOGIT,
            feature_names=feature_names,
        )
        self.initial_prior_summary = self.summarize_prior_vector(prior, stage=self.current_prior_stage, top_k=12)
        self.latest_prior_summary = dict(self.initial_prior_summary)
        self.prior_summary_history = [{
            "event": "init",
            "stage": self.current_prior_stage,
            "summary": dict(self.initial_prior_summary),
        }]
        self.logger.info(
            f"prior_init | stage={self.current_prior_stage} | mean={self.initial_prior_summary['mean']:.4f} | "
            f"std={self.initial_prior_summary['std']:.4f} | min={self.initial_prior_summary['min']:.4f} | "
            f"max={self.initial_prior_summary['max']:.4f} | nonzero_frac={self.initial_prior_summary['nonzero_fraction']:.4f}"
        )

        self.warmup_feature_optimizer = optim.Adam([self.model.feature_logits], lr=self.learning_rate)
        self.ppo_optimizer = optim.Adam(
            [
                {"params": [self.model.feature_logits], "lr": PHASE2_FEATURE_LR},
                {
                    "params": [
                        self.model.continue_logit,
                        self.model.stop_logit,
                        self.model.stop_count_scale,
                        self.model.stop_quality_scale,
                        self.model.raw_stop_gain_scale,
                        self.model.stop_advantage_scale,
                    ],
                    "lr": PHASE2_STOP_LR,
                },
                {
                    "params": [
                        *self.model.feature_state_encoder.parameters(),
                        *self.model.value_head.parameters(),
                    ],
                    "lr": PPO_LR,
                },
            ]
        )

        self.reward_history = []
        self.inner_auc_history = []
        self.inner_auc_warmup_history = []
        self.selected_size_history = []
        self.moving_avg_history = []
        self.stop_reason_history = []

        self.penalty_scale_history = []
        self.joint_progress_history = []
        self.effective_penalty_history = []
        self.active_stop_reward_history = []
        self.synergy_bonus_history = []

        self.policy_stop_history = []
        self.policy_stop_ratio_history = []
        self.logits_std_history = []
        self.sparsity_penalty_history = []
        self.cap_penalty_history = []
        self.late_length_penalty_history = []
        self.stop_bonus_history = []
        self.compact_bonus_history = []
        self.feature_signal_history = []
        self.feature_bonus_history = []
        self.coarse_pair_term_history = []
        self.step_structure_bonus_history = []
        self.step_cost_history = []
        self.continue_logit_history = []
        self.stop_logit_history = []
        self.stop_count_scale_history = []
        self.stop_quality_scale_history = []
        self.raw_stop_gain_scale_history = []
        self.stop_gain_scale_history = []
        self.stop_prev_reward_scale_history = []
        self.stop_advantage_scale_history = []
        self.stop_reward_trend_scale_history = []
        self.phase_history = []

        self.top25_logits_mean_history = []
        self.top25_logits_std_history = []
        self.top50_logits_mean_history = []
        self.top50_logits_std_history = []
        self.top25_minus_top50_gap_history = []
        self.top50_minus_top100_gap_history = []
        self.max_minus_median_logit_history = []
        self.max_median_gap_history = []
        self.top50_overlap_recent_history = []
        self.top100_overlap_recent_history = []

        self.shaping_score_history = []
        self.pair_pos_ratio_history = []
        self.pair_neg_ratio_history = []
        self.pair_count_history = []
        self.pair_term_history = []

        self.feature_logits_snapshots = []
        self.feature_snapshot_episodes = []
        self.feature_snapshot_every = 10
        self.joint_feature_temperature_history = []
        self.joint_stop_temperature_history = []
        self.joint_stop_step_records = []
        self.final_stop_probability_history = []
        self.final_stop_selected_count_history = []
        self.joint_stop_barrier_history = []
        self.joint_stop_applied_barrier_history = []
        self.ppo_feat_entropy_coef_history = []
        self.ppo_stop_entropy_coef_history = []
        self.joint_stage_label_history = []

        self.final_subset_logit_trajectories = {}
        self.final_subset_rank_trajectories = {}
        self.final_subset_rank_summary = {}
        self.stop_prob_by_selected_count_summary = {}

        self.best_reward = -np.inf
        self.best_inner_auc = 0.5
        self.best_subset = None
        self.best_episode = -1
        self.best_selection_score = (-np.inf, -np.inf, -np.inf)
        self.best_stop_reason = None
        self.top_episode_candidates = []
        self.refit_selection_summary = {}
        self.start_episode = 0
        self.joint_refit_episode_history = []

        self.final_reward = -np.inf
        self.final_inner_auc = 0.5
        self.final_subset = None
        self.final_policy_logits = None

        self.warmup_completed = False
        self.warmup_end_episode = None
        self.warmup_end_reason = None
        self.joint_max_episodes = None
        self.joint_episode_count = 0
        self.joint_temperature_history = []
        self.feature_rank_snapshots = []
        self.warmup_transition_history = []
        self.joint_refit_readiness_history = []
        self.joint_refit_episode = None
        self.joint_refit_reason = None
        self.ppo_rollout_buffer = []
        self.ppo_update_history = []
        self.joint_checkpoint_history = []
        self.recent_joint_subsets = []

        self.high_reward_feature_credit = np.zeros(self.n_features, dtype=np.float32)
        self.recent_high_reward_subsets = []
        self.high_reward_feature_boost_history = []

        self.ema_baseline = 0.0
        self.ema_adv_sq = 1.0

    # ------------------------------------------------------------------
    # Prior helpers
    # ------------------------------------------------------------------

    def compute_stage_prior(self, stage: str) -> np.ndarray:
        prior = np.asarray(build_network_prior(self.feature_names, stage=stage), dtype=np.float32)
        if prior.shape[0] != self.n_features:
            raise ValueError(
                f"Prior length mismatch for stage={stage}: got {prior.shape[0]}, expected {self.n_features}"
            )
        return prior

    def summarize_prior_vector(self, prior: np.ndarray, stage: str, top_k: int = 10) -> dict:
        prior = np.asarray(prior, dtype=np.float32)
        if prior.ndim != 1:
            prior = prior.reshape(-1)
        top_k = max(1, min(int(top_k), int(prior.shape[0])))
        order = np.argsort(-prior)
        top_idx = order[:top_k]
        top_features = [
            {
                "feature_idx": int(idx),
                "feature_name": str(self.feature_names[int(idx)]),
                "prior": float(prior[int(idx)]),
            }
            for idx in top_idx
        ]
        return {
            "stage": str(stage),
            "mean": float(np.mean(prior)) if prior.size > 0 else 0.0,
            "std": float(np.std(prior)) if prior.size > 0 else 0.0,
            "min": float(np.min(prior)) if prior.size > 0 else 0.0,
            "max": float(np.max(prior)) if prior.size > 0 else 0.0,
            "nonzero_fraction": float(np.mean(np.abs(prior) > 1e-12)) if prior.size > 0 else 0.0,
            "top_features": top_features,
        }

    def apply_stage_prior(self, stage: str, rebase_feature_logits: bool = True):
        if stage == self.current_prior_stage:
            return
        previous_stage = self.current_prior_stage
        new_prior = self.compute_stage_prior(stage)
        old_prior = self.model.prior_logits_buffer.detach().cpu().numpy().copy()
        delta_prior = new_prior - old_prior

        new_prior_tensor = torch.tensor(
            new_prior,
            dtype=self.model.prior_logits_buffer.dtype,
            device=self.model.prior_logits_buffer.device,
        )
        delta_prior_tensor = torch.tensor(
            delta_prior,
            dtype=self.model.feature_logits.dtype,
            device=self.model.feature_logits.device,
        )
        with torch.no_grad():
            self.model.prior_logits_buffer.copy_(new_prior_tensor)
            if rebase_feature_logits:
                scale = 1.0
                if previous_stage == "warmup" and stage == "joint":
                    scale = float(JOINT_PRIOR_DELTA_SCALE)
                self.model.feature_logits.add_(scale * delta_prior_tensor)

        self.current_prior_stage = stage
        old_summary = self.summarize_prior_vector(old_prior, stage=previous_stage, top_k=8)
        new_summary = self.summarize_prior_vector(new_prior, stage=stage, top_k=8)
        delta_summary = self.summarize_prior_vector(delta_prior, stage=f"delta_{previous_stage}_to_{stage}", top_k=8)
        self.latest_prior_summary = dict(new_summary)
        self.prior_summary_history.append({
            "event": "stage_switch",
            "from_stage": previous_stage,
            "to_stage": stage,
            "rebase_feature_logits": bool(rebase_feature_logits),
            "old_summary": old_summary,
            "new_summary": new_summary,
            "delta_summary": delta_summary,
        })
        self.logger.info(
            f"prior_switch | from={previous_stage} | to={stage} | rebase={rebase_feature_logits} | "
            f"old_mean={old_summary['mean']:.4f} | new_mean={new_summary['mean']:.4f} | "
            f"delta_mean={delta_summary['mean']:.4f} | delta_std={delta_summary['std']:.4f} | "
            f"delta_min={delta_summary['min']:.4f} | delta_max={delta_summary['max']:.4f}"
        )

    # ------------------------------------------------------------------
    # Feature rank tracking
    # ------------------------------------------------------------------

    @staticmethod
    def get_feature_ranks_from_logits(logits: np.ndarray) -> np.ndarray:
        order = np.argsort(-logits)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(logits) + 1)
        return ranks

    def update_logit_summary_stats(self, feat_logits: np.ndarray):
        sorted_logits_desc = np.sort(feat_logits)[::-1]
        top25 = sorted_logits_desc[:min(25, len(sorted_logits_desc))]
        top50 = sorted_logits_desc[:min(50, len(sorted_logits_desc))]
        top100 = sorted_logits_desc[:min(100, len(sorted_logits_desc))]

        top25_mean = float(top25.mean()) if top25.size > 0 else 0.0
        top25_std = float(top25.std()) if top25.size > 0 else 0.0
        top50_mean = float(top50.mean()) if top50.size > 0 else 0.0
        top50_std = float(top50.std()) if top50.size > 0 else 0.0
        top100_mean = float(top100.mean()) if top100.size > 0 else 0.0
        max_median_gap = float(sorted_logits_desc[0] - np.median(feat_logits)) if sorted_logits_desc.size > 0 else 0.0

        self.top25_logits_mean_history.append(top25_mean)
        self.top25_logits_std_history.append(top25_std)
        self.top50_logits_mean_history.append(top50_mean)
        self.top50_logits_std_history.append(top50_std)
        self.top25_minus_top50_gap_history.append(float(top25_mean - top50_mean))
        self.top50_minus_top100_gap_history.append(float(top50_mean - top100_mean))
        self.max_minus_median_logit_history.append(max_median_gap)
        self.max_median_gap_history.append(max_median_gap)

    def maybe_store_feature_snapshot(self, episode: int, feat_logits: np.ndarray):
        if (episode + 1) % self.feature_snapshot_every != 0:
            return
        self.feature_logits_snapshots.append(feat_logits.copy())
        self.feature_snapshot_episodes.append(int(episode + 1))
        top50_features = np.argsort(-feat_logits)[:50].tolist()
        self.feature_rank_snapshots.append({
            "episode": int(episode + 1),
            "top50_features": top50_features,
        })

    def compute_recent_rank_overlap(self, top_k: int = 50, recent_n: int = 3):
        if len(self.feature_logits_snapshots) < int(recent_n):
            return 0.0
        recent_snaps = self.feature_logits_snapshots[-int(recent_n):]
        overlap_vals = []
        for i in range(len(recent_snaps) - 1):
            a = set(np.argsort(-recent_snaps[i])[:int(top_k)].tolist())
            b = set(np.argsort(-recent_snaps[i + 1])[:int(top_k)].tolist())
            union = a.union(b)
            overlap = len(a.intersection(b)) / float(len(union)) if len(union) > 0 else 0.0
            overlap_vals.append(overlap)
        return float(np.mean(overlap_vals)) if overlap_vals else 0.0

    def build_final_subset_logit_trajectories(self, final_subset: np.ndarray):
        trajectories = {}
        subset_list = [int(x) for x in final_subset.tolist()]
        for feat_idx in subset_list:
            trajectories[feat_idx] = []
        for ep, logits in zip(self.feature_snapshot_episodes, self.feature_logits_snapshots):
            for feat_idx in subset_list:
                trajectories[feat_idx].append({"episode": int(ep), "logit": float(logits[feat_idx])})
        return trajectories

    def build_final_subset_rank_trajectories(self, final_subset: np.ndarray):
        rank_trajectories = {}
        subset_list = [int(x) for x in final_subset.tolist()]
        for feat_idx in subset_list:
            rank_trajectories[feat_idx] = []
        for ep, logits in zip(self.feature_snapshot_episodes, self.feature_logits_snapshots):
            ranks = self.get_feature_ranks_from_logits(logits)
            for feat_idx in subset_list:
                rank_trajectories[feat_idx].append({"episode": int(ep), "rank": int(ranks[feat_idx])})
        return rank_trajectories

    def build_final_subset_rank_summary(self, final_subset: np.ndarray):
        summary = {}
        subset_list = [int(x) for x in final_subset.tolist()]
        if len(self.feature_snapshot_episodes) == 0:
            return summary
        target_eps = [1, self.warmup_end_episode or 1, (self.warmup_end_episode or 0) + max(1, self.joint_episode_count)]
        nearest_eps = []
        for target in target_eps:
            nearest = min(self.feature_snapshot_episodes, key=lambda x: abs(x - target))
            nearest_eps.append(nearest)
        nearest_eps = sorted(set(nearest_eps))
        snapshot_map = {ep: logits for ep, logits in zip(self.feature_snapshot_episodes, self.feature_logits_snapshots)}
        for feat_idx in subset_list:
            summary[feat_idx] = {}
            for ep in nearest_eps:
                ranks = self.get_feature_ranks_from_logits(snapshot_map[ep])
                summary[feat_idx][f"episode_{ep}"] = int(ranks[feat_idx])
        return summary

    def summarize_stop_prob_by_selected_count(self):
        bucket = defaultdict(list)
        for ep_record in self.joint_stop_step_records:
            for step in ep_record["trace"]:
                c = int(step["selected_count_before_decision"])
                bucket[c].append(float(step["p_stop"]))
        summary = {}
        for c, vals in sorted(bucket.items()):
            summary[c] = {
                "mean_p_stop": float(np.mean(vals)),
                "std_p_stop": float(np.std(vals)),
                "n": int(len(vals)),
            }
        return summary

    # ------------------------------------------------------------------
    # Warmup transition detection
    # ------------------------------------------------------------------

    def warmup_transition_metrics(self):
        rewards = np.asarray(self.reward_history, dtype=np.float32)
        if rewards.size == 0:
            return {
                "n_rewards": 0, "mean_first25": 0.0, "mean_last25": 0.0,
                "mean_prev25": 0.0, "global_reward_median": 0.0,
                "last50_above_global_median_fraction": 0.0,
                "max_median_gap": 0.0, "top25_top50_gap": 0.0,
                "logits_std": 0.0, "top50_overlap_recent": 0.0,
            }

        first_n = min(WARMUP_REWARD_WINDOW, rewards.size)
        last_n = min(WARMUP_REWARD_WINDOW, rewards.size)
        first_rewards = rewards[:first_n]
        last_rewards = rewards[-last_n:]

        if rewards.size >= 2 * WARMUP_PLATEAU_WINDOW:
            prev_rewards = rewards[-2 * WARMUP_PLATEAU_WINDOW:-WARMUP_PLATEAU_WINDOW]
        else:
            prev_rewards = rewards[:last_n]

        global_median = float(np.median(rewards))
        above_global_frac = float(np.mean(last_rewards > global_median)) if last_rewards.size > 0 else 0.0
        max_median_gap = float(self.max_minus_median_logit_history[-1]) if self.max_minus_median_logit_history else 0.0
        top25_top50_gap = float(self.top25_minus_top50_gap_history[-1]) if self.top25_minus_top50_gap_history else 0.0
        logits_std = float(self.logits_std_history[-1]) if self.logits_std_history else 0.0

        top50_overlap_recent = 0.0
        if hasattr(self, "feature_rank_snapshots") and len(self.feature_rank_snapshots) >= 3:
            recent_snaps = self.feature_rank_snapshots[-3:]
            overlap_vals = []
            for i in range(len(recent_snaps) - 1):
                a = set(recent_snaps[i]["top50_features"])
                b = set(recent_snaps[i + 1]["top50_features"])
                if len(a) > 0 and len(b) > 0:
                    overlap = len(a.intersection(b)) / float(len(a.union(b))) if len(a.union(b)) > 0 else 0.0
                    overlap_vals.append(overlap)
            if overlap_vals:
                top50_overlap_recent = float(np.mean(overlap_vals))

        return {
            "n_rewards": int(rewards.size),
            "mean_first25": float(first_rewards.mean()) if first_rewards.size > 0 else 0.0,
            "mean_last25": float(last_rewards.mean()) if last_rewards.size > 0 else 0.0,
            "mean_prev25": float(prev_rewards.mean()) if prev_rewards.size > 0 else 0.0,
            "global_reward_median": global_median,
            "last50_above_global_median_fraction": above_global_frac,
            "max_median_gap": max_median_gap,
            "top25_top50_gap": top25_top50_gap,
            "logits_std": logits_std,
            "top50_overlap_recent": top50_overlap_recent,
        }

    def should_end_warmup(self):
        metrics = self.warmup_transition_metrics()
        n_rewards = metrics["n_rewards"]
        if n_rewards < ADAPTIVE_WARMUP_MIN_EPISODES:
            metrics["adaptive_warmup_ready"] = False
            metrics["adaptive_warmup_reason"] = "minimum_warmup_not_reached"
            return False, "minimum_warmup_not_reached", metrics

        strong_ready = (
            metrics["max_median_gap"] >= WARMUP_MAX_MEDIAN_GAP_THRESHOLD
            and metrics["top25_top50_gap"] >= WARMUP_TOP25_TOP50_GAP_THRESHOLD
            and metrics["last50_above_global_median_fraction"] >= WARMUP_ABOVE_MEDIAN_FRACTION_THRESHOLD
            and metrics["top50_overlap_recent"] >= WARMUP_TOP50_OVERLAP_THRESHOLD
        )
        if strong_ready:
            metrics["adaptive_warmup_ready"] = True
            metrics["adaptive_warmup_reason"] = "criteria_met_and_plateau"
            return True, "criteria_met_and_plateau", metrics

        if n_rewards >= max(ADAPTIVE_WARMUP_MIN_EPISODES, 2 * WARMUP_NO_IMPROVEMENT_WINDOW):
            recent = np.asarray(self.reward_history[-WARMUP_NO_IMPROVEMENT_WINDOW:], dtype=np.float32)
            prev = np.asarray(
                self.reward_history[-2 * WARMUP_NO_IMPROVEMENT_WINDOW:-WARMUP_NO_IMPROVEMENT_WINDOW],
                dtype=np.float32,
            )
            no_improvement = (
                recent.size > 0
                and prev.size > 0
                and float(recent.mean() - prev.mean()) <= WARMUP_NO_IMPROVEMENT_DELTA_THRESHOLD
            )
            weak_ready = (
                metrics["max_median_gap"] >= WARMUP_FALLBACK_MAX_MEDIAN_GAP_THRESHOLD
                and metrics["top25_top50_gap"] >= WARMUP_FALLBACK_TOP25_TOP50_GAP_THRESHOLD
                and metrics["top50_overlap_recent"] >= WARMUP_FALLBACK_TOP50_OVERLAP_THRESHOLD
            )
            if no_improvement and weak_ready:
                metrics["adaptive_warmup_ready"] = True
                metrics["adaptive_warmup_reason"] = "recent_reward_no_improvement"
                metrics["recent50_minus_prev50_delta"] = float(recent.mean() - prev.mean())
                return True, "recent_reward_no_improvement", metrics

        metrics["adaptive_warmup_ready"] = False
        metrics["adaptive_warmup_reason"] = "continue_warmup"
        return False, "continue_warmup", metrics

    # ------------------------------------------------------------------
    # Joint refit readiness
    # ------------------------------------------------------------------

    def compute_joint_refit_metrics(self):
        joint_rewards = [float(r) for r, p in zip(self.reward_history, self.phase_history) if p == "joint"]
        joint_sizes = [int(s) for s, p in zip(self.selected_size_history, self.phase_history) if p == "joint"]
        metrics = {
            "n_joint_episodes": int(len(joint_rewards)),
            "recent_reward_mean": 0.0, "prev_reward_mean": 0.0, "reward_delta": 0.0,
            "recent_reward_plateau": False, "recent_subset_size_mean": 0.0,
            "recent_subset_size_std": 0.0, "subset_length_stable": False,
            "recent_greedy_subset_overlap": 0.0, "recent_sampled_subset_overlap": 0.0,
            "subset_overlap_ready": False, "checkpoint_refit_auc_mean": 0.0,
            "checkpoint_refit_auc_std": 0.0, "checkpoint_subset_size_std": 0.0,
            "checkpoint_refit_stable": False, "joint_refit_ready": False,
            "joint_refit_reason": "minimum_joint_not_reached",
        }
        if len(joint_rewards) < max(JOINT_REFIT_MIN_EPISODES, JOINT_REWARD_PLATEAU_WINDOW):
            return metrics

        reward_win = int(JOINT_REWARD_PLATEAU_WINDOW)
        recent_rewards = np.asarray(joint_rewards[-reward_win:], dtype=np.float32)
        prev_rewards = (
            np.asarray(joint_rewards[-2 * reward_win:-reward_win], dtype=np.float32)
            if len(joint_rewards) >= 2 * reward_win
            else np.asarray(joint_rewards[:reward_win], dtype=np.float32)
        )
        metrics["recent_reward_mean"] = float(recent_rewards.mean()) if recent_rewards.size > 0 else 0.0
        metrics["prev_reward_mean"] = float(prev_rewards.mean()) if prev_rewards.size > 0 else metrics["recent_reward_mean"]
        metrics["reward_delta"] = float(metrics["recent_reward_mean"] - metrics["prev_reward_mean"])
        metrics["recent_reward_plateau"] = abs(metrics["reward_delta"]) <= float(JOINT_REWARD_PLATEAU_DELTA)

        recent_sizes = np.asarray(joint_sizes[-reward_win:], dtype=np.float32)
        metrics["recent_subset_size_mean"] = float(recent_sizes.mean()) if recent_sizes.size > 0 else 0.0
        metrics["recent_subset_size_std"] = float(recent_sizes.std()) if recent_sizes.size > 0 else 0.0
        metrics["subset_length_stable"] = metrics["recent_subset_size_std"] <= float(JOINT_LENGTH_STD_MAX)

        recent_ckpts = self.joint_checkpoint_history[-int(max(JOINT_READY_MIN_CHECKPOINTS, JOINT_CHECKPOINT_STABILITY_WINDOW)):]
        if len(recent_ckpts) > 0:
            greedy_overlaps = [float(x.get("greedy_overlap_with_previous", 0.0)) for x in recent_ckpts if x.get("greedy_overlap_with_previous") is not None]
            sampled_overlaps = [float(x.get("sampled_overlap_mean", 0.0)) for x in recent_ckpts]
            if greedy_overlaps:
                metrics["recent_greedy_subset_overlap"] = float(np.mean(greedy_overlaps))
            if sampled_overlaps:
                metrics["recent_sampled_subset_overlap"] = float(np.mean(sampled_overlaps))
            metrics["subset_overlap_ready"] = (
                metrics["recent_greedy_subset_overlap"] >= float(JOINT_GREEDY_OVERLAP_THRESHOLD)
                and metrics["recent_sampled_subset_overlap"] >= float(JOINT_SAMPLED_OVERLAP_THRESHOLD)
            )

            chosen_aucs, chosen_sizes = [], []
            for ck in recent_ckpts:
                chosen_method = ck.get("chosen_refit_method", self.refit_subset_source)
                refit_eval = ck.get("refit_evaluations", {}).get(chosen_method, {})
                if "refit_cv_auc" in refit_eval:
                    chosen_aucs.append(float(refit_eval["refit_cv_auc"]))
                    chosen_sizes.append(int(refit_eval.get("subset_size", 0)))
            if chosen_aucs:
                metrics["checkpoint_refit_auc_mean"] = float(np.mean(chosen_aucs))
                metrics["checkpoint_refit_auc_std"] = float(np.std(chosen_aucs))
                if len(chosen_aucs) >= 2:
                    metrics["checkpoint_refit_stable"] = (
                        float(np.max(chosen_aucs) - np.min(chosen_aucs)) <= float(JOINT_CHECKPOINT_REFIT_AUC_DELTA_MAX)
                    )
            if chosen_sizes:
                metrics["checkpoint_subset_size_std"] = float(np.std(chosen_sizes))
                metrics["checkpoint_refit_stable"] = bool(
                    metrics["checkpoint_refit_stable"]
                    and metrics["checkpoint_subset_size_std"] <= float(JOINT_CHECKPOINT_SIZE_STD_MAX)
                )

        ready = (
            metrics["recent_reward_plateau"]
            and metrics["subset_length_stable"]
            and metrics["subset_overlap_ready"]
            and metrics["checkpoint_refit_stable"]
            and len(self.joint_checkpoint_history) >= int(JOINT_READY_MIN_CHECKPOINTS)
        )
        metrics["joint_refit_ready"] = bool(ready)
        metrics["joint_refit_reason"] = "criteria_met" if ready else "continue_joint"
        return metrics

    def should_enter_refit_from_joint(self):
        metrics = self.compute_joint_refit_metrics()
        return bool(metrics["joint_refit_ready"]), str(metrics["joint_refit_reason"]), metrics

    # ------------------------------------------------------------------
    # Reward trend state
    # ------------------------------------------------------------------

    def get_reward_trend_state(self):
        if len(self.reward_history) == 0:
            return torch.zeros(7, dtype=torch.float32)
        prev_reward = float(self.reward_history[-1])
        moving_avg = float(self.moving_avg_history[-1]) if self.moving_avg_history else prev_reward
        prev_advantage = prev_reward - moving_avg

        recent_n = min(5, len(self.reward_history))
        recent_rewards = np.asarray(self.reward_history[-recent_n:], dtype=np.float32)
        recent_reward_gain_trend = float(np.mean(np.diff(recent_rewards))) if recent_rewards.size >= 2 else 0.0

        recent_struct_n = min(10, len(self.step_structure_bonus_history))
        if recent_struct_n > 0:
            recent_structure = np.asarray(self.step_structure_bonus_history[-recent_struct_n:], dtype=np.float32)
            recent_structure_bonus_mean = float(np.mean(recent_structure))
            recent_structure_bonus_trend = float(np.mean(np.diff(recent_structure))) if recent_structure.size >= 2 else 0.0
        else:
            recent_structure_bonus_mean = 0.0
            recent_structure_bonus_trend = 0.0

        recent_stop_n = min(10, len(self.policy_stop_ratio_history))
        recent_policy_stop_ratio = float(np.mean(self.policy_stop_ratio_history[-recent_stop_n:])) if recent_stop_n > 0 else 0.0

        return torch.tensor(
            [prev_reward, prev_advantage, recent_reward_gain_trend, moving_avg,
             recent_structure_bonus_mean, recent_structure_bonus_trend, recent_policy_stop_ratio],
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Step record helper
    # ------------------------------------------------------------------

    def _build_step_record(self, selected, selected_mask, reward_trend_state, stage1_probs, barrier_debug=None):
        if barrier_debug is None:
            barrier_debug = {}
        p_continue = float(stage1_probs[0].detach().cpu().item())
        p_stop = float(stage1_probs[1].detach().cpu().item())
        selected_count_before = int(len(selected))
        selected_ratio = float(selected_count_before / max(self.k_stop_ref, 1))

        state_features = self.model.build_state_features(selected_mask, reward_trend_state=reward_trend_state)
        stop_debug = self.model._compute_stop_term_breakdown(selected_mask, reward_trend_state)
        effective_logits = self.model._compute_effective_feature_logits_from_state(state_features).detach().cpu().numpy()

        if len(selected) > 0:
            current_selected_logits = effective_logits[selected]
            selected_mean_logit = float(np.mean(current_selected_logits))
            selected_max_logit = float(np.max(current_selected_logits))
            selected_q75_logit = float(
                self.model._stable_selected_summary(current_selected_logits).detach().cpu().item()
            )
            current_selected_prior = self.model.prior_logits_buffer.detach().cpu().numpy()[selected]
            selected_mean_prior = float(np.mean(current_selected_prior))
        else:
            selected_mean_logit = selected_max_logit = selected_q75_logit = selected_mean_prior = 0.0

        remaining_indices = np.where(~selected_mask.detach().cpu().numpy())[0]
        if len(remaining_indices) > 0:
            remaining_logits_np = effective_logits[remaining_indices]
            topm = min(STOP_REMAINING_TOP_M, int(len(remaining_logits_np)))
            remaining_topm_logits = np.sort(remaining_logits_np)[-topm:]
            remaining_q75_logit = float(np.quantile(remaining_topm_logits, 0.75))
        else:
            remaining_q75_logit = 0.0
        gain_gap = float(remaining_q75_logit - selected_q75_logit)

        coverage_ratio, cross_ratio = self.model._compute_network_state_stats(selected_mask.detach().clone().bool())
        nonbarrier_stop_total = (
            stop_debug["base_stop_logit"]
            + stop_debug["stop_gain_term"]
            + stop_debug["stop_progress_term"]
            + stop_debug["stop_plateau_term"]
            + stop_debug["stop_coverage_term"]
            + stop_debug["stop_cross_term"]
            - stop_debug["continue_trend_term"]
            - stop_debug["early_stop_penalty"]
        )

        return {
            "selected_count_before_decision": selected_count_before,
            "selected_ratio": selected_ratio,
            "p_continue": p_continue,
            "p_stop": p_stop,
            "continue_logit_param": float(self.model.continue_logit.detach().cpu().item()),
            "stop_logit_param": float(self.model.stop_logit.detach().cpu().item()),
            "stop_count_scale_param": float(self.model.stop_count_scale.detach().cpu().item()),
            "selected_mean_logit": selected_mean_logit,
            "selected_q75_logit": selected_q75_logit,
            "selected_max_logit": selected_max_logit,
            "selected_mean_prior": selected_mean_prior,
            "remaining_q75_logit": remaining_q75_logit,
            "gain_gap": float(stop_debug["gain_gap"]),
            "network_coverage_ratio": float(coverage_ratio.detach().cpu().item()),
            "cross_network_ratio": float(cross_ratio.detach().cpu().item()),
            "stop_quality_scale_param": float(self.model.stop_quality_scale.detach().cpu().item()),
            "raw_stop_gain_scale_param": float(self.model.raw_stop_gain_scale.detach().cpu().item()),
            "stop_advantage_scale_param": float(self.model.stop_advantage_scale.detach().cpu().item()),
            "base_continue_logit": float(stop_debug["base_continue_logit"].detach().cpu().item()),
            "base_stop_logit": float(stop_debug["base_stop_logit"].detach().cpu().item()),
            "nonbarrier_stop_total": nonbarrier_stop_total.detach().cpu().item(),
            "continue_logit_total": float(stop_debug["raw_continue_logit_total"].detach().cpu().item()),
            "stop_logit_total": float(stop_debug["raw_stop_logit_total"].detach().cpu().item()),
            "stop_progress_term": float(stop_debug["stop_progress_term"].detach().cpu().item()),
            "stop_gain_term": float(stop_debug["stop_gain_term"].detach().cpu().item()),
            "stop_plateau_term": float(stop_debug["stop_plateau_term"].detach().cpu().item()),
            "stop_coverage_term": float(stop_debug["stop_coverage_term"].detach().cpu().item()),
            "stop_cross_term": float(stop_debug["stop_cross_term"].detach().cpu().item()),
            "continue_trend_term": float(stop_debug["continue_trend_term"].detach().cpu().item()),
            "early_stop_penalty": float(stop_debug["early_stop_penalty"]),
            "stop_gain_signal": float(stop_debug["stop_gain_signal"].detach().cpu().item()),
            "stop_gain_scale": float(stop_debug["stop_gain_scale"].detach().cpu().item()),
            "extra_neg_gain": float(stop_debug["extra_neg_gain"].detach().cpu().item()),
            "stop_plateau_signal": float(stop_debug["stop_plateau_signal"].detach().cpu().item()),
            "raw_continue_logit": barrier_debug.get("raw_continue_logit"),
            "raw_stop_logit": barrier_debug.get("raw_stop_logit"),
            "raw_margin": barrier_debug.get("raw_margin"),
            "raw_p_continue": barrier_debug.get("raw_p_continue"),
            "raw_p_stop": barrier_debug.get("raw_p_stop"),
            "raw_argmax": barrier_debug.get("raw_argmax"),
            "post_barrier_continue_logit": barrier_debug.get("post_barrier_continue_logit"),
            "post_barrier_stop_logit": barrier_debug.get("post_barrier_stop_logit"),
            "post_barrier_margin": barrier_debug.get("post_barrier_margin"),
            "post_barrier_p_continue": barrier_debug.get("post_barrier_p_continue"),
            "post_barrier_p_stop": barrier_debug.get("post_barrier_p_stop"),
            "post_barrier_argmax": barrier_debug.get("post_barrier_argmax"),
            "barrier_value": barrier_debug.get("barrier_value"),
            "barrier_flipped_stop_to_continue": barrier_debug.get("barrier_flipped_stop_to_continue"),
            "final_continue_logit": barrier_debug.get("final_continue_logit"),
            "final_stop_logit": barrier_debug.get("final_stop_logit"),
            "final_margin": barrier_debug.get("final_margin"),
            "final_argmax": barrier_debug.get("final_argmax"),
            "stop_temperature": barrier_debug.get("stop_temperature"),
        }

    # ------------------------------------------------------------------
    # Joint phase schedule helpers
    # ------------------------------------------------------------------

    def get_joint_progress_ratio(self) -> float:
        denom = max(1, int(self.joint_max_episodes))
        return float(self.joint_episode_count) / float(denom)

    def get_joint_feature_temperature(self) -> float:
        j = int(self.joint_episode_count)
        if j < 60:
            return float(JOINT_FEATURE_TEMP_EARLY)
        elif j < 180:
            return float(JOINT_FEATURE_TEMP_MID)
        return float(JOINT_FEATURE_TEMP_LATE)

    def get_joint_stop_temperature(self) -> float:
        j = int(self.joint_episode_count)
        if j < 60:
            return float(JOINT_STOP_TEMP_EARLY)
        elif j < 180:
            return float(JOINT_STOP_TEMP_MID)
        return float(JOINT_STOP_TEMP_LATE)

    def get_joint_stop_barrier(self, selected_count: int) -> float:
        j = int(self.joint_episode_count)
        if j < 60:
            phase_barrier = float(JOINT_STOP_BARRIER_EARLY)
        elif j < 180:
            phase_barrier = float(JOINT_STOP_BARRIER_MID)
        else:
            phase_barrier = float(JOINT_STOP_BARRIER_LATE)
        missing = max(int(MIN_FEATURES_BEFORE_STOP) - int(selected_count), 0)
        soft_count_barrier = 1.0 * (float(missing) / max(float(MIN_FEATURES_BEFORE_STOP), 1.0))
        return phase_barrier + soft_count_barrier

    def get_entropy_coefs(self):
        j = int(self.joint_episode_count)
        if j < 60:
            return float(PPO_STOP_ENTROPY_COEF_EARLY), float(PPO_FEAT_ENTROPY_COEF_EARLY)
        elif j < 180:
            return float(PPO_STOP_ENTROPY_COEF_MID), float(PPO_FEAT_ENTROPY_COEF_MID)
        return float(PPO_STOP_ENTROPY_COEF_LATE), float(PPO_FEAT_ENTROPY_COEF_LATE)

    def apply_stop_barrier(self, continue_stop_logits, subset_size_before):
        logits = continue_stop_logits.clone()
        base_barrier = float(self.get_joint_stop_barrier(int(subset_size_before)))
        barrier = float(BARRIER_SCALE) * base_barrier
        if int(subset_size_before) < 8:
            barrier += 0.5 * (8 - float(subset_size_before)) / 8.0
        logits[1] = logits[1] - barrier
        return logits

    # ------------------------------------------------------------------
    # Subset sampling
    # ------------------------------------------------------------------

    def sample_subset(self, greedy: bool = False, warmup: bool = False):
        selected = []
        log_probs = []
        stop_reason = "unknown"
        selected_mask = torch.zeros(self.n_features, dtype=torch.bool)
        episode_stop_trace = []
        rollout_steps = []

        if warmup:
            target_len = int(np.random.randint(int(WARMUP_MIN_STEPS), int(WARMUP_MAX_STEPS) + 1))
            while len(selected) < target_len:
                _, feature_logits, _, _ = self.model(selected_mask, reward_trend_state=None)
                available_features = int((~selected_mask).sum().item())
                if available_features == 0:
                    stop_reason = "no_available_features"
                    break
                scaled_feature_logits = feature_logits / WARMUP_SOFTMAX_TEMPERATURE
                stage2_dist = torch.distributions.Categorical(logits=scaled_feature_logits)
                feat_action = torch.argmax(scaled_feature_logits) if greedy else stage2_dist.sample()
                stage2_log_prob = stage2_dist.log_prob(feat_action)
                feat_idx = int(feat_action.item())
                if selected_mask[feat_idx]:
                    continue
                selected.append(feat_idx)
                selected_mask = selected_mask.clone()
                selected_mask[feat_idx] = True
                log_probs.append(stage2_log_prob)
            if stop_reason == "unknown":
                stop_reason = "warmup_forced_stop"
            subset = np.array(sorted(set(selected)), dtype=int)
            return subset, log_probs, stop_reason, episode_stop_trace, rollout_steps

        max_steps = self.k_stop_ref + MAX_SELECTION_STEPS_BUFFER
        for _ in range(max_steps):
            reward_trend_state = self.get_reward_trend_state()
            continue_stop_logits, feature_logits, state_value, state_features = self.model(
                selected_mask, reward_trend_state=reward_trend_state
            )
            continue_stop_logits = continue_stop_logits.clone()

            raw_continue_stop_logits = continue_stop_logits.clone()
            raw_continue_logit = float(raw_continue_stop_logits[0].detach().cpu().item())
            raw_stop_logit = float(raw_continue_stop_logits[1].detach().cpu().item())
            raw_margin = float((raw_continue_stop_logits[1] - raw_continue_stop_logits[0]).detach().cpu().item())
            raw_probs = torch.softmax(raw_continue_stop_logits, dim=0)
            raw_p_continue = float(raw_probs[0].detach().cpu().item())
            raw_p_stop = float(raw_probs[1].detach().cpu().item())
            raw_argmax = int(torch.argmax(raw_continue_stop_logits).detach().cpu().item())

            stop_temperature = self.get_joint_stop_temperature()
            feature_temperature = self.get_joint_feature_temperature()
            continue_stop_logits = self.apply_stop_barrier(continue_stop_logits, len(selected))

            post_barrier_logits = continue_stop_logits.clone()
            post_barrier_continue_logit = float(post_barrier_logits[0].detach().cpu().item())
            post_barrier_stop_logit = float(post_barrier_logits[1].detach().cpu().item())
            post_barrier_margin = float((post_barrier_logits[1] - post_barrier_logits[0]).detach().cpu().item())
            barrier_value = raw_stop_logit - post_barrier_stop_logit
            post_barrier_probs = torch.softmax(post_barrier_logits, dim=0)
            post_barrier_p_continue = float(post_barrier_probs[0].detach().cpu().item())
            post_barrier_p_stop = float(post_barrier_probs[1].detach().cpu().item())
            post_barrier_argmax = int(torch.argmax(post_barrier_logits).detach().cpu().item())
            barrier_flipped_stop_to_continue = int(raw_argmax == 1 and post_barrier_argmax == 0)

            scaled_stage1_logits = continue_stop_logits / stop_temperature
            stage1_dist = torch.distributions.Categorical(logits=scaled_stage1_logits)
            stage1_probs = torch.softmax(scaled_stage1_logits, dim=0)

            final_continue_logit = float(scaled_stage1_logits[0].detach().cpu().item())
            final_stop_logit = float(scaled_stage1_logits[1].detach().cpu().item())
            final_margin = float((scaled_stage1_logits[1] - scaled_stage1_logits[0]).detach().cpu().item())
            final_argmax = int(torch.argmax(scaled_stage1_logits).detach().cpu().item())

            barrier_debug = {
                "raw_continue_logit": raw_continue_logit, "raw_stop_logit": raw_stop_logit,
                "raw_margin": raw_margin, "raw_p_continue": raw_p_continue,
                "raw_p_stop": raw_p_stop, "raw_argmax": raw_argmax,
                "post_barrier_continue_logit": post_barrier_continue_logit,
                "post_barrier_stop_logit": post_barrier_stop_logit,
                "post_barrier_margin": post_barrier_margin,
                "post_barrier_p_continue": post_barrier_p_continue,
                "post_barrier_p_stop": post_barrier_p_stop,
                "post_barrier_argmax": post_barrier_argmax,
                "barrier_value": barrier_value,
                "barrier_flipped_stop_to_continue": barrier_flipped_stop_to_continue,
                "final_continue_logit": final_continue_logit, "final_stop_logit": final_stop_logit,
                "final_margin": final_margin, "final_argmax": final_argmax,
                "stop_temperature": float(stop_temperature),
            }

            step_record = self._build_step_record(
                selected, selected_mask, reward_trend_state, stage1_probs, barrier_debug=barrier_debug,
            )

            stage1_action = torch.argmax(scaled_stage1_logits) if greedy else stage1_dist.sample()
            stage1_log_prob = stage1_dist.log_prob(stage1_action)
            entropy = stage1_dist.entropy()

            rollout_step = {
                "selected_mask": selected_mask.clone().detach(),
                "reward_trend_state": reward_trend_state.clone().detach(),
                "stage1_action": int(stage1_action.item()),
                "stage2_action": None,
                "old_log_prob": float(stage1_log_prob.detach().cpu().item()),
                "value": float(state_value.detach().cpu().item()),
                "done": False,
                "subset_size_before": int(len(selected)),
                "step_structure_bonus": 0.0,
                "step_structure_meta": {},
            }

            if int(stage1_action.item()) == 1:
                step_record["decision"] = "stop"
                episode_stop_trace.append(step_record)
                log_probs.append(stage1_log_prob)
                stop_reason = "policy_stop"
                rollout_step["done"] = True
                rollout_steps.append(rollout_step)
                break

            step_record["decision"] = "continue"
            episode_stop_trace.append(step_record)
            available_features = int((~selected_mask).sum().item())
            if available_features == 0:
                log_probs.append(stage1_log_prob)
                stop_reason = "no_available_features"
                rollout_step["done"] = True
                rollout_steps.append(rollout_step)
                break

            scaled_feature_logits = feature_logits / feature_temperature
            stage2_dist = torch.distributions.Categorical(logits=scaled_feature_logits)
            feat_action = torch.argmax(scaled_feature_logits) if greedy else stage2_dist.sample()
            stage2_log_prob = stage2_dist.log_prob(feat_action)
            entropy = entropy + stage2_dist.entropy()
            feat_idx = int(feat_action.item())
            if selected_mask[feat_idx]:
                continue

            selected_before = np.array(selected, dtype=int)
            selected.append(feat_idx)
            selected_mask = selected_mask.clone()
            selected_mask[feat_idx] = True
            joint_log_prob = stage1_log_prob + stage2_log_prob
            log_probs.append(joint_log_prob)

            step_structure_bonus, step_structure_meta = self.compute_step_structure_bonus(
                selected_before=selected_before, new_feature_idx=feat_idx, phase="joint",
            )
            rollout_step["stage2_action"] = feat_idx
            rollout_step["old_log_prob"] = float(joint_log_prob.detach().cpu().item())
            rollout_step["entropy"] = float(entropy.detach().cpu().item())
            rollout_step["step_structure_bonus"] = float(step_structure_bonus)
            rollout_step["step_structure_meta"] = dict(step_structure_meta)
            rollout_steps.append(rollout_step)
        else:
            stop_reason = "safety_cap"

        if stop_reason == "unknown":
            stop_reason = "safety_cap"
        subset = np.array(sorted(set(selected)), dtype=int)
        return subset, log_probs, stop_reason, episode_stop_trace, rollout_steps

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _recompute_step_outputs(self, step):
        selected_mask = step["selected_mask"].to(dtype=torch.bool)
        reward_trend_state = step["reward_trend_state"].to(dtype=torch.float32)
        continue_stop_logits, feature_logits, state_value, _ = self.model(selected_mask, reward_trend_state=reward_trend_state)
        continue_stop_logits = continue_stop_logits.clone()
        continue_stop_logits = self.apply_stop_barrier(continue_stop_logits, int(step.get("subset_size_before", 0)))

        stop_temperature = self.get_joint_stop_temperature()
        feature_temperature = self.get_joint_feature_temperature()
        scaled_stage1_logits = continue_stop_logits / stop_temperature

        stage1_dist = torch.distributions.Categorical(logits=scaled_stage1_logits)
        stage1_action = torch.tensor(int(step["stage1_action"]), dtype=torch.long, device=scaled_stage1_logits.device)
        log_prob = stage1_dist.log_prob(stage1_action)

        stop_entropy = stage1_dist.entropy()
        feature_entropy_raw = torch.tensor(0.0, device=scaled_stage1_logits.device)
        feature_entropy_norm = torch.tensor(0.0, device=scaled_stage1_logits.device)
        max_feature_prob = torch.tensor(0.0, device=scaled_stage1_logits.device)
        top5_prob_mass = torch.tensor(0.0, device=scaled_stage1_logits.device)

        if step.get("stage2_action") is not None and int(step["stage1_action"]) == 0:
            scaled_feature_logits = feature_logits / feature_temperature
            stage2_dist = torch.distributions.Categorical(logits=scaled_feature_logits)
            stage2_action = torch.tensor(int(step["stage2_action"]), dtype=torch.long, device=scaled_feature_logits.device)
            log_prob = log_prob + stage2_dist.log_prob(stage2_action)
            feature_entropy_raw = stage2_dist.entropy()
            feature_probs = torch.softmax(scaled_feature_logits, dim=0)
            num_available = int(feature_probs.numel())
            if num_available > 1:
                feature_entropy_norm = feature_entropy_raw / torch.log(
                    torch.tensor(float(num_available), device=feature_probs.device))
            else:
                feature_entropy_norm = torch.tensor(0.0, device=feature_probs.device)
            max_feature_prob = feature_probs.max()
            k = min(5, num_available)
            top5_prob_mass = torch.topk(feature_probs, k=k).values.sum()

        total_entropy = stop_entropy + feature_entropy_raw
        return {
            "log_prob": log_prob, "total_entropy": total_entropy, "stop_entropy": stop_entropy,
            "feature_entropy_raw": feature_entropy_raw, "feature_entropy_norm": feature_entropy_norm,
            "max_feature_prob": max_feature_prob, "top5_prob_mass": top5_prob_mass,
            "state_value": state_value.squeeze(),
        }

    def _prepare_ppo_rollout_batch(self, reward: float, rollout_steps: list):
        if len(rollout_steps) == 0:
            return []
        traj_len = float(len(rollout_steps))
        prepared = []
        for step in rollout_steps:
            value = float(step.get("value", 0.0))
            step_structure_bonus = float(step.get("step_structure_bonus", 0.0))
            if PPO_LENGTH_NORMALIZE_ADVANTAGE:
                base_advantage = (float(reward) - value) / max(traj_len, 1.0)
            else:
                base_advantage = float(reward) - value
            advantage = base_advantage + step_structure_bonus
            ret = float(reward) + step_structure_bonus
            prepared.append({
                **step,
                "base_advantage": float(base_advantage),
                "step_structure_bonus": float(step_structure_bonus),
                "advantage": float(advantage),
                "return": float(ret),
            })
        return prepared

    def ppo_update(self, rollout_batch: list):
        if len(rollout_batch) == 0:
            return {"updated": False, "n_steps": 0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        advantages = np.asarray([float(x["advantage"]) for x in rollout_batch], dtype=np.float32)
        if advantages.size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = np.asarray([float(x["return"]) for x in rollout_batch], dtype=np.float32)
        old_log_probs = np.asarray([float(x["old_log_prob"]) for x in rollout_batch], dtype=np.float32)

        update_advs, update_returns, update_values = [], [], []
        update_stop_entropies, update_feat_entropies, update_feat_entropies_norm = [], [], []
        update_max_feature_probs, update_top5_prob_mass = [], []
        update_kls, update_clip_flags, update_ratios, update_grad_norms = [], [], [], []
        policy_losses, value_losses, entropies = [], [], []
        last_stats = None

        for _ in range(int(PPO_EPOCHS)):
            if int(PPO_MINI_BATCH_SIZE) and int(PPO_MINI_BATCH_SIZE) > 0:
                order = np.arange(len(rollout_batch))
                np.random.shuffle(order)
                batches = [order[i:i + int(PPO_MINI_BATCH_SIZE)] for i in range(0, len(order), int(PPO_MINI_BATCH_SIZE))]
            else:
                batches = [np.arange(len(rollout_batch))]

            for batch_idx in batches:
                batch_policy_losses, batch_value_losses, batch_entropies = [], [], []
                batch_advs, batch_returns, batch_values = [], [], []
                batch_stop_entropies, batch_feat_entropies, batch_feat_entropies_norm = [], [], []
                batch_max_feature_probs, batch_top5_prob_mass = [], []
                batch_kls, batch_clip_flags, batch_ratios = [], [], []

                for idx in batch_idx.tolist():
                    step = rollout_batch[int(idx)]
                    out = self._recompute_step_outputs(step)
                    new_log_prob = out["log_prob"]
                    entropy = out["total_entropy"]
                    state_value = out["state_value"]

                    old_lp = torch.tensor(old_log_probs[int(idx)], dtype=torch.float32, device=new_log_prob.device)
                    adv = torch.tensor(float(advantages[int(idx)]), dtype=torch.float32, device=new_log_prob.device)
                    ret = torch.tensor(float(returns[int(idx)]), dtype=torch.float32, device=new_log_prob.device)

                    ratio = torch.exp(new_log_prob - old_lp)
                    clipped_ratio = torch.clamp(ratio, 1.0 - float(PPO_CLIP_EPS), 1.0 + float(PPO_CLIP_EPS))
                    policy_loss = -torch.min(ratio * adv, clipped_ratio * adv)
                    value_loss = (state_value - ret) ** 2

                    batch_policy_losses.append(policy_loss)
                    batch_value_losses.append(value_loss)
                    stop_entropy_coef, feat_entropy_coef = self.get_entropy_coefs()
                    weighted_entropy = (
                        float(stop_entropy_coef) * out["stop_entropy"]
                        + float(feat_entropy_coef) * out["feature_entropy_norm"]
                    )
                    batch_entropies.append(weighted_entropy)

                    batch_advs.append(adv.item())
                    batch_returns.append(ret.item())
                    batch_values.append(state_value.item())
                    batch_stop_entropies.append(float(out["stop_entropy"].detach().cpu().item()))
                    batch_feat_entropies.append(float(out["feature_entropy_raw"].detach().cpu().item()))
                    batch_feat_entropies_norm.append(float(out["feature_entropy_norm"].detach().cpu().item()))
                    batch_max_feature_probs.append(float(out["max_feature_prob"].detach().cpu().item()))
                    batch_top5_prob_mass.append(float(out["top5_prob_mass"].detach().cpu().item()))
                    batch_kls.append(float((old_lp - new_log_prob).detach().cpu().item()))
                    batch_ratios.append(float(ratio.detach().cpu().item()))
                    batch_clip_flags.append(
                        float((torch.abs(ratio - 1.0) > float(PPO_CLIP_EPS)).float().detach().cpu().item())
                    )

                policy_loss = torch.stack(batch_policy_losses).mean()
                value_loss = torch.stack(batch_value_losses).mean()
                entropy_bonus = torch.stack(batch_entropies).mean()

                update_advs.extend(batch_advs)
                update_returns.extend(batch_returns)
                update_values.extend(batch_values)
                update_stop_entropies.extend(batch_stop_entropies)
                update_feat_entropies.extend(batch_feat_entropies)
                update_feat_entropies_norm.extend(batch_feat_entropies_norm)
                update_max_feature_probs.extend(batch_max_feature_probs)
                update_top5_prob_mass.extend(batch_top5_prob_mass)
                update_kls.extend(batch_kls)
                update_clip_flags.extend(batch_clip_flags)
                update_ratios.extend(batch_ratios)

                total_loss = policy_loss + float(PPO_VALUE_COEF) * value_loss - entropy_bonus
                self.ppo_optimizer.zero_grad()
                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(PPO_MAX_GRAD_NORM))
                update_grad_norms.append(float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm))
                self.ppo_optimizer.step()

                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropies.append(float(entropy_bonus.detach().cpu().item()))

            self.logger.info(
                f"[PPO_MON] "
                f"policy_loss={np.mean(policy_losses):.4f}, "
                f"value_loss={np.mean(value_losses):.4f}, "
                f"entropy_total={np.mean(entropies):.4f}, "
                f"stop_entropy={np.mean(update_stop_entropies):.4f}, "
                f"feat_entropy_raw={np.mean(update_feat_entropies):.4f}, "
                f"feat_entropy_norm={np.mean(update_feat_entropies_norm):.4f}, "
                f"max_feat_prob={np.mean(update_max_feature_probs):.4f}, "
                f"top5_mass={np.mean(update_top5_prob_mass):.4f}, "
                f"approx_kl={np.mean(update_kls):.6f}, "
                f"clip_frac={np.mean(update_clip_flags):.4f}, "
                f"ratio_mean={np.mean(update_ratios):.4f}, "
                f"grad_norm={np.mean(update_grad_norms):.4f}"
            )
            last_stats = {
                "updated": True,
                "n_steps": int(len(rollout_batch)),
                "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
                "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
                "entropy": float(np.mean(entropies)) if entropies else 0.0,
                "stop_entropy": float(np.mean(update_stop_entropies)) if update_stop_entropies else 0.0,
                "feat_entropy_raw": float(np.mean(update_feat_entropies)) if update_feat_entropies else 0.0,
                "feat_entropy_norm": float(np.mean(update_feat_entropies_norm)) if update_feat_entropies_norm else 0.0,
                "max_feat_prob": float(np.mean(update_max_feature_probs)) if update_max_feature_probs else 0.0,
                "top5_prob_mass": float(np.mean(update_top5_prob_mass)) if update_top5_prob_mass else 0.0,
                "approx_kl": float(np.mean(update_kls)) if update_kls else 0.0,
                "clip_fraction": float(np.mean(update_clip_flags)) if update_clip_flags else 0.0,
                "ratio_mean": float(np.mean(update_ratios)) if update_ratios else 0.0,
                "grad_norm": float(np.mean(update_grad_norms)) if update_grad_norms else 0.0,
            }
        return last_stats

    # ------------------------------------------------------------------
    # Network pair shaping
    # ------------------------------------------------------------------

    def get_selected_network_pairs(self, selected_features):
        pairs = set()
        for idx in selected_features:
            if int(idx) < 0 or int(idx) >= len(self.feature_names):
                continue
            fname = str(self.feature_names[int(idx)])
            try:
                n1, n2 = parse_edge_networks(fname)
            except Exception:
                continue
            if n1 == "UNKNOWN" or n2 == "UNKNOWN":
                continue
            pairs.add(tuple(sorted((n1, n2))))
        return pairs

    def compute_coarse_network_pair_shaping_score(self, selected_features, phase: str):
        pairs = self.get_selected_network_pairs(selected_features)
        if len(pairs) == 0:
            return 0.0, 0.0, 0.0, 0
        prior_dict = NETWORK_PAIR_PRIOR_WARMUP if phase == "warmup" else NETWORK_PAIR_PRIOR_JOINT
        max_weight = max(prior_dict.values()) if prior_dict else 1.0

        total_pos = 0.0
        neg_count = 0
        for pair in pairs:
            pair = tuple(sorted(pair))
            total_pos += float(prior_dict.get(pair, 0.0))
            if any(net in NON_TARGET_NETWORKS for net in pair):
                neg_count += 1

        total = float(len(pairs))
        pos_ratio = (total_pos / total) / max(max_weight, 1e-8)
        neg_ratio = neg_count / total
        score = pos_ratio
        return float(score), float(pos_ratio), float(neg_ratio), int(len(pairs))

    def compute_subnetwork_pair_shaping_score(self, selected_features, phase: str = "joint"):
        pairs = get_selected_subnetwork_pairs(selected_features, self.feature_names)
        if len(pairs) == 0:
            return 0.0, 0.0, 0.0, 0

        prior_dict = SUBNETWORK_PAIR_PRIOR_JOINT
        total_score = 0.0
        neg_count = 0
        for a, b in pairs:
            pair = tuple(sorted((a, b)))
            weight = prior_dict.get(pair, 0.0)
            total_score += weight
            if ("Vis" in pair) or ("SomMot" in pair):
                neg_count += 1

        total = float(len(pairs))
        max_weight = max(prior_dict.values()) if prior_dict else 1.0
        pos_ratio = (total_score / total) / max(max_weight, 1e-8)
        neg_ratio = neg_count / total
        score = pos_ratio - JOINT_POSITIVE_SHAPING_NEG_COEF * neg_ratio
        return float(score), float(pos_ratio), float(neg_ratio), int(len(pairs))

    def compute_step_structure_bonus(self, selected_before, new_feature_idx: int, phase: str = "joint"):
        selected_before = np.asarray(selected_before, dtype=int)
        selected_after = np.concatenate([selected_before, np.asarray([int(new_feature_idx)], dtype=int)])

        coarse_before, coarse_pos_before, coarse_neg_before, coarse_cnt_before = self.compute_coarse_network_pair_shaping_score(selected_before, phase=phase)
        coarse_after, coarse_pos_after, coarse_neg_after, coarse_cnt_after = self.compute_coarse_network_pair_shaping_score(selected_after, phase=phase)
        coarse_delta = float(coarse_after - coarse_before)

        sub_before, sub_pos_before, sub_neg_before, sub_cnt_before = self.compute_subnetwork_pair_shaping_score(selected_before, phase=phase)
        sub_after, sub_pos_after, sub_neg_after, sub_cnt_after = self.compute_subnetwork_pair_shaping_score(selected_after, phase=phase)
        sub_delta = float(sub_after - sub_before)

        mixed_delta = (
            (1.0 - float(PPO_STEP_STRUCTURE_SUBNETWORK_MIX)) * coarse_delta
            + float(PPO_STEP_STRUCTURE_SUBNETWORK_MIX) * sub_delta
        )
        bonus = float(PPO_STEP_STRUCTURE_BONUS_SCALE) * float(mixed_delta)

        meta = {
            "phase": str(phase), "new_feature_idx": int(new_feature_idx),
            "selected_count_before": int(len(selected_before)),
            "selected_count_after": int(len(selected_after)),
            "coarse_before": float(coarse_before), "coarse_after": float(coarse_after),
            "coarse_delta": float(coarse_delta), "coarse_pos_before": float(coarse_pos_before),
            "coarse_pos_after": float(coarse_pos_after), "coarse_neg_before": float(coarse_neg_before),
            "coarse_neg_after": float(coarse_neg_after),
            "coarse_pair_count_before": int(coarse_cnt_before),
            "coarse_pair_count_after": int(coarse_cnt_after),
            "sub_before": float(sub_before), "sub_after": float(sub_after), "sub_delta": float(sub_delta),
            "sub_pos_before": float(sub_pos_before), "sub_pos_after": float(sub_pos_after),
            "sub_neg_before": float(sub_neg_before), "sub_neg_after": float(sub_neg_after),
            "sub_pair_count_before": int(sub_cnt_before), "sub_pair_count_after": int(sub_cnt_after),
            "mix_weight_subnetwork": float(PPO_STEP_STRUCTURE_SUBNETWORK_MIX),
            "mixed_delta": float(mixed_delta), "step_structure_bonus": float(bonus),
        }
        return float(bonus), meta

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def compute_reward(self, subset, inner_cv, stop_reason: str, phase: str, moving_baseline: float = 0.0,
                       episode_idx: int = None, final_stop_step: dict | None = None):
        inner_auc = evaluate_subset_auc_cv(self.X_train, self.y_train, subset, inner_cv)
        inner_auc_warmup = evaluate_subset_auc_cv_warmup(self.X_train, self.y_train, subset, inner_cv)

        subset_ratio = len(subset) / float(self.k_stop_ref)
        quadratic_term = subset_ratio ** 2
        sparsity_shape = (SPARSITY_LINEAR_WEIGHT * subset_ratio + (1.0 - SPARSITY_LINEAR_WEIGHT) * quadratic_term)
        sparsity_penalty = self.lambda_reg * sparsity_shape
        step_cost_penalty = STEP_COST * float(len(subset))
        late_excess = max(0.0, subset_ratio - CAP_LATE_THRESHOLD)
        late_length_penalty = LATE_LENGTH_PENALTY_SCALE * (late_excess ** 2)

        if stop_reason == "safety_cap":
            cap_penalty = CAP_BASE_PENALTY + CAP_LATE_SCALE * (late_excess ** 2)
        else:
            cap_penalty = 0.0

        quality_term = max(0.0, inner_auc - 0.60)
        if len(subset) > 0:
            subset_np = np.asarray(subset, dtype=int)
            base_logits_np = self.model.feature_logits.detach().cpu().numpy()[subset_np]
            mean_base_logit = float(np.mean(base_logits_np))
            q75_base_logit = float(np.percentile(base_logits_np, 75))
        else:
            mean_base_logit = q75_base_logit = 0.0

        selected_q75_effective = 0.0
        if final_stop_step is not None:
            selected_q75_effective = float(final_stop_step.get("selected_q75_logit", 0.0))

        feature_signal = (
            0.08 * quality_term
            + 0.004 * max(0.0, mean_base_logit)
            + 0.004 * max(0.0, q75_base_logit)
            + 0.006 * max(0.0, selected_q75_effective)
        )

        baseline_advantage = max(0.0, inner_auc - moving_baseline)
        baseline_advantage_warmup = max(0.0, inner_auc_warmup - moving_baseline)

        if phase == "warmup":
            feature_bonus = 0.0
        else:
            feature_bonus = JOINT_FEATURE_BONUS_SCALE * baseline_advantage

        compact_term = max(0.0, 1.0 - subset_ratio)
        quality_gate = min(1.0, quality_term / 0.10)
        compact_bonus = 0.0005 * compact_term * quality_gate

        stop_bonus = active_stop_reward = synergy_bonus = tiny_subset_penalty = 0.0

        if stop_reason == "policy_stop" and final_stop_step is not None:
            gain_gap_at_stop = float(final_stop_step.get("gain_gap", 0.0))
            coverage_at_stop = float(final_stop_step.get("network_coverage_ratio", 0.0))
            cross_ratio_at_stop = float(final_stop_step.get("cross_network_ratio", 0.0))
            final_k = int(len(subset))

            structure_score = 0.5 * coverage_at_stop + 0.5 * cross_ratio_at_stop
            min_useful_k = float(MIN_FEATURES_BEFORE_STOP)
            if final_k < min_useful_k:
                early_stop_penalty = 0.15 * (min_useful_k - final_k) / float(min_useful_k)
            else:
                early_stop_penalty = 0.0

            gap_tau = 0.35
            stop_gap_signal = float(np.exp(-((gain_gap_at_stop / gap_tau) ** 2)))
            stop_bonus = 0.015 * stop_gap_signal + 0.01 * structure_score

            if final_k <= 5:
                tiny_subset_penalty = 0.02 * max(0, 6 - final_k)

            stop_margin_lo, stop_margin_hi = 0.20, 0.80
            if final_k < 6:
                active_stop_reward = -0.03
            elif gain_gap_at_stop <= -stop_margin_hi:
                active_stop_reward = 0.04
            elif gain_gap_at_stop <= -stop_margin_lo:
                active_stop_reward = 0.025
            elif gain_gap_at_stop <= stop_margin_lo:
                active_stop_reward = 0.005
            elif gain_gap_at_stop <= stop_margin_hi:
                active_stop_reward = -0.015
            else:
                active_stop_reward = -0.03
            active_stop_reward = active_stop_reward - tiny_subset_penalty

        if phase == "warmup":
            warmup_size_penalty = 0.02 * subset_ratio
            shaping_score, pos_ratio, neg_ratio, pair_count = self.compute_coarse_network_pair_shaping_score(subset, phase="warmup")
            pair_term = WARMUP_COARSE_PAIR_SHAPING_SCALE * shaping_score
            reward = inner_auc_warmup - warmup_size_penalty + pair_term
            sparsity_penalty = warmup_size_penalty
            step_cost_penalty = late_length_penalty = cap_penalty = compact_bonus = stop_bonus = 0.0
            feature_signal = total_stop_reward = applied_step_cost_penalty = applied_feature_signal = 0.0
            joint_progress = 0.0
            penalty_scale = 1.0
            penalty_total = warmup_size_penalty
            effective_penalty = penalty_scale * penalty_total
        else:
            joint_budget = float(max(1, self.joint_max_episodes if self.joint_max_episodes is not None else 1))
            joint_progress = float(self.joint_episode_count) / joint_budget
            joint_progress = max(0.0, min(1.0, joint_progress))

            if self.joint_episode_count < 60:
                penalty_scale = 0.25
            else:
                progress = (self.joint_episode_count - 60) / max(1, self.joint_max_episodes - 60)
                progress = max(0.0, min(1.0, progress))
                penalty_scale = 0.25 + 0.20 * progress

            shaping_score, pos_ratio, neg_ratio, pair_count = self.compute_subnetwork_pair_shaping_score(subset, phase="joint")
            pair_term = NETWORK_PAIR_SHAPING_COEF * shaping_score
            penalty_total = sparsity_penalty + step_cost_penalty + late_length_penalty + cap_penalty
            effective_penalty = penalty_scale * penalty_total
            reward = (
                inner_auc
                - effective_penalty - tiny_subset_penalty
                + feature_signal + feature_bonus + compact_bonus
                + stop_bonus + active_stop_reward + synergy_bonus
                + pair_term
            )
            total_stop_reward = stop_bonus + active_stop_reward + synergy_bonus
            applied_step_cost_penalty = step_cost_penalty
            applied_feature_signal = feature_signal

        return (
            float(reward), float(inner_auc), float(inner_auc_warmup), float(sparsity_penalty),
            float(cap_penalty), float(late_length_penalty), float(total_stop_reward),
            float(applied_step_cost_penalty), float(compact_bonus), float(applied_feature_signal),
            float(feature_bonus), float(shaping_score), float(pos_ratio), float(neg_ratio),
            int(pair_count), float(pair_term), float(penalty_scale), float(joint_progress),
            float(effective_penalty), float(active_stop_reward), float(synergy_bonus),
        )

    # ------------------------------------------------------------------
    # Episode candidate tracking
    # ------------------------------------------------------------------

    def update_top_episode_candidates(self, episode: int, subset: np.ndarray, inner_auc: float, reward: float, stop_reason: str):
        if subset is None or len(subset) == 0:
            return
        subset = sanitize_subset(subset, self.n_features)
        key = tuple(int(x) for x in subset.tolist())
        entry = {
            "episode": int(episode), "subset": subset.tolist(), "subset_size": int(len(subset)),
            "inner_auc": float(inner_auc), "reward": float(reward), "stop_reason": str(stop_reason),
            "selection_score": [float(inner_auc), float(reward), -float(len(subset))],
        }
        replaced = False
        for idx, existing in enumerate(self.top_episode_candidates):
            if tuple(existing["subset"]) == key:
                if tuple(entry["selection_score"]) > tuple(existing["selection_score"]):
                    self.top_episode_candidates[idx] = entry
                replaced = True
                break
        if not replaced:
            self.top_episode_candidates.append(entry)
        self.top_episode_candidates = sorted(
            self.top_episode_candidates,
            key=lambda d: tuple(d["selection_score"]),
            reverse=True,
        )[:TOP_EPISODES_FOR_REFIT]

    def update_joint_refit_episode_history(self, episode: int, subset: np.ndarray, inner_auc: float, reward: float, stop_reason: str):
        subset = sanitize_subset(subset, self.n_features)
        if len(subset) == 0:
            return
        self.joint_refit_episode_history.append({
            "episode": int(episode), "subset": subset.tolist(), "subset_size": int(len(subset)),
            "inner_auc": float(inner_auc), "reward": float(reward), "stop_reason": str(stop_reason),
        })

    # ------------------------------------------------------------------
    # Refit subset selection
    # ------------------------------------------------------------------

    def select_refit_subset_from_feature_ranking(self):
        logits = self.model.feature_logits.detach().cpu().numpy()
        if logits.size == 0:
            return None
        ranked_indices = np.argsort(-logits)
        top_k = min(int(self.k_stop_ref), self.n_features)
        selected_subset = sanitize_subset(ranked_indices[:top_k], self.n_features)
        if len(selected_subset) == 0:
            return None
        selected_logits = logits[selected_subset]
        self.refit_selection_summary = {
            "method": "feature_ranking_top_k",
            "refit_top_k": int(top_k),
            "selected_subset": selected_subset.tolist(),
            "selected_subset_size": int(len(selected_subset)),
            "selected_logits": [float(x) for x in selected_logits.tolist()],
            "max_logit": float(np.max(selected_logits)),
            "min_logit": float(np.min(selected_logits)),
            "mean_logit": float(np.mean(selected_logits)),
        }
        return selected_subset

    def rollout_greedy_subset(self):
        subset, _, stop_reason, _, _ = self.sample_subset(greedy=True, warmup=False)
        subset = sanitize_subset(subset, self.n_features)
        if len(subset) == 0:
            logits = self.model.feature_logits.detach().cpu().numpy()
            subset = np.array([int(np.argmax(logits))], dtype=int)
        return subset, stop_reason

    def multiple_rollout_consensus_subset(self, n_rollouts: int = PPO_CONSENSUS_ROLLOUTS):
        subsets = []
        for _ in range(int(max(1, n_rollouts))):
            subset, _, _, _, _ = self.sample_subset(greedy=False, warmup=False)
            subset = sanitize_subset(subset, self.n_features)
            if len(subset) > 0:
                subsets.append(subset)
        if len(subsets) == 0:
            return self.rollout_greedy_subset()[0]
        counts = np.zeros(self.n_features, dtype=np.float32)
        lengths = []
        for subset in subsets:
            counts[subset] += 1.0
            lengths.append(len(subset))
        target_k = max(1, int(round(float(np.median(lengths)))))
        ranked = np.argsort(-counts)
        return sanitize_subset(ranked[:target_k], self.n_features)

    def hybrid_score_subset(self, n_rollouts: int = PPO_CONSENSUS_ROLLOUTS, alpha: float = HYBRID_SCORE_ALPHA):
        logits = self.model.feature_logits.detach().cpu().numpy().astype(np.float32)
        subsets = []
        for _ in range(int(max(1, n_rollouts))):
            subset, _, _, _, _ = self.sample_subset(greedy=False, warmup=False)
            subset = sanitize_subset(subset, self.n_features)
            if len(subset) > 0:
                subsets.append(subset)
        freq = np.zeros(self.n_features, dtype=np.float32)
        lengths = []
        for subset in subsets:
            freq[subset] += 1.0
            lengths.append(len(subset))
        if subsets:
            freq /= float(len(subsets))
            target_k = max(1, int(round(float(np.median(lengths)))))
        else:
            target_k = min(int(self.k_stop_ref), self.n_features)
        logits_norm = (logits - logits.min()) / (logits.max() - logits.min() + 1e-8)
        score = float(alpha) * logits_norm + (1.0 - float(alpha)) * freq
        return sanitize_subset(np.argsort(-score)[:target_k], self.n_features)

    def build_refit_subset_by_method(self, method: str):
        method = str(method)
        if method == "greedy_rollout_subset":
            subset, stop_reason = self.rollout_greedy_subset()
            return subset, {"method": method, "stop_reason": stop_reason}
        if method == "multiple_rollout_consensus_subset":
            subset = self.multiple_rollout_consensus_subset()
            return subset, {"method": method, "n_rollouts": int(PPO_CONSENSUS_ROLLOUTS)}
        if method == "hybrid_score":
            subset = self.hybrid_score_subset()
            return subset, {"method": method, "n_rollouts": int(PPO_CONSENSUS_ROLLOUTS), "alpha": float(HYBRID_SCORE_ALPHA)}
        if method == "feature_ranking_top_k":
            subset = self.select_refit_subset_from_feature_ranking()
            return subset, {"method": method}
        raise ValueError(f"Unknown refit subset method: {method}")

    def get_refit_subset(self):
        selected_subset, meta = self.build_refit_subset_by_method(self.refit_subset_source)
        selected_subset = sanitize_subset(selected_subset, self.n_features)
        final_method = str(meta.get("method", self.refit_subset_source))
        final_meta = dict(meta)

        if len(selected_subset) == 0:
            if self.best_subset is not None:
                selected_subset = sanitize_subset(self.best_subset, self.n_features)
                final_method = "best_subset_fallback"
                final_meta = {"method": final_method}
            else:
                greedy_subset, _ = self.rollout_greedy_subset()
                selected_subset = sanitize_subset(greedy_subset, self.n_features)
                final_method = "greedy_subset_fallback"
                final_meta = {"method": final_method}

        self.refit_selection_summary = {
            "method": final_method,
            "selected_subset": selected_subset.tolist(),
            "selected_subset_size": int(len(selected_subset)),
            **final_meta,
        }
        return selected_subset

    def _pairwise_subset_overlap(self, subsets):
        subsets = [sanitize_subset(x, self.n_features) for x in subsets if len(x) > 0]
        if len(subsets) <= 1:
            return 1.0 if len(subsets) == 1 else 0.0
        vals = []
        for i in range(len(subsets)):
            for j in range(i + 1, len(subsets)):
                vals.append(jaccard_similarity(subsets[i].tolist(), subsets[j].tolist()))
        return float(np.mean(vals)) if vals else 0.0

    def evaluate_policy_checkpoint(self, episode: int, inner_cv):
        greedy_subset, greedy_stop_reason = self.rollout_greedy_subset()
        sampled_subsets = []
        for _ in range(int(max(2, PPO_CHECKPOINT_SAMPLED_ROLLOUTS))):
            subset, _, _, _, _ = self.sample_subset(greedy=False, warmup=False)
            subset = sanitize_subset(subset, self.n_features)
            if len(subset) > 0:
                sampled_subsets.append(subset)
        sampled_overlap_mean = self._pairwise_subset_overlap(sampled_subsets)
        prev_greedy = self.joint_checkpoint_history[-1]["greedy_subset"] if self.joint_checkpoint_history else None
        greedy_overlap = None if prev_greedy is None else float(jaccard_similarity(prev_greedy, greedy_subset.tolist()))

        refit_methods = ["greedy_rollout_subset", "multiple_rollout_consensus_subset", "hybrid_score", "feature_ranking_top_k"]
        refit_evaluations = {}
        for method in refit_methods:
            subset, meta = self.build_refit_subset_by_method(method)
            subset = sanitize_subset(subset, self.n_features)
            refit_auc = float(evaluate_subset_auc_cv(self.X_train, self.y_train, subset, inner_cv)) if len(subset) > 0 else 0.5
            refit_evaluations[method] = {
                "subset": subset.tolist(), "subset_size": int(len(subset)), "refit_cv_auc": refit_auc, **meta,
            }

        checkpoint = {
            "episode": int(episode),
            "greedy_subset": greedy_subset.tolist(),
            "greedy_stop_reason": greedy_stop_reason,
            "greedy_overlap_with_previous": greedy_overlap,
            "sampled_subsets": [x.tolist() for x in sampled_subsets],
            "sampled_overlap_mean": float(sampled_overlap_mean),
            "refit_evaluations": refit_evaluations,
            "chosen_refit_method": str(self.refit_subset_source),
        }
        self.joint_checkpoint_history.append(checkpoint)
        return checkpoint

    # ------------------------------------------------------------------
    # High-reward feature credit
    # ------------------------------------------------------------------

    def update_high_reward_feature_credit(self, subset: np.ndarray, reward: float, moving_baseline: float):
        subset = sanitize_subset(subset, self.n_features)
        if len(subset) == 0:
            return 0.0
        positive_advantage = max(0.0, float(reward) - float(moving_baseline))
        if positive_advantage <= 0.0:
            return 0.0
        per_feature_credit = positive_advantage / float(len(subset))
        decay = float(HIGH_REWARD_FEATURE_CREDIT_EMA)
        self.high_reward_feature_credit *= decay
        self.high_reward_feature_credit[subset] += (1.0 - decay) * per_feature_credit
        self.recent_high_reward_subsets.append(subset.astype(np.int32).copy())
        recent_window = int(HIGH_REWARD_RECENT_WINDOW)
        if recent_window > 0 and len(self.recent_high_reward_subsets) > recent_window:
            self.recent_high_reward_subsets = self.recent_high_reward_subsets[-recent_window:]
        return float(per_feature_credit)

    def get_stable_high_reward_features(self) -> np.ndarray:
        if len(self.recent_high_reward_subsets) == 0:
            return np.asarray([], dtype=int)
        occurrence = np.zeros(self.n_features, dtype=np.int32)
        for recent_subset in self.recent_high_reward_subsets:
            recent_subset = sanitize_subset(recent_subset, self.n_features)
            if len(recent_subset) > 0:
                occurrence[recent_subset] += 1
        min_occ = max(1, int(HIGH_REWARD_STABLE_MIN_OCCURRENCES))
        stable_candidates = np.where(occurrence >= min_occ)[0]
        if stable_candidates.size == 0:
            return np.asarray([], dtype=int)
        candidate_credit = self.high_reward_feature_credit[stable_candidates]
        positive_mask = candidate_credit > 0.0
        stable_candidates = stable_candidates[positive_mask]
        candidate_credit = candidate_credit[positive_mask]
        if stable_candidates.size == 0:
            return np.asarray([], dtype=int)
        top_k = max(1, int(HIGH_REWARD_STABLE_TOP_K))
        order = np.argsort(candidate_credit)[::-1]
        return sanitize_subset(stable_candidates[order[:top_k]], self.n_features)

    def apply_high_reward_feature_boost(self, subset: np.ndarray, phase: str):
        subset = sanitize_subset(subset, self.n_features)
        if len(subset) == 0:
            return 0.0
        stable_features = self.get_stable_high_reward_features()
        if len(stable_features) == 0:
            return 0.0
        boost_targets = sanitize_subset(np.intersect1d(subset, stable_features, assume_unique=False), self.n_features)
        if len(boost_targets) == 0:
            return 0.0
        target_credit = self.high_reward_feature_credit[boost_targets]
        max_credit = float(np.max(target_credit)) if target_credit.size > 0 else 0.0
        if max_credit <= 0.0:
            return 0.0
        normalized_credit = target_credit / (max_credit + 1e-8)
        boost_scale = float(HIGH_REWARD_FEATURE_CREDIT_WARMUP_SCALE) if phase == "warmup" else float(HIGH_REWARD_FEATURE_CREDIT_JOINT_SCALE)
        boost = boost_scale * normalized_credit
        boost_tensor = torch.tensor(boost, dtype=self.model.feature_logits.dtype, device=self.model.feature_logits.device)
        boost_targets_tensor = torch.tensor(boost_targets, dtype=torch.long, device=self.model.feature_logits.device)
        with torch.no_grad():
            self.model.feature_logits[boost_targets_tensor] += boost_tensor
        return float(np.mean(boost))

    # ------------------------------------------------------------------
    # Checkpoint / status I/O
    # ------------------------------------------------------------------

    def save_status(self, episode: int, completed: bool, stop_reason: str):
        status = {
            "seed": self.seed, "current_episode": episode,
            "best_reward": float(self.best_reward), "best_inner_auc": float(self.best_inner_auc),
            "best_subset_size": int(len(self.best_subset)) if self.best_subset is not None else 0,
            "is_completed": completed, "stop_reason": stop_reason, "params": self.params,
        }
        save_json(status, self.run_dir / STATUS_FILENAME)

    def save_training_checkpoint(self, episode: int):
        checkpoint = {
            "episode": episode,
            "joint_feature_temperature_history": self.joint_feature_temperature_history,
            "joint_stop_barrier_history": self.joint_stop_barrier_history,
            "joint_stop_applied_barrier_history": self.joint_stop_applied_barrier_history,
            "joint_stop_temperature_history": self.joint_stop_temperature_history,
            "ppo_feat_entropy_coef_history": self.ppo_feat_entropy_coef_history,
            "ppo_stop_entropy_coef_history": self.ppo_stop_entropy_coef_history,
            "model_state_dict": self.model.state_dict(),
            "warmup_feature_optimizer_state_dict": self.warmup_feature_optimizer.state_dict(),
            "ppo_optimizer_state_dict": self.ppo_optimizer.state_dict(),
            "joint_stop_step_records": self.joint_stop_step_records,
            "reward_history": self.reward_history,
            "inner_auc_history": self.inner_auc_history,
            "penalty_scale_history": self.penalty_scale_history,
            "joint_progress_history": self.joint_progress_history,
            "effective_penalty_history": self.effective_penalty_history,
            "active_stop_reward_history": self.active_stop_reward_history,
            "synergy_bonus_history": self.synergy_bonus_history,
            "inner_auc_warmup_history": self.inner_auc_warmup_history,
            "selected_size_history": self.selected_size_history,
            "moving_avg_history": self.moving_avg_history,
            "stop_reason_history": self.stop_reason_history,
            "best_reward": self.best_reward, "best_inner_auc": self.best_inner_auc,
            "best_subset": self.best_subset, "best_episode": self.best_episode,
            "best_selection_score": self.best_selection_score,
            "best_stop_reason": self.best_stop_reason,
            "top_episode_candidates": self.top_episode_candidates,
            "refit_selection_summary": self.refit_selection_summary,
            "joint_refit_episode_history": self.joint_refit_episode_history,
            "params": self.params, "seed": self.seed,
            "policy_stop_history": self.policy_stop_history,
            "policy_stop_ratio_history": self.policy_stop_ratio_history,
            "logits_std_history": self.logits_std_history,
            "sparsity_penalty_history": self.sparsity_penalty_history,
            "cap_penalty_history": self.cap_penalty_history,
            "late_length_penalty_history": self.late_length_penalty_history,
            "stop_bonus_history": self.stop_bonus_history,
            "compact_bonus_history": self.compact_bonus_history,
            "feature_signal_history": self.feature_signal_history,
            "feature_bonus_history": self.feature_bonus_history,
            "step_cost_history": self.step_cost_history,
            "continue_logit_history": self.continue_logit_history,
            "step_structure_bonus_history": self.step_structure_bonus_history,
            "stop_logit_history": self.stop_logit_history,
            "stop_count_scale_history": self.stop_count_scale_history,
            "stop_quality_scale_history": self.stop_quality_scale_history,
            "raw_stop_gain_scale_history": self.raw_stop_gain_scale_history,
            "stop_gain_scale_history": self.stop_gain_scale_history,
            "stop_prev_reward_scale_history": self.stop_prev_reward_scale_history,
            "stop_advantage_scale_history": self.stop_advantage_scale_history,
            "stop_reward_trend_scale_history": self.stop_reward_trend_scale_history,
            "phase_history": self.phase_history,
            "top25_logits_mean_history": self.top25_logits_mean_history,
            "top25_logits_std_history": self.top25_logits_std_history,
            "top50_logits_mean_history": self.top50_logits_mean_history,
            "top50_logits_std_history": self.top50_logits_std_history,
            "top25_minus_top50_gap_history": self.top25_minus_top50_gap_history,
            "top50_minus_top100_gap_history": self.top50_minus_top100_gap_history,
            "max_minus_median_logit_history": self.max_minus_median_logit_history,
            "max_median_gap_history": self.max_median_gap_history,
            "top50_overlap_recent_history": self.top50_overlap_recent_history,
            "top100_overlap_recent_history": self.top100_overlap_recent_history,
            "shaping_score_history": self.shaping_score_history,
            "pair_pos_ratio_history": self.pair_pos_ratio_history,
            "pair_neg_ratio_history": self.pair_neg_ratio_history,
            "pair_count_history": self.pair_count_history,
            "pair_term_history": self.pair_term_history,
            "feature_logits_snapshots": self.feature_logits_snapshots,
            "feature_snapshot_episodes": self.feature_snapshot_episodes,
            "final_stop_probability_history": self.final_stop_probability_history,
            "final_stop_selected_count_history": self.final_stop_selected_count_history,
            "warmup_completed": self.warmup_completed,
            "warmup_end_episode": self.warmup_end_episode,
            "warmup_end_reason": self.warmup_end_reason,
            "joint_episode_count": self.joint_episode_count,
            "warmup_transition_history": self.warmup_transition_history,
            "joint_refit_readiness_history": self.joint_refit_readiness_history,
            "joint_refit_episode": self.joint_refit_episode,
            "joint_refit_reason": self.joint_refit_reason,
            "current_prior_stage": self.current_prior_stage,
            "initial_prior_summary": self.initial_prior_summary,
            "latest_prior_summary": self.latest_prior_summary,
            "prior_summary_history": self.prior_summary_history,
            "high_reward_feature_credit": self.high_reward_feature_credit.tolist(),
            "recent_high_reward_subsets": [x.tolist() for x in self.recent_high_reward_subsets],
            "high_reward_feature_boost_history": self.high_reward_feature_boost_history,
            "joint_checkpoint_history": self.joint_checkpoint_history,
            "ppo_update_history": self.ppo_update_history,
            "refit_subset_source": self.refit_subset_source,
        }
        save_checkpoint(checkpoint, self.run_dir / CHECKPOINT_FILENAME)

    def maybe_resume(self):
        ckpt_path = self.run_dir / CHECKPOINT_FILENAME
        if not ckpt_path.exists():
            return
        ckpt = load_checkpoint(ckpt_path)
        try:
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "warmup_feature_optimizer_state_dict" in ckpt:
                self.warmup_feature_optimizer.load_state_dict(ckpt["warmup_feature_optimizer_state_dict"])
            if "ppo_optimizer_state_dict" in ckpt:
                self.ppo_optimizer.load_state_dict(ckpt["ppo_optimizer_state_dict"])
        except Exception as e:
            self.logger.warning(
                f"Checkpoint at {ckpt_path} is incompatible with the current policy; "
                f"starting from scratch. Details: {e}"
            )
            return

        self.reward_history = ckpt.get("reward_history", [])
        self.joint_stop_temperature_history = ckpt.get("joint_stop_temperature_history", [])
        self.joint_feature_temperature_history = ckpt.get("joint_feature_temperature_history", [])
        self.joint_stop_barrier_history = ckpt.get("joint_stop_barrier_history", [])
        self.joint_stop_applied_barrier_history = ckpt.get("joint_stop_applied_barrier_history", [])
        self.ppo_feat_entropy_coef_history = ckpt.get("ppo_feat_entropy_coef_history", [])
        self.ppo_stop_entropy_coef_history = ckpt.get("ppo_stop_entropy_coef_history", [])
        self.inner_auc_history = ckpt.get("inner_auc_history", [])
        self.inner_auc_warmup_history = ckpt.get("inner_auc_warmup_history", [])
        self.selected_size_history = ckpt.get("selected_size_history", [])
        self.penalty_scale_history = ckpt.get("penalty_scale_history", [])
        self.joint_progress_history = ckpt.get("joint_progress_history", [])
        self.effective_penalty_history = ckpt.get("effective_penalty_history", [])
        self.active_stop_reward_history = ckpt.get("active_stop_reward_history", [])
        self.synergy_bonus_history = ckpt.get("synergy_bonus_history", [])
        self.moving_avg_history = ckpt.get("moving_avg_history", [])
        self.stop_reason_history = ckpt.get("stop_reason_history", [])
        self.best_reward = ckpt.get("best_reward", -np.inf)
        self.best_inner_auc = ckpt.get("best_inner_auc", 0.5)
        self.best_subset = ckpt.get("best_subset", None)
        self.best_episode = ckpt.get("best_episode", -1)
        self.best_selection_score = tuple(ckpt.get("best_selection_score", (-np.inf, -np.inf, -np.inf)))
        self.best_stop_reason = ckpt.get("best_stop_reason", None)
        self.top_episode_candidates = ckpt.get("top_episode_candidates", [])
        self.refit_selection_summary = ckpt.get("refit_selection_summary", {})
        self.joint_refit_episode_history = ckpt.get("joint_refit_episode_history", [])
        self.start_episode = int(ckpt.get("episode", -1)) + 1
        self.policy_stop_history = ckpt.get("policy_stop_history", [])
        self.step_structure_bonus_history = ckpt.get("step_structure_bonus_history", [])
        self.policy_stop_ratio_history = ckpt.get("policy_stop_ratio_history", [])
        self.logits_std_history = ckpt.get("logits_std_history", [])
        self.sparsity_penalty_history = ckpt.get("sparsity_penalty_history", [])
        self.cap_penalty_history = ckpt.get("cap_penalty_history", [])
        self.late_length_penalty_history = ckpt.get("late_length_penalty_history", [])
        self.stop_bonus_history = ckpt.get("stop_bonus_history", [])
        self.compact_bonus_history = ckpt.get("compact_bonus_history", [])
        self.feature_signal_history = ckpt.get("feature_signal_history", [])
        self.feature_bonus_history = ckpt.get("feature_bonus_history", [])
        self.step_cost_history = ckpt.get("step_cost_history", [])
        self.continue_logit_history = ckpt.get("continue_logit_history", [])
        self.stop_logit_history = ckpt.get("stop_logit_history", [])
        self.stop_count_scale_history = ckpt.get("stop_count_scale_history", [])
        self.stop_quality_scale_history = ckpt.get("stop_quality_scale_history", [])
        self.raw_stop_gain_scale_history = ckpt.get("raw_stop_gain_scale_history", [])
        self.stop_gain_scale_history = ckpt.get("stop_gain_scale_history", [])
        self.stop_prev_reward_scale_history = ckpt.get("stop_prev_reward_scale_history", [])
        self.stop_advantage_scale_history = ckpt.get("stop_advantage_scale_history", [])
        self.stop_reward_trend_scale_history = ckpt.get("stop_reward_trend_scale_history", [])
        self.phase_history = ckpt.get("phase_history", [])
        self.top25_logits_mean_history = ckpt.get("top25_logits_mean_history", [])
        self.top25_logits_std_history = ckpt.get("top25_logits_std_history", [])
        self.top50_logits_mean_history = ckpt.get("top50_logits_mean_history", [])
        self.top50_logits_std_history = ckpt.get("top50_logits_std_history", [])
        self.top25_minus_top50_gap_history = ckpt.get("top25_minus_top50_gap_history", [])
        self.top50_minus_top100_gap_history = ckpt.get("top50_minus_top100_gap_history", [])
        self.max_minus_median_logit_history = ckpt.get("max_minus_median_logit_history", [])
        self.max_median_gap_history = ckpt.get("max_median_gap_history", list(self.max_minus_median_logit_history))
        self.top50_overlap_recent_history = ckpt.get("top50_overlap_recent_history", [])
        self.top100_overlap_recent_history = ckpt.get("top100_overlap_recent_history", [])
        self.shaping_score_history = ckpt.get("shaping_score_history", [])
        self.pair_pos_ratio_history = ckpt.get("pair_pos_ratio_history", [])
        self.pair_neg_ratio_history = ckpt.get("pair_neg_ratio_history", [])
        self.pair_count_history = ckpt.get("pair_count_history", [])
        self.pair_term_history = ckpt.get("pair_term_history", [])
        self.feature_logits_snapshots = ckpt.get("feature_logits_snapshots", [])
        self.feature_snapshot_episodes = ckpt.get("feature_snapshot_episodes", [])
        self.joint_stop_step_records = ckpt.get("joint_stop_step_records", [])
        self.final_stop_probability_history = ckpt.get("final_stop_probability_history", [])
        self.final_stop_selected_count_history = ckpt.get("final_stop_selected_count_history", [])
        self.warmup_completed = ckpt.get("warmup_completed", False)
        self.warmup_end_episode = ckpt.get("warmup_end_episode", None)
        self.warmup_end_reason = ckpt.get("warmup_end_reason", None)
        self.joint_episode_count = ckpt.get("joint_episode_count", 0)
        self.warmup_transition_history = ckpt.get("warmup_transition_history", [])
        self.joint_refit_readiness_history = ckpt.get("joint_refit_readiness_history", [])
        self.joint_refit_episode = ckpt.get("joint_refit_episode", None)
        self.joint_refit_reason = ckpt.get("joint_refit_reason", None)
        self.current_prior_stage = ckpt.get("current_prior_stage", self.current_prior_stage)
        self.initial_prior_summary = ckpt.get("initial_prior_summary", getattr(self, "initial_prior_summary", {}))
        self.latest_prior_summary = ckpt.get("latest_prior_summary", getattr(self, "latest_prior_summary", {}))
        self.prior_summary_history = ckpt.get("prior_summary_history", getattr(self, "prior_summary_history", []))
        self.high_reward_feature_credit = np.asarray(
            ckpt.get("high_reward_feature_credit", np.zeros(self.n_features, dtype=np.float32)),
            dtype=np.float32,
        )
        if self.high_reward_feature_credit.shape[0] != self.n_features:
            self.high_reward_feature_credit = np.zeros(self.n_features, dtype=np.float32)
        self.recent_high_reward_subsets = [
            sanitize_subset(np.asarray(x, dtype=int), self.n_features)
            for x in ckpt.get("recent_high_reward_subsets", [])
        ]
        recent_window = int(HIGH_REWARD_RECENT_WINDOW)
        if recent_window > 0 and len(self.recent_high_reward_subsets) > recent_window:
            self.recent_high_reward_subsets = self.recent_high_reward_subsets[-recent_window:]
        self.high_reward_feature_boost_history = ckpt.get("high_reward_feature_boost_history", [])
        self.joint_checkpoint_history = ckpt.get("joint_checkpoint_history", [])
        self.ppo_update_history = ckpt.get("ppo_update_history", [])
        self.refit_subset_source = ckpt.get("refit_subset_source", self.refit_subset_source)
        self.logger.info(f"Resumed from checkpoint at episode {self.start_episode}.")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, max_episodes: int):
        self.maybe_resume()
        self.joint_max_episodes = int(max_episodes)
        inner_cv = make_inner_cv(self.y_train, seed=self.seed + self.trial_idx)
        train_start_time = time.time()

        episode = self.start_episode
        last_ppo_stats = {"updated": False, "n_steps": 0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        while self.joint_episode_count < max_episodes:
            if should_stop_for_walltime(train_start_time):
                self.logger.info("Walltime safety stop triggered. Saving checkpoint and exiting trial early.")
                self.save_training_checkpoint(max(episode - 1, 0))
                self.save_status(max(episode - 1, 0), completed=False, stop_reason="walltime_safety_stop")
                return False

            is_warmup = not self.warmup_completed
            phase_name = "warmup" if is_warmup else "joint"
            subset, log_probs, episode_stop_reason, episode_stop_trace, rollout_steps = self.sample_subset(
                greedy=False, warmup=is_warmup,
            )
            prev_baseline = self.ema_baseline
            final_stop_step = episode_stop_trace[-1] if episode_stop_trace else None
            (reward, inner_auc, inner_auc_warmup, sparsity_penalty, cap_penalty, late_length_penalty,
             total_stop_reward, step_cost_penalty, compact_bonus, applied_feature_signal, feature_bonus,
             pair_shape_score, pair_pos_ratio, pair_neg_ratio, pair_count, pair_term, penalty_scale,
             joint_progress, effective_penalty, active_stop_reward, synergy_bonus,
             ) = self.compute_reward(
                subset, inner_cv, stop_reason=episode_stop_reason, phase=phase_name,
                moving_baseline=prev_baseline, episode_idx=episode, final_stop_step=final_stop_step,
            )

            beta = 0.95 if phase_name == "warmup" else 0.90
            self.ema_baseline = beta * prev_baseline + (1.0 - beta) * reward
            advantage = reward - prev_baseline
            beta_var = 0.9
            self.ema_adv_sq = beta_var * self.ema_adv_sq + (1.0 - beta_var) * (advantage ** 2)
            adv_std = float(np.sqrt(self.ema_adv_sq) + 1e-6)
            norm_advantage = float(np.clip(advantage / adv_std, -2.0, 2.0))

            if is_warmup:
                if len(log_probs) == 0:
                    raise RuntimeError("No log_probs collected for the sampled trajectory.")
                loss = -norm_advantage * torch.stack(log_probs).sum()
                self.warmup_feature_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([self.model.feature_logits], max_norm=GRAD_CLIP_NORM)
                self.warmup_feature_optimizer.step()
            else:
                prepared_rollout = self._prepare_ppo_rollout_batch(reward=reward, rollout_steps=rollout_steps)
                self.ppo_rollout_buffer.extend(prepared_rollout)
                if prepared_rollout:
                    self.step_structure_bonus_history.append(
                        float(np.mean([float(x.get("step_structure_bonus", 0.0)) for x in prepared_rollout]))
                    )
                else:
                    self.step_structure_bonus_history.append(0.0)
                if self.joint_episode_count > 0 and self.joint_episode_count % int(PPO_UPDATE_EVERY) == 0:
                    last_ppo_stats = self.ppo_update(self.ppo_rollout_buffer)
                    last_ppo_stats.update({"episode": int(episode), "joint_episode_count": int(self.joint_episode_count)})
                    self.ppo_update_history.append(dict(last_ppo_stats))
                    self.ppo_rollout_buffer = []

            self.update_high_reward_feature_credit(subset=subset, reward=reward, moving_baseline=prev_baseline)
            high_reward_feature_boost = self.apply_high_reward_feature_boost(subset=subset, phase=phase_name)

            self.reward_history.append(float(reward))
            feat_temp = float(self.get_joint_feature_temperature())
            stop_temp = float(self.get_joint_stop_temperature())
            base_barrier = float(self.get_joint_stop_barrier(len(subset)))
            applied_barrier = float(BARRIER_SCALE) * base_barrier
            if int(len(subset)) < 8:
                applied_barrier += 0.5 * (8 - float(len(subset))) / 8.0
            stop_coef, feat_coef = self.get_entropy_coefs()

            self.joint_feature_temperature_history.append(feat_temp)
            self.joint_stop_temperature_history.append(stop_temp)
            self.joint_stop_barrier_history.append(base_barrier)
            self.joint_stop_applied_barrier_history.append(applied_barrier)
            self.ppo_feat_entropy_coef_history.append(float(feat_coef))
            self.ppo_stop_entropy_coef_history.append(float(stop_coef))
            self.inner_auc_history.append(float(inner_auc))
            self.inner_auc_warmup_history.append(float(inner_auc_warmup))
            self.selected_size_history.append(int(len(subset)))
            self.penalty_scale_history.append(float(penalty_scale))
            self.joint_progress_history.append(float(joint_progress))
            self.effective_penalty_history.append(float(effective_penalty))
            self.active_stop_reward_history.append(float(active_stop_reward))
            self.synergy_bonus_history.append(float(synergy_bonus))
            self.stop_reason_history.append(str(episode_stop_reason))
            self.policy_stop_history.append(1 if episode_stop_reason == "policy_stop" else 0)
            self.policy_stop_ratio_history.append(float(np.mean(self.policy_stop_history)))
            self.sparsity_penalty_history.append(float(sparsity_penalty))
            self.cap_penalty_history.append(float(cap_penalty))
            self.late_length_penalty_history.append(float(late_length_penalty))
            self.stop_bonus_history.append(float(total_stop_reward))
            self.compact_bonus_history.append(float(compact_bonus))
            self.feature_signal_history.append(float(applied_feature_signal))
            self.feature_bonus_history.append(float(feature_bonus))
            self.coarse_pair_term_history.append(float(pair_term))
            self.shaping_score_history.append(float(pair_shape_score))
            self.pair_pos_ratio_history.append(float(pair_pos_ratio))
            self.pair_neg_ratio_history.append(float(pair_neg_ratio))
            self.pair_count_history.append(int(pair_count))
            self.pair_term_history.append(float(pair_term))
            if is_warmup:
                self.step_structure_bonus_history.append(0.0)
            self.high_reward_feature_boost_history.append(float(high_reward_feature_boost))
            self.step_cost_history.append(float(step_cost_penalty))

            with torch.no_grad():
                feat_logits = self.model.feature_logits.detach().cpu().numpy()
                continue_logit = float(self.model.continue_logit.detach().cpu().item())
                stop_logit = float(self.model.stop_logit.detach().cpu().item())
                stop_count_scale = float(self.model.stop_count_scale.detach().cpu().item())
                stop_quality_scale = float(self.model.stop_quality_scale.detach().cpu().item())
                raw_stop_gain_scale = float(self.model.raw_stop_gain_scale.detach().cpu().item())
                stop_gain_scale = float(torch.nn.functional.softplus(self.model.raw_stop_gain_scale).detach().cpu().item())
                stop_advantage_scale = float(self.model.stop_advantage_scale.detach().cpu().item())

            self.update_logit_summary_stats(feat_logits)
            self.maybe_store_feature_snapshot(episode, feat_logits)
            self.top50_overlap_recent_history.append(self.compute_recent_rank_overlap(top_k=50, recent_n=3))
            self.top100_overlap_recent_history.append(self.compute_recent_rank_overlap(top_k=100, recent_n=3))
            self.logits_std_history.append(float(feat_logits.std()))
            self.continue_logit_history.append(continue_logit)
            self.stop_logit_history.append(stop_logit)
            self.stop_count_scale_history.append(stop_count_scale)
            self.stop_quality_scale_history.append(stop_quality_scale)
            self.raw_stop_gain_scale_history.append(raw_stop_gain_scale)
            self.stop_gain_scale_history.append(stop_gain_scale)
            self.stop_advantage_scale_history.append(stop_advantage_scale)
            self.phase_history.append(phase_name)

            if not is_warmup:
                self.joint_episode_count += 1
                self.recent_joint_subsets.append(sanitize_subset(subset, self.n_features))
                if len(self.recent_joint_subsets) > int(JOINT_REWARD_PLATEAU_WINDOW):
                    self.recent_joint_subsets = self.recent_joint_subsets[-int(JOINT_REWARD_PLATEAU_WINDOW):]
                self.joint_stop_step_records.append({
                    "episode": int(episode + 1), "phase": "joint",
                    "stop_reason": str(episode_stop_reason),
                    "final_subset_size": int(len(subset)),
                    "trace": episode_stop_trace,
                })
                if episode_stop_trace:
                    last_step = episode_stop_trace[-1]
                    self.final_stop_probability_history.append(float(last_step["p_stop"]))
                    self.final_stop_selected_count_history.append(int(last_step["selected_count_before_decision"]))
                    stop_steps = [x for x in episode_stop_trace if x.get("decision") == "stop"]
                    if stop_steps:
                        s = stop_steps[-1]
                        self.logger.info(
                            "[STOP_TRACE] "
                            f"ep={episode} | "
                            f"k_before={s.get('selected_count_before_decision', -1)} | "
                            f"p_stop={s.get('p_stop', 0.0):.6f} | "
                            f"sel_q75_mean={s.get('selected_q75_logit', 0.0):.4f} | "
                            f"remain_q75_mean={s.get('remaining_q75_logit', 0.0):.4f} | "
                            f"gain_gap={s.get('gain_gap', 0.0):.4f} | "
                            f"coverage={s.get('network_coverage_ratio', 0.0):.4f} | "
                            f"cross_ratio={s.get('cross_network_ratio', 0.0):.4f}"
                        )

            self.moving_avg_history = moving_average(self.reward_history, window=MOVING_AVG_WINDOW)

            if is_warmup:
                should_switch, switch_reason, transition_metrics = self.should_end_warmup()
                transition_metrics.update({"episode": int(episode), "phase": "warmup"})
                self.warmup_transition_history.append(transition_metrics)
                if should_switch:
                    self.warmup_completed = True
                    self.warmup_end_episode = int(episode)
                    self.warmup_end_reason = str(switch_reason)
                    self.apply_stage_prior(JOINT_PRIOR_STAGE, rebase_feature_logits=True)
                    self.logger.info(
                        f"Adaptive warmup finished at episode={episode} | reason={switch_reason} | "
                        f"max_median_gap={transition_metrics['max_median_gap']:.4f} | "
                        f"top25_top50_gap={transition_metrics['top25_top50_gap']:.4f} | "
                        f"above_global_median_frac={transition_metrics['last50_above_global_median_fraction']:.4f} | "
                        f"top50_overlap_recent={transition_metrics['top50_overlap_recent']:.4f} | "
                        f"prior_stage={self.current_prior_stage}"
                    )

            if not is_warmup:
                current_score = (float(inner_auc), float(reward), -float(len(subset)))
                self.update_top_episode_candidates(episode=episode, subset=subset, inner_auc=inner_auc, reward=reward, stop_reason=episode_stop_reason)
                self.update_joint_refit_episode_history(episode=episode, subset=subset, inner_auc=inner_auc, reward=reward, stop_reason=episode_stop_reason)
                if current_score > self.best_selection_score:
                    self.best_selection_score = current_score
                    self.best_reward = float(reward)
                    self.best_inner_auc = float(inner_auc)
                    self.best_subset = subset.tolist()
                    self.best_episode = episode
                    self.best_stop_reason = str(episode_stop_reason)

                if self.joint_episode_count > 0 and self.joint_episode_count % int(JOINT_CHECKPOINT_EVAL_EVERY) == 0:
                    checkpoint_eval = self.evaluate_policy_checkpoint(episode=episode, inner_cv=inner_cv)
                    self.logger.info(
                        f"joint_checkpoint | episode={episode} | chosen_method={self.refit_subset_source} | "
                        f"greedy_overlap_prev={checkpoint_eval.get('greedy_overlap_with_previous', None)} | "
                        f"sampled_overlap={checkpoint_eval.get('sampled_overlap_mean', 0.0):.4f} | "
                        f"chosen_refit_auc={checkpoint_eval['refit_evaluations'][self.refit_subset_source]['refit_cv_auc']:.4f}"
                    )

            if episode % LOG_EVERY == 0:
                msg = (
                    f"Episode={episode} | phase={phase_name} | reward={reward:.4f} | inner_auc={inner_auc:.4f} | "
                    f"joint_stop_temperature={float(self.get_joint_stop_temperature())} | "
                    f"joint_feature_temperature={float(self.get_joint_feature_temperature())} | "
                    f"joint_stop_barrier={float(self.get_joint_stop_barrier(len(subset)))} | "
                    f"k={len(subset)} | stop_reason={episode_stop_reason} | best_reward={self.best_reward:.4f} | "
                    f"policy_stop_ratio={self.policy_stop_ratio_history[-1]:.4f} | logits_std={self.logits_std_history[-1]:.4f} | "
                    f"step_struct={self.step_structure_bonus_history[-1]:.4f} | "
                    f"pair_score={pair_shape_score:.4f} | pair_pos={pair_pos_ratio:.3f} | pair_neg={pair_neg_ratio:.3f} | "
                    f"pair_cnt={pair_count} | pair_term={pair_term} | "
                    f"penalty_scale={penalty_scale:.3f} | joint_progress={joint_progress:.3f} | "
                    f"effective_penalty={effective_penalty:.3f} | active_stop_reward={active_stop_reward:.3f} | "
                    f"synergy_bonus={synergy_bonus:.3f}"
                )
                if not is_warmup:
                    top50_top100_gap = self.top50_minus_top100_gap_history[-1] if self.top50_minus_top100_gap_history else 0.0
                    max_median_gap = self.max_median_gap_history[-1] if self.max_median_gap_history else 0.0
                    top50_overlap_recent = self.top50_overlap_recent_history[-1] if self.top50_overlap_recent_history else 0.0
                    top100_overlap_recent = self.top100_overlap_recent_history[-1] if self.top100_overlap_recent_history else 0.0
                    msg += (
                        f" | top50_top100_gap={top50_top100_gap:.4f}"
                        f" | max_median_gap={max_median_gap:.4f}"
                        f" | top50_overlap_recent={top50_overlap_recent:.4f}"
                        f" | top100_overlap_recent={top100_overlap_recent:.4f}"
                        f" | ppo_policy_loss={last_ppo_stats.get('policy_loss', 0.0):.4f}"
                        f" | ppo_value_loss={last_ppo_stats.get('value_loss', 0.0):.4f}"
                        f" | ppo_entropy={last_ppo_stats.get('entropy', 0.0):.4f}"
                    )
                self.logger.info(msg)
                if not is_warmup:
                    _, _, joint_metrics = self.should_enter_refit_from_joint()
                    self.logger.info(
                        f"joint_refit_transition | ready={joint_metrics['joint_refit_ready']} | "
                        f"reason={joint_metrics['joint_refit_reason']} | reward_delta={joint_metrics['reward_delta']:.4f} | "
                        f"subset_size_std={joint_metrics['recent_subset_size_std']:.4f} | "
                        f"greedy_overlap={joint_metrics['recent_greedy_subset_overlap']:.4f} | "
                        f"sampled_overlap={joint_metrics['recent_sampled_subset_overlap']:.4f} | "
                        f"checkpoint_refit_auc_mean={joint_metrics['checkpoint_refit_auc_mean']:.4f} | "
                        f"checkpoint_refit_auc_std={joint_metrics['checkpoint_refit_auc_std']:.4f}"
                    )

            if episode % CHECKPOINT_EVERY == 0:
                self.save_training_checkpoint(episode)
                self.save_status(episode, completed=False, stop_reason="running")

            if not is_warmup:
                should_refit, refit_reason, joint_metrics = self.should_enter_refit_from_joint()
                joint_metrics.update({"episode": int(episode), "phase": "joint"})
                self.joint_refit_readiness_history.append(joint_metrics)
                if should_refit:
                    self.joint_refit_episode = int(episode)
                    self.joint_refit_reason = str(refit_reason)
                    self.save_training_checkpoint(episode)
                    self.save_status(episode, completed=False, stop_reason="joint_ready_for_refit")
                    break

            episode += 1

        final_subset = self.get_refit_subset()
        (final_reward, final_inner_auc, *_) = self.compute_reward(
            final_subset, inner_cv, stop_reason="refit_eval", phase="joint",
            moving_baseline=0.0, episode_idx=episode, final_stop_step=None,
        )
        self.final_subset = final_subset.tolist()
        self.final_reward = float(final_reward)
        self.final_inner_auc = float(final_inner_auc)
        self.final_policy_logits = {
            "feature_logits": self.model.feature_logits.detach().cpu().numpy().tolist(),
            "continue_logit": float(self.model.continue_logit.detach().cpu().item()),
            "stop_logit": float(self.model.stop_logit.detach().cpu().item()),
            "stop_count_scale": float(self.model.stop_count_scale.detach().cpu().item()),
            "stop_quality_scale": float(self.model.stop_quality_scale.detach().cpu().item()),
            "raw_stop_gain_scale": float(self.model.raw_stop_gain_scale.detach().cpu().item()),
            "stop_advantage_scale": float(self.model.stop_advantage_scale.detach().cpu().item()),
            "feature_delta_scale": float(self.model.feature_delta_scale),
        }
        final_episode_idx = max(episode - 1, 0)
        self.save_training_checkpoint(final_episode_idx)
        self.save_status(final_episode_idx, completed=True, stop_reason="trial_finished")
        return True
