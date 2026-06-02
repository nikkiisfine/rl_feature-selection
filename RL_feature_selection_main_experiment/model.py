import numpy as np
import torch
import torch.nn as nn

from .config import (
    HYBRID_FEATURE_DELTA_SCALE,
    INIT_CONTINUE_LOGIT,
    INIT_STOP_COUNT_SCALE,
    INIT_STOP_LOGIT,
    MIN_FEATURES_BEFORE_STOP,
    STOP_REMAINING_TOP_M,
    STOP_STATE_DIM,
)
from .utils import parse_edge_networks


class SequentialPolicy(nn.Module):
    """
    Sequential feature-selection policy with a joint stop/continue + feature head.

    At each step the policy outputs:
      - continue_stop_logits: 2-vector [continue, stop]
      - feature_logits:       masked logit per feature (already-selected → -inf)
      - state_value:          scalar baseline for PPO
      - state_features:       the state encoding fed to all heads
    """

    def __init__(
        self,
        n_features: int,
        prior_logits: np.ndarray,
        k_stop_ref: int,
        init_stop_logit: float = INIT_STOP_LOGIT,
        init_stop_count_scale: float = INIT_STOP_COUNT_SCALE,
        init_continue_logit: float = INIT_CONTINUE_LOGIT,
        feature_names=None,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.k_stop_ref = float(k_stop_ref)
        self.feature_delta_scale = float(HYBRID_FEATURE_DELTA_SCALE)
        self.stop_coverage_scale = 0.4
        self.stop_cross_scale = 0.2

        prior_tensor = torch.tensor(prior_logits, dtype=torch.float32)
        self.feature_logits = nn.Parameter(prior_tensor.clone())
        self.register_buffer("prior_logits_buffer", prior_tensor.clone())

        self.continue_logit = nn.Parameter(torch.tensor(init_continue_logit, dtype=torch.float32))
        self.stop_logit = nn.Parameter(torch.tensor(init_stop_logit, dtype=torch.float32))
        self.stop_count_scale = nn.Parameter(torch.tensor(init_stop_count_scale, dtype=torch.float32))
        self.stop_quality_scale = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.raw_stop_gain_scale = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.stop_advantage_scale = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

        feature_names = (
            list(feature_names) if feature_names is not None
            else [f"feature_{i}" for i in range(self.n_features)]
        )
        self.feature_name_list = [str(x) for x in feature_names]

        network_vocab = {"UNKNOWN": 0}
        left_ids, right_ids = [], []
        for fname in self.feature_name_list:
            left_net, right_net = parse_edge_networks(str(fname))
            if left_net not in network_vocab:
                network_vocab[left_net] = len(network_vocab)
            if right_net not in network_vocab:
                network_vocab[right_net] = len(network_vocab)
            left_ids.append(network_vocab[left_net])
            right_ids.append(network_vocab[right_net])
        self.network_vocab = dict(network_vocab)
        self.total_known_networks = max(1, len(network_vocab) - 1)
        self.register_buffer("edge_left_network_ids", torch.tensor(left_ids, dtype=torch.long))
        self.register_buffer("edge_right_network_ids", torch.tensor(right_ids, dtype=torch.long))

        self.stage1_state_encoder = nn.Sequential(
            nn.Linear(STOP_STATE_DIM, 16),
            nn.Tanh(),
            nn.Linear(16, 2),
        )
        self.feature_state_encoder = nn.Sequential(
            nn.Linear(STOP_STATE_DIM, 32),
            nn.Tanh(),
            nn.Linear(32, self.n_features),
        )
        self.value_head = nn.Sequential(
            nn.Linear(STOP_STATE_DIM, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_effective_feature_logits_from_state(self, state_features: torch.Tensor) -> torch.Tensor:
        feature_delta_logits = self.feature_state_encoder(state_features)
        return self.feature_logits + self.feature_delta_scale * feature_delta_logits

    def _compute_network_state_stats(self, state_mask: torch.Tensor):
        device = self.feature_logits.device
        if not state_mask.any():
            return (
                torch.tensor(0.0, dtype=torch.float32, device=device),
                torch.tensor(0.0, dtype=torch.float32, device=device),
            )
        left_ids = self.edge_left_network_ids[state_mask]
        right_ids = self.edge_right_network_ids[state_mask]
        all_ids = torch.cat([left_ids, right_ids], dim=0)
        known_ids = all_ids[all_ids > 0]
        if known_ids.numel() == 0:
            coverage_ratio = torch.tensor(0.0, dtype=torch.float32, device=device)
        else:
            coverage_ratio = torch.unique(known_ids).numel() / float(max(self.total_known_networks, 1))
            coverage_ratio = torch.tensor(float(coverage_ratio), dtype=torch.float32, device=device)
        cross_edges = ((left_ids != right_ids) & (left_ids > 0) & (right_ids > 0)).float()
        cross_ratio = cross_edges.mean() if cross_edges.numel() > 0 else torch.tensor(0.0, dtype=torch.float32, device=device)
        return coverage_ratio, cross_ratio

    def _robust_q75(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=self.feature_logits.device)
        return torch.quantile(x.float(), 0.75)

    def _stable_selected_summary(self, x) -> torch.Tensor:
        device = self.feature_logits.device
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
        elif not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        else:
            x = x.to(device=device, dtype=torch.float32)
        if x.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=device)
        n = int(x.numel())
        if n <= 5:
            return x.mean()
        if n <= 8:
            return 0.5 * x.mean() + 0.5 * torch.median(x)
        return torch.quantile(x, 0.75)

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def build_state_features(
        self,
        selected_mask: torch.Tensor,
        reward_trend_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state_mask = selected_mask.detach().clone().bool()
        device = self.feature_logits.device
        selected_count = state_mask.float().sum()
        selected_ratio = selected_count / max(self.k_stop_ref, 1.0)
        coverage_ratio, cross_ratio = self._compute_network_state_stats(state_mask)

        if reward_trend_state is None:
            prev_reward_advantage = torch.tensor(0.0, dtype=torch.float32, device=device)
        else:
            reward_trend_state = reward_trend_state.to(device=device, dtype=torch.float32)
            if reward_trend_state.numel() < 2:
                reward_trend_state = torch.cat([
                    reward_trend_state,
                    torch.zeros(2 - reward_trend_state.numel(), dtype=torch.float32, device=device),
                ])
            prev_reward_advantage = reward_trend_state[1]

        if state_mask.any():
            raw_selected_logits = self.feature_logits[state_mask]
            raw_selected_q75 = self._stable_selected_summary(raw_selected_logits)
        else:
            raw_selected_q75 = torch.tensor(0.0, dtype=torch.float32, device=device)

        remaining_mask = ~state_mask
        if remaining_mask.any():
            raw_remaining_logits = self.feature_logits[remaining_mask]
            raw_topm = min(STOP_REMAINING_TOP_M, int(raw_remaining_logits.numel()))
            raw_remaining_topm_logits = torch.topk(raw_remaining_logits, k=raw_topm).values
            raw_remaining_q75 = torch.quantile(raw_remaining_topm_logits.float(), 0.75)
        else:
            raw_remaining_q75 = torch.tensor(0.0, dtype=torch.float32, device=device)

        selected_q75_logit = raw_selected_q75
        remaining_q75_logit = raw_remaining_q75
        raw_gain_gap = remaining_q75_logit - selected_q75_logit
        gain_gap = torch.clamp(raw_gain_gap, min=-4.0, max=4.0)

        state_features = torch.stack([
            selected_ratio,
            selected_q75_logit,
            gain_gap,
            prev_reward_advantage,
            coverage_ratio,
            cross_ratio,
        ])
        return state_features

    # ------------------------------------------------------------------
    # Stop / continue term breakdown (diagnostic + forward)
    # ------------------------------------------------------------------

    def _compute_stop_term_breakdown(
        self,
        selected_mask: torch.Tensor,
        reward_trend_state: torch.Tensor | None = None,
    ) -> dict:
        state_mask = selected_mask.detach().clone().bool()
        state_features = self.build_state_features(state_mask, reward_trend_state)

        selected_count = state_mask.float().sum()
        progress_numer = torch.clamp(selected_count - float(MIN_FEATURES_BEFORE_STOP), min=0.0)
        progress_denom = max(float(self.k_stop_ref) - float(MIN_FEATURES_BEFORE_STOP), 1.0)

        prev_reward_advantage = state_features[3]
        coverage_ratio = state_features[4]
        cross_ratio = state_features[5]
        gain_gap = state_features[2]

        progress_center = float(MIN_FEATURES_BEFORE_STOP) + 1.0
        progress_tau = 1.5
        progress_signal = torch.sigmoid((selected_count - progress_center) / progress_tau)
        stop_progress_term = 0.2 * self.stop_count_scale * progress_signal

        gain_center = -0.2
        gain_tau = 0.4
        gain_shifted = gain_gap - gain_center
        stop_gain_signal = torch.tanh(-gain_shifted / gain_tau)
        extra_neg_gain = torch.relu(-gain_shifted - 0.3)
        stop_gain_scale = torch.nn.functional.softplus(self.raw_stop_gain_scale)
        pos_part = torch.relu(stop_gain_signal)
        neg_part = torch.relu(-stop_gain_signal)
        stop_gain_term = stop_gain_scale * (
            0.22 * pos_part
            + 0.10 * extra_neg_gain
            - 0.05 * neg_part
        )

        stop_plateau_signal = torch.exp(-(gain_gap / 0.5) ** 2)
        stop_plateau_term = self.stop_quality_scale * (0.01 * stop_plateau_signal)

        stop_coverage_term = 0.05 * self.stop_coverage_scale * coverage_ratio
        stop_cross_term = 0.05 * self.stop_cross_scale * cross_ratio

        continue_trend_signal = torch.clamp(prev_reward_advantage, min=0.0, max=1.0)
        continue_trend_term = 0.15 * self.stop_advantage_scale * continue_trend_signal

        early_stop_penalty = 0.0

        stop_logit_total = (
            self.stop_logit
            + stop_progress_term
            + stop_plateau_term
            + stop_coverage_term
            + stop_cross_term
            + stop_gain_term
            - continue_trend_term
            - early_stop_penalty
        )
        continue_logit_total = self.continue_logit

        return {
            "state_features": state_features,
            "raw_continue_logit_total": continue_logit_total,
            "raw_stop_logit_total": stop_logit_total,
            "base_continue_logit": self.continue_logit,
            "base_stop_logit": self.stop_logit,
            "stop_progress_term": stop_progress_term,
            "stop_gain_term": stop_gain_term,
            "stop_plateau_term": stop_plateau_term,
            "stop_coverage_term": stop_coverage_term,
            "stop_cross_term": stop_cross_term,
            "continue_trend_term": continue_trend_term,
            "early_stop_penalty": early_stop_penalty,
            "stop_gain_signal": stop_gain_signal,
            "stop_plateau_signal": stop_plateau_signal,
            "continue_trend_signal": continue_trend_signal,
            "gain_gap": gain_gap,
            "selected_count": selected_count,
            "extra_neg_gain": extra_neg_gain,
            "raw_stop_gain_scale": self.raw_stop_gain_scale,
            "stop_gain_scale": stop_gain_scale,
        }

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        selected_mask: torch.Tensor,
        reward_trend_state: torch.Tensor | None = None,
    ):
        debug = self._compute_stop_term_breakdown(selected_mask, reward_trend_state)
        state_features = debug["state_features"]

        continue_stop_logits = torch.stack([
            debug["raw_continue_logit_total"],
            debug["raw_stop_logit_total"],
        ]) + 0.0 * self.stage1_state_encoder(state_features)

        feature_logits = self._compute_effective_feature_logits_from_state(state_features)
        masked_feature_logits = feature_logits.masked_fill(selected_mask, -1e9)

        state_value = self.value_head(state_features).squeeze(-1)
        return continue_stop_logits, masked_feature_logits, state_value, state_features
