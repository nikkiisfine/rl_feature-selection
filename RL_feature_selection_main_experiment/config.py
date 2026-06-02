from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
# For HPC cluster runs, replace PROJECT_ROOT with your cluster path, e.g.:
# PROJECT_ROOT = Path("/home/daxiaoniqiu/AD_main_experiment")
PROJECT_ROOT = Path("/Users/daxiaoniqiu/anaconda3/AI4H/pythonProject/myvenv/thesis/thesis/AD_main_experiment")

DATA_DIR = PROJECT_ROOT / "connectivity_dataset_out"
X_PATH = DATA_DIR / "X.npy"
Y_PATH = DATA_DIR / "y.npy"
FEATURE_NAMES_PATH = DATA_DIR / "feature_names.npy"

OUTER_FOLDS_PATH = PROJECT_ROOT / "experiments" / "outer_folds.json"

RESULTS_ROOT = PROJECT_ROOT / "results_2" / "rl_main"
LOG_ROOT = PROJECT_ROOT / "logs_2" / "rl_main"

# =============================================================================
# Cross-validation settings
# =============================================================================
OUTER_N_SPLITS = 5
INNER_N_SPLITS = 3

# =============================================================================
# RL array / seed mapping (for SLURM array jobs)
# =============================================================================
RL_SEEDS = [2]
N_FOLDS_FOR_ARRAY = OUTER_N_SPLITS

# =============================================================================
# Hyperparameter search space for nested CV
# =============================================================================
SEARCH_SPACE = {
    "k_stop": [20],
    "lambda_reg": [0.06],
    "learning_rate": [6e-3],
    "refit_subset_source": [
        "greedy_rollout_subset",
    ],
}

PROFILE_SEARCH_SPACE = {
    "k_stop": [10, 20],
    "lambda_reg": [0.001, 0.01],
    "learning_rate": [1e-2],
    "prior_beta": [0.0, 0.5],
    "refit_subset_source": [
        "greedy_rollout_subset",
        "multiple_rollout_consensus_subset",
        "hybrid_score",
    ],
}

# =============================================================================
# Network / subnetwork pair priors (AD-specific domain knowledge)
# =============================================================================
BASE_PRIOR_WARMUP = 0.005
BASE_PRIOR_JOINT = 0.005

NETWORK_PAIR_PRIOR_WARMUP = {
    ("Default", "Limbic"): 0.040,
    ("Default", "Default"): 0.035,
    ("Default", "Cont"): 0.035,
    ("Default", "SalVentAttn"): 0.025,
    ("Default", "DorsAttn"): 0.025,
    ("Limbic", "Cont"): 0.030,
    ("Limbic", "SalVentAttn"): 0.020,
    ("Cont", "SalVentAttn"): 0.020,
    ("Cont", "DorsAttn"): 0.020,
    ("Vis", "SomMot"): 0.010,
    ("Default", "Vis"): 0.010,
    ("Default", "SomMot"): 0.010,
    ("Cont", "Vis"): 0.010,
    ("Cont", "SomMot"): 0.010,
    ("DorsAttn", "Vis"): 0.010,
    ("DorsAttn", "SomMot"): 0.010,
}

NETWORK_PAIR_PRIOR_JOINT = {
    ("Default", "Limbic"): 0.070,
    ("Default", "Default"): 0.060,
    ("Default", "Cont"): 0.055,
    ("Default", "SalVentAttn"): 0.040,
    ("Default", "DorsAttn"): 0.040,
    ("Limbic", "Cont"): 0.040,
    ("Limbic", "SalVentAttn"): 0.030,
    ("Cont", "SalVentAttn"): 0.030,
    ("Cont", "DorsAttn"): 0.030,
    ("Vis", "SomMot"): 0.010,
    ("Default", "Vis"): 0.010,
    ("Default", "SomMot"): 0.010,
    ("Cont", "Vis"): 0.010,
    ("Cont", "SomMot"): 0.010,
    ("DorsAttn", "Vis"): 0.010,
    ("DorsAttn", "SomMot"): 0.010,
}

SUBNETWORK_PAIR_PRIOR_WARMUP = {
    ("Default_pCunPCC", "Limbic_TempPole"): 0.10,
    ("Default_Temp", "Limbic_TempPole"): 0.08,
    ("Default_pCunPCC", "Default_PFC"): 0.08,
    ("Default_Temp", "Default_PFC"): 0.05,
    ("Default_pCunPCC", "Cont_PFC"): 0.05,
    ("Limbic_OFC", "Cont_PFC"): 0.05,
    ("Default_PFC", "DorsAttn_Post"): 0.03,
    ("Default_Temp", "DorsAttn_Post"): 0.03,
    ("DorsAttn_Post", "Limbic_TempPole"): 0.03,
    ("Cont_PFCl", "Default_PFC"): 0.03,
}

SUBNETWORK_PAIR_PRIOR_JOINT = {
    ("Default_pCunPCC", "Limbic_TempPole"): 0.13,
    ("Default_Temp", "Limbic_TempPole"): 0.13,
    ("Default_pCunPCC", "Default_PFC"): 0.13,
    ("Default_Temp", "Default_PFC"): 0.09,
    ("Default_pCunPCC", "Cont_PFC"): 0.09,
    ("Limbic_OFC", "Cont_PFC"): 0.09,
    ("Default_PFC", "DorsAttn_Post"): 0.05,
    ("Default_Temp", "DorsAttn_Post"): 0.05,
    ("DorsAttn_Post", "Limbic_TempPole"): 0.05,
    ("Cont_PFCl", "Default_PFC"): 0.05,
}

NETWORK_PAIR_SHAPING_COEF = 0.02

AD_POSITIVE_PAIRS = {
    ("Default", "Limbic"),
    ("Default", "Default"),
    ("Default", "Cont"),
    ("Default", "SalVentAttn"),
    ("Default", "DorsAttn"),
    ("Limbic", "Cont"),
    ("Limbic", "SalVentAttn"),
    ("Cont", "SalVentAttn"),
    ("Cont", "DorsAttn"),
}

NON_TARGET_NETWORKS = {"Vis", "SomMot"}
NETWORK_PAIR_NEG_PENALTY = 0.0

# Network keyword weights (used in prior construction)
NETWORK_KEYWORDS = {
    "Default": 2.0,
    "Cont": 1.5,
    "SalVentAttn": 1.5,
    "DorsAttn": 1.2,
    "Limbic": 1.0,
    "SomMot": 0.8,
    "Vis": 0.6,
}
UNKNOWN_NETWORK_WEIGHT = 0.0

# =============================================================================
# Training settings
# =============================================================================
MAX_EPISODES = 300
MIN_EPISODES_BEFORE_STOP = 100
MOVING_AVG_WINDOW = 50
PLATEAU_PATIENCE = 40
PLATEAU_STD_TOL = 1e-4

# Sparsity / length penalties
SPARSITY_LINEAR_WEIGHT = 0.5
CAP_BASE_PENALTY = 0.03
CAP_LATE_THRESHOLD = 0.6
CAP_LATE_SCALE = 0.05
LATE_LENGTH_PENALTY_SCALE = 0.1
MIN_FEATURES_BEFORE_STOP = 15
STEP_COST = 0.004

# Policy initialization
INIT_STOP_LOGIT = -1.0
INIT_STOP_COUNT_SCALE = 0.12
SOFTMAX_TEMPERATURE = 1.3

# Warmup phase sampling range
WARMUP_MIN_STEPS = 15
WARMUP_MAX_STEPS = 35
MAX_SELECTION_STEPS_BUFFER = 0

# Gradient update
GRAD_CLIP_NORM = 5.0

# =============================================================================
# Downstream classifier settings
# =============================================================================
CLASSIFIER_MAX_ITER = 500
CLASS_WEIGHT = "balanced"

# =============================================================================
# Reproducibility
# =============================================================================
GLOBAL_SEED = 42

# =============================================================================
# Checkpoint / logging cadence
# =============================================================================
CHECKPOINT_EVERY = 10
LOG_EVERY = 10

# =============================================================================
# Walltime safety (hours)
# =============================================================================
WALLTIME_LIMIT_HOURS = 10
SAFETY_STOP_HOURS = 9.0

# =============================================================================
# Output file names
# =============================================================================
RESULT_FILENAME = "final_result.json"
STATUS_FILENAME = "status.json"
CHECKPOINT_FILENAME = "checkpoint.pt"
TRIALS_FILENAME = "trial_summaries.json"

# =============================================================================
# Nested-CV model selection
# =============================================================================
PARAM_SELECTION_METRIC = "mean_val_auc"

# =============================================================================
# Hybrid PPO joint-stage settings
# =============================================================================
HYBRID_PPO_ENABLED = True
HYBRID_FEATURE_DELTA_SCALE = 2
PPO_UPDATE_EVERY = 8
PPO_EPOCHS = 2
PPO_CLIP_EPS = 0.2
PPO_VALUE_COEF = 0.5
PPO_MAX_GRAD_NORM = 5.0
PPO_MINI_BATCH_SIZE = 3
PPO_LENGTH_NORMALIZE_ADVANTAGE = True

# =============================================================================
# Joint -> refit readiness thresholds
# =============================================================================
JOINT_REWARD_PLATEAU_WINDOW = 30
JOINT_REWARD_PLATEAU_DELTA = 0.005
JOINT_LENGTH_STD_MAX = 4.0
JOINT_GREEDY_OVERLAP_THRESHOLD = 0.65
JOINT_SAMPLED_OVERLAP_THRESHOLD = 0.40
JOINT_CHECKPOINT_EVAL_EVERY = 20
JOINT_CHECKPOINT_STABILITY_WINDOW = 3
JOINT_CHECKPOINT_REFIT_AUC_DELTA_MAX = 0.05
JOINT_CHECKPOINT_SIZE_STD_MAX = 4.0
JOINT_READY_MIN_CHECKPOINTS = 3
PPO_CHECKPOINT_SAMPLED_ROLLOUTS = 5
PPO_CONSENSUS_ROLLOUTS = 5
HYBRID_SCORE_ALPHA = 0.6
REFIT_SUBSET_SOURCE = "multiple_rollout_consensus_subset"

# =============================================================================
# Phase-level learning rates (defined here for visibility alongside PPO params)
# =============================================================================
PHASE2_FEATURE_LR = 1.5e-3
PHASE2_STOP_LR = 1.5e-3
PPO_LR = 1e-3

# =============================================================================
# Adaptive warmup thresholds
# =============================================================================
ADAPTIVE_WARMUP_MIN_EPISODES = 200

WARMUP_REWARD_WINDOW = 50
WARMUP_PLATEAU_WINDOW = 50
WARMUP_NO_IMPROVEMENT_WINDOW = 50

# Strong-ready criteria (primary entry)
WARMUP_MAX_MEDIAN_GAP_THRESHOLD = 0.45
WARMUP_TOP25_TOP50_GAP_THRESHOLD = 0.02
WARMUP_ABOVE_MEDIAN_FRACTION_THRESHOLD = 0.5
WARMUP_TOP50_OVERLAP_THRESHOLD = 0.63

# Fallback criteria (secondary entry)
WARMUP_NO_IMPROVEMENT_DELTA_THRESHOLD = 0.015
WARMUP_FALLBACK_MAX_MEDIAN_GAP_THRESHOLD = 0.4
WARMUP_FALLBACK_TOP25_TOP50_GAP_THRESHOLD = 0.015
WARMUP_FALLBACK_TOP50_OVERLAP_THRESHOLD = 0.60

# =============================================================================
# Reward shaping scales
# =============================================================================
WARMUP_FEATURE_BONUS_SCALE = 0.05
JOINT_FEATURE_BONUS_SCALE = 0.015

HIGH_REWARD_FEATURE_CREDIT_EMA = 0.90
HIGH_REWARD_FEATURE_CREDIT_WARMUP_SCALE = 0.02
HIGH_REWARD_FEATURE_CREDIT_JOINT_SCALE = 0.04

HIGH_REWARD_STABLE_TOP_K = 10
HIGH_REWARD_RECENT_WINDOW = 15
HIGH_REWARD_STABLE_MIN_OCCURRENCES = 3

STOP_REMAINING_TOP_M = 5

WARMUP_COARSE_PAIR_SHAPING_SCALE = NETWORK_PAIR_SHAPING_COEF * 0.35
JOINT_POSITIVE_SHAPING_NEG_COEF = 0.0
PPO_STEP_STRUCTURE_BONUS_SCALE = 0.4
PPO_STEP_STRUCTURE_SUBNETWORK_MIX = 0.7

TOP_EPISODES_FOR_REFIT = 5
TOP_EPISODE_REEVAL_REPEATS = 5

# =============================================================================
# Joint-phase temperature / barrier schedules
# =============================================================================
WARMUP_SOFTMAX_TEMPERATURE = float(SOFTMAX_TEMPERATURE)
JOINT_SOFTMAX_TEMPERATURE = 0.25
JOINT_PRIOR_DELTA_SCALE = 0
STOP_STAGE1_DELTA_SCALE = 0.0
BARRIER_SCALE = 0.1

JOINT_FEATURE_TEMP_EARLY = 0.52
JOINT_FEATURE_TEMP_MID = 0.55
JOINT_FEATURE_TEMP_LATE = 0.58

JOINT_STOP_TEMP_EARLY = 0.2
JOINT_STOP_TEMP_MID = 0.30
JOINT_STOP_TEMP_LATE = 0.36

JOINT_STOP_BARRIER_EARLY = 0.40
JOINT_STOP_BARRIER_MID = 0.20
JOINT_STOP_BARRIER_LATE = 0.1

PPO_STOP_ENTROPY_COEF_EARLY = 0.01
PPO_STOP_ENTROPY_COEF_MID = 0.008
PPO_STOP_ENTROPY_COEF_LATE = 0.006

PPO_FEAT_ENTROPY_COEF_EARLY = 0.0010
PPO_FEAT_ENTROPY_COEF_MID = 0.0012
PPO_FEAT_ENTROPY_COEF_LATE = 0.0015

# =============================================================================
# Stage labels and joint readiness thresholds
# =============================================================================
WARMUP_PRIOR_STAGE = "warmup"
JOINT_PRIOR_STAGE = "joint"

JOINT_REFIT_MIN_EPISODES = 200
JOINT_VAL_PLATEAU_WINDOW = 50
JOINT_VAL_PLATEAU_DELTA_THRESHOLD = 0.005
JOINT_BEST_VAL_IMPROVEMENT_THRESHOLD = 0.01
JOINT_TOP25_TOP50_GAP_MIN = 0.010
JOINT_TOP25_TOP50_GAP_STD_MAX = 0.003
JOINT_MAX_MEDIAN_GAP_MIN = 0.45
JOINT_RANK_OVERLAP_WINDOW = 5
JOINT_TOPK_OVERLAP_THRESHOLD = 0.75

# =============================================================================
# Policy state / initialization
# =============================================================================
STOP_STATE_DIM = 6
INIT_CONTINUE_LOGIT = -0.9
