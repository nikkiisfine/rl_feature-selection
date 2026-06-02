# RL-Based Feature Selection for Alzheimer's Disease fMRI Connectivity

A reinforcement learning framework for selecting discriminative functional connectivity features from resting-state fMRI data for Alzheimer's Disease classification.

The policy is trained with Proximal Policy Optimization (PPO) using a hybrid warmup–joint phase strategy. It sequentially selects ROI-pair connectivity features, guided by AD-specific network priors (Default Mode, Limbic, SalVentAttn, Cont, DorsAttn, Vis, SomMot), and is evaluated via cross-validated AUC on a downstream logistic regression classifier.

---

## Repository Structure

```
.
├── RL_feature_selection_main_experiment/   # Core RL feature selection package
├── RL_feature_selection_fMRI_Prep/        # fMRI preprocessing pipeline
├── RL_feature_selection_sample_filter/    # Subject QC and sample filtering
└── RL_baseline_feature_selection/         # Classical baseline methods
```

---

### `RL_feature_selection_main_experiment/`

The main Python package implementing the PPO-based sequential feature selector.

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters, paths, network priors, training schedules |
| `model.py` | `SequentialPolicy` — PyTorch model with stop/continue + feature selection heads |
| `trainer.py` | `RLSelectorTrainer` — full training loop (warmup → joint PPO), reward computation, refit logic |
| `experiment.py` | `run_single_trial`, `build_search_grid`, `nested_select_best_params` |
| `utils.py` | Data loading, CV utilities, AUC evaluation, network parsing, I/O helpers |
| `main.py` | CLI entry point with argparse; supports SLURM array job indexing |

**Key design:**
- **Warmup phase**: supervised on network priors, feature logits trained with AUC-based reward
- **Joint PPO phase**: full policy gradient with GAE advantage, entropy regularization, adaptive stop barrier
- **Nested CV**: inner 3-fold CV for hyperparameter selection; outer 5-fold CV for generalization
- **Reward**: combines AUC signal, sparsity penalty, network structure bonus, stop quality term

**Run:**
```bash
# Local test (fold 0)
python -m RL_feature_selection_main_experiment.main --task-id 0 --episodes 300

# SLURM array job
sbatch submit_array.sh
```

---

### `RL_feature_selection_fMRI_Prep/`

Preprocessing pipeline from raw fMRIPrep derivatives to connectivity matrices.

| File | Description |
|------|-------------|
| `build_connectivity_dataset.py` | Main pipeline: scrubbing → confound regression → ROI time series → Pearson connectivity |
| `connectivity_pipeline.py` | MVP-scale connectivity extraction with Nilearn + Schaefer atlas |
| `X_y_generator.py` | Assembles final `X.npy`, `y.npy`, `feature_names.npy` from connectivity CSVs |
| `prep_features_and_labels.py` | Feature matrix preparation and label alignment |
| `qc_subjects_from_confounds.py` | Per-run QC: flags subjects based on FD/DVARS thresholds |
| `run_fmriprep_mvp.sh` | Shell script to run fMRIPrep on MVP subset |

**Pipeline summary:**
1. QC subjects via `qc_subjects_from_confounds.py` (FD + std_dvars thresholds)
2. Confound regression, bandpass filtering, standardization via Nilearn
3. ROI time series extraction using Schaefer 200-ROI atlas
4. Pearson correlation → flattened upper-triangle feature vector
5. Assemble `X`, `y`, `feature_names` arrays for downstream experiments

---

### `RL_feature_selection_sample_filter/`

Subject selection and DICOM-to-BIDS conversion for ADNI data.

| File | Description |
|------|-------------|
| `DCM2BIDS.py` | Batch DICOM-to-BIDS conversion using `dcm2bids` |
| `select_mvp_subjects.py` | Filters subjects meeting age/diagnosis/scan-quality criteria |
| `subject_select_script.py` | Utilities for date parsing, cohort construction, exclusion criteria |

---

### `RL_baseline_feature_selection/`

Classical baseline feature selection methods for benchmarking.

| File | Description |
|------|-------------|
| `run_baselines_nested_cv_5seeds_parallel.py` | Runs Lasso and ElasticNet baselines with 5 seeds × nested CV in parallel |
| `config_baseline_5seeds_parallel.py` | Paths, grid search parameters, parallelism settings |
| `data_utils_baseline_5seeds_parallel.py` | Data loading, outer fold creation, inner CV |
| `submit_rl_array.sh` | SLURM submission script |

Baselines use `sklearn` `LogisticRegression` (L1/ElasticNet) with `GridSearchCV` and the same outer-fold structure as the RL method for fair comparison.

---

## Dependencies

```
torch
numpy
scikit-learn
nilearn
pandas
joblib
```

---

## Data

This repository uses resting-state fMRI data from the [ADNI](https://adni.loni.usc.edu/) (Alzheimer's Disease Neuroimaging Initiative) dataset. Data access requires registration at the ADNI portal. Raw data and preprocessed outputs are **not** included in this repository.

---

