#!/bin/bash
#SBATCH --job-name=rl_main
#SBATCH --partition=cpu_large
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-24
#SBATCH --output=logs/rl_main/slurm_%A_%a.out
#SBATCH --error=logs/rl_main/slurm_%A_%a.err

# ==============================================
# RL Main Experiment Array Submission Script
# 1 task = 1 seed x 1 outer fold
# 5 seeds x 5 folds = 25 tasks -> array 0-24
# ==============================================

set -e

PROJECT_ROOT=/home/h24/qini8808/AD_main_experiment
cd "$PROJECT_ROOT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

mkdir -p logs/rl_main
mkdir -p results/rl_main

python experiments/rl_main_experiment.py
