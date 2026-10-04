#!/bin/bash
# MIMIQ final benchmark, GPU (Exaqt with cuStateVec). One array task per molecule; its 9 active
# spaces run one after another and give one PKL per molecule. Same layout and resources as
# the CUDA-Q final GPU run. Needs mimiq-exaqt installed with the [gpu] extra. Give the account,
# QOS and partition on the command line, from the repository root (mkdir -p logs once first):
#     sbatch -A <account> --qos=<qos> -p <gpu partition> scripts/run_gpu_final.sh
#     SPACE_IDX=7 sbatch -A <account> --qos=<qos> -p <gpu partition> --array=0 scripts/run_gpu_final.sh   # short test
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH -J MIMIQ_GPU_final
#SBATCH --output=logs/mimiq_gpu_%A_%a.log
#SBATCH --error=logs/mimiq_gpu_err_%A_%a.log
#SBATCH --array=0-11

TARGET="exaqt-gpu"
OUT_DIR="pkl_results/final_2026/gpu"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load slurm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vqe_final

source scripts/final_run_common.sh
