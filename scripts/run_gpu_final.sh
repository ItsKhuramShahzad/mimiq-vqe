#!/bin/bash
# MIMIQ final benchmark, GPU (Exaqt with cuStateVec). One array task per molecule; its 9 active
# spaces run one after another and give one PKL per molecule. Same layout and resources as
# the CUDA-Q final GPU run. Needs mimiq-exaqt installed with the [gpu] extra. Give the account,
# QOS and partition on the command line, from the repository root (mkdir -p logs once first):
#     sbatch -A <account> --qos=<qos> -p <gpu partition> scripts/run_gpu_final.sh
#     SPACE_IDX=7 sbatch -A <account> --qos=<qos> -p <gpu partition> --array=0 scripts/run_gpu_final.sh   # short test
# Any ansatz (default: spin UCCSD in vqe_final), e.g. ANSATZ=lucj CONDA_ENV=mimiq sbatch ... ;
# see scripts/final_run_common.sh for the settings.
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
ANSATZ=${ANSATZ:-spin}
CONDA_ENV=${CONDA_ENV:-vqe_final}
# the final run keeps its folder; any other ansatz or env gets its own, e.g. gpu_lucj_local_r2_mimiq
if [ "$ANSATZ" = "spin" ] && [ "$CONDA_ENV" = "vqe_final" ]; then
  OUT_DIR="pkl_results/final_2026/gpu"
elif [ "$ANSATZ" = "lucj" ]; then
  OUT_DIR="pkl_results/final_2026/gpu_lucj_${LUCJ_PAIRS:-local}_r${LUCJ_REPS:-2}_$CONDA_ENV"
else
  OUT_DIR="pkl_results/final_2026/gpu_uccsd_${ANSATZ}_$CONDA_ENV"
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load slurm
source ~/miniconda3/etc/profile.d/conda.sh
export JAX_PLATFORMS=cpu                          # ffsim (LUCJ) imports jax; keep it on the CPU
conda activate "$CONDA_ENV"

source scripts/final_run_common.sh
