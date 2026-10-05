#!/bin/bash
# MIMIQ final benchmark, CPU (Exaqt state vector). One array task per molecule; its 9 active
# spaces run one after another and give one PKL per molecule. Same layout and resources as
# the CUDA-Q final CPU run. Give the account and partition on the command line, from the
# repository root (create logs/ once first: mkdir -p logs):
#     sbatch -A <account> -p <partition> scripts/run_cpu_final.sh                         # all 12 molecules
#     SPACE_IDX=7 sbatch -A <account> -p <partition> --array=0 scripts/run_cpu_final.sh   # short test: Ethylene (2,3)
# Any ansatz (default: spin UCCSD in vqe_final, the final run). LUCJ needs the mimiq env:
#     ANSATZ=lucj CONDA_ENV=mimiq sbatch -A <account> -p <partition> scripts/run_cpu_final.sh
#     ANSATZ=spin CONDA_ENV=mimiq sbatch -A <account> -p <partition> scripts/run_cpu_final.sh   # UCCSD, same env as LUCJ
#     ANSATZ=lucj CONDA_ENV=mimiq LUCJ_PAIRS=full LUCJ_REPS=1 sbatch ...                        # other LUCJ settings
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=14-00:00:00
#SBATCH -J MIMIQ_CPU_final
#SBATCH --output=logs/mimiq_cpu_%A_%a.log
#SBATCH --error=logs/mimiq_cpu_err_%A_%a.log
#SBATCH --array=0-11

TARGET="exaqt-cpu"
ANSATZ=${ANSATZ:-spin}
CONDA_ENV=${CONDA_ENV:-vqe_final}
# the final run keeps its folder; any other ansatz or env gets its own, e.g. cpu_lucj_local_r2_mimiq
if [ "$ANSATZ" = "spin" ] && [ "$CONDA_ENV" = "vqe_final" ]; then
  OUT_DIR="pkl_results/final_2026/cpu"
elif [ "$ANSATZ" = "lucj" ]; then
  OUT_DIR="pkl_results/final_2026/cpu_lucj_${LUCJ_PAIRS:-local}_r${LUCJ_REPS:-2}_$CONDA_ENV"
else
  OUT_DIR="pkl_results/final_2026/cpu_uccsd_${ANSATZ}_$CONDA_ENV"
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK     # Exaqt's own threads (it ignores OMP_NUM_THREADS)
source ~/miniconda3/etc/profile.d/conda.sh
export JAX_PLATFORMS=cpu                          # ffsim (LUCJ) imports jax; keep it on the CPU
conda activate "$CONDA_ENV"

source scripts/final_run_common.sh
