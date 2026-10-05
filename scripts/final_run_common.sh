# Shared part of run_cpu_final.sh and run_gpu_final.sh (sourced, not run on its own).
#
# Set by the calling script: TARGET (exaqt-cpu or exaqt-gpu) and OUT_DIR.
# Set when you submit (all optional):
#   ANSATZ      spin (default), singlet or lucj
#   CONDA_ENV   vqe_final (default; the env of the CUDA-Q final run) or mimiq (needed for lucj:
#               ffsim is only there)
#   LUCJ_REPS   LUCJ layers (default 2)        LUCJ_PAIRS  local (default) or full
#   SPACE_IDX   run one active space only (a short test; output goes to final_2026_test)
#   VERSIONS    which version table to check (default: the env name)

# Same molecules in the same order as the CUDA-Q final run (array index 0-11)
MOLECULES=(
  "Ethylene" "Methanamide" "NH2-" "Benzene" "Naphthalene" "Tetracene"
  "Pentacene" "Adenine" "Thymine" "Guanine" "Cytosine" "Uracil"
)
ANSATZ=${ANSATZ:-spin}
CONDA_ENV=${CONDA_ENV:-vqe_final}
LUCJ_REPS=${LUCJ_REPS:-2}
LUCJ_PAIRS=${LUCJ_PAIRS:-local}
export VERSIONS=${VERSIONS:-$CONDA_ENV}
export ANSATZ

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
if [ ! -f src/run_single.py ]; then
  echo "Submit from the mimiq-vqe repository root." >&2; exit 1
fi
if [ "$ANSATZ" = "lucj" ] && [ "$VERSIONS" != "mimiq" ]; then
  echo "ANSATZ=lucj needs CONDA_ENV=mimiq (ffsim is not in $CONDA_ENV)." >&2; exit 1
fi

# 1) the exact versions of the chosen env; the MIMIQ packages are the same in both
python - <<'PYEOF' || exit 1
import os, sys
from importlib.metadata import version, PackageNotFoundError
TABLES = {
    # the CUDA-Q final-run env, identical on fe and lyra
    "vqe_final": {"python": "3.11.13", "scipy": "1.16.0", "numpy": "1.26.4", "pyscf": "2.6.2",
                  "openfermion": "1.6.1"},
    # the MIMIQ env, identical on the laptop and the HPC; needed for LUCJ (ffsim)
    "mimiq": {"python": "3.11.16", "scipy": "1.17.1", "numpy": "2.4.6", "pyscf": "2.14.0",
              "openfermion": "1.8.1", "ffsim": "0.0.84", "qiskit": "2.5.2",
              "mimiq-qiskit": "0.2.0"},
}
table = os.environ["VERSIONS"]
if table not in TABLES:
    sys.exit(f"no version table for {table!r}; use one of {list(TABLES)}")
want = dict(TABLES[table], **{"mimiqcircuits": "0.28.0", "mimiq-exaqt": "0.3.0"})
if os.environ["ANSATZ"] != "lucj":
    for k in ("ffsim", "qiskit", "mimiq-qiskit"):
        want.pop(k, None)
def have_of(name):
    if name == "python":
        return sys.version.split()[0]
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"
have = {k: have_of(k) for k in want}
bad = {k: (have[k], v) for k, v in want.items() if have[k] != v}
print(f"versions ({table}):", have)
if bad:
    sys.exit(f"wrong versions (have, want): {bad}")
PYEOF

# 2) the code is exactly the checked-out commit
if ! git diff --quiet HEAD -- src; then
  echo "src/ has uncommitted changes; commit or reset them first." >&2; exit 1
fi
echo "commit: $(git rev-parse HEAD)"

MOLECULE=${MOLECULES[$SLURM_ARRAY_TASK_ID]}
EXTRA=(--ansatz "$ANSATZ")
if [ "$ANSATZ" = "lucj" ]; then
  EXTRA+=(--lucj-reps "$LUCJ_REPS" --lucj-pairs "$LUCJ_PAIRS")
else
  EXTRA+=(--trotter_order 1)
fi
if [ -n "$SPACE_IDX" ]; then
  EXTRA+=(--space_idx "$SPACE_IDX")
  OUT_DIR="${OUT_DIR/final_2026/final_2026_test}"
fi
mkdir -p "$OUT_DIR"

echo "molecule: $MOLECULE | target: $TARGET | env: $CONDA_ENV | out: $OUT_DIR | job: $SLURM_JOB_ID[$SLURM_ARRAY_TASK_ID]"
echo "node: $HOSTNAME | OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} | RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-unset} | ${EXTRA[*]}"
[ "$TARGET" = "exaqt-gpu" ] && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

python -m src.run_single \
  --molecule "$MOLECULE" --integrals integrals --target "$TARGET" \
  --out_dir "$OUT_DIR" "${EXTRA[@]}"
