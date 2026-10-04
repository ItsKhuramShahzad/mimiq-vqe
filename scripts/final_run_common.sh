# Shared part of run_cpu_final.sh and run_gpu_final.sh (sourced, not run on its own).
# Expects TARGET and OUT_DIR; optional SPACE_IDX runs one active space only (test).

# Same molecules in the same order as the CUDA-Q final run (array index 0-11)
MOLECULES=(
  "Ethylene" "Methanamide" "NH2-" "Benzene" "Naphthalene" "Tetracene"
  "Pentacene" "Adenine" "Thymine" "Guanine" "Cytosine" "Uracil"
)

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
if [ ! -f src/run_single.py ]; then
  echo "Submit from the mimiq-vqe repository root." >&2; exit 1
fi

# 1) the vqe_final versions (same as the CUDA-Q final run) plus the MIMIQ packages
python - <<'EOF' || exit 1
import sys, scipy, numpy, pyscf, openfermion, mimiqcircuits, exaqt
want = {"python": "3.11.13", "scipy": "1.16.0", "numpy": "1.26.4", "pyscf": "2.6.2",
        "openfermion": "1.6.1", "mimiqcircuits": "0.28.0", "exaqt": "0.3.0"}
have = {"python": sys.version.split()[0], "scipy": scipy.__version__,
        "numpy": numpy.__version__, "pyscf": pyscf.__version__,
        "openfermion": openfermion.__version__,
        "mimiqcircuits": mimiqcircuits.__version__, "exaqt": exaqt.__version__}
bad = {k: (have[k], v) for k, v in want.items() if have[k] != v}
print("versions:", have)
if bad:
    sys.exit(f"wrong versions (have, want): {bad}")
EOF

# 2) the code is exactly the checked-out commit
if ! git diff --quiet HEAD -- src; then
  echo "src/ has uncommitted changes; commit or reset them first." >&2; exit 1
fi
echo "commit: $(git rev-parse HEAD)"

MOLECULE=${MOLECULES[$SLURM_ARRAY_TASK_ID]}
EXTRA=()
if [ -n "$SPACE_IDX" ]; then
  EXTRA+=(--space_idx "$SPACE_IDX")
  OUT_DIR="${OUT_DIR/final_2026/final_2026_test}"
fi
mkdir -p "$OUT_DIR"

echo "molecule: $MOLECULE | target: $TARGET | out: $OUT_DIR | job: $SLURM_JOB_ID[$SLURM_ARRAY_TASK_ID]"
echo "node: $HOSTNAME | OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} | RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-unset} | ${EXTRA[*]}"
[ "$TARGET" = "exaqt-gpu" ] && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

python -m src.run_single \
  --molecule "$MOLECULE" --integrals integrals --target "$TARGET" \
  --ansatz spin --trotter_order 1 --out_dir "$OUT_DIR" "${EXTRA[@]}"
