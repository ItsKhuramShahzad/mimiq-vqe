#!/usr/bin/env bash
# Build the combined MIMIQ + chemistry environment.
# Neither the existing `mimiq` nor `qchem` env is sufficient on its own
# Shortest path is extending `mimiq`.
set -euo pipefail

ENV_NAME="${1:-mimiq}"
echo "Extending conda env: $ENV_NAME"

conda run -n "$ENV_NAME" pip install \
    openfermion openfermionpyscf pyscf quantanium matplotlib pandas

echo "--- verifying ---"
conda run -n "$ENV_NAME" python - <<'PY'
mods = ["mimiqcircuits", "openfermion", "openfermionpyscf", "pyscf", "quantanium"]
ok = True
for m in mods:
    try:
        mod = __import__(m)
        print(f"  OK   {m}: {getattr(mod, '__version__', '?')}")
    except Exception as e:
        ok = False
        print(f"  FAIL {m}: {e}")
try:
    import cudaq
    print(f"  OK   cudaq: {cudaq.__version__} (reference runs available)")
except Exception:
    print("  --   cudaq not in this env; use `qchem` for CUDA-Q reference runs")
raise SystemExit(0 if ok else 1)
PY
echo "Environment ready."
