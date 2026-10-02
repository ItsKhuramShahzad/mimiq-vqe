""" Shared helper, copied verbation from reference/2026_JUNE_Optimized_VQE.py
These must stay byte-identical to the CUDA-Q reference. snitize_name feed the pkl filename
convention the analysis chain parses, and _stable_hash seeds the jitter RNG , if the two beackends dra different jitter, the benchmakr measures luck rether than backend (docs/12 12.2)
"""

import re
import pickle
import hashlib

def sanitize_name(name: str ) -> str:
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\-\+]", "", name)
    return name

def save_pkl(obj, path: str):
    with open (path, "wb") as f:
        pickle.dump(obj, f, protocol= pickle.HIGHEST_PROTOCOL)
        
def _stable_hash(obj) -> int:
    """ deterministicn has across pthon versions, for seeding the jitter RNG. 
    Pyhton built-in hand() randomzes string hashing pre-process unless
    hash(mol_name, ....) differes b/w runs and b./w cpu/gpu machines/ hashlib.sha256() is stable across python versions and platforms, so we use it to seed the jitter RNG.     
    """
    
    return int(hashlib.sha256(repr(obj).encode()).hexdigest(), 16) % (2**31)

def run_metadata() -> dict:
    """Software versions, machine, SLURM job and a hash of the code, as in the CUDA-Q
    package (vqe_cudaq.utils.run_metadata), so MIMIQ and CUDA-Q runs record the same.

    ``script_sha256`` hashes all ``src/*.py`` files; ``git_commit`` is the repository
    commit and ``git_dirty`` is True if ``src/`` differed from it (None outside git).
    """
    import os
    import glob
    import platform
    import socket
    import subprocess
    import numpy as np
    import scipy
    import pyscf
    import openfermion
    import mimiqcircuits
    import exaqt
    import mimiq_openfermion

    src_dir = os.path.dirname(os.path.abspath(__file__))
    code = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(src_dir, "*.py"))):
        with open(path, "rb") as fh:
            code.update(os.path.basename(path).encode())
            code.update(fh.read())

    cpu_model = None
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    meta = {
        "python": platform.python_version(),
        "mimiqcircuits": getattr(mimiqcircuits, "__version__", None),
        "exaqt": getattr(exaqt, "__version__", None),
        "mimiq_openfermion": getattr(mimiq_openfermion, "__version__", None),
        "pyscf": pyscf.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "openfermion": openfermion.__version__,
        "hostname": socket.gethostname(),
        "cpu_model": cpu_model,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),   # Exaqt's own threads
        "os_cpu_count": os.cpu_count(),
        "script_name": "mimiq_vqe",
        "script_sha256": code.hexdigest(),
    }

    # git commit of the repository and whether src/ had uncommitted changes
    repo = os.path.dirname(src_dir)
    try:
        def git(*args):
            return subprocess.run(["git", "-C", repo, *args], capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        meta["git_commit"] = git("rev-parse", "HEAD") or None
        meta["git_dirty"] = (bool(git("status", "--porcelain", "--", "src"))
                             if meta["git_commit"] else None)
    except Exception:
        meta["git_commit"] = meta["git_dirty"] = None

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        meta["gpu"] = out.stdout.strip() or None
    except Exception:
        meta["gpu"] = None
    return meta
