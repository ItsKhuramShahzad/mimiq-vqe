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

