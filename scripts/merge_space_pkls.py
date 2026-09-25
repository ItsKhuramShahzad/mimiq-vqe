"""
Merge the per-actiev space pkls of a slurm array run inot one single pkl per molecule, for easier plotting and analysis.

each array task run one active spaces and writes its pkl into its own folder, e.g. `reports/seed_recovery_cudaq/1/active_space_1.pkl`, `reports/seed_recovery_cudaq/1/active_space_2.pkl`, etc.
This script merges all the active space pkls into one pkl per molecule, e.g.
    <in >/<molecule>/<space_idx_*><i>/<name>_VQE_results .pkl
    
    This puts the spaces of each molecules back together, in the order of config/molecules_daya.py, into the same layout a full run_single.py run writes, 
    so analysis and plotting can be done the same way as for a full run_single.py run.
    
    python scripts/merge_space_pkls.py --in pkl_results/mimiq_exaqt/spaces --out pkl_results/mimiq_exaqt/
    
"""
import argparse
import os
import glob
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.molecules_data import molecules
from src.schema import validate_payload
from src.utils import sanitize_name, save_pkl

def space_key(space):
    return int(space["ncore"]), int (space["nele_cas"]), int (space["norb_cas"])

def newest_pkl(folder):
    """ The finished pkl in one space folder (the newest, if the space was run twice)."""
    pkls = glob.glob(os.path.join(folder, "*.pkl"))
    if not pkls:
        return None
    return max(pkls, key=os.path.getmtime)

def merge_molecule(name, in_dir):
    
    folders = sorted(glob.glob(os.path.join(in_dir, name, "space_idx_*")))
    parts =[]
    
    for folder in folders:
        path = newest_pkl(folder)
        if path is None:
            print(f"Warning: no pkl found in {folder}, skipping")
            continue
        with open(path, "rb") as fh:
            res= pickle.load(fh)[name]
        if not res.get("skipped"):
            parts.append(res)
    if not parts:
        print(f"Warning: no valid pkl found for {name}, skipping")
        return None, []
    
    runs = {}
    for part in parts:
        for run in part["active_space_runs"]:
            runs[space_key(run["space"])] = run
    all_spaces = molecules[name]["valid_active_spaces"]
    
    order = [space_key(s) for s in all_spaces]
    
    merged = dict(parts[0])
    merged["input_spec"] = dict(merged["input_spec"], valid_active_spaces=all_spaces)   
    merged["active_space_runs"] = [runs[k] for k in order if k in runs]                 

    merged["timing"]= dict(merged["timing"], pyscf_run_scf_seconds=sum(p["timing"]["pyscf_run_scf_seconds"] for p in parts))
    merged["integral_files_made"]= [f for p in parts for f in p.get("integral_files_made", [])]
    
    missing = [k for k in order if k not in runs]
    
    return merged, missing

def main ():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest= "in_dir", default="pkl_results/mimiq_exaqt/spaces", help="folder with the per-space pkls, e.g. pkl_results/mimiq_exaqt/spaces/<molecule>/space_idx_<i>/")
    p.add_argument("--out", default = "pkl_results/mimiq_exaqt", help="folder to write the merged molecule pkls to, e.g. pkl_results/mimiq_exaqt/spaces/<molecule>/space_idx_<i>/"   )
    p.add_argument("--molecule", nargs="*", default = None, help="molecule(s) to merge, default: all molecules in config/molecules_data.py")
    
    a = p.parse_args()
    names = a.molecule or sorted (d for d in os.listdir(a.in_dir) if os.path.isdir(os.path.join (a.in_dir, d)))
    
    os.makedirs(a.out, exist_ok=True)
    
    for name in names:
        
        merged, missing = merge_molecule(name, a.in_dir)
        if merged is None:
            print(f"Warning: no valid pkl found for {name}, skipping")
            continue
        
        payload = {name: merged}
        validate_payload(payload)
        file_name= (f"{merged['tag']}_{sanitize_name(name)}_{sanitize_name(merged['basis'])}"
                    f"_{sanitize_name(merged['target'])}_{sanitize_name(merged['optimizer'])}_VQE_results.pkl")
        out_path = os.path.join(a.out, file_name)
        save_pkl(payload, out_path)
        done = len (merged["active_space_runs"])
        
        total = len(molecules[name]["valid_active_spaces"])
        line = f"{name}: {done}/{total} active spaces -> {out_path}"
        if missing:
            line += f" (missing {len(missing)} active spaces: {missing})"
        print(line)
if __name__ == "__main__":
    main()