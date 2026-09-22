"""
Compute and save active-space integrals for every molecule, active space and basis.
Meant for the HPC. One .npz per active space:

    <out>/<basis>/<molecule>/space_01_ncore_5_nele_6_norb_4.npz

    python scripts/dump_active_integrals.py --basis sto-3g 6-31g cc-pVDZ --molecule Ethylene
    python scripts/dump_active_integrals.py --ccsd ...    # also store CCSD amplitudes
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.molecules_data import molecules
from src.integrals import compute_active_space, run_ccsd, run_scf, save_active_space


def dump_molecule(name, basis, out_dir, max_memory, with_ccsd):
    entry = molecules[name]
    t0 = time.time()
    mf = run_scf(entry, basis, max_memory)
    print(f"  SCF E={mf.e_tot:.10f} nmo={mf.mo_coeff.shape[1]} {time.time() - t0:.0f}s", flush=True)

    ccsd = None
    if with_ccsd:
        t_cc = time.time()
        ccsd = run_ccsd(mf, max_memory)
        print(f"  CCSD E={ccsd[0]:.10f} {time.time() - t_cc:.0f}s", flush=True)

    mol_dir = os.path.join(out_dir, basis, name)
    os.makedirs(mol_dir, exist_ok=True)

    for i, space in enumerate(entry["valid_active_spaces"], start=1):
        ncore, nele, norb = int(space["ncore"]), int(space["nele_cas"]), int(space["norb_cas"])
        tag = f"space_{i:02d}_ncore_{ncore}_nele_{nele}_norb_{norb}"
        try:
            data = compute_active_space(mf, ncore, nele, norb, ccsd=ccsd)
        except ValueError as exc:
            print(f"  skip {tag}: {exc}", flush=True)
            continue
        save_active_space(os.path.join(mol_dir, tag + ".npz"), data,
                          molecule=name, basis=basis,
                          charge=int(entry["charge"]), multiplicity=int(entry["multiplicity"]))
        print(f"  {tag}: {2 * norb} qubits, E_CASCI={data['e_casci']:.10f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--basis", nargs="+", default=["sto-3g", "6-31g", "cc-pVDZ"])
    p.add_argument("--molecule", nargs="*", default=None, help="default: all closed shell")
    p.add_argument("--out", default="integrals")
    p.add_argument("--max-memory", type=int, default=16000, help="PySCF memory limit in MB")
    p.add_argument("--ccsd", action="store_true",
                   help="also store the CCSD amplitudes cut to each active space")
    a = p.parse_args()

    names = a.molecule or [k for k, v in molecules.items() if int(v["multiplicity"]) == 1]
    for basis in a.basis:
        for name in names:
            print(f"\n=== {basis} / {name} ===", flush=True)
            try:
                dump_molecule(name, basis, a.out, a.max_memory, a.ccsd)
            except Exception as exc:
                print(f"  FAILED {name} {basis}: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
