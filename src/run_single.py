"""
run vqe for one molecule on mimiq and write a pkl.

mapping of run_one_molecule() and main from reference/2026_JUNE_Optimized_VQE.py,. 

same cli, same constants, same pkl layout, so the snalysis scripts read mimiq and cuda-q results side by side.


run from the repo root with:

python -m src.run_single --molecule Benzene --space_idx 0

"""

import argparse
import os 
import itertools
import sys
import time

import numpy as np
import openfermion
from openfermion import get_fermion_operator, jordan_wigner, uccsd_singlet_paramsize

from pyscf import cc, mcscf

import exaqt
import mimiqcircuits

import mimiq_openfermion
import openfermionpyscf
from config.molecules_data import molecules
from src.mimiq_backend import make_energy_fn
from src.mimiq_driver import best_of_jitters_one_chunk, vqe_until_converged
from src.mimiq_hamiltonian import openfermion_to_mimiq_hamiltonian
from src.integrals import (compute_active_space, find_integral_file, load_integrals,
                           qubit_hamiltonian, run_ccsd, run_scf, save_active_space)
from src.schema import validate_payload
from src.utils import _stable_hash, sanitize_name, save_pkl

# same constants as the cuda-q script

BASIS= "cc-pVDZ"
TARGET= "exaqt"
OPTIMIZER= "COBYLA"

SEED =12345
TOL = 1e-10
COBYLA_RHOBEG = 0.2
THETA_SCALE= 1.0
N_JITTER_RESTARTS= 3 
JITTER_SCALE= 5E-3

MAX_MEMORY = 16000   # MB, PySCF limit when a missing integral file has to be made

VQE_EPS_E=1E-6
VQE_PATIENCE=3
VQE_MAX_CYCLES =25
VQE_CHUNK_MAXITER= 600 
VQE_JITTER_BETWEEN_CYCLES= True
VQE_JITTER_BETWEEN_SCALE= 5E-4

HEAVY_PARAM_THRESHOLD= 150
HEAVY_RESTARTS= 0
HEAVY_RHOBEG= 0.05

PRINT_EVERY_CYCLE= False

# Heavy mode decided from the cuda-q parameter count not mimiq, so both 
# backends, always use identical optimizer settign for the same active space.
# (mimiq singlet uccsd has roughly half the parameters of cuda-q.)

HEAVY_FROM_CUDAQ_COUNT= True

# helpers

def cudaq_uccsd_num_parameters(n_ele_cas,qubit_count):
    """
    cuda-q spin orbital uccsd paraneter count, closed shell.
    
    Used only to decide heavy mode, so both backends get identical optimizer setting for the same active space 
    (mimiq singlet uccsd has about half  as many parameters as cuda-q spin orbital uccsd.)
    
    Ask cuda-q when it is installed. the closed form fallback lets his run on machine without cuda-q; it is verified against the library for 4-16 qubits
    whenever both are available. The closed form is derived from the combinatorial formula for the number of single and double excitations in a closed shell system.
        singles    2*n_occ * n_vir
        mixed doubles (n_occ* n_vir)^2
        same-spin 2* C(nocc,2) * C(nvir,2)
    
    """
    n_occ = n_ele_cas//2
    n_vir = qubit_count//2 - n_occ
    
    n_params = (2 * n_occ * n_vir
                + (n_occ * n_vir) ** 2
                + 2 * (n_occ * (n_occ - 1) // 2) * (n_vir * (n_vir - 1) // 2))
    try: 
        import cudaq
    except ImportError: 
        return n_params
    n_cudaq = int(cudaq.kernels.uccsd_num_parameters(n_ele_cas, qubit_count)
                         )    
    if n_cudaq != n_params:
        raise RuntimeError(
            f"CUDA-Q parameter count {n_cudaq} disagrees with the closed form "
            f"{n_params} for nele_cas={n_ele_cas}, qubit_count={qubit_count}; "
            f"cudaq {cudaq.__version__}")
    return n_cudaq
    
    
def slice_ccsd_to_active(t1, t2, nocc, active_orbs):
    """" copied from the referenec cuda-q script, slice the full ccsd amplitudes to the active space. """
  # this function cutt full molecule ccsd amplitudes down to just the activae space.
    t1= np.asarray(t1)  # shape(nocc, nvir)  one electron jumps i -> a
    t2= np.asarray(t2)  # shape (nocc, nocc, nvir, nvir) two electron jumps i, j -> a, b
    
    # here i and j are occupied orbitals, a and b are virtual orbitals both count start from 0.
    
    
    active_occ= [p for p in active_orbs if p< nocc]
    active_vir= [p-nocc for p in active_orbs if p >= nocc]
    # check if the active space has no occupied or no virtual orbitals, then no excitations, in which case the uccsd amplitudes are empty.
    if len(active_occ) ==0 or len(active_vir)==0:
        return (np.zeros((len(active_occ), len(active_vir))),
                np.zeros((len(active_occ),)*2+ (len(active_vir),)*2))
    #sclice the full ccsd amplitudes to the active space, using numpy advanced indexing.
    t1_active= t1[np.ix_(active_occ,active_vir )]
    t2_active = t2[np.ix_(active_occ, active_occ, active_vir, active_vir)   ]
    
    return t1_active, t2_active

def pack_ccsd_singlet(t1_active, t2_active, scale = 1.0):
    """
    Packed restricted ccsd amplitudes into Openfermion  singlet uccssd order
    
    Derived excatly: T-T^dagger built from (t1, t2) equals
    uccsd_singlet_generator (packed) with residual ~1e-16. no factor fo 2 on doubles; 
    taht factor in the cuda-q packer os a cuda q kernel artefact.
    
    singles t1[i, a] -> packed[i, a]
    doubles_1 0.5* t2[i,i, a, a]    same spin, same orbital(ONE SPIN UP AND ONE SPIN DOWN) 
    doubles_2 t2[i, j, a, b]  (two different pairs)
    
    On Ethylene cc-pVDZ this see recovers ~94.5% of active space correlation energy before any optimization. 
    
    
    """
    t1 = np.asarray(t1_active, dtype=float)
    t2 = np.asarray(t2_active, dtype=float)
    nocc, nvir = t1.shape
    # LIST OF  all occupied-virtual pairs. 
    # openfermion own loop order: virtual, occupoed inner.
    ov = list (itertools.product(range(nvir), range(nocc))) # Openfermion order
    
    # singles
    singles = [t1[occ, vir] for vir, occ in ov]
    doubles_1 =[0.5 * t2[occ, occ, vir, vir] for vir, occ in ov]
    doubles_2= [t2[occ1, occ2, vir1, vir2] for (vir1, occ1), (vir2, occ2) in itertools.combinations(ov, 2)]
    
    return scale * np.array(singles+ doubles_1 + doubles_2)





def collect_pyscf_insights(mf, molecule_of):
    """Copied from the reference."""
    mol = mf.mol
    return {
        "pyscf": {
            "nelec": int(mol.nelectron),
            "charge": int(mol.charge),
            "spin_2S": int(mol.spin),
            "basis": str(mol.basis),
            "nao_nr": int(mol.nao_nr()),
            "energy_scf_total": float(mf.e_tot),
            "converged": bool(getattr(mf, "converged", False)),
            "mo_energy": np.array(mf.mo_energy, dtype=float),
            "mo_occ": np.array(mf.mo_occ, dtype=float),
            "mo_coeff_shape": tuple(mf.mo_coeff.shape),
        },
        "openfermion": {
            "hf_energy": float(molecule_of.hf_energy),
            "n_orbitals": int(molecule_of.n_orbitals),
            "n_electrons": int(molecule_of.n_electrons),
        },
    }


def backend_versions():
    return {
        "mimiqcircuits": getattr(mimiqcircuits, "__version__", "?"),
        "exaqt": getattr(exaqt, "__version__", "?"),
        "mimiq_openfermion": getattr(mimiq_openfermion, "__version__", "?"),
        "openfermion": getattr(openfermion, "__version__", "?"),
        "python": sys.version.split()[0],
    }


# -----------------------------
# Run one molecule
# -----------------------------
def setup_from_geometry(spec):
    """SCF and CCSD on the whole molecule, from the geometry (the original route)."""
    moldata = openfermion.MolecularData(
        spec["geometry"], BASIS, int(spec["multiplicity"]), int(spec["charge"]))
    t0 = time.time()
    molecule = openfermionpyscf.run_pyscf(moldata, run_scf=True, run_fci=False)
    t1 = time.time()

    mf = molecule._pyscf_data["scf"]
    pyscf_info = collect_pyscf_insights(mf, molecule)

    nmo = mf.mo_coeff.shape[1]
    nocc = mf.mol.nelectron // 2
    nvir = nmo - nocc
    HF_FULL = float(molecule.hf_energy)

    # Every VQE starts from CCSD amplitudes, so CCSD must run and converge.
    mycc = cc.CCSD(mf)
    ecc_corr, t1amp, t2amp = mycc.kernel()
    if not mycc.converged:
        raise RuntimeError("CCSD did not converge; no amplitudes to start the VQE from")
    E_CCSD_FULL = float(mf.e_tot + ecc_corr)
    ccsd_block = {
        "computed": True,
        "E_ccsd_total": E_CCSD_FULL,
        "E_ccsd_corr": float(ecc_corr),
        "t1_norm": float(np.linalg.norm(t1amp)),
        "t2_norm": float(np.linalg.norm(t2amp)),
        "note": "CCSD full-system amplitudes used for theta0 slicing.",
    }

    return dict(molecule=molecule, mf=mf, nelec=int(mf.mol.nelectron), nmo=int(nmo),
                e_hf=HF_FULL, e_ccsd=E_CCSD_FULL, ccsd_block=ccsd_block,
                pyscf_info=pyscf_info, t1=t1amp, t2=t2amp, scf_seconds=float(t1 - t0))


class IntegralFiles:
    """The integral files of one molecule: read them, and make the missing ones.

    A missing file is computed the same way scripts/dump_active_integrals.py does it
    (PySCF CASCI integrals, norb^4 memory, plus CCSD amplitudes), saved where the next
    run will find it, and then read like any other file. SCF and CCSD run at most once
    per molecule, and only if some file is actually missing."""

    def __init__(self, root, mol_name, spec):
        self.root, self.mol_name, self.spec = root, mol_name, spec
        self.mf = self.ccsd = None
        self.seconds = 0.0
        self.made = []

    def get(self, ncore, nele_cas, norb_cas):
        """The file for this space, with CCSD amplitudes. A missing file, or one saved
        without amplitudes, is (re)computed and saved first."""
        try:
            data = load_integrals(self.root, BASIS, self.mol_name, ncore, nele_cas, norb_cas)
            if "t1_active" in data:
                return data
            print(f"[MAKE ] {data['path']} has no CCSD amplitudes: recomputing it", flush=True)
        except FileNotFoundError:
            pass
        self.make(ncore, nele_cas, norb_cas)
        return load_integrals(self.root, BASIS, self.mol_name, ncore, nele_cas, norb_cas)

    def make(self, ncore, nele_cas, norb_cas):
        if self.mf is None:
            print(f"[MAKE ] {self.mol_name} / {BASIS}: running SCF + CCSD once "
                  f"to write the missing integral file(s)", flush=True)
            t0 = time.time()
            self.mf = run_scf(self.spec, BASIS, MAX_MEMORY)
            self.ccsd = run_ccsd(self.mf, MAX_MEMORY)
            self.seconds += time.time() - t0

        data = compute_active_space(self.mf, ncore, nele_cas, norb_cas, ccsd=self.ccsd)

        mol_dir = os.path.join(self.root, BASIS, self.mol_name)
        os.makedirs(mol_dir, exist_ok=True)
        try:                        # replacing a file without amplitudes: keep its name
            path = find_integral_file(self.root, BASIS, self.mol_name, ncore, nele_cas, norb_cas)
            tag = os.path.basename(path)[:-len(".npz")]
        except FileNotFoundError:
            tag = (f"space_{self.number(ncore, nele_cas, norb_cas):02d}"
                   f"_ncore_{ncore}_nele_{nele_cas}_norb_{norb_cas}")
            path = os.path.join(mol_dir, tag + ".npz")
        # write under a private name, then rename: two jobs making the same file at once
        # can never leave a half-written one where load_integrals looks
        tmp = os.path.join(mol_dir, f".{tag}.{os.uname().nodename}.{os.getpid()}.npz")
        save_active_space(tmp, data, molecule=self.mol_name, basis=BASIS,
                          charge=int(self.spec["charge"]),
                          multiplicity=int(self.spec["multiplicity"]))
        os.replace(tmp, path)
        self.made.append(path)
        print(f"[MAKE ] saved {path}", flush=True)

    def number(self, ncore, nele_cas, norb_cas):
        """Same numbering as the dump script: position in the molecule's full list, from 1."""
        full = molecules.get(self.mol_name, self.spec)["valid_active_spaces"]
        for i, s in enumerate(full, start=1):
            if (int(s["ncore"]), int(s["nele_cas"]), int(s["norb_cas"])) == (ncore, nele_cas, norb_cas):
                return i
        return 0


def setup_from_integrals(files, spec):
    """Same information as setup_from_geometry, read from the integral files.
    The amplitudes are per active space, so they are read inside the loop."""
    spaces = [(int(s["ncore"]), int(s["nele_cas"]), int(s["norb_cas"]))
              for s in spec["valid_active_spaces"]]
    first = None
    for space in spaces:
        try:
            first = files.get(*space)
            break
        except ValueError:                    # this space does not fit the molecule/basis
            continue
    if first is None:
        raise ValueError(f"no active space of {files.mol_name} fits the {BASIS} basis")

    e_ccsd = first.get("e_ccsd")
    ccsd_block = {"computed": e_ccsd is not None,
                  "E_ccsd_total": e_ccsd,
                  "E_ccsd_corr": (e_ccsd - first["e_hf"]) if e_ccsd is not None else None,
                  "t1_norm": None, "t2_norm": None,
                  "note": "read from integral files; amplitudes stored per active space only."}
    return dict(molecule=None, mf=None, nelec=first["n_electrons"], nmo=int(first["nmo"]),
                e_hf=first["e_hf"], e_ccsd=e_ccsd, ccsd_block=ccsd_block,
                pyscf_info={"source": f"integral files: {files.root}"},
                t1=None, t2=None, scf_seconds=files.seconds)


def run_one_molecule(mol_name, spec, checkpoint_path=None, integrals_dir=None):
    mol_name_clean = sanitize_name(mol_name)

    if int(spec["multiplicity"]) != 1 or int(spec["Total Electrons"]) % 2 != 0:
        return {"molecule_name": mol_name, "skipped": True,
                "skip_reason": "open-shell or odd electrons: out of scope"}

    # With --integrals, everything comes from the files; missing ones are made and saved.
    files = None
    if integrals_dir is None:
        setup = setup_from_geometry(spec)
    else:
        files = IntegralFiles(integrals_dir, mol_name, spec)
        setup = setup_from_integrals(files, spec)

    molecule, mf = setup["molecule"], setup["mf"]
    nelec, nmo = setup["nelec"], setup["nmo"]
    nocc = nelec // 2
    nvir = nmo - nocc
    HF_FULL, E_CCSD_FULL = setup["e_hf"], setup["e_ccsd"]
    ccsd_block, pyscf_info = setup["ccsd_block"], setup["pyscf_info"]
    t1amp, t2amp = setup["t1"], setup["t2"]

    molecule_results = {
        "molecule_name": mol_name,
        "molecule_name_clean": mol_name_clean,
        "skipped": False,
        "tag": time.strftime("%d_%b_%Y").upper(),
        "basis": BASIS,
        "target": TARGET,
        "target_precision_option": None,
        "cudaq_precision": None,
        "optimizer": OPTIMIZER,
        "seed": int(SEED),
        "input_spec": spec,
        "timing": {"pyscf_run_scf_seconds": setup["scf_seconds"]},
        "references": {"E_hf_full": HF_FULL, "E_ccsd_full": E_CCSD_FULL},
        "pyscf_insights": pyscf_info,
        "ccsd": ccsd_block,
        "system_sizes": {"nmo": int(nmo), "nocc": int(nocc), "nvir": int(nvir)},
        "backend_versions": backend_versions(),
        "active_space_runs": [],
    }

    for space in spec["valid_active_spaces"]:
        ncore = int(space["ncore"])
        nele_cas = int(space["nele_cas"])
        norb_cas = int(space["norb_cas"])
        qubit_count = 2 * norb_cas

        local_rng = np.random.default_rng(
            SEED + _stable_hash((mol_name, ncore, nele_cas, norb_cas)))

        occ = list(range(ncore))
        act = list(range(ncore, ncore + norb_cas))

        if len(act) == 0 or act[-1] >= nmo:
            molecule_results["active_space_runs"].append({
                "space": space, "skipped": True,
                "skip_reason": f"active indices exceed nmo={nmo}"})
            continue

        if (2 * ncore + nele_cas) != nelec:
            molecule_results["active_space_runs"].append({
                "space": space, "skipped": True,
                "skip_reason": (f"CASCI sanity fail: 2*ncore+nele_cas="
                                f"{2 * ncore + nele_cas} != nelec={nelec}")})
            continue

        print(f"[SPACE] {mol_name} ncore={ncore} nele={nele_cas} norb={norb_cas} "
              f"-> {qubit_count} qubits", flush=True)



        data = None
        if files is not None:
            data = files.get(ncore, nele_cas, norb_cas)
            E_CASCI = data["e_casci"]
            qop = qubit_hamiltonian(data)
            molecule_results["timing"]["pyscf_run_scf_seconds"] = files.seconds
            molecule_results["integral_files_made"] = list(files.made)
        else:
            casci = mcscf.CASCI(mf, norb_cas, nele_cas)
            casci.ncore = ncore
            E_CASCI = float(casci.kernel()[0])
            molecular_ham = molecule.get_molecular_hamiltonian(
                occupied_indices=occ, active_indices=act)
            qop = jordan_wigner(get_fermion_operator(molecular_ham))

        H, c0, _ = openfermion_to_mimiq_hamiltonian(qop)

        # constant=0.0: the driver works with E_nc, c0 is added below.
        energy_fn, execute_times  = make_energy_fn(
            H, constant=0.0, n_qubits=qubit_count, n_electrons=nele_cas)

        expected = int(uccsd_singlet_paramsize(qubit_count, nele_cas))

        # theta0 always comes from the CCSD amplitudes, never from zeros
        if data is not None:
            theta0 = pack_ccsd_singlet(data["t1_active"], data["t2_active"], scale=THETA_SCALE)
            theta0_source = "CCSD-sliced (singlet packer, from integral file)"
        else:
            t1_act, t2_act = slice_ccsd_to_active(t1amp, t2amp, nocc, act)
            theta0 = pack_ccsd_singlet(t1_act, t2_act, scale=THETA_SCALE)
            theta0_source = "CCSD-sliced (singlet packer)"
        if len(theta0) != expected:
            raise RuntimeError(f"CCSD seed has {len(theta0)} parameters, the ansatz needs "
                               f"{expected} (ncore={ncore}, nele={nele_cas}, norb={norb_cas})")
        E_theta0= float(c0 + energy_fn(theta0))
        corr_active = HF_FULL - E_CASCI
        seed_pct = 100.0* (HF_FULL - E_theta0)/ corr_active if abs(corr_active) > 1e-12 else None
        print(f"[SEED ] {theta0_source}: E_theta0={E_theta0:.10f} "
              f"recovers {seed_pct:5.1f}% of active correlation "
              f"({1000*corr_active:.2f} mHa), {1000*(E_theta0-E_CASCI):.2f} mHa above CASCI",
              flush=True)
        heavy_count = (cudaq_uccsd_num_parameters(nele_cas, qubit_count)
                       if HEAVY_FROM_CUDAQ_COUNT else expected)
        is_heavy = heavy_count > HEAVY_PARAM_THRESHOLD
        local_restarts = HEAVY_RESTARTS if is_heavy else N_JITTER_RESTARTS
        local_rhobeg = HEAVY_RHOBEG if is_heavy else COBYLA_RHOBEG

        if local_restarts > 0:
            seed_out = best_of_jitters_one_chunk(
                energy_fn, theta0, local_rng,
                n_restarts=local_restarts, jitter_scale=JITTER_SCALE,
                chunk_maxiter=VQE_CHUNK_MAXITER,
                method=OPTIMIZER, tol=TOL, rhobeg=local_rhobeg)
            theta_seed = seed_out["theta_opt"]
            best_init_index = int(seed_out["best_init_index"])
        else:
            seed_out = None
            theta_seed = theta0.copy()
            best_init_index = -1
        n_exec_before_vqe = len(execute_times)
        vqe_out = vqe_until_converged(
            energy_fn, theta_seed, local_rng,
            eps_E=VQE_EPS_E, patience=VQE_PATIENCE, max_cycles=VQE_MAX_CYCLES,
            chunk_maxiter=VQE_CHUNK_MAXITER,
            jitter_between_cycles=VQE_JITTER_BETWEEN_CYCLES,
            jitter_scale=VQE_JITTER_BETWEEN_SCALE,
            method=OPTIMIZER, tol=TOL, rhobeg=local_rhobeg,
            verbose_cycles=PRINT_EVERY_CYCLE)
        vqe_out["best_init_index"] = best_init_index

        E_VQE = float(c0 + vqe_out["E_nc_opt"])

        print(f"[SPACE] done E_VQE={E_VQE:.10f} E_CASCI={E_CASCI:.10f} "
              f"diff={E_VQE - E_CASCI:+.2e} cycles={vqe_out['cycles']} "
              f"t={vqe_out['runtime_total']:.1f}s", flush=True)

        molecule_results["active_space_runs"].append({
            "space": space,
            "skipped": False,
            "sizes": {
                "qubits": int(qubit_count),
                "uccsd_num_parameters": expected,
                "cudaq_uccsd_num_parameters": cudaq_uccsd_num_parameters(nele_cas, qubit_count),
                "heavy_mode": bool(is_heavy),
            },
            "active_indices": {"occupied_indices": occ, "active_indices": act},
            "casci": {"E_casci_total": E_CASCI},
            "hamiltonian": {
                "c0": float(c0),
                "num_qubit_terms_nonconstant": int(H.num_terms()),
            },
            "theta0": {
                "source": theta0_source,
                "theta_scale": float(THETA_SCALE),
                "theta0_norm": float(np.linalg.norm(theta0)),
                "E_theta0": E_theta0,
            },
            "seed_search": seed_out,
            "vqe": {
                "E_nc_opt": float(vqe_out["E_nc_opt"]),
                "E_total": E_VQE,
                "theta_opt": np.array(vqe_out["theta_opt"], dtype=float),
                "converged": bool(vqe_out["converged"]),
                "cycles": int(vqe_out["cycles"]),
                "runtime": float(vqe_out["runtime_total"]),
                "simulated_quantum_runtime": float(vqe_out["runtime_quantum_sum"]),
                "simulated_quantum_runtime": float(vqe_out["runtime_quantum_sum"]),
                "exaqt_execute_seconds": float(sum(execute_times[n_exec_before_vqe:])),
                "exaqt_execute_calls": int(len(execute_times) - n_exec_before_vqe),

                "optimizer_runtime": float(vqe_out["runtime_optimizer"]),
                "quantum_times": list(vqe_out["quantum_times"]),
                "energy_convergence": list(vqe_out["energy_convergence"]),
                "best_energy_per_cycle": list(vqe_out["best_energy_per_cycle"]),
                "cycle_summaries": list(vqe_out["cycle_summaries"]),
                "success_any": bool(vqe_out["success"]),
                "message": str(vqe_out["message"]),
                "best_init_index": int(vqe_out["best_init_index"]),
                "nit_total": int(vqe_out["nit"]),
                "nfev_total": int(vqe_out["nfev"]),
            },
            "compare": {
                "d_vqe_minus_casci": float(E_VQE - E_CASCI),
                "d_vqe_minus_hf_full": float(E_VQE - HF_FULL),
                "d_vqe_minus_ccsd_full": (float(E_VQE - E_CCSD_FULL)
                                          if E_CCSD_FULL is not None else None),
            },
        })
        
        if checkpoint_path is not None:
            save_pkl({mol_name: molecule_results}, checkpoint_path)
            print(f"[SAVE ] {len(molecule_results['active_space_runs'])} spaces so far "
                  f"-> {checkpoint_path}", flush=True)

    return molecule_results
                    

# -----------------------------
# CLI, same arguments as the reference
# -----------------------------
def main():
    global BASIS, TARGET, OPTIMIZER, MAX_MEMORY

    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", required=True)
    parser.add_argument("--basis", default=BASIS)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--precision", default="default",
                        choices=["default", "fp32", "fp64"],
                        help="accepted for CLI parity with CUDA-Q; ignored on MIMIQ")
    parser.add_argument("--optimizer", default=OPTIMIZER)
    parser.add_argument("--out_dir", default="pkl_results/mimiq_exaqt")
    parser.add_argument("--space_idx", type=int, default=None)
    parser.add_argument("--integrals", default=None,
                        help="read Hamiltonian, reference energies and CCSD amplitudes "
                             "from this integrals folder instead of running PySCF; "
                             "missing files are made and saved there")
    parser.add_argument("--max-memory", type=int, default=MAX_MEMORY,
                        help="PySCF memory limit in MB when a missing integral file is made")
    args = parser.parse_args()

    BASIS = args.basis
    TARGET = args.target
    OPTIMIZER = args.optimizer
    MAX_MEMORY = args.max_memory

    if args.molecule not in molecules:
        raise ValueError(f"Molecule '{args.molecule}' not found!")

    spec = dict(molecules[args.molecule])
    if args.space_idx is not None:
        spaces = spec.get("valid_active_spaces", [])
        if not 0 <= args.space_idx < len(spaces):
            raise ValueError(f"--space_idx {args.space_idx} out of range (0-{len(spaces) - 1})")
        spec["valid_active_spaces"] = [spaces[args.space_idx]]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[RUN] {args.molecule} | BASIS={BASIS} | TARGET={TARGET} | OPT={OPTIMIZER}", flush=True)
    tag = time.strftime("%d_%b_%Y").upper()
    file_name = (f"{tag}_{sanitize_name(args.molecule)}_{sanitize_name(BASIS)}_"
                 f"{sanitize_name(TARGET)}_{sanitize_name(OPTIMIZER)}_VQE_results.pkl")
    out_path = os.path.join(args.out_dir, file_name)
    checkpoint_path = out_path + ".partial"
    
    mol_res = run_one_molecule(args.molecule, spec, checkpoint_path=checkpoint_path,
                               integrals_dir=args.integrals)
    payload = {args.molecule: mol_res}

    validate_payload(payload)

    save_pkl(payload, out_path)
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print(f"[DONE] Saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
    
    