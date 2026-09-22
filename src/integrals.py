"""
Active-space integrals: compute them once, save to .npz, build the Hamiltonian
from the file alone.

compute_active_space() uses PySCF CASCI (get_h1eff, get_h2eff), so memory is
norb_cas^4 instead of the two full nmo^4 copies that
openfermionpyscf.compute_integrals() makes. Same Hamiltonian, checked to 1e-13.

Each file stores the integrals plus e_hf and e_casci, so a file can be checked
against the molecule it claims to be.
"""

import numpy as np
import pyscf
from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner
from openfermion.chem.molecular_data import spinorb_from_spatial
from pyscf import ao2mo, fci, gto, mcscf, scf

REQUIRED_KEYS = {"ncore", "nele_cas", "norb_cas", "n_electrons",
                 "e_core", "h1", "eri", "e_hf", "e_casci"}


def run_scf(entry, basis, max_memory=16000):
    mol = gto.M(atom=entry["geometry"], basis=basis, charge=int(entry["charge"]),
                spin=int(entry["multiplicity"]) - 1, max_memory=max_memory, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    return mf


def compute_active_space(mf, ncore, nele_cas, norb_cas):
    """Active-space integrals and CASCI energy for one active space."""
    nelec = mf.mol.nelectron
    nmo = mf.mo_coeff.shape[1]
    if 2 * ncore + nele_cas != nelec:
        raise ValueError(f"2*ncore+nele_cas = {2 * ncore + nele_cas}, "
                         f"but the molecule has {nelec} electrons")
    if ncore + norb_cas > nmo:
        raise ValueError(f"needs {ncore + norb_cas} orbitals, basis has {nmo}")

    cas = mcscf.CASCI(mf, norb_cas, nele_cas)
    cas.ncore = ncore
    cas.verbose = 0
    h1, e_core = cas.get_h1eff()
    eri = ao2mo.restore(1, cas.get_h2eff(), norb_cas)
    e_casci = float(cas.kernel()[0])

    # the integrals must give back the CASCI energy
    e_check = fci.direct_spin1.kernel(h1, eri, norb_cas, nele_cas)[0] + e_core
    if abs(e_check - e_casci) > 1e-8:
        raise RuntimeError(f"integrals give {e_check}, CASCI gave {e_casci}")

    return dict(ncore=ncore, nele_cas=nele_cas, norb_cas=norb_cas,
                n_electrons=nelec, nmo=nmo, e_core=float(e_core), h1=h1, eri=eri,
                e_hf=float(mf.e_tot), e_casci=e_casci, pyscf_version=pyscf.__version__)


def save_active_space(path, data, **metadata):
    np.savez_compressed(path, **data, **metadata)


def load_active_space(path):
    """Load a file and check it is consistent before returning it."""
    z = np.load(path, allow_pickle=True)
    missing = REQUIRED_KEYS - set(z.files)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    data = {k: z[k] for k in z.files}
    for k in ("ncore", "nele_cas", "norb_cas", "n_electrons"):
        data[k] = int(data[k])
    for k in ("e_core", "e_hf", "e_casci"):
        data[k] = float(data[k])

    if 2 * data["ncore"] + data["nele_cas"] != data["n_electrons"]:
        raise ValueError(f"{path}: 2*ncore+nele_cas does not equal n_electrons")
    n = data["norb_cas"]
    if data["h1"].shape != (n, n) or data["eri"].shape != (n, n, n, n):
        raise ValueError(f"{path}: integral shapes do not match norb_cas={n}")
    return data


def qubit_hamiltonian(data):
    """Jordan-Wigner qubit Hamiltonian from the integrals alone."""
    one, two = spinorb_from_spatial(
        data["h1"], np.asarray(data["eri"].transpose(0, 2, 3, 1), order="C"))
    op = InteractionOperator(data["e_core"], one, 0.5 * two)
    return jordan_wigner(get_fermion_operator(op))
