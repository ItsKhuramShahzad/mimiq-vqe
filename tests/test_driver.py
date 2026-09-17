"""
H2 test: seed search + vqe loop should reach the exact energy.
"""

import numpy as np
import pytest
from openfermion import (MolecularData, get_fermion_operator, get_sparse_operator,
                         jordan_wigner, uccsd_singlet_paramsize)
from openfermionpyscf import run_pyscf
from scipy.sparse.linalg import eigsh

from src.mimiq_ansatz import build_uccsd
from src.mimiq_backend import make_energy_fn
from src.mimiq_driver import best_of_jitters_one_chunk, vqe_until_converged
from src.mimiq_hamiltonian import openfermion_to_mimiq_hamiltonian
from src.utils import _stable_hash


@pytest.fixture(scope="module")
def h2():
    geom = [("H", (0, 0, 0)), ("H", (0, 0, 0.74))]
    mol = run_pyscf(MolecularData(geom, "sto-3g", 1, 0), run_scf=True)
    qop = jordan_wigner(get_fermion_operator(mol.get_molecular_hamiltonian()))
    H, c0, _ = openfermion_to_mimiq_hamiltonian(qop)
    E_exact = eigsh(get_sparse_operator(qop, n_qubits=4), k=1, which="SA")[0][0]
    return H, c0, E_exact


def test_sparse_theta_does_not_crash():
    # The library builder raises here; ours must not.
    n = uccsd_singlet_paramsize(8, 4)
    for i in range(n):
        build_uccsd(8, 4, np.eye(n)[i] * 0.1)


def test_driver_reaches_exact_energy(h2):
    H, c0, E_exact = h2
    energy_fn, _ = make_energy_fn(H, constant=0.0, n_qubits=4, n_electrons=2)
    rng = np.random.default_rng(12345 + _stable_hash(("H2", 0, 2, 2)))
    theta0 = np.zeros(uccsd_singlet_paramsize(4, 2))

    seed = best_of_jitters_one_chunk(energy_fn, theta0, rng, n_restarts=3)
    out = vqe_until_converged(energy_fn, seed["theta_opt"], rng)

    E_total = c0 + out["E_nc_opt"]
    assert out["converged"]
    assert abs(E_total - E_exact) < 1e-8
    assert E_total >= E_exact - 1e-10
