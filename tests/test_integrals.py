"""
Integrals saved to .npz must give the same Hamiltonian as the normal
geometry -> PySCF -> OpenFermion route, and VQE from the file alone
must reach the stored CASCI energy.
"""

import numpy as np
import pytest
from openfermion import MolecularData, get_fermion_operator, jordan_wigner, uccsd_singlet_paramsize
from openfermionpyscf import run_pyscf

from config.molecules_data import molecules
from src.integrals import (compute_active_space, load_active_space, qubit_hamiltonian,
                           run_scf, save_active_space)
from src.mimiq_backend import make_energy_fn
from src.mimiq_driver import vqe_until_converged
from src.mimiq_hamiltonian import openfermion_to_mimiq_hamiltonian

BASIS = "sto-3g"
NAME = "Ethylene"
NCORE, NELE, NORB = 5, 6, 4


@pytest.fixture(scope="module")
def saved_file(tmp_path_factory):
    entry = molecules[NAME]
    mf = run_scf(entry, BASIS)
    data = compute_active_space(mf, NCORE, NELE, NORB)
    path = tmp_path_factory.mktemp("integrals") / "space.npz"
    save_active_space(path, data, molecule=NAME, basis=BASIS)
    return path


def test_file_matches_pipeline_hamiltonian(saved_file):
    q_file = qubit_hamiltonian(load_active_space(saved_file))

    entry = molecules[NAME]
    mol = run_pyscf(MolecularData(entry["geometry"], BASIS, 1, 0), run_scf=True)
    q_pipe = jordan_wigner(get_fermion_operator(mol.get_molecular_hamiltonian(
        occupied_indices=list(range(NCORE)), active_indices=list(range(NCORE, NCORE + NORB)))))

    keys = set(q_file.terms) | set(q_pipe.terms)
    diff = max(abs(q_file.terms.get(k, 0) - q_pipe.terms.get(k, 0)) for k in keys)
    assert diff < 1e-10


def test_vqe_from_file_alone(saved_file):
    data = load_active_space(saved_file)
    H, c0, _ = openfermion_to_mimiq_hamiltonian(qubit_hamiltonian(data))
    energy_fn, _ = make_energy_fn(H, constant=0.0, n_qubits=2 * NORB, n_electrons=NELE)
    theta0 = np.zeros(uccsd_singlet_paramsize(2 * NORB, NELE))

    assert abs(c0 + energy_fn(theta0) - data["e_hf"]) < 1e-10

    out = vqe_until_converged(energy_fn, theta0, np.random.default_rng(0))
    assert abs(c0 + out["E_nc_opt"] - data["e_casci"]) < 1e-8


def test_rejects_wrong_electron_count(saved_file, tmp_path):
    data = dict(np.load(saved_file, allow_pickle=True))
    data["ncore"] = NCORE - 1          # the mistake in the old integral files
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **data)
    with pytest.raises(ValueError, match="n_electrons"):
        load_active_space(bad)


def test_rejects_space_that_does_not_fit():
    mf = run_scf(molecules[NAME], BASIS)
    with pytest.raises(ValueError, match="electrons"):
        compute_active_space(mf, NCORE - 1, NELE, NORB)
