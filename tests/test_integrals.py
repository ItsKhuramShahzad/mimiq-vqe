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


# ---------------------------------------------------------------------------
# CCSD amplitudes stored with the integrals (the VQE starting point)
# ---------------------------------------------------------------------------

from openfermion import uccsd_singlet_paramsize as _paramsize
from src.integrals import run_ccsd, slice_ccsd_to_active
from src.run_single import pack_ccsd_singlet


@pytest.fixture(scope="module")
def saved_file_ccsd(tmp_path_factory):
    mf = run_scf(molecules[NAME], BASIS)
    ccsd = run_ccsd(mf)
    data = compute_active_space(mf, NCORE, NELE, NORB, ccsd=ccsd)
    path = tmp_path_factory.mktemp("integrals_ccsd") / "space.npz"
    save_active_space(path, data, molecule=NAME, basis=BASIS)
    return path, mf, ccsd


def test_amplitudes_have_active_space_shape(saved_file_ccsd):
    path, _, _ = saved_file_ccsd
    data = load_active_space(path)
    nocc_act, nvir_act = NELE // 2, NORB - NELE // 2
    assert data["t1_active"].shape == (nocc_act, nvir_act)
    assert data["t2_active"].shape == (nocc_act, nocc_act, nvir_act, nvir_act)
    assert data["e_ccsd"] < data["e_hf"]


def test_amplitudes_equal_full_ccsd_sliced(saved_file_ccsd):
    """The stored amplitudes are exactly the full-molecule CCSD cut to the active space."""
    path, mf, (_, t1, t2) = saved_file_ccsd
    data = load_active_space(path)
    nocc = mf.mol.nelectron // 2
    t1a, t2a = slice_ccsd_to_active(t1, t2, nocc, list(range(NCORE, NCORE + NORB)))
    assert np.array_equal(data["t1_active"], t1a)
    assert np.array_equal(data["t2_active"], t2a)


def test_seed_from_file_matches_run_single(saved_file_ccsd):
    """Starting point from the file = starting point run_single.py builds today."""
    path, mf, (_, t1, t2) = saved_file_ccsd
    data = load_active_space(path)

    theta_file = pack_ccsd_singlet(data["t1_active"], data["t2_active"])
    assert len(theta_file) == _paramsize(2 * NORB, NELE)

    from src.run_single import slice_ccsd_to_active as rs_slice
    nocc = mf.mol.nelectron // 2
    theta_run_single = pack_ccsd_singlet(*rs_slice(t1, t2, nocc, list(range(NCORE, NCORE + NORB))))
    assert np.array_equal(theta_file, theta_run_single)

    H, c0, _ = openfermion_to_mimiq_hamiltonian(qubit_hamiltonian(data))
    energy_fn, _ = make_energy_fn(H, constant=0.0, n_qubits=2 * NORB, n_electrons=NELE)
    e_seed = c0 + energy_fn(theta_file)
    assert data["e_casci"] - 1e-10 < e_seed < data["e_hf"]


def test_rejects_wrong_amplitude_shape(saved_file_ccsd, tmp_path):
    path, _, _ = saved_file_ccsd
    data = dict(np.load(path, allow_pickle=True))
    data["t1_active"] = data["t1_active"][:, :0]
    bad = tmp_path / "bad_ccsd.npz"
    np.savez_compressed(bad, **data)
    with pytest.raises(ValueError, match="amplitude"):
        load_active_space(bad)


# ---------------------------------------------------------------------------
# run_single.py reading the files instead of running PySCF (--integrals)
# ---------------------------------------------------------------------------

import os

import src.run_single as run_single
from src.integrals import find_integral_file, load_integrals


@pytest.fixture(scope="module")
def integral_tree(tmp_path_factory):
    """A small <root>/<basis>/<molecule>/ tree, laid out as the dump script writes it."""
    root = tmp_path_factory.mktemp("tree")
    mol_dir = root / BASIS / NAME
    mol_dir.mkdir(parents=True)
    mf = run_scf(molecules[NAME], BASIS)
    ccsd = run_ccsd(mf)
    data = compute_active_space(mf, NCORE, NELE, NORB, ccsd=ccsd)
    save_active_space(mol_dir / f"space_01_ncore_{NCORE}_nele_{NELE}_norb_{NORB}.npz",
                      data, molecule=NAME, basis=BASIS)
    return str(root)


def test_load_integrals_finds_the_file(integral_tree):
    data = load_integrals(integral_tree, BASIS, NAME, NCORE, NELE, NORB)
    assert os.path.basename(data["path"]).startswith("space_01_")
    assert (data["ncore"], data["nele_cas"], data["norb_cas"]) == (NCORE, NELE, NORB)
    assert "t1_active" in data


def test_load_integrals_complains_when_the_space_is_missing(integral_tree):
    with pytest.raises(FileNotFoundError):
        find_integral_file(integral_tree, BASIS, NAME, NCORE, NELE, NORB + 1)


def test_run_single_from_files_matches_geometry(integral_tree, monkeypatch):
    """--integrals must give the same energies as the geometry route, with no SCF."""
    monkeypatch.setattr(run_single, "BASIS", BASIS)
    spec = dict(molecules[NAME])
    spec["valid_active_spaces"] = [{"ncore": NCORE, "nele_cas": NELE, "norb_cas": NORB}]

    from_geom = run_single.run_one_molecule(NAME, spec)
    from_file = run_single.run_one_molecule(NAME, spec, integrals_dir=integral_tree)

    g = from_geom["active_space_runs"][0]
    f = from_file["active_space_runs"][0]
    assert abs(g["theta0"]["E_theta0"] - f["theta0"]["E_theta0"]) < 1e-9
    assert abs(g["casci"]["E_casci_total"] - f["casci"]["E_casci_total"]) < 1e-9
    assert abs(g["vqe"]["E_total"] - f["vqe"]["E_total"]) < 1e-6
    assert abs(from_geom["references"]["E_hf_full"] - from_file["references"]["E_hf_full"]) < 1e-9
    assert "from integral file" in f["theta0"]["source"]
    assert from_file["timing"]["pyscf_run_scf_seconds"] == 0.0
