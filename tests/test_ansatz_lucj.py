"""
LUCJ ansatz (src/mimiq_lucj.py): theta = 0 gives Hartree-Fock, the MIMIQ circuit gives the same
energy as ffsim's exact simulation, the parameter count matches ffsim, and the CCSD start is
below Hartree-Fock. Skipped where ffsim is not installed.
"""

import numpy as np
import pytest

ffsim = pytest.importorskip("ffsim")

from src import mimiq_lucj as lucj
from src.integrals import load_integrals
from src.mimiq_backend import make_energy_fn
from src.mimiq_hamiltonian import openfermion_to_mimiq_hamiltonian

ROOT, BASIS, MOLECULE = "integrals", "cc-pVDZ", "Ethylene"
NCORE, NELE, NORB = 6, 4, 4          # 8 qubits: quick
N_REPS = 2


@pytest.fixture(scope="module")
def space():
    data = load_integrals(ROOT, BASIS, MOLECULE, NCORE, NELE, NORB)
    H, c0, _ = openfermion_to_mimiq_hamiltonian(lucj.qubit_hamiltonian_alpha_then_beta(data))
    nelec = (NELE // 2, NELE // 2)
    H_ffsim = ffsim.linear_operator(
        ffsim.MolecularHamiltonian(data["h1"], data["eri"], constant=float(data["e_core"])),
        norb=NORB, nelec=nelec)
    return {"data": data, "H": H, "c0": c0, "H_ffsim": H_ffsim, "nelec": nelec}


def mimiq_energy(space, params, pairs_kind):
    builder = lambda n_qubits, n_electrons, params: lucj.build_lucj(
        n_qubits, n_electrons, params, n_reps=N_REPS, pairs_kind=pairs_kind)
    energy_fn, _ = make_energy_fn(space["H"], space["c0"], n_qubits=2 * NORB,
                                  n_electrons=NELE, ansatz=builder)
    return energy_fn(params)


def ffsim_energy(space, params, pairs_kind):
    op = ffsim.UCJOpSpinBalanced.from_parameters(
        np.asarray(params), norb=NORB, n_reps=N_REPS,
        interaction_pairs=lucj.lucj_pairs(NORB, pairs_kind), with_final_orbital_rotation=True)
    vec = ffsim.apply_unitary(ffsim.hartree_fock_state(NORB, space["nelec"]), op,
                              norb=NORB, nelec=space["nelec"])
    return float(np.real(np.vdot(vec, space["H_ffsim"] @ vec)))


@pytest.mark.parametrize("pairs_kind", ["local", "full"])
def test_parameter_count_matches_ffsim(space, pairs_kind):
    d = space["data"]
    theta0 = lucj.lucj_start(d["t1_active"], d["t2_active"], N_REPS, pairs_kind)
    assert len(theta0) == lucj.lucj_num_parameters(NORB, N_REPS, pairs_kind)


@pytest.mark.parametrize("pairs_kind", ["local", "full"])
def test_zero_parameters_give_hartree_fock(space, pairs_kind):
    n = lucj.lucj_num_parameters(NORB, N_REPS, pairs_kind)
    assert mimiq_energy(space, np.zeros(n), pairs_kind) == pytest.approx(
        float(space["data"]["e_hf"]), abs=1e-10)


@pytest.mark.parametrize("pairs_kind", ["local", "full"])
def test_mimiq_equals_ffsim(space, pairs_kind):
    d = space["data"]
    theta0 = lucj.lucj_start(d["t1_active"], d["t2_active"], N_REPS, pairs_kind)
    rng = np.random.default_rng(1)
    for params in (theta0, theta0 + 0.1 * rng.standard_normal(len(theta0))):
        assert mimiq_energy(space, params, pairs_kind) == pytest.approx(
            ffsim_energy(space, params, pairs_kind), abs=1e-8)


def test_ccsd_start_is_below_hartree_fock(space):
    d = space["data"]
    theta0 = lucj.lucj_start(d["t1_active"], d["t2_active"], N_REPS, "local")
    e0 = mimiq_energy(space, theta0, "local")
    assert float(d["e_casci"]) - 1e-10 <= e0 < float(d["e_hf"])
