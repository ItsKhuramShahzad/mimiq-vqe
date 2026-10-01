"""
Spin orbital uccsd (build_uccsd_spin): same parameters count as cuda-q, 
HF at theta =0, and same ccsd start as the singlet ansatz.

"""

import numpy as np
import pytest

from src.integrals import load_integrals, qubit_hamiltonian
from src.mimiq_ansatz import build_uccsd, build_uccsd_spin, singlet_to_spin_params, spin_excitations
from src.mimiq_backend import make_energy_fn
from src.mimiq_hamiltonian import openfermion_to_mimiq_hamiltonian
from src.run_single import cudaq_uccsd_num_parameters, pack_ccsd_singlet

ROOT, BASIS, MOLECULE = 'integrals', 'cc-pVDZ', "Ethylene"
NCORE, NELE, NORB = 6, 4, 4


@pytest.mark.parametrize("n_qubits, nelectrons", [(6, 2), (8, 4), (10, 6), (12, 6), (14, 6)])
def test_parameter_count(n_qubits, nelectrons):
    assert len(spin_excitations(n_qubits, nelectrons)) == cudaq_uccsd_num_parameters(nelectrons, n_qubits)
    
    
def test_excitations_conserve_spin():
    # even qubit = alpha , odd qubit = beta: each electorn keep its spin
    
    for e in spin_excitations(14, 6):
        
        for to , frm in zip(e[0::2], e[1::2]):
            assert to % 2 == frm % 2

def test_wrong_number_of_parameters_refused():
    with pytest.raises(ValueError):
        build_uccsd_spin(8, 4, np.zeros(14))  # 14 is singlet count, spin needs 26
        
@pytest.fixture(scope="module")
def space():
    data = load_integrals(ROOT, BASIS, MOLECULE, NCORE, NELE, NORB)
    H, c0, _ = openfermion_to_mimiq_hamiltonian(qubit_hamiltonian(data))
    packed = pack_ccsd_singlet(data["t1_active"], data["t2_active"])
    return {"data": data, "H": H, "c0": c0, "packed": packed, "n_qubits": 2 *NORB}

def energy(space, ansatz, theta):
    energy_fn, _ = make_energy_fn(space["H"], constant=space["c0"], n_qubits=space["n_qubits"], n_electrons=NELE, ansatz=ansatz)
    return energy_fn(theta)

def test_zero_amplitudes_gives_hf_energy(space):
    n = len(spin_excitations(space["n_qubits"], NELE))
    hf = energy(space, build_uccsd, np.zeros(len(space["packed"])))
    assert energy(space, build_uccsd_spin, np.zeros(n)) == pytest.approx(hf, abs=1e-10)
    
def test_ccsd_start_gives_same_energy(space):
    theta_spin = singlet_to_spin_params(space["packed"], space["n_qubits"], NELE)
    e_singlet = energy(space, build_uccsd, space["packed"])
    e_spin = energy(space, build_uccsd_spin, theta_spin) 
    assert e_spin == pytest.approx(e_singlet, abs=1e-6)
    assert e_spin < energy(space, build_uccsd_spin, np.zeros(len(theta_spin)))  # better than HF
    
    
def test_ccsd_start_is_not_below_casci(space):
    theta_spin = singlet_to_spin_params(space["packed"], space["n_qubits"], NELE)
    e_spin = energy(space, build_uccsd_spin, theta_spin) 
    assert e_spin > float(space["data"]["e_casci"]) - 1e-10  # not below CASCI