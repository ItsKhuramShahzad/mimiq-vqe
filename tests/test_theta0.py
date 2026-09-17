"""
The ccsd seed packer: pack_uccsd_singlet mutst put each amplitude in the slot Openfermion's 
uccsd_singlet_generator expects. The packed amplitudes are then used to build the uccsd_singlet_generator operator, which is then exponentiated to get the uccsd_singlet ansatz circuit.
The packed amplitudes are also used to build the uccsd_singlet ansatz circuit

A wrong packing raise no error, if just start vqe from a bad gueses, so these test check the mapping direclty rather than checking vqe convergences
"""


import itertools
import numpy as np
import pytest
from openfermion import (FermionOperator, MolecularData, get_fermion_operator, hermitian_conjugated, jordan_wigner, normal_ordered, uccsd_singlet_generator, uccsd_singlet_paramsize)

from openfermionpyscf import run_pyscf

from pyscf import cc

from src.mimiq_backend import make_energy_fn
from src.mimiq_hamiltonian import openfermion_to_mimiq_hamiltonian
from src.run_single import pack_ccsd_singlet, slice_ccsd_to_active

SIZES= [(1,1), (2,2), (2,3), (3,3)] 
def _random_rccsd_amplitudes(nocc, nvir, seed=0):
    """
    Random amplitudes with RCCSD symmetry t2[i, j, a, b] = t2[j,i,b,a].
    """
    rng = np.random.default_rng(seed)
    t1= rng.normal(size = (nocc, nvir))
    t2= rng.normal(size= (nocc, nocc, nvir, nvir))
    
    return t1, 0.5*(t2+t2.transpose(1,0,3,2))  # enforce t2[i,j,a,b] = t2[j,i,b,a]


def _ccsd_cluster_operator(t1_active, t2_active):
    """
    T-T^dagger for restricted ccsd amplitudes, with the same ordering as uccsd_singlet_generator expects. 
    """
    nocc, nvir= t1_active.shape
    
    T= FermionOperator()
    
    for i in range(nocc):
        for a in range(nvir):
            A= nocc+a
            for s in (0,1):  # spin up and down
                T+= FermionOperator(((2*A+s, 1), (2*i+s, 0)), t1_active[i,a])
                
    for i , j in itertools.product(range(nocc), repeat =2):
        for a, b in itertools.product(range(nvir), repeat=2):
            A, B = nocc+a, nocc+b
            for s, t in itertools.product((0,1), repeat=2):
                T+= FermionOperator(((2*A+s, 1), (2*B+t, 1), (2*j+t, 0), (2*i+s, 0)), 0.5*t2_active[i,j,a,b])
    
    return normal_ordered(T- hermitian_conjugated(T))  # T-T^dagger


@pytest.mark.parametrize("nocc, nvir", SIZES)

def test_pack_uccsd_singlet(nocc, nvir):
    t1, t2= _random_rccsd_amplitudes(nocc, nvir)
    n_qubits, n_electron= 2*(nocc+nvir), 2*nocc
    assert len(pack_ccsd_singlet(t1,t2)) == uccsd_singlet_paramsize(n_qubits, n_electron)
    
@pytest.mark.parametrize("nocc, nvir", SIZES)

def test_packed_generator_equals_ccsd_operator(nocc, nvir):
    
    """
    Packed parameters should give back the same T - T^dagger.
    """
    t1, t2= _random_rccsd_amplitudes(nocc, nvir)
    n_qubits, n_electron = 2*(nocc+nvir), 2*nocc
    
    packed = pack_ccsd_singlet(t1, t2   )
    built = normal_ordered(uccsd_singlet_generator(packed, n_qubits, n_electron, True))
    target = _ccsd_cluster_operator(t1, t2)
    
    diff = built - target
    
    worst = max((abs(c) for c in diff.terms.values()), default=0.0)
    scale = max((abs(c) for c in target.terms.values()), default = 1.0)
    assert worst/scale < 1e-15, f"worst relative error {worst/scale} for nocc={nocc}, nvir={nvir}"  
    

def test_scale_is_linear():
    """
    The packed amplitudes must scale linearly with the input amplitudes.
    """
    t1, t2= _random_rccsd_amplitudes(2, 2)
    assert np.allclose(pack_ccsd_singlet(t1, t2, scale =0.5), 
                       0.5* pack_ccsd_singlet(t1,t2))
    
    assert np.allclose(pack_ccsd_singlet(t1, t2, scale=0.0), 0.0)

def test_seed_lower_energy_on_lih():
    """
    On a real molecule, the ccsd seed must be well below Hartree fock energy, and close to the ccsd energy. 
    """
    
    geometry = [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.6))]
    mol = run_pyscf(MolecularData(geometry, "sto-3g", 1, 0), run_scf=True)
    mf = mol._pyscf_data["scf"]
    _, t1, t2 = cc.CCSD(mf).kernel()
    
    
    ncore, norb_cas, nele_cas = 1, 3, 2
    act = list(range(ncore, ncore+norb_cas))
    
    nocc= mf.mol.nelectron//2
    
    qop = jordan_wigner(get_fermion_operator(mol.get_molecular_hamiltonian(occupied_indices = list(range(ncore)), active_indices = act)))
    
    H, c0, _ = openfermion_to_mimiq_hamiltonian(qop)
    
    theta0= pack_ccsd_singlet(*slice_ccsd_to_active(t1, t2, nocc, act))
    energy, _ = make_energy_fn(H, constant=c0, n_qubits=2*norb_cas, n_electrons=nele_cas)
    
    E_hf= energy(np.zeros_like(theta0))
    E_seed = energy(theta0)
    E_flipped = energy(-theta0)
    
    assert E_seed < E_hf, "CSSD seed is not below Hsrtree-Fock"
    assert E_flipped > E_hf, "sign flipped seed should be worsel check the sign convention"