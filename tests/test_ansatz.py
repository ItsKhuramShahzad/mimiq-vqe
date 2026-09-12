import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np
import pytest
from openfermion import (MolecularData, uccsd_singlet_paramsize, uccsd_singlet_get_packed_amplitudes)

from openfermionpyscf import run_pyscf
from openfermion.transforms import jordan_wigner
import mimiq_openfermion
from mimiq_openfermion import build_uccsd_singlet_ansatz
from mimiqcircuits import Add
from exaqt import ExaqtQCS

from scipy.optimize import minimize

from mimiq_hamiltonian import openfermion_to_mimiq_hamiltonian

CEHM_ACC= 1.6E-3 # HATREE

@pytest.fixture(scope="module")
def h2_molecule():
    """ H2/sto-3g molecule at 0.735 Angstrom bond length """
    molecule = MolecularData(
        geometry=[('H', (0, 0, 0)), ('H', (0, 0, 0.735))],
        basis='sto-3g',
        multiplicity=1,
        charge=0
    )
    mol= run_pyscf(molecule, run_scf=True, run_ccsd=True, run_fci=True)
    qop = jordan_wigner(mol.get_molecular_hamiltonian())
    h = mimiq_openfermion.to_mimiq(qop)
    
    return mol, h

def make_energy_function(mol, h):
    """ Returns f(theta) = <0|U(theta)^dagger H U(theta)|0> -> energy 
    Two workarounds are required:
    params must be passed explicitly; params=None does not respect n_qubits
    .decompose() is required, otherwise the simulator raises on composite gates
    """
    sim = ExaqtQCS()
    
    def energy(theta):
        cir, _= build_uccsd_singlet_ansatz(n_qubits=mol.n_qubits, n_electrons=mol.n_electrons, params=list(theta))
        
        cir = cir.decompose()
        cir.push_expval(h, *range(mol.n_qubits))
        # cir.push(Add(2, c=constant), 0,0)
        
        return float(np.real(sim.execute(cir, nsamples=1).zstates[0][0]))
    return energy

def ccsd_seed(mol):
    """ Returns the CCSD amplitudes as a seed for the UCCSD ansatz """    
    return np.asarray(
        uccsd_singlet_get_packed_amplitudes(
            mol.ccsd_single_amps,
            mol.ccsd_double_amps,
            mol.n_qubits,
            mol.n_electrons,
        ),
        dtype=np.float64,
    )

#Test-1: Theta =0 must gives Hartree-Fock energy

def test_T1_zero_params_gives_HF_energy(h2_molecule):
    mol, H= h2_molecule
    energy= make_energy_function(mol, H)
    
    npar= uccsd_singlet_paramsize(mol.n_qubits, mol.n_electrons)
    E= energy(np.zeros(npar))
    print(f"\n T1: E(theta=0) = {E:.6f} Hartree, E(HF) = {mol.hf_energy:.6f} Hartree")
    assert abs(E - mol.hf_energy) < 1e-9, \
        f"Energy at theta=0 ({E:.6f}) does not match Hartree-Fock energy ({mol.hf_energy:.6f})"
    

# Test 4: the CCSD seed must sit below HF energy

def test_T4_ccsd_seed_below_hartree_fock(h2_molecule):
    mol, H= h2_molecule
    energy= make_energy_function(mol, H)
    E_seed= energy(ccsd_seed(mol))
    print(f"\n T4: E(ccsd seed) = {E_seed:.6f} Hartree, E(HF) = {mol.hf_energy:.6f} Hartree")
    assert E_seed < mol.hf_energy, \
        f"CCSD seed energy ({E_seed:.6f}) is not below Hartree-Fock energy ({mol.hf_energy:.6f})"


# Test 2: the gate test:>= 99% correlation energy at the seed

def test_T2_ccsd_seed_recovers_correlation(h2_molecule):
    """ The strongest test. T1 passes for a wrong sign and a wrong packing;
    this dose not happen for T2. The CCSD seed must recover at least 99% of the correlation energy """
    mol, H= h2_molecule
    energy = make_energy_function(mol, H)
    E_seed= energy(ccsd_seed(mol))
    
    corr_total= mol.hf_energy- mol.fci_energy
    corr_seed= mol.hf_energy- E_seed
    corr_ratio= corr_seed/corr_total
    print(f"'\nT2: Correlation energy recovered by ccsd seed: {corr_ratio*100:.2f}% (must be>= 99.00%)")
    assert corr_ratio>= 0.99, f"only {corr_ratio*100:.2f} % recovered (need>= 99.00%)"
    

# test 3: converge VQE must reach FCI energy within chemical accuracy

def test_T3_vqe_converges_to_fci(h2_molecule):
    
    mol, H= h2_molecule
    energy= make_energy_function(mol, H )
    
    calls= {"n": 0}
    
    def counted (theta):
        calls['n']+=1
        return energy(theta)
    result= minimize(counted, ccsd_seed(mol), method= 'COBYLA', options={'maxiter': 1000,  'rhobeg': 0.2})
    err= abs(result.fun- mol.fci_energy)
    print(f"\n T3: VQE converged in {calls['n']} calls to E= {result.fun:.6f} Hartree, FCI= {mol.fci_energy:.6f} Hartree , error= {err:.6f} Hartree")
    assert err <CEHM_ACC, f"VQE did not converge to FCI within chemical accuracy: error= {err:.6f} Hartree" 
    assert result.fun>= mol.fci_energy -1e-9, f"VQE converged to {result.fun:.6f} Hartree, which is below FCI energy {mol.fci_energy:.6f} Hartree"
    
    
@pytest.mark.xfail(strict=True, 
                       reason= "zero-amplitude ansatz collapses to n_electrons qubits"
                       "instead of n_qubits, so the ansatz is not valid for theta=0.")
def test_zero_amplitude_ansatz_has_correct_width(h2_molecule):
    
    """ Regression guard: the builder must respect n_qubits    """
        
    mol, h= h2_molecule
    npar= uccsd_singlet_paramsize(mol.n_qubits,mol.n_electrons)
    circ, _= build_uccsd_singlet_ansatz(n_qubits= mol.n_qubits, n_electrons= mol.n_electrons, params= np.zeros(npar))
    assert circ.n_qubits == mol.n_qubits, f"Zero-amplitude ansatz has {circ.n_qubits} qubits, expected {mol.n_qubits}"    
         