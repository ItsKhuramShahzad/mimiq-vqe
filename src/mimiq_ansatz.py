"""
uccsd singlet ansatz on mimiq

same construction on mimiq_openfermion.build_uccsd_singlet_ansatz
with one fix. 

The library passes range(n_qubits) to push_suzukitrotter, but to_mimiq()
 sizes the generator by heigest qubits it actually touches. 
 
 When the non zero apmlitudes do not reach the top qubit, the two disagree aand it raises "number of qubitsdose not match Hamiltonian". that happen for 
 sparse theta which coblya prodces on it first steps and ccsd seed contian by symmetry. 
 
 Verified: identical energies to the library buider (|dE|= 0.0) whreveer the library works, and no faliurs on 199 single amplitudes cases up to 14 qubits

"""


import numpy as np
import mimiqcircuits as mc
from openfermion import jordan_wigner, uccsd_singlet_generator, uccsd_singlet_paramsize

from mimiq_openfermion import to_mimiq, hartree_fock_state, hartree_fock_occupation


def build_uccsd(n_qubits, n_electrons, params, trotter_steps=1, trotter_order=2):
    """Build a UCCSD singlet ansatz circuit on mimiq.

    Args:
        n_qubits (int): number of qubits
        n_electrons (int): number of electrons
        params (array-like): parameters for the ansatz
        trotter_steps (int): number of Trotter steps
        trotter_order (int): order of the Trotter expansion

    Returns:
        mc.Circuit: the constructed UCCSD singlet ansatz circuit
    """
    params = np.asarray(params, dtype=float)
    n_params = uccsd_singlet_paramsize(n_qubits, n_electrons)
    if params.shape != (n_params,):
        raise ValueError(
            f"expected {n_params} amplitudes for n_qubits={n_qubits}, "
            f"n_electrons={n_electrons}; got shape {params.shape}"
        )
    fop = uccsd_singlet_generator(params, n_qubits, n_electrons, anti_hermitian=True)
    
    # T - T&=^dagger is anti-hermitian: times i gives a Hermitian henerator
    ham = to_mimiq(jordan_wigner(fop) *1j, n_qubits=n_qubits)
    
    circuit = mc.Circuit()
    
    hartree_fock_state(
        n_qubits, hartree_fock_occupation(n_qubits, n_electrons), circuit= circuit
        
    )
        
    if ham.num_terms()>0:
        # the fix, size the target range by generator . not by n_qubits
        circuit.push_suzukitrotter(ham, tuple(range(ham.num_qubits())), t=1.0, steps= trotter_steps, order=trotter_order)

    return circuit