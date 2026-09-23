"""
Energy evaluation on MIMIQ via the Exaqt local statevector simulator.

This is the single seed b/w build a circuit at theta and get a number. 

The driver call make_energy_fn once per active space, then hands the retunred energy_fn to scipy.optimize.minimize() to do the optimization."""


import time 
import numpy as np
from mimiqcircuits import Add
from exaqt import ExaqtQCS
from src.mimiq_ansatz import build_uccsd

def make_energy_fn(H, constant=0.0, n_qubits=None, n_electrons=None):
    """ Build the vqe cost funtion for one molecue and active space.
     Args: 
        H: mimiqcircuits. Hamiotonian for the active space, identity term removed.
        n_qubits: size of qubits register (2* n_active_orbitals)
        n_electrons: number of electrons in the active space, sets HF reference state.
        constant: identity coefficient split out by Hamiltionan bridge.
            Folded back in so return energy is the total energy of the molecule, sireclty comparable to CASCI and FCI
    
    Returns:
        energy_fn: function that takes a vector of parameters and returns the energy expectation value.
        quanutm_times: list of times for each quantum circuit evaluation, useful for profiling. 
        
     """
     
    if H.num_qubits() != n_qubits:
         raise ValueError(f"Hamilttonian spans {H.num_qubits()} qubits, but expected {n_qubits} qubits. Please specify n_qubits.")
    if n_electrons is None:
         raise ValueError("Number of electrons not specified. Please provide n_electrons.")
    if constant is None:
         raise ValueError("Constant not specified. Please provide constant.")
    sim =ExaqtQCS() 
     
    quantum_times=[]
    def energy_fn(theta):
         circ = build_uccsd(
              n_qubits=n_qubits, 
              n_electrons= n_electrons,
              params=list(theta))
         circ = circ.decompose()  # Decompose the circuit into basic gates for simulation     
         circ.push_expval(H, *range(H.num_qubits()))  # Add the expectation value measurement for the Hamiltonian
         if abs(constant)>0.0:
              circ.push(Add(2, c=constant), 0, 0)
              
         t0= time.perf_counter()
         result= sim.execute(circ, nsamples=1)
         
         quantum_times.append(time.perf_counter() - t0)
         
         return float(np.real(result.zstates[0][0]))         
    
    return energy_fn, quantum_times
                               
                               
              
         
    
    