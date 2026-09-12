"""
Energy evaluation on MIMIQ via the Exaqt local statevector simulator.

This is te single seam b/w build a circuit at theta and get a number. 

The driver call make_energy_fn once per active space, then hands the retunred energy_fn to scipy.optimize.minimize() to do the optimization."""


import time 
import numpy as np
from mimiq_circuits import Add
from exaqt import ExaqtQCS
from mimiq_openfermion import build_uccsd_singlet_ansatz

def make_energy_fn(mol, H, constant=0.0, n_qubits=None, n_electrons=None):
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
     
    if H.num_qubits is None:
         raise ValueError(f"Hamilttonian spans{H.num_qubits} qubits, but expected {n_qubits} qubits. Please specify n_qubits.")
    if n_electrons is None:
         raise ValueError("Number of electrons not specified. Please provide n_electrons.")
    if constant is None:
         raise ValueError("Constant not specified. Please provide constant.")
    sim =ExaqtQCS(n_qubits) 
     
    quantum_times=[]
    
    