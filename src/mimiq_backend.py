"""
Energy evaluation on MIMIQ via the Exaqt local statevector simulator.

This is the single step  b/w building a circuit at theta and get a number. 

The driver call make_energy_fn once per active space, then hands the retunred energy_fn to scipy.optimize.minimize() to do the optimization."""


import time 
import numpy as np
from mimiqcircuits import Add
from exaqt import ExaqtQCS
from src.mimiq_ansatz import build_uccsd


TARGETS= {"exaqt-cpu", "exaqt-gpu"}
def make_energy_fn(H, constant=0.0, n_qubits=None, n_electrons=None, ansatz= build_uccsd, target= "exaqt-cpu"):
    """ Build the vqe cost funtion for one molecue and active space.
     Args: 
        H: mimiqcircuits. Hamiotonian for the active space, identity term removed.
        n_qubits: size of qubits register (2* n_active_orbitals)
        n_electrons: number of electrons in the active space, sets HF reference state.
        constant: identity coefficient split out by Hamiltionan bridge.
            Folded back in so return energy is the total energy of the molecule, sireclty comparable to CASCI and FCI
        ansatz: function taht build the circuit, called as
               ansatz(nqubits, nleectron, parrams=...)
        build_uccsd(singlet, default) or build_uccsd_spin (spin-orbitals
        same parametes as cuda-q.theta must haev ansatz length.)   
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
    sim = select_target(target)
     
    quantum_times=[]
    exaqt_timings=[]
    def energy_fn(theta):
         circ = ansatz(
              n_qubits=n_qubits, 
              n_electrons= n_electrons,
              params=list(theta))
         circ = circ.decompose()  # Decompose the circuit into basic gates for simulation     
         circ.push_expval(H, *range(H.num_qubits()))  # Add the expectation value measurement for the Hamiltonian
         if abs(constant)>0.0:
              circ.push(Add(2, c=constant), 0, 0)              
         result= sim.execute(circ, nsamples=1)
         timings = {k: float(v) for k, v in result.timings.items()}
     
         quantum_times.append(timings['total'])
         exaqt_timings.append(timings)
         
         return float(np.real(result.zstates[0][0]))         
    energy_fn.quantum_times= quantum_times
    energy_fn.exaqt_timings= exaqt_timings
    return energy_fn, quantum_times
                               
def select_target(target: str):
    """Select the target for Exaqt simulation.

    Args:
        target (str): The target to select. Must be one of "exaqt-cpu = ExaqtQCS" or "exaqt-gpu = ExaqtQCSGpu (cyStateVec)".

    Raises:
        ValueError: If the target is not one of the allowed values.
    """
    if target =="exaqt-cpu":
        return ExaqtQCS()
    if target =="exaqt-gpu":
        import exaqt
        if not exaqt.gpu_available():
            raise ValueError("Exaqt GPU target selected, but no GPU is available.")
        from exaqt import ExaqtQCSGpu
        return ExaqtQCSGpu()
    raise ValueError(f"Invalid target '{target}'. Must be one of {TARGETS}.")
              
         
    
    