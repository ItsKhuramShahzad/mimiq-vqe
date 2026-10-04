"""
uccsd singlet ansatz and spin uccsd on mimiq

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
from openfermion import (FermionOperator,  jordan_wigner, normal_ordered, uccsd_generator, uccsd_singlet_generator, uccsd_singlet_paramsize)

from mimiq_openfermion import to_mimiq, hartree_fock_state, hartree_fock_occupation


def push_trotter(circuit, ham, trotter_steps=1, trotter_order=1):
    """Apply exp(-i * ham) = exp(T - T^dagger) to the circuit as Pauli rotations.

    trotter_order 1: Lie-Trotter (push_lietrotter), each Pauli rotation once: one
    first-order step, like the cudaq.kernels.uccsd kernel. The rotations follow the order
    of the generator's Pauli terms, not CUDA-Q's excitation-by-excitation order.
    trotter_order 2, 4, ...: symmetric Suzuki (push_suzukitrotter), each rotation applied
    more than once.
    The target range is sized by the generator, not by n_qubits (the ISSUE-07 fix).
    """
    qubits = tuple(range(ham.num_qubits()))
    if trotter_order == 1:
        circuit.push_lietrotter(ham, qubits, t=1.0, steps=trotter_steps)
    elif trotter_order >= 2 and trotter_order % 2 == 0:
        circuit.push_suzukitrotter(ham, qubits, t=1.0, steps=trotter_steps, order=trotter_order)
    else:
        raise ValueError(f"trotter_order must be 1 or an even number >= 2, got {trotter_order}")
    return circuit


def build_uccsd(n_qubits, n_electrons, params, trotter_steps=1, trotter_order=1):
    """Build a UCCSD singlet ansatz circuit on mimiq.

    Args:
        n_qubits (int): number of qubits
        n_electrons (int): number of electrons
        params (array-like): parameters for the ansatz
        trotter_steps (int): number of Trotter steps
        trotter_order (int): 1 = first order, like CUDA-Q (default); 2, 4, ... = Suzuki

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
        push_trotter(circuit, ham, trotter_steps, trotter_order)

    return circuit

# spin conservation excitations, in cuda-q order: singles alpha | single beta | double mixed| double alpha | double beta.
def spin_excitations(n_qubits, n_electrons):
    """Spin conserving singles and doubles, in CUDA-'s orders:
        singles alpha | single beta | double mixed| double alpha | double beta.
        
    Qubit numbering (Openfermion / Jordan-Wigner) is 0,1,2,...,n_qubits-1.
        spatial orbitals p -> alpha spin orbital 2p, beta spin orbital 2p+1
        e.g orbital 0- > qubits 0 (alpha), 1 (beta), orbital 1 -> qubits 2 (alpha), 3 (beta), etc.

    Hartree-fock fills the lowest orbitals, so for 6 electron in 7 orbitals:
        occupied orbitals 0, 1,2 (qubits 0-5)
        virtual orbitals 3,4,5,6 (qubits 6-13)
    Exach excitation moves electron form occupied to virtual orbital, so: 
        single (a, i) : one electron from occupied orbital i to virtual orbital a
        double (a, i, b, j): two electrons from occupied orbitals i,j to virtual orbitals a,b.
        
        each entry is (a, i) for i-> a, or (a, i, b, j) for i, j -> a,b.
    "Spin conserving": an alpha electron stays alpha , a beta electron stays beta. 
    
    The list is in cuda-q order, so parametes k mean the same excitation:
    as cuda-q parameter k:
            singles alpha | single beta | double mixed| double alpha | double beta.

  
    Args:
        n_qubits (int): number of qubits
        n_electrons (int): number of electrons
    """
    
    nocc, norb = n_electrons//2 , n_qubits//2
    occ, vir = range(nocc), range(nocc, norb)
    
    alpha, beta = (lambda p: 2*p), lambda p: 2*p+1
    
    # singles: one electron from occupied orbital i to virtual orbital a, same spin
    singles = [ (alpha(a), (alpha(i))) for i in occ for a in vir] # alpha -> alpha
    singles +=[ (beta (a), beta(i)) for i in occ for a in vir] # beta -> beta
    
    #mixed doubles: one alpha electron (i-> a) and one beta electron (j-< b)
    # loop order i, j, b , a is the one cuda q uses
    doubles = [(alpha(a), (alpha(i)), beta (b), beta(j)) for i in occ for j in occ for b in vir for a in vir]
    
    # smae-spin doubles: two alpha (then two beta) electrons , i< j -> a < b
    # i< j and a<b so every pair is counted once
    
    for spin in (alpha , beta):
        doubles+= [(spin(a), spin(i), spin(b), spin(j)) 
                   for i in occ for  j in  occ if i< j
                   for a in vir for b in vir if a< b]
    
    return singles+doubles

    
def build_uccsd_spin(n_qubits, n_electrons, params, trotter_steps=1, trotter_order=1):
    """ Spin-orbital uccsd circuit |psi> = exp(T-T^dagger) |HF> with T the spin conserving singles and doubles generator.
    params[k] is the amplitudes of excitation from spin_excitations(n_qubits, n_electrons)[k]
    
    Same steps as build_uccsd: only generator is different, and the number of parameters is different.
    
    """            
    excitations = spin_excitations(n_qubits, n_electrons)
    #one amplitude per excitation, otherwise stop
    params = np.asarray(params, dtype= float)
    if params.shape!= (len(excitations),):
        raise ValueError(f"expected {len(excitations)} amplitudes for n_qubits={n_qubits}, "
                         f"n_electrons={n_electrons}; got shape {params.shape}")
        
    # uccsd_generator wants pairs [excitation, amplitude] , singles and doubles apart
    
    singles = [[list(e), float(t)] for e, t in zip(excitations, params) if len(e) ==2]
    doubles =[[list(e) , float(t)] for e, t in zip(excitations, params) if len(e) ==4]
    
    # fermion operator T-T^dagger (anti-hermitain =True) subtract the dagger part, so the result is anti-hermitian. multiply by i to make it hermitian 
    # T-T^dagger is anti-hermitian, so we multiply by i to make it hermitian
    fop = uccsd_generator (singles, doubles,anti_hermitian=True)
    # map to qubits (jordan_wigner ) T-T^dagger is anti-hermitian, time i make it hermitian
    # it hermitian , which is what mimiq time evolition expects:
    # exp(-i*H*t) with H = i(T-T^dagger), t = 1 gives exp(T-T^dagger) 
    
    ham = to_mimiq(jordan_wigner(fop)  *1j, n_qubits = n_qubits)
    #start for the Hartree-fock state, the lowest n_electron qubits set to 1, the rest to 0
    circuit = mc.Circuit()
    hartree_fock_state(n_qubits, hartree_fock_occupation(n_qubits, n_electrons), circuit= circuit)
    
    # apply exp(T-T^dagger) as Pauli rotations (one Trotter step, first order by default)
    # all zero amplitudes give an empty generator: then the circuit stay HF.
    
    if ham.num_terms()>0:
        # same fix as build_uccsd: size the target range by generator . not by n_qubits
        push_trotter(circuit, ham, trotter_steps, trotter_order)
        
    return circuit

def singlet_to_spin_params(packed, n_qubits, n_electrons):
    """
    CCSD start for the spin orbital ansatz.
    
    we already have ccsd amplitudes packed for the singlet generator
    (packed_ccsd_singlet). instead of writing a new packer, build that singlet generator give off how much of each spin orbital excitation it contian. 
    the results gives exactly the same opertor T-T^dagger, just written with one parameter 
    per spin orbital excitation
    
    NO factotr of 2 on doubles: cuda-q needs x2 only because its kernel applies theta/2. this circuit applies rotaion as written, so no factor of 2 is needed.
    
    """
    # the singlet_generator with ccsd amplitudes  in normal order
    # normal order = a fixed way of writign each term, so terms can be matched

    target = normal_ordered(uccsd_singlet_generator(packed, n_qubits, n_electrons, anti_hermitian=True))
    theta =[]
    
    for e in spin_excitations(n_qubits, n_electrons):
        # this excitation as a fermion operator, a^i or a^i b^j 
        
        term = ((e[0],1), (e[1], 0)) if len(e)== 2 else ((e[0], 1), (e[1], 0), (e[2], 1 ), (e[3], 0))
        
        # normal order it too. reordering creation/annihilation operatoes ca
        # flip the sing, so keep the to undo it below
        
        (key, sign), = normal_ordered(FermionOperator(term)).terms.items()
        
        # cofficient of this term n the singlet generator = its amplitude
        #(0 if the singlet generator deose not contain this spin orbital excitation)
        
        theta.append(float(np.real(target.terms.get(key, 0.0)/sign)))
    
    return np.asarray(theta)