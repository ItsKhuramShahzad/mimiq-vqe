"""
LUCJ ansatz (local unitary cluster Jastrow) on MIMIQ, built with ffsim.

    |psi(theta)> = prod_k  U_k exp(i J_k) U_k^dagger  |HF>        k = 1 .. n_reps

U_k are orbital rotations (Givens rotations, XX+YY gates) and J_k diagonal Coulomb
(Jastrow) operators (controlled-phase and phase gates). "local" lets J_k couple only
neighbouring orbitals: alpha-alpha (p, p+1) and alpha-beta (p, p), as in the SQD work;
"full" couples every pair (UCJ). Spin-balanced: J_aa = J_bb and J_ab = J_ba.

The start comes from the CCSD amplitudes of the integral file
(ffsim.UCJOpSpinBalanced.from_t_amplitudes): t2 is factorised into the U_k and J_k, t1 gives
a final orbital rotation. The circuit is ffsim's Qiskit circuit, decomposed into x,
xx_plus_yy, cp and p and converted into a MIMIQ circuit with QPerfect's converter
(mimiq_qiskit.qiskit_to_mimiq); there is no Trotter step.

Qubit order: ffsim puts all alpha orbitals first (qubits 0 .. norb-1), then all beta
(norb .. 2 norb-1). The Hamiltonian must be built in that order (qubit_hamiltonian_alpha_then_beta),
not in the interleaved order the UCCSD ansatz uses.

ffsim and mimiq_qiskit are imported here only, so runs without --ansatz lucj do not need them.
"""

import numpy as np
import ffsim
import mimiq_qiskit
from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner, reorder
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.utils import up_then_down
from qiskit import QuantumCircuit

PAIR_KINDS = ("local", "full")


def alpha_then_beta(fermion_op, norb):
    """Reorder a FermionOperator from interleaved spin orbitals (2p = alpha, 2p+1 = beta)
    to ffsim's order: alpha 0 .. norb-1, then beta norb .. 2 norb-1."""
    return reorder(fermion_op, up_then_down, num_modes=2 * norb)


def qubit_hamiltonian_alpha_then_beta(data):
    """Jordan-Wigner Hamiltonian from the integral file, qubits in ffsim's order.
    The same operator as src.integrals.qubit_hamiltonian, only the qubits renumbered."""
    norb = data["h1"].shape[0]
    one, two = spinorb_from_spatial(
        data["h1"], np.asarray(data["eri"].transpose(0, 2, 3, 1), order="C"))
    fop = get_fermion_operator(InteractionOperator(data["e_core"], one, 0.5 * two))
    return jordan_wigner(alpha_then_beta(fop, norb))


def lucj_pairs(norb, kind="local"):
    """Interaction pairs: "local" = alpha-alpha (p, p+1), alpha-beta (p, p); "full" = all (None)."""
    if kind == "full":
        return None
    if kind == "local":
        return ([(p, p + 1) for p in range(norb - 1)], [(p, p) for p in range(norb)])
    raise ValueError(f"unknown LUCJ pairs {kind!r}; use one of {PAIR_KINDS}")


def lucj_num_parameters(norb, n_reps, pairs_kind="local"):
    """Number of LUCJ parameters, including the final orbital rotation from t1."""
    return ffsim.UCJOpSpinBalanced.n_params(
        norb, n_reps, interaction_pairs=lucj_pairs(norb, pairs_kind),
        with_final_orbital_rotation=True)


def lucj_start(t1, t2, n_reps, pairs_kind="local"):
    """CCSD start: the LUCJ parameters from the active-space t1 and t2 (plain factorisation)."""
    norb = t1.shape[0] + t1.shape[1]
    pairs = lucj_pairs(norb, pairs_kind)
    op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, t1=t1, n_reps=n_reps, interaction_pairs=pairs)
    # the pairs must be given here too, or the vector has the entries of every pair
    return op.to_parameters(interaction_pairs=pairs)


def build_lucj(n_qubits, n_electrons, params, n_reps=2, pairs_kind="local"):
    """MIMIQ circuit: Hartree-Fock state, then the LUCJ operator with these parameters.
    Same call as build_uccsd, so make_energy_fn(..., ansatz=partial(build_lucj, ...)) works."""
    norb = n_qubits // 2
    nelec = (n_electrons // 2, n_electrons // 2)
    op = ffsim.UCJOpSpinBalanced.from_parameters(
        np.asarray(params, dtype=float), norb=norb, n_reps=n_reps,
        interaction_pairs=lucj_pairs(norb, pairs_kind), with_final_orbital_rotation=True)
    qc = QuantumCircuit(n_qubits)
    qc.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), range(n_qubits))
    qc.append(ffsim.qiskit.UCJOpSpinBalancedJW(op), range(n_qubits))
    return mimiq_qiskit.qiskit_to_mimiq(qc.decompose(reps=2))
