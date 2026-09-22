"""Open Fermion Qubit Operator to -> MIMIQ Hmamiltonian conversion functions."""

# import re


def openfermion_to_mimiq_hamiltonian(qubit_op, tol=1e-12, imag_tol=1e-6):
    """
    Convert an OpenFermion QubitOperator to a MIMIQ Hamiltonian.

    Args:
        qubit_op (QubitOperator): The input OpenFermion QubitOperator.
        tol (float): Tolerance for filtering out small coefficients.
        imag_tol (float): Tolerance for filtering out imaginary parts of coefficients.

    Returns:
        Hamiltonian: A MIMIQ Hamiltonian object.
    """
    from mimiqcircuits import Hamiltonian , PauliString

    H = Hamiltonian()
    constant = 0.0
    n_kept= 0
    n_dropped = 0

    max_weight=0; 
    weights={}
    
    for term, coeff in qubit_op.terms.items():
        c = complex(coeff)
        if abs(c.imag) > imag_tol:
            raise ValueError(f"Imaginary part of coefficient {c} for term {term} exceeds tolerance {imag_tol}.")
        
        c = float(c.real)
        if  len(term) == 0:  # Identity -> constant term, scalar offset
            constant += c
            continue
        if abs(c) < tol:
            n_dropped += 1
            continue
        # Sort by qubit index -> compact Pauli string, not identity padded
        idx_ops= sorted(term, key=lambda t: t[0])  
        qubits= tuple(q for q, p in idx_ops)
        paulis= "".join(p for _, p in idx_ops)
        
        H.push(c, PauliString(paulis), *qubits)
        n_kept += 1
        weight = len(term)
        max_weight = max(max_weight, weight)
        weights[weight] = weights.get(weight, 0) + 1
    stats = {
        "n_terms_kept": n_kept,
        "n_terms_dropped": n_dropped,
        "constant": constant,
        "num_qubits": H.num_qubits(),
        "max_pauli_weight": max_weight,
        "weight_histogram": dict(sorted(weights.items())),
    }
    return H, constant, stats

def attach_cost(circuit, H, constant=0.0, zreg=0):
    """
    Attach a cost function to a MIMIQ circuit based on a Hamiltonian.

    Args:
        circuit (Circuit): The MIMIQ circuit to which the cost function will be attached.
        H (Hamiltonian): The MIMIQ Hamiltonian representing the cost function.
        constant (float): A constant term to be added to the cost function.
        zreg (int): The index of the Z register in the circuit.

    Returns:
        Circuit: The modified circuit with the attached cost function.
    """
    from mimiqcircuits import Add
    nq= H.num_qubits()
    circuit.push_expval(H, *range(nq))
    if abs(constant) > 0:
        circuit.push(Add(2, c=constant), zreg, zreg)
    return circuit