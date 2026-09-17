# mimiq-vqe

VQE for molecular ground state energies on QPerfect MIMIQ. This is a port
of my CUDA-Q VQE benchmark, so both can be compared on the same molecules.

Uses PySCF and OpenFermion for the chemistry, mimiq-openfermion for the
UCCSD ansatz, and the Exaqt simulator.

Status: work in progress. Runs end to end, tested on H2, LiH and Ethylene
(cc-pVDZ, up to 10 qubits so far).

Run one molecule:

    python -m src.run_single --molecule Ethylene --space_idx 5

Tests:

    pytest

Khuram Shahzad
