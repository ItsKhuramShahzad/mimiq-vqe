# MIMIQ VQE

Molecular ground-state energies with the Variational Quantum Eigensolver (VQE),
running on [QPerfect MIMIQ](https://qperfect.io).

This project ports a molecular VQE benchmark originally written for NVIDIA CUDA-Q
onto MIMIQ, so that the same set of molecules, the same active spaces and the same
optimisation settings can be run on both platforms and compared directly.

## Why

VQE results depend heavily on the simulator underneath: how the Hamiltonian is
represented, how expectation values are evaluated, how the ansatz is compiled, and
how all of that scales with qubit count. Running an identical workload on two
independent stacks separates what is physics from what is implementation.

The CUDA-Q version of this benchmark already exists and produces reference numbers.
This repository is the MIMIQ side of that comparison.

## Method

For each molecule and each active space:

1. **Mean field.** Restricted Hartree-Fock with PySCF in the `cc-pVDZ` basis.
2. **Active space.** A CAS is selected around the Fermi level, freezing `ncore`
   orbitals and correlating `nele_cas` electrons in `norb_cas` orbitals.
3. **Second quantisation.** The active-space one- and two-electron integrals are
   transformed into a fermionic Hamiltonian with OpenFermion.
4. **Qubit mapping.** Jordan-Wigner, giving `2 x norb_cas` qubits.
5. **Conversion.** The OpenFermion `QubitOperator` becomes a MIMIQ `Hamiltonian`.
6. **State preparation.** The Hartree-Fock determinant, then a UCCSD singlet ansatz
   whose amplitudes are the variational parameters.
7. **Energy.** The expectation value is evaluated exactly on the state vector, with
   no shot noise, and read back from the simulator's classical register.
8. **Optimisation.** SciPy minimises the energy, seeded from CCSD amplitudes rather
   than from zero or from random values, which reduces iteration count considerably.

Every energy is checked against CASCI in the same active space, and against FCI
where the full problem is small enough to diagonalise.

## Molecules

Twelve closed-shell, even-electron molecules. Each has several active spaces, from
6 up to 14 qubits.

| Molecule | Formula | Electrons | Spatial orbitals |
|---|---|---|---|
| Ethylene | C2H4 | 16 | 26 |
| Methanamide | CHONH2 | 24 | 33 |
| NH2- | NH2- | 10 | 13 |
| Benzene | C6H6 | 42 | 66 |
| Naphthalene | C10H8 | 68 | 106 |
| Benzaanthracene | C18H12 | 120 | 186 |
| Pentacene | C22H14 | 146 | 226 |
| Adenine | C5H5N5 | 70 | 100 |
| Guanine | C5H5N5O | 78 | 109 |
| Cytosine | C4H5N3O | 58 | 82 |
| Thymine | C5H6N2O2 | 66 | 93 |
| Uracil | C4H4N2O2 | 58 | 80 |

The set spans small molecules where FCI is available, aromatic systems, and the
four DNA and RNA nucleobases. The larger members are included because their
mean-field step is expensive even though their active spaces are small, which is
itself part of what the benchmark measures.

Two open-shell species in `config/molecules_data.py`, NH3+ and methylene, are
excluded from the current scope. They need an unrestricted reference and a
different ansatz.

## Status

Working and tested:

- Hamiltonian conversion from PySCF through OpenFermion into MIMIQ. Validated
  three independent ways. For H2 the converted operator reproduces the PySCF
  energy to ~1e-16.
- Energy evaluation on the Exaqt state-vector simulator, exact rather than sampled.
- UCCSD ansatz with a Hartree-Fock reference, giving the CASCI energy for H2.

In progress:

- The batch driver that sweeps molecules and active spaces.
- The result file format, matched to the CUDA-Q output so the two can be compared
  without reformatting.

Planned:

- Full production runs across all twelve molecules.
- Cost and timing characterisation as a function of qubit count and term count.
- Extension past 14 qubits once the smaller cases are complete.

## Requirements

- Python 3.11
- `mimiqcircuits`, `mimiq-openfermion`, and the Exaqt simulator, from QPerfect
- `openfermion`, `openfermionpyscf`, `pyscf`
- `numpy`, `scipy`

```bash
bash env/setup_env.sh
conda activate mimiq
```

The MIMIQ packages are distributed by QPerfect and are not on PyPI. Access is
arranged separately.

## Layout

| Path | Contents |
|---|---|
| `src/mimiq_hamiltonian.py` | OpenFermion `QubitOperator` to MIMIQ `Hamiltonian` |
| `src/mimiq_backend.py` | energy evaluation on the Exaqt simulator |
| `src/mimiq_driver.py` | optimisation loop over one active space |
| `src/run_single.py` | command-line entry point for a single run |
| `src/schema.py` | result record definition |
| `src/utils.py` | shared helpers |
| `config/molecules_data.py` | geometries, charges, and active spaces |
| `tests/` | validation against PySCF, CASCI and FCI |

## Tests

```bash
pytest tests/ -v
```

The suite checks that the converted Hamiltonian has the same spectrum as the
OpenFermion original, that the ansatz has the correct width and particle number,
that the energy at zero amplitudes equals the Hartree-Fock energy, and that the
optimised energy reaches CASCI while respecting the variational bound.

## Author

Khuram Shahzad

## Licence

Not yet licensed. Please ask before reusing.
