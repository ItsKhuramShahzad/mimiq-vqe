# mimiq-vqe

VQE for molecular ground state energies on QPerfect MIMIQ.

This is a port of my CUDA-Q VQE benchmark to MIMIQ, so the same molecules and
active spaces can be run on both and compared. The chemistry part (PySCF,
OpenFermion) is the same as in the CUDA-Q version. The ansatz comes from
mimiq-openfermion and the circuits run on the Exaqt state vector simulator.

## Status

Work in progress.

- Runs end to end: SCF, CCSD, CASCI, Hamiltonian, UCCSD ansatz, VQE, results file.
- Tested on H2, LiH and Ethylene (cc-pVDZ, active spaces up to 10 qubits).
  VQE reaches the CASCI energy to about 1e-12 Ha.
- Not yet run on all molecules. 14 qubit active spaces are slow at the moment,
  about 3 s per energy evaluation.
- Closed shell molecules only.

## Requirements

Tested with:

    python             3.11
    mimiqcircuits      0.27.1
    exaqt              0.2.1
    mimiq-openfermion  0.1.0
    openfermion        1.8.1
    openfermionpyscf   0.5
    pyscf              2.14.0
    numpy              2.4
    scipy              1.17

exaqt and mimiq-openfermion are QPerfect packages and are not on PyPI.

cudaq is optional. If it is installed it is only used to count the CUDA-Q
UCCSD parameters (see Notes). Without it a closed form count is used.

## Installation

    git clone https://github.com/ItsKhuramShahzad/mimiq-vqe.git
    cd mimiq-vqe

    conda create -n mimiq python=3.11
    conda activate mimiq

    pip install mimiqcircuits
    pip install openfermion openfermionpyscf pyscf numpy scipy pytest

Then install exaqt and mimiq-openfermion from QPerfect.

Check the install:

    python -c "import mimiqcircuits, exaqt, mimiq_openfermion, openfermion, pyscf; print('ok')"

## Running

Run from the repo root with `python -m`, because the code imports `src.` and
`config.`.

One active space of one molecule:

    python -m src.run_single --molecule Ethylene --space_idx 5

All active spaces of a molecule:

    python -m src.run_single --molecule Ethylene

Molecules, geometries and active spaces are in `config/molecules_data.py`.
`--space_idx` is the index into `valid_active_spaces` for that molecule.

Output is one pkl file per molecule in `pkl_results/mimiq_exaqt/`, for example

    16_SEP_2026_Ethylene_cc-pVDZ_exaqt_COBYLA_VQE_results.pkl

The layout of the pkl is the same as in the CUDA-Q version, so the same
analysis scripts can read both.

For each active space it prints the energy of the starting point and the final
VQE energy next to CASCI:

    [SPACE] Ethylene ncore=6 nele=4 norb=3 -> 6 qubits
    [SEED ] CCSD-sliced (singlet packer): E_theta0=-78.0563945040 ...
    [SPACE] done E_VQE=-78.0574268190 E_CASCI=-78.0574268190 diff=-4.83e-13 cycles=3 t=14.6s

## How it works

For each molecule:

1. SCF (RHF) and CCSD with PySCF on the full molecule, cc-pVDZ.

Then for each active space:

2. CASCI with PySCF, used as the reference energy.
3. Active space Hamiltonian from OpenFermion, Jordan-Wigner mapping.
4. Convert the OpenFermion QubitOperator to a MIMIQ Hamiltonian. The identity
   term is taken out as a constant. (`src/mimiq_hamiltonian.py`)
5. Starting parameters: the CCSD amplitudes are cut to the active space and
   put in the order that OpenFermion's `uccsd_singlet_generator` expects.
   (`slice_ccsd_to_active` and `pack_ccsd_singlet` in `src/run_single.py`)
6. Energy function: build the UCCSD singlet circuit for the parameters, add
   `push_expval` for the Hamiltonian, run on Exaqt, read `zstates[0][0]`.
   (`src/mimiq_backend.py`, `src/mimiq_ansatz.py`)
7. Optimisation with COBYLA: a seed search from the CCSD point and a few
   jittered copies of it, then repeated optimisation cycles until the energy
   stops improving. (`src/mimiq_driver.py`)
8. Check the result (`src/schema.py`) and save the pkl.

The optimizer settings are the same as in the CUDA-Q script: COBYLA, tol 1e-10,
rhobeg 0.2, 3 restarts, cycles of 600 iterations, stop after 3 cycles that
improve by less than 1e-6 Ha.

## Notes

**decompose() before running.** Exaqt cannot run the composite gates made by
`push_suzukitrotter` yet, so the circuit is decomposed before every run.

**Ansatz builder.** `build_uccsd_singlet_ansatz` in mimiq-openfermion fails
with "Number of qubits does not match Hamiltonian" when the non zero amplitudes
do not reach the highest qubit. This happens for sparse parameter vectors, for
example on the first COBYLA steps. `src/mimiq_ansatz.py` builds the same
circuit but uses the generator's own qubit count for the Trotter step. The
energies are the same as with the library builder wherever that one works.

**Starting point.** The CCSD amplitudes already give most of the correlation
energy before any optimisation: about 94% for Ethylene, 85-92% for Benzene,
73-82% for Cytosine. The factor of 2 on the doubles used in the CUDA-Q version
is not needed here. It comes from how the CUDA-Q UCCSD kernel applies the
rotations.

**Parameter count.** The singlet UCCSD in OpenFermion has fewer parameters
than the CUDA-Q UCCSD (14 qubits, 8 electrons: 90 vs 204). The CUDA-Q script
uses no restarts and a smaller step when there are more than 150 parameters.
To keep the settings the same on both, this code decides that from the CUDA-Q
count, not from its own.

**Speed.** The circuit is rebuilt at every energy evaluation. At 14 qubits this
takes more than half of the time per evaluation.

## Tests

    pytest tests/test_ansatz.py tests/test_driver.py tests/test_theta0.py

- `test_ansatz.py`: H2. Zero parameters give the Hartree-Fock energy, the CCSD
  start is below Hartree-Fock and recovers most of the correlation, VQE
  converges to FCI.
- `test_driver.py`: the full optimisation loop on H2, and the ansatz builder
  with sparse parameters.
- `test_theta0.py`: the CCSD packing gives back the same cluster operator, and
  the CCSD start is below Hartree-Fock on LiH.

`tests/test_hamiltonian_bridge.py` compares the spectrum of the MIMIQ
Hamiltonian with OpenFermion. It has a LiH case that takes a long time.

## Files

    config/molecules_data.py     molecules, geometries, active spaces
    src/run_single.py            command line entry point, one molecule
    src/mimiq_hamiltonian.py     OpenFermion QubitOperator to MIMIQ Hamiltonian
    src/mimiq_ansatz.py          UCCSD singlet circuit
    src/mimiq_backend.py         energy function on Exaqt
    src/mimiq_driver.py          COBYLA optimisation loop
    src/schema.py                checks the result before saving
    src/utils.py                 file names, pkl save, stable hash for seeding
    tests/                       tests
    env/                         conda environment files

## Author

Khuram Shahzad
