# Results

VQE results on MIMIQ (Exaqt state vector simulator, CPU) for the 12 molecules
in the main README, cc-pVDZ basis, all 9 active spaces of each molecule
(6 to 14 qubits): 108 active spaces.

    results/mimiq_exaqt_uccsd_singlet/    one pkl per molecule, singlet UCCSD

## How they were made

- Run on an HPC cluster, 24-25 September 2026, one SLURM job per molecule with
  its 9 active spaces side by side (2 threads each), then merged into one pkl per
  molecule with `scripts/merge_space_pkls.py`.
- Hamiltonian, CASCI energy and CCSD amplitudes read from the saved integral
  files (`--integrals`), so every space starts from the CCSD amplitudes.
- Singlet UCCSD (90 parameters at 14 qubits), one second order Trotter step.
- COBYLA, tol 1e-10, cycles of 600 iterations, stop after 3 cycles that improve
  by less than 1e-6 Ha; same settings as the CUDA-Q runs.
- Software: python 3.11.16, mimiqcircuits 0.27.1, exaqt 0.2.1,
  mimiq-openfermion 0.1.0, openfermion 1.8.1 (also stored in each pkl under
  `backend_versions`).

Each run can be repeated with

    python -m src.run_single --molecule Ethylene --integrals integrals --ansatz singlet

## Summary

All 108 active spaces reach the CASCI energy within chemical accuracy (1.6 mHa).
Gap = E_VQE - E_CASCI, largest over the 9 spaces. The last three columns are
for the 14 qubit space (6 electrons, 7 orbitals).

| Molecule | Spaces | Largest gap (mHa) | Parameters (14q) | Energy evaluations (14q) | VQE time (14q, h) |
|---|---:|---:|---:|---:|---:|
| NH2- | 9 | 0.190 | 90 | 5400 | 9.8 |
| Ethylene | 9 | 0.008 | 90 | 3600 | 6.1 |
| Methanamide | 9 | 0.295 | 90 | 4200 | 7.6 |
| Benzene | 9 | 0.014 | 90 | 3600 | 6.6 |
| Naphthalene | 9 | 0.178 | 90 | 3600 | 6.2 |
| Tetracene | 9 | 0.249 | 90 | 4200 | 7.1 |
| Pentacene | 9 | 0.375 | 90 | 3600 | 6.5 |
| Uracil | 9 | 0.024 | 90 | 3600 | 6.5 |
| Cytosine | 9 | 0.015 | 90 | 3600 | 6.7 |
| Thymine | 9 | 0.018 | 90 | 3600 | 6.5 |
| Adenine | 9 | 0.102 | 90 | 4200 | 7.8 |
| Guanine | 9 | 0.052 | 90 | 4200 | 7.8 |

The times are wall times on shared cluster nodes, so they are indicative only.

## Reading a pkl

The layout is the same as the CUDA-Q version, so the same analysis scripts read
both.

    import pickle

    with open("results/mimiq_exaqt_uccsd_singlet/24_SEP_2026_Ethylene_cc-pVDZ_exaqt-cpu_COBYLA_VQE_results.pkl", "rb") as f:
        raw = pickle.load(f)

    mol = raw["Ethylene"]
    print(mol["references"])                 # E_hf_full, E_ccsd_full
    print(mol["backend_versions"])

    for run in mol["active_space_runs"]:
        if run.get("skipped"):
            continue
        print(run["space"],                          # ncore, nele_cas, norb_cas
              run["sizes"]["qubits"],
              run["sizes"]["uccsd_num_parameters"],
              run["casci"]["E_casci_total"],
              run["vqe"]["E_total"],                 # best VQE energy, Ha
              run["vqe"]["nfev_total"])              # energy evaluations

Useful keys in `run["vqe"]`:

| Key | Meaning |
|---|---|
| `E_total` | best VQE energy, including the constant, Ha |
| `theta_opt` | optimised parameters |
| `energy_convergence` | energy at every evaluation, without the constant; add `run["hamiltonian"]["c0"]` |
| `best_energy_per_cycle` | best energy after each optimisation cycle |
| `nfev_total` | number of energy evaluations |
| `runtime` | VQE wall time in seconds |
| `quantum_times` | time of each circuit execution, seconds |

`run["theta0"]` has the CCSD starting point and its source, and
`run["casci"]["E_casci_total"]` the CASCI reference.

Only load pkl files you trust: unpickling can run code.
