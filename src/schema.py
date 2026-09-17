"""
Checks the result dict before saving to pkl. Keys are the same as in the
CUDA-Q script so the analysis scripts can read both.
"""

import numpy as np


TOP_KEYS = {
    "molecule_name", "molecule_name_clean", "skipped", "tag", "basis",
    "target", "optimizer", "seed", "input_spec", "timing", "references",
    "pyscf_insights", "ccsd", "system_sizes", "active_space_runs",
}

# Added to the CUDA-Q script mid-June; PKLs from 3-6 June do not have them.
# MIMIQ runs should still write them, set to None.
OPTIONAL_TOP_KEYS = {"target_precision_option", "cudaq_precision"}

RUN_KEYS = {
    "space", "skipped", "sizes", "active_indices", "casci",
    "hamiltonian", "theta0", "seed_search", "vqe", "compare",
}

SKIPPED_RUN_KEYS = {"space", "skipped", "skip_reason"}

VQE_KEYS = {
    "E_nc_opt", "E_total", "theta_opt", "converged", "cycles", "runtime",
    "simulated_quantum_runtime", "optimizer_runtime", "quantum_times",
    "energy_convergence", "best_energy_per_cycle", "cycle_summaries",
    "success_any", "message", "best_init_index", "nit_total", "nfev_total",
}

# mimiq does not check the optimizer name, "COBLYA" also runs
KNOWN_OPTIMIZERS = {"COBYLA", "L-BFGS-B", "BFGS", "Nelder-Mead", "Powell", "SLSQP"}

VARIATIONAL_TOL = 1e-8
CONSTANT_TOL = 1e-10


class SchemaError(ValueError):
    pass


def _require(d, keys, where):
    missing = keys - set(d)
    if missing:
        raise SchemaError(f"{where}: missing keys {sorted(missing)}")


def _num_parameters(sizes, where):
    # Older PKLs use "num_parameters", newer ones "uccsd_num_parameters".
    n = sizes.get("uccsd_num_parameters", sizes.get("num_parameters"))
    if n is None:
        raise SchemaError(f"{where}.sizes: no parameter count")
    return n


def validate_run(run, where="run"):
    """Check one entry of active_space_runs."""
    if run.get("skipped"):
        _require(run, SKIPPED_RUN_KEYS, where)
        return

    _require(run, RUN_KEYS, where)
    _require(run["vqe"], VQE_KEYS, f"{where}.vqe")

    vqe = run["vqe"]
    E_total = vqe["E_total"]
    E_casci = run["casci"]["E_casci_total"]

    if not np.isfinite(E_total):
        raise SchemaError(f"{where}: E_total is not finite ({E_total})")

    # VQE in the active space can never go below CASCI in the same space.
    # If it does, the constant is almost certainly wrong.
    if E_total < E_casci - VARIATIONAL_TOL:
        raise SchemaError(
            f"{where}: variational bound violated, "
            f"E_total={E_total:.10f} < E_casci={E_casci:.10f} "
            f"(by {E_casci - E_total:.2e})"
        )

    n_params = _num_parameters(run["sizes"], where)
    if len(vqe["theta_opt"]) != n_params:
        raise SchemaError(
            f"{where}: theta_opt has {len(vqe['theta_opt'])} values, "
            f"expected {n_params}"
        )

    # Catches the constant being added twice, or not at all.
    c0 = run["hamiltonian"]["c0"]
    diff = abs((c0 + vqe["E_nc_opt"]) - E_total)
    if diff > CONSTANT_TOL:
        raise SchemaError(f"{where}: E_total != c0 + E_nc_opt (diff {diff:.2e})")


def validate_payload(payload):
    """Check a whole PKL payload: {mol_name: molecule_results}."""
    if not isinstance(payload, dict) or len(payload) != 1:
        raise SchemaError("payload must be {mol_name: molecule_results}")

    (mol_name, res), = payload.items()

    if res.get("skipped"):
        return

    _require(res, TOP_KEYS, mol_name)

    if res["optimizer"] not in KNOWN_OPTIMIZERS:
        raise SchemaError(
            f"{mol_name}: unknown optimizer {res['optimizer']!r} "
            f"(MIMIQ does not validate this)"
        )

    for i, run in enumerate(res["active_space_runs"]):
        validate_run(run, where=f"{mol_name}.active_space_runs[{i}]")
