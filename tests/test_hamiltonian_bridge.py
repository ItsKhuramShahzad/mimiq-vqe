import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from openfermion import MolecularData, get_sparse_operator
from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator, jordan_wigner

from mimiq_hamiltonian import openfermion_to_mimiq_hamiltonian

TEST_MOLECULES = {
    "H2":  dict(geometry=[("H", (0,0,0)), ("H", (0,0,0.74))], basis="sto-3g", multiplicity=1, charge=0),
    # "LiH": dict(geometry=[("Li", (0,0,0)), ("H", (0,0,1.6))], basis="sto-3g", multiplicity=1, charge=0),
}

def test_hamiltonian_bridge_matches_openfermion():
    """Spectrum check: eigenvalues must match OpenFermion's own representation.
    Validated against get_sparse_operator, never remembered constants."""
    for name, spec in TEST_MOLECULES.items():
        moldata = MolecularData(spec["geometry"], spec["basis"],
                                spec["multiplicity"], spec["charge"])
        molecule = run_pyscf(moldata, run_scf=True, run_fci=False)
        qop = jordan_wigner(molecule.get_molecular_hamiltonian())

        H, constant, stats = openfermion_to_mimiq_hamiltonian(qop)

        mimiq_eigs = np.linalg.eigvalsh(np.array(H.matrix()).astype(complex))
        mimiq_ground = mimiq_eigs[0].real + constant

        ref_eigs = np.linalg.eigvalsh(np.array(get_sparse_operator(qop).todense()))
        ref_ground = ref_eigs[0].real

        diff = abs(mimiq_ground - ref_ground)
        print(f"{name}: mimiq={mimiq_ground:.10f}  openfermion={ref_ground:.10f}  diff={diff:.2e}")
        assert diff < 1e-10, f"{name}: bridge disagrees with OpenFermion by {diff}"
        assert np.allclose(mimiq_eigs + constant, ref_eigs, atol=1e-10), \
            f"{name}: spectra differ beyond the ground state"

