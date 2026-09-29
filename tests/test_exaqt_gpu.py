"""
The exaqt gpu backend must give smae vqe energies as cpu backend.
This test fille will skipped where these is no gpu available.
"""

import numpy as np
import pytest

exaqt = pytest.importorskip("exaqt")

if not hasattr(exaqt, "gpu_available") or not exaqt.gpu_available():
    pytest.skip("no usable gpu for exaqt", allow_module_level=True) 
    
    
from exaqt import ExaqtQCS, ExaqtQCSGpu, ExaqtSVGpu

from exaqt.passes import ExaqtFlattenContainersPass
from mimiqcircuits import Add
from mimiqcircuits.backends import PassPipeline

from src.integrals import load_integrals, qubit_hamiltonian
from src.mimiq_ansatz import build_uccsd
from src.mimiq_hamiltonian import openfermion_to_mimiq_hamiltonian
from src.run_single import pack_ccsd_singlet

ROOT, BASIS, MOLECULE = "integrals", "cc-pVDZ", "Ethylene"

SPACES = [(7,2,3)# 6 QUBITS. QUICK
          
          , (5,6,7)] # 14 QUBITS  long

ATOL = 1e-9
NO_FUSION =PassPipeline([ExaqtFlattenContainersPass()])

@pytest.fixture(scope="module", params=SPACES, ids = lambda s: f"{2*s[2]}q")

def space(request):
    ncore, nele, norb= request.param
    data = load_integrals(ROOT, BASIS, MOLECULE, ncore, nele, norb)
    H, c0, _= openfermion_to_mimiq_hamiltonian(qubit_hamiltonian(data))
    theta0 = pack_ccsd_singlet(data["t1_active"], data["t2_active"])
    rng = np.random.default_rng(1234)
    theta = [theta0] + [theta0 + 0.5*rng.standard_normal(size=len(theta0)) for _ in range(2)]
    return {"H": H, "c0": c0, "theta0": theta0, "theta": theta, "ncore": ncore, "nele": nele, "norb": norb, "nqubits": 2*norb}

@pytest.fixture(scope="module")
def cpu():
    return ExaqtQCS()
@pytest.fixture(scope="module")
def gpu():
    return ExaqtQCSGpu()

def energy(sim, space, theta, passes =None):
    
    circ= build_uccsd(n_qubits= space["nqubits"], n_electrons=space["nele"], params=list(theta))
    circ = circ.decompose()
    circ.push_expval(space["H"], *range(space["H"].num_qubits()))
    circ.push(Add(2, c= space["c0"]), 0 ,0)
    
    result = sim.execute(circ, nsamples=1, passes = passes)
    return float(np.real(result.zstates[0][0]))

def test_gpu_holds_the_largest_space():
    assert ExaqtSVGpu.max_qubits_on_device()>= max(2*norb for _, _, norb in SPACES)
    
def test_energy_matches_cpu_gpu(space, cpu, gpu):
    for theta in space["theta"]:
        assert energy(gpu, space, theta) == pytest.approx(energy(cpu, space, theta), abs = ATOL)

def test_no_fusion_gives_the_same_energy(space, gpu):
    theta = space["theta"][1]
    assert energy (gpu, space, theta, passes = NO_FUSION) == pytest.approx(energy(gpu, space, theta), abs = ATOL)
    
    
def test_same_theta_twice_gives_the_same_energy(space, gpu):
    theta = space["theta"][1]
    assert energy (gpu, space, theta) == pytest.approx(energy(gpu, space, theta), abs = ATOL)  

