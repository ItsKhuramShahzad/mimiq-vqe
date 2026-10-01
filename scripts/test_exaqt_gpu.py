"""
check the exaqt gpu backend against the cpu one on real vqe energy
"""


import os
import sys

import time

import numpy as np
import exaqt

from exaqt import ExaqtQCS
from mimiqcircuits import Add

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.integrals import load_integrals, qubit_hamiltonian
from src.mimiq_ansatz import build_uccsd
from src.mimiq_hamiltonian import openfermion_to_mimiq_hamiltonian
from src.run_single import pack_ccsd_singlet

MOLECULE, BASIS = "Ethylene", "cc-pVDZ"
NCORE, NELE, NORB = 5, 6, 7  # 14 qubits

REPEATS = 5

# 1. Is the gpu there?
print("exaqt", exaqt.__version__)
print("exaqt gpu available:", exaqt.gpu_available())


if not exaqt.gpu_available():
    sys.exit("No exaqt gpu available, stopping test_exaqt_gpu.py")

from exaqt import ExaqtQCSGpu, ExaqtSVGpu

print("largest state that fits on the gpu:", ExaqtSVGpu.max_qubits_on_device(), "qubits")

# 2. the same energy on cpu and gpu, at the ccsd starting point
data = load_integrals(f"{REPO}/integrals", BASIS, MOLECULE, NCORE, NELE, NORB)

nele, n_qubits = NELE, 2 * NORB

h, c0, _ = openfermion_to_mimiq_hamiltonian(qubit_hamiltonian(data))

theta0 = pack_ccsd_singlet(data["t1_active"], data["t2_active"])

print(f"{MOLECULE} {BASIS} ncore={NCORE} {NELE}e {NORB}o : {n_qubits} qubits, {len(theta0)} parameters")


def circuit(theta):
    circ = build_uccsd(n_qubits=n_qubits, n_electrons=nele, params=list(theta))
    circ = circ.decompose()
    circ.push_expval(h, *range(h.num_qubits()))
    circ.push(Add(2, c=c0), 0, 0)
    return circ


def energy(sim, theta):
    t0 = time.perf_counter()
    result = sim.execute(circuit(theta), nsamples=1)
    return float(np.real(result.zstates[0][0])), time.perf_counter() - t0


cpu, gpu = ExaqtQCS(), ExaqtQCSGpu()
energy(gpu, theta0)  # first gpu call opens the cuda context, keep it out of the timing

rng = np.random.default_rng(1234)

thetas = [theta0] + [theta0 + 0.05 * rng.standard_normal(len(theta0)) for _ in range(REPEATS - 1)]

t_cpu, t_gpu = [], []
print(f"Running {REPEATS} tests on {n_qubits} qubits, {len(theta0)} parameters\n")

for i, theta in enumerate(thetas):
    e_cpu, tc = energy(cpu, theta)
    t_cpu.append(tc)
    e_gpu, tg = energy(gpu, theta)
    t_gpu.append(tg)
    assert np.isclose(e_cpu, e_gpu, rtol=0, atol=1e-9), f"CPU and GPU energies differ: {e_cpu} vs {e_gpu}"

    name = "theta0" if i == 0 else f"random{i}"
    print(f"{name:>8} : CPU {e_cpu:.12f} vs GPU {e_gpu:.12f} (diff {e_gpu - e_cpu:.1e})"
          f" in {tc:.3f}s vs {tg:.3f}s")


print(f"\ncasci energy: {float(data['e_casci']):.12f}")
print(f"s per energy    CPU {np.median(t_cpu):.3f}   GPU {np.median(t_gpu):.3f}")
