
# -*- coding: utf-8 -*-
"""
End-to-end verifier for VQAC(n,m) using the chain-based E/G construction.

Functionality:
1. Generate chained E/G using eg_generator.py
2. Construct the plaintext circuit U_plain
3. Construct the chained circuit U_encdec = initial encryption -> global G sequence -> final decryption
4. Perform end-to-end verification on basis states / random basis states

Notes:
- This verifier is generic and supports arbitrary n and m
- However, because it uses statevector simulation, the total qubit count
  q = 2*n*lambda + 1 can quickly become very large
- It is therefore best suited for small-scale verification, e.g.
  VQAC(2,2), VQAC(2,3), VQAC(3,2), VQAC(3,3)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# =========================
# Dynamic module loading
# =========================
def load_module(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _candidate_builder_paths() -> List[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    return [
        os.path.join(here, "vqac_builder.py"),
        os.path.join(cwd, "vqac_builder.py"),
        "/mnt/data/vqac_builder.py",
    ]


def _candidate_chain_paths() -> List[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    return [
        os.path.join(here, "eg_generator.py"),
        os.path.join(cwd, "eg_generator.py"),
        "/mnt/data/eg_generator.py",
    ]


def _find_existing_path(candidates: List[str], desc: str) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{desc} not found")


BUILDER_PATH = _find_existing_path(_candidate_builder_paths(), "vqac_builder.py")
CHAIN_PATH = _find_existing_path(_candidate_chain_paths(), "eg_generator.py")

vqac_builder = load_module(BUILDER_PATH, "vqac_builder_for_nm_verifier")
chain_mod = load_module(CHAIN_PATH, "chain_mod_for_nm_verifier")

append_embedded_gate = chain_mod.append_embedded_gate
inverse_clifford_circuit = chain_mod.inverse_clifford_circuit
generate_E_and_G = chain_mod.generate_E_and_G
lambda_len = vqac_builder.lambda_len


# =========================
# Basic utilities
# =========================
def basis_state_from_bits(bits: List[int]) -> Statevector:
    qc = QuantumCircuit(len(bits))
    for i, b in enumerate(bits):
        if b:
            qc.x(i)
    return Statevector.from_instruction(qc)


def states_equivalent(a: Statevector, b: Statevector, atol: float = 1e-8) -> Tuple[bool, float]:
    av = np.asarray(a.data, dtype=complex)
    bv = np.asarray(b.data, dtype=complex)

    idx = None
    for i in range(len(av)):
        if abs(av[i]) > 1e-12 and abs(bv[i]) > 1e-12:
            idx = i
            break

    if idx is not None:
        phase = av[idx] / bv[idx]
        bv = bv * phase

    diff = float(np.max(np.abs(av - bv)))
    return diff <= atol, diff


def state_overlap_abs(a: Statevector, b: Statevector) -> float:
    return float(abs(np.vdot(a.data, b.data)))


def estimate_statevector_memory_gb(num_qubits: int) -> float:
    """
    Estimate memory usage assuming complex double precision (about 16 bytes).
    """
    return (2 ** num_qubits) * 16 / (1024 ** 3)


# =========================
# Construct plaintext / encrypt-compute-decrypt circuits
# =========================
def build_global_wire_order(N_regs: List[List[str]], M_regs: List[List[str]], c0: str) -> List[str]:
    out: List[str] = []
    for i in range(len(N_regs)):
        out.extend(N_regs[i])
        out.extend(M_regs[i])
    out.append(c0)
    return out


def compose_subcircuit_on_global(
    global_qc: QuantumCircuit,
    sub_qc: QuantumCircuit,
    sub_wire_order: List[str],
    global_wire_order: List[str]
):
    pos = {q: i for i, q in enumerate(global_wire_order)}
    qubits = [pos[q] for q in sub_wire_order]
    global_qc.compose(sub_qc, qubits=qubits, inplace=True)


def build_plain_global_circuit(result: Dict[str, Any]) -> Tuple[QuantumCircuit, List[str]]:
    builder = result["builder"]
    N_regs = result["N_regs"]
    M_regs = result["M_regs"]
    c0 = result["c0"]

    global_wire_order = build_global_wire_order(N_regs, M_regs, c0)
    qc = QuantumCircuit(len(global_wire_order))

    for gate in builder.gates:
        append_embedded_gate(qc, gate, global_wire_order)

    return qc, global_wire_order


def build_chain_encdec_global_circuit(result: Dict[str, Any]) -> Tuple[QuantumCircuit, List[str]]:
    builder = result["builder"]
    N_regs = result["N_regs"]
    M_regs = result["M_regs"]
    c0 = result["c0"]
    E = result["E"]
    global_G_by_index = result["global_G_by_index"]
    client_wire_map = result["client_wire_map"]

    global_wire_order = build_global_wire_order(N_regs, M_regs, c0)
    qc = QuantumCircuit(len(global_wire_order))

    # Initial encryption
    for p in sorted(E.keys(), key=lambda x: int(x[1:])):
        compose_subcircuit_on_global(qc, E[p][0].circuit, client_wire_map[p], global_wire_order)

    # Apply G in global gate order
    for global_idx in range(1, len(builder.gates) + 1):
        g = global_G_by_index[global_idx]
        compose_subcircuit_on_global(qc, g.circuit, g.wire_order, global_wire_order)

    # Final decryption
    for p in sorted(E.keys(), key=lambda x: int(x[1:])):
        final_e = E[p][-1]
        dec = inverse_clifford_circuit(final_e.clifford)
        compose_subcircuit_on_global(qc, dec, client_wire_map[p], global_wire_order)

    return qc, global_wire_order


# =========================
# Test case generation
# =========================
def build_test_cases(q: int, num_random_tests: int, seed: int) -> List[List[int]]:
    fixed_tests = [
        [0] * q,
        [1] * q,
        [0, 1] * (q // 2) + ([0] if q % 2 else []),
        [1, 0] * (q // 2) + ([1] if q % 2 else []),
        [1 if i % 3 == 0 else 0 for i in range(q)],
        [1 if i % 3 == 1 else 0 for i in range(q)],
        [1 if i % 3 == 2 else 0 for i in range(q)],
    ]
    rng = random.Random(seed + 999)
    random_tests = [[rng.randint(0, 1) for _ in range(q)] for _ in range(num_random_tests)]
    return fixed_tests + random_tests


# =========================
# Main verification logic
# =========================
def verify_vqac_nm_chain(
    n: int,
    m: int,
    seed: int = 42,
    num_random_tests: int = 32,
    random_initial_key: bool = False,
    atol: float = 1e-8,
    verbose: bool = True,
):
    result = generate_E_and_G(
        n=n,
        m=m,
        seed=seed,
        build_operator_for_G=False,
        random_initial_key=random_initial_key,
    )

    plain_qc, global_wire_order = build_plain_global_circuit(result)
    encdec_qc, _ = build_chain_encdec_global_circuit(result)

    lam = result["lambda"]
    q = len(global_wire_order)
    est_mem_gb = estimate_statevector_memory_gb(q)

    tests = build_test_cases(q=q, num_random_tests=num_random_tests, seed=seed)

    passed = 0
    failed_cases: List[Tuple[int, List[int], float, float]] = []

    for idx, bits in enumerate(tests, start=1):
        init_sv = basis_state_from_bits(bits)
        sv_plain = init_sv.evolve(plain_qc)
        sv_encdec = init_sv.evolve(encdec_qc)

        ok, diff = states_equivalent(sv_plain, sv_encdec, atol=atol)
        ov = state_overlap_abs(sv_plain, sv_encdec)

        if ok:
            passed += 1
        else:
            failed_cases.append((idx, bits, ov, diff))
            if verbose:
                print(f"[FAIL] case#{idx}, overlap={ov:.12f}, maxdiff={diff:.3e}, bits={bits}")

    if verbose:
        print(f"===== End-to-End Verification for VQAC({n},{m}) =====")
        print(f"lambda = {lam}")
        print(f"Total qubits = {q}")
        print(f"Estimated statevector memory = {est_mem_gb:.4f} GB")
        for p in sorted(result["E"].keys(), key=lambda x: int(x[1:])):
            print(f"{p} local gate count = {len(result['E'][p]) - 1}")
        print(f"Global gate count = {len(result['builder'].gates)}")
        print(f"plain depth = {plain_qc.depth()}, encdec depth = {encdec_qc.depth()}")
        print(f"random_initial_key = {random_initial_key}")
        print(f"Passed {passed}/{len(tests)}")
        if failed_cases:
            print("Failed cases (showing up to 5):")
            for idx, bits, ov, diff in failed_cases[:5]:
                print(f"  case#{idx}: overlap={ov:.12f}, maxdiff={diff:.3e}, bits={bits}")
        else:
            print("All test cases passed.")
        print()
        print("Global wire order:")
        print(global_wire_order)

    return {
        "passed": passed,
        "total": len(tests),
        "failed_cases": failed_cases,
        "plain_qc": plain_qc,
        "encdec_qc": encdec_qc,
        "global_wire_order": global_wire_order,
        "result": result,
        "estimated_statevector_memory_gb": est_mem_gb,
    }


def recommend_small_scale(n: int, m: int) -> str:
    lam = lambda_len(n, m)
    q = 2 * n * lam + 1
    mem = estimate_statevector_memory_gb(q)
    if q <= 18:
        return f"Direct full verification is recommended. Current q={q}, estimated statevector memory is about {mem:.4f} GB."
    if q <= 24:
        return f"This is still feasible, but it may become noticeably slower. Current q={q}, estimated statevector memory is about {mem:.4f} GB."
    return f"Full statevector end-to-end verification is not recommended. Current q={q}, estimated statevector memory is about {mem:.4f} GB."


def main():
    parser = argparse.ArgumentParser(description="End-to-end verifier for VQAC(n,m) using chain-based E/G")
    parser.add_argument("--n", type=int, required=True, help="Number of parties n")
    parser.add_argument("--m", type=int, required=True, help="Plaintext bit width m")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-random-tests", type=int, default=32, help="Number of random basis-state tests")
    parser.add_argument("--random-initial-key", action="store_true", help="Use a random Clifford for E_i[0]")
    parser.add_argument("--atol", type=float, default=1e-8, help="Equivalence tolerance")
    args = parser.parse_args()

    print(recommend_small_scale(args.n, args.m))
    print()

    verify_vqac_nm_chain(
        n=args.n,
        m=args.m,
        seed=args.seed,
        num_random_tests=args.num_random_tests,
        random_initial_key=args.random_initial_key,
        atol=args.atol,
        verbose=True,
    )


if __name__ == "__main__":
    verify_vqac_nm_chain(
        n=2,
        m=2,
        seed=42,
        num_random_tests=1,
        random_initial_key=False,
        atol=1e-8,
        verbose=True,
    )
