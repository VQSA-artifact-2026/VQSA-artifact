# -*- coding: utf-8 -*-
"""
General VQAC(n,m) execution script with two random bit-flip attack experiments.

Based on:
- vqac_builder.py
- eg_generator.py

Capabilities:
1. Build the complete quantum circuit for generic VQAC(n,m)
2. Accept arbitrary integer inputs for n clients
3. Print all register inputs and outputs
4. Run locally with AerSimulator
5. Support two statistical attack settings:

A. Global random attack
   - Randomly flip positions across the global circuit
   - Randomly choose the number of flips
   - Randomly choose the injection time
   - Count how many clients detect the tampering

B. P1-targeted attack
   - Randomly flip qubits only in P1's registers N1 and M1
   - Compare only P1's verifiable register M1
   - Report P1's detection rate

Detection semantics:
- Client 1 checks M1_out == p1
- Client 2 checks N2_out == p2
- Client j (j>=3) checks Nj_out == pj and Mj_out == pj
- c0 is logged separately but is not counted toward the number of detecting clients
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from eg_generator import (
    generate_E_and_G,
    inverse_clifford_circuit,
)


# =========================
# Basic utilities
# =========================

def build_global_wire_order(N_regs: List[List[str]], M_regs: List[List[str]], c0: str) -> List[str]:
    wires: List[str] = []
    for i in range(len(N_regs)):
        wires.extend(N_regs[i])
        wires.extend(M_regs[i])
    wires.append(c0)
    return wires


def compose_subcircuit_on_global(
    global_qc: QuantumCircuit,
    sub_qc: QuantumCircuit,
    sub_wire_order: List[str],
    global_wire_order: List[str],
):
    pos = {q: i for i, q in enumerate(global_wire_order)}
    qubits = [pos[q] for q in sub_wire_order]
    global_qc.compose(sub_qc, qubits=qubits, inplace=True)


def int_to_bits_be(x: int, width: int) -> List[int]:
    if x < 0:
        raise ValueError("This version supports only non-negative integer inputs")
    if x >= (1 << width):
        raise ValueError(f"Input {x} exceeds the {width}-bit representable range [0, {2**width - 1}]")
    s = format(x, f"0{width}b")
    return [int(ch) for ch in s]


def bits_to_int_be(bits: List[int]) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | b
    return out


def get_reg_bits_from_bitstring(bitstring: str, global_wire_order: List[str], reg_names: List[str]) -> List[int]:
    """
    Qiskit count bitstrings are displayed in big-endian order.
    However, when QuantumCircuit.measure(range(n), range(n)) is used, the rightmost bit corresponds to qubit 0.
    We therefore reverse the bitstring before parsing it according to global_wire_order.
    """
    bits_le = [int(ch) for ch in bitstring[::-1]]
    pos = {q: i for i, q in enumerate(global_wire_order)}
    return [bits_le[pos[name]] for name in reg_names]


# =========================
# Input construction (generic)
# =========================

def build_input_value_map(
    N_regs: List[List[str]],
    M_regs: List[List[str]],
    c0: str,
    inputs: List[int],
) -> Dict[str, int]:
    """
    Generic input mapping:
      N_i = 0^lambda
      M_i = the lambda-bit big-endian representation of inputs[i]
      c0 = 0
    """
    if len(inputs) != len(M_regs):
        raise ValueError("The length of inputs must equal the number of clients n")

    lam = len(M_regs[0])
    value_map: Dict[str, int] = {}

    for i in range(len(N_regs)):
        for q in N_regs[i]:
            value_map[q] = 0

        bits = int_to_bits_be(inputs[i], lam)
        for q, b in zip(M_regs[i], bits):
            value_map[q] = b

    value_map[c0] = 0
    return value_map


def prepare_input_state(
    qc: QuantumCircuit,
    global_wire_order: List[str],
    value_map: Dict[str, int],
):
    for i, q in enumerate(global_wire_order):
        if value_map.get(q, 0) == 1:
            qc.x(i)


def pretty_input_registers(
    N_regs: List[List[str]],
    M_regs: List[List[str]],
    inputs: List[int],
) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    lam = len(M_regs[0])
    for i in range(len(N_regs)):
        out[f"N{i+1}"] = [0] * lam
        out[f"M{i+1}"] = int_to_bits_be(inputs[i], lam)
    out["c0"] = [0]
    return out


# =========================
# Output parsing (generic)
# =========================

def parse_all_registers_from_bitstring(
    bitstring: str,
    global_wire_order: List[str],
    N_regs: List[List[str]],
    M_regs: List[List[str]],
    c0: str,
) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}

    for i in range(len(N_regs)):
        out[f"N{i+1}"] = get_reg_bits_from_bitstring(bitstring, global_wire_order, N_regs[i])
        out[f"M{i+1}"] = get_reg_bits_from_bitstring(bitstring, global_wire_order, M_regs[i])

    out["c0"] = get_reg_bits_from_bitstring(bitstring, global_wire_order, [c0])
    return out


def print_registers(title: str, regs: Dict[str, List[int]]):
    print(title)

    def key_fn(k: str):
        if k == "c0":
            return (2, 9999)
        prefix = k[0]
        idx = int(k[1:])
        return (0 if prefix == "N" else 1, idx)

    for name in sorted(regs.keys(), key=key_fn):
        bits = regs[name]
        if len(bits) == 1:
            print(f"  {name:>3} bits = {bits}")
        else:
            print(f"  {name:>3} bits = {bits} (int={bits_to_int_be(bits)})")
    print()


# =========================
# Build the complete quantum circuit (generic)
# =========================

def build_vqac_quantum_circuit(
    n: int,
    m: int,
    inputs: List[int],
    seed: int = 42,
    random_initial_key: bool = True,
):
    result = generate_E_and_G(
        n=n,
        m=m,
        seed=seed,
        build_operator_for_G=False,
        random_initial_key=random_initial_key,
    )

    N_regs = result["N_regs"]
    M_regs = result["M_regs"]
    c0 = result["c0"]
    E = result["E"]
    global_G_by_index = result["global_G_by_index"]
    builder = result["builder"]
    client_wire_map = result["client_wire_map"]

    global_wire_order = build_global_wire_order(N_regs, M_regs, c0)
    qc = QuantumCircuit(len(global_wire_order), len(global_wire_order))

    # 1. Input state
    value_map = build_input_value_map(N_regs, M_regs, c0, inputs)
    prepare_input_state(qc, global_wire_order, value_map)

    # 2. Initial encryption E[p][0]
    for p in sorted(E.keys(), key=lambda x: int(x[1:])):
        compose_subcircuit_on_global(qc, E[p][0].circuit, client_wire_map[p], global_wire_order)

    # 3. Server applies all G operators
    for global_idx in range(1, len(builder.gates) + 1):
        g = global_G_by_index[global_idx]
        compose_subcircuit_on_global(qc, g.circuit, g.wire_order, global_wire_order)

    # 4. Final decryption
    for p in sorted(E.keys(), key=lambda x: int(x[1:])):
        final_e = E[p][-1]
        dec = inverse_clifford_circuit(final_e.clifford)
        compose_subcircuit_on_global(qc, dec, client_wire_map[p], global_wire_order)

    # 5. Measurement
    qc.measure(range(len(global_wire_order)), range(len(global_wire_order)))

    return qc, global_wire_order, result


# =========================
# Error injection
# =========================

def _inject_x_errors_by_targets(
    qc: QuantumCircuit,
    global_wire_order: List[str],
    flip_targets: List[str],
    when: str = "middle",
) -> QuantumCircuit:
    pos = {q: i for i, q in enumerate(global_wire_order)}

    new_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    ops = qc.data

    if when == "after_enc":
        split_point = len(ops) // 4
    elif when == "middle":
        split_point = len(ops) // 2
    else:
        raise ValueError("'when' only supports 'after_enc' or 'middle'")

    for i in range(split_point):
        inst = ops[i]
        new_qc.append(inst.operation, inst.qubits, inst.clbits)

    for qname in flip_targets:
        qid = pos[qname]
        new_qc.x(qid)

    for i in range(split_point, len(ops)):
        inst = ops[i]
        new_qc.append(inst.operation, inst.qubits, inst.clbits)

    return new_qc


def inject_bit_flip(
    qc: QuantumCircuit,
    global_wire_order: List[str],
    target_qubit_name: str,
    when: str = "middle",
) -> QuantumCircuit:
    if target_qubit_name not in set(global_wire_order):
        raise ValueError(f"Target qubit {target_qubit_name} is not in global_wire_order")
    print(f"⚠️ Injecting X error on qubit: {target_qubit_name}, location: {when}")
    return _inject_x_errors_by_targets(qc, global_wire_order, [target_qubit_name], when)


def inject_random_bit_flips(
    qc: QuantumCircuit,
    global_wire_order: List[str],
    num_flips: int,
    when: str = "middle",
) -> Tuple[QuantumCircuit, List[str]]:
    if num_flips < 1:
        raise ValueError("num_flips must be >= 1")
    if num_flips > len(global_wire_order):
        raise ValueError("num_flips cannot exceed the total number of qubits")

    all_qubits = list(global_wire_order)
    flip_targets = random.sample(all_qubits, num_flips)
    print(f"⚠️ Randomly flipping {num_flips} bits, injection time={when}, targets={flip_targets}")
    new_qc = _inject_x_errors_by_targets(qc, global_wire_order, flip_targets, when)
    return new_qc, flip_targets


def inject_random_bit_flips_on_subset(
    qc: QuantumCircuit,
    global_wire_order: List[str],
    candidate_qubits: List[str],
    num_flips: int,
    when: str = "middle",
) -> Tuple[QuantumCircuit, List[str]]:
    """
    Randomly flip bits only within the candidate_qubits subset
    """
    if not candidate_qubits:
        raise ValueError("candidate_qubits cannot be empty")
    if num_flips < 1:
        raise ValueError("num_flips must be >= 1")
    if num_flips > len(candidate_qubits):
        raise ValueError("num_flips cannot exceed the number of candidate qubits")

    flip_targets = random.sample(candidate_qubits, num_flips)
    print(f"⚠️ Randomly flipping {num_flips} bits within the subset, injection time={when}, subset_targets={flip_targets}")
    new_qc = _inject_x_errors_by_targets(qc, global_wire_order, flip_targets, when)
    return new_qc, flip_targets


# =========================
# Detection logic for verifiable registers (global)
# =========================

def expected_verifiable_registers(inputs: List[int], lam: int) -> Dict[str, List[int]]:
    """
    Expected verifiable output registers for server/clients:

    - M1_out = p1
    - N2_out = p2
    - For j>=3:
        Nj_out = pj
        Mj_out = pj
    - c0_out = 0
    """
    n = len(inputs)
    expected: Dict[str, List[int]] = {}

    if n >= 1:
        expected["M1"] = int_to_bits_be(inputs[0], lam)

    if n >= 2:
        expected["N2"] = int_to_bits_be(inputs[1], lam)

    for j in range(3, n + 1):
        bits = int_to_bits_be(inputs[j - 1], lam)
        expected[f"N{j}"] = bits
        expected[f"M{j}"] = bits

    expected["c0"] = [0]
    return expected


def get_detecting_clients(
    output_regs: Dict[str, List[int]],
    inputs: List[int],
    lam: int,
) -> Tuple[int, Dict[int, List[str]], bool]:
    """
    Returns:
    - detected_client_count: number of clients that detect the attack
    - detected_by_client: mismatched registers for each client
    - c0_mismatch: whether c0 is abnormal (logged separately and not counted toward client detections)
    """
    n = len(inputs)
    detected_by_client: Dict[int, List[str]] = {}

    # client 1 -> check M1
    if n >= 1:
        expected_m1 = int_to_bits_be(inputs[0], lam)
        if output_regs["M1"] != expected_m1:
            detected_by_client[1] = ["M1"]

    # client 2 -> check N2
    if n >= 2:
        expected_n2 = int_to_bits_be(inputs[1], lam)
        if output_regs["N2"] != expected_n2:
            detected_by_client[2] = ["N2"]

    # client j >= 3 -> check Nj, Mj
    for j in range(3, n + 1):
        mismatches: List[str] = []
        expected_bits = int_to_bits_be(inputs[j - 1], lam)

        if output_regs[f"N{j}"] != expected_bits:
            mismatches.append(f"N{j}")
        if output_regs[f"M{j}"] != expected_bits:
            mismatches.append(f"M{j}")

        if mismatches:
            detected_by_client[j] = mismatches

    c0_mismatch = (output_regs["c0"] != [0])

    detected_client_count = len(detected_by_client)
    return detected_client_count, detected_by_client, c0_mismatch


def print_verification_report(
    output_regs: Dict[str, List[int]],
    inputs: List[int],
    lam: int,
):
    expected = expected_verifiable_registers(inputs, lam)
    print("Verifiable register check:")
    for reg_name, exp_bits in expected.items():
        got_bits = output_regs[reg_name]
        ok = (got_bits == exp_bits)
        print(
            f"  {reg_name:>3}: got={got_bits}"
            f"{f' (int={bits_to_int_be(got_bits)})' if len(got_bits) > 1 else ''}"
            f", expected={exp_bits}"
            f"{f' (int={bits_to_int_be(exp_bits)})' if len(exp_bits) > 1 else ''}"
            f", match={ok}"
        )
    print()


# =========================
# P1-targeted detection logic
# =========================

def get_p1_register_qubits(result: Dict[str, object]) -> List[str]:
    """
    Return all qubit names corresponding to P1's two registers N1 and M1
    """
    n1 = result["N_regs"][0]
    m1 = result["M_regs"][0]
    return list(n1) + list(m1)


def get_p1_detection(
    output_regs: Dict[str, List[int]],
    inputs: List[int],
    lam: int,
) -> Tuple[bool, List[str]]:
    """
    P1 checks only whether its verifiable register M1 still equals p1
    Returns:
    - detected_by_p1: bool
    - mismatch_regs: List[str]
    """
    mismatch_regs: List[str] = []
    expected_m1 = int_to_bits_be(inputs[0], lam)
    if output_regs["M1"] != expected_m1:
        mismatch_regs.append("M1")
    return len(mismatch_regs) > 0, mismatch_regs


# =========================
# Execution
# =========================

def run_local_simulation(qc: QuantumCircuit, shots: int = 1024):
    backend = AerSimulator(method="matrix_product_state")
    job = backend.run(qc, shots=shots)
    result = job.result()
    return result.get_counts()


def run_and_print(
    n: int,
    m: int,
    inputs: List[int],
    shots: int = 1024,
    seed: int = 42,
    random_initial_key: bool = False,
    inject_error: bool = False,
    error_qubit_name: Optional[str] = None,
    error_when: str = "middle",
):
    qc, global_wire_order, result = build_vqac_quantum_circuit(
        n=n,
        m=m,
        inputs=inputs,
        seed=seed,
        random_initial_key=random_initial_key,
    )

    if inject_error:
        if error_qubit_name is None:
            raise ValueError("When inject_error=True, error_qubit_name must be provided")
        qc = inject_bit_flip(
            qc,
            global_wire_order,
            target_qubit_name=error_qubit_name,
            when=error_when,
        )

    N_regs = result["N_regs"]
    M_regs = result["M_regs"]
    c0 = result["c0"]
    lam = result["lambda"]

    print("\n================ Generic VQAC Experiment =================")
    print(f"Number of clients n = {n}")
    print(f"Parameter m = {m}")
    print(f"lambda = {lam}")
    print(f"Number of qubits = {qc.num_qubits}")
    print(f"Circuit depth = {qc.depth()}")
    print(f"Total gate count = {qc.size()}")
    print(f"Input integers = {inputs}")
    print(f"random_initial_key = {random_initial_key}")
    print(f"Inject error = {inject_error}")
    if inject_error:
        print(f"Error location = {error_qubit_name}, injection time = {error_when}")
    print()

    input_regs = pretty_input_registers(N_regs, M_regs, inputs)
    print_registers("Input registers:", input_regs)

    counts = run_local_simulation(qc, shots=shots)
    best = max(counts, key=counts.get)

    print(f"Measured bitstring = {best}\n")

    output_regs = parse_all_registers_from_bitstring(
        bitstring=best,
        global_wire_order=global_wire_order,
        N_regs=N_regs,
        M_regs=M_regs,
        c0=c0,
    )
    print_registers("Output registers:", output_regs)

    print_verification_report(output_regs, inputs, lam)

    detected_client_count, detected_by_client, c0_mismatch = get_detecting_clients(
        output_regs=output_regs,
        inputs=inputs,
        lam=lam,
    )

    p1_detected, p1_mismatches = get_p1_detection(output_regs, inputs, lam)

    print(f"Number of clients detecting the attack = {detected_client_count}")
    print(f"Detection details by client = {detected_by_client}")
    print(f"c0 abnormal = {c0_mismatch}")
    print(f"P1 detected attack = {p1_detected}")
    print(f"P1 mismatched registers = {p1_mismatches}")
    print()

    print("Measurement distribution (top 5):")
    for k, v in sorted(counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {k}: {v}")

    print("\n===============================================\n")


# =========================
# A. Global random bit-flip attack experiments
# =========================

def run_single_attack_experiment(
    n: int,
    m: int,
    inputs: List[int],
    max_flips: int,
    shots: int = 1,
    circuit_seed: int = 42,
    random_initial_key: bool = False,
    verbose: bool = True,
) -> Dict[str, object]:
    qc, global_wire_order, result = build_vqac_quantum_circuit(
        n=n,
        m=m,
        inputs=inputs,
        seed=circuit_seed,
        random_initial_key=random_initial_key,
    )

    total_qubits = len(global_wire_order)
    actual_max_flips = min(max_flips, total_qubits)
    num_flips = random.randint(1, actual_max_flips)
    when = random.choice(["after_enc", "middle"])

    qc, flipped_qubits = inject_random_bit_flips(
        qc,
        global_wire_order,
        num_flips=num_flips,
        when=when,
    )

    counts = run_local_simulation(qc, shots=shots)
    best = max(counts, key=counts.get)

    output_regs = parse_all_registers_from_bitstring(
        bitstring=best,
        global_wire_order=global_wire_order,
        N_regs=result["N_regs"],
        M_regs=result["M_regs"],
        c0=result["c0"],
    )

    lam = result["lambda"]

    detected_client_count, detected_by_client, c0_mismatch = get_detecting_clients(
        output_regs=output_regs,
        inputs=inputs,
        lam=lam,
    )

    detected = (detected_client_count > 0)

    # if verbose:
        #print("--------------------------------------------------")
        #print(f"Input integers                  = {inputs}")
        #print(f"Number of flips                  = {num_flips}")
        #print(f"Injection time                  = {when}")
        #print(f"Flipped qubits              = {flipped_qubits}")
        #print(f"Measured bitstring             = {best}")
        #print_registers("Output registers:", output_regs)
        #print_verification_report(output_regs, inputs, lam)
        #print(f"Number of clients detecting the attack   = {detected_client_count}")
        #print(f"Detection details by client           = {detected_by_client}")
        #print(f"c0 abnormal                = {c0_mismatch}")
        #print(f"Detected by at least one client  = {detected}")

    return {
        "detected": detected,
        "detected_client_count": detected_client_count,
        "detected_by_client": detected_by_client,
        "c0_mismatch": c0_mismatch,
        "num_flips": num_flips,
        "when": when,
        "flipped_qubits": flipped_qubits,
        "bitstring": best,
    }


def run_bit_flip_attack_experiments(
    n: int,
    m: int,
    inputs: List[int],
    num_trials: int = 10,
    max_flips: Optional[int] = None,
    shots: int = 1,
    circuit_seed: int = 42,
    random_seed: Optional[int] = 1234,
    random_initial_key: bool = False,
    verbose_each_trial: bool = True,
) -> Dict[str, float]:
    """
    Global random attack statistics:
    1. Fraction of attacks detected by at least one client
    2. Average number of detecting clients per trial
    3. Average number of flipped bits

    Note:
    When max_flips is None, it defaults to the full set of qubits accessible to the server,
    i.e., the total number of qubits in the global quantum state.
    """
    if random_seed is not None:
        random.seed(random_seed)

    qc_tmp, global_wire_order, result = build_vqac_quantum_circuit(
        n=n,
        m=m,
        inputs=inputs,
        seed=circuit_seed,
        random_initial_key=random_initial_key,
    )

    total_qubits = len(global_wire_order)
    lambda_val = result["lambda"]

    if max_flips is None:
        max_flips = total_qubits

    detected_trials = 0
    total_detected_clients = 0
    total_flips = 0
    c0_mismatch_trials = 0

    print("\n================ Global Random Bit-Flip Attack Statistics =================")
    print(f"Number of clients n = {n}")
    print(f"Parameter m = {m}")
    print(f"lambda = {lambda_val}")
    print(f"Total qubits total_qubits = {total_qubits}")
    print(f"Input integers = {inputs}")
    print(f"Number of trials num_trials = {num_trials}")
    print(f"Maximum flips max_flips = {max_flips}")
    print(f"shots = {shots}")
    print(f"Circuit seed circuit_seed = {circuit_seed}")
    print(f"Attack random seed random_seed = {random_seed}")
    print(f"random_initial_key = {random_initial_key}")
    print("=============================================================\n")

    for t in range(num_trials):
        if verbose_each_trial:
            print(f"[Trial {t + 1}/{num_trials}]")

        trial_result = run_single_attack_experiment(
            n=n,
            m=m,
            inputs=inputs,
            max_flips=max_flips,
            shots=shots,
            circuit_seed=circuit_seed,
            random_initial_key=random_initial_key,
            verbose=verbose_each_trial,
        )

        if trial_result["detected"]:
            detected_trials += 1
        total_detected_clients += trial_result["detected_client_count"]
        total_flips += trial_result["num_flips"]
        if trial_result["c0_mismatch"]:
            c0_mismatch_trials += 1

        if verbose_each_trial:
            print()

    detection_rate = detected_trials / num_trials if num_trials > 0 else 0.0
    avg_detected_clients = total_detected_clients / num_trials if num_trials > 0 else 0.0
    avg_flips = total_flips / num_trials if num_trials > 0 else 0.0
    c0_mismatch_rate = c0_mismatch_trials / num_trials if num_trials > 0 else 0.0

    print("================ Global Experiment Summary ================")
    print(f"Total trials                    = {num_trials}")
    print(f"Trials detected by at least one client = {detected_trials}")
    print(f"Detection rate                  = {detection_rate:.4f}")
    print(f"Average number of detecting clients = {avg_detected_clients:.4f}")
    print(f"Average flipped bits            = {avg_flips:.4f}")
    print(f"c0 mismatch rate               = {c0_mismatch_rate:.4f}")
    print("================================================\n")

    return {
        "detection_rate": detection_rate,
        "avg_detected_clients": avg_detected_clients,
        "avg_flips": avg_flips,
        "c0_mismatch_rate": c0_mismatch_rate,
    }


# =========================
# B. P1-targeted attack experiments
# =========================

def run_single_p1_targeted_attack_experiment(
    n: int,
    m: int,
    inputs: List[int],
    max_flips_on_p1: Optional[int] = None,
    shots: int = 1,
    circuit_seed: int = 42,
    random_initial_key: bool = False,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Randomly apply bit flips only on P1's two registers N1 and M1,
    then check only whether P1's verifiable register M1 is abnormal
    """
    qc, global_wire_order, result = build_vqac_quantum_circuit(
        n=n,
        m=m,
        inputs=inputs,
        seed=circuit_seed,
        random_initial_key=random_initial_key,
    )

    p1_qubits = get_p1_register_qubits(result)
    if max_flips_on_p1 is None:
        max_flips_on_p1 = len(p1_qubits)

    actual_max_flips = min(max_flips_on_p1, len(p1_qubits))
    num_flips = random.randint(1, actual_max_flips)
    when = random.choice(["after_enc", "middle"])

    qc, flipped_qubits = inject_random_bit_flips_on_subset(
        qc,
        global_wire_order,
        candidate_qubits=p1_qubits,
        num_flips=num_flips,
        when=when,
    )

    counts = run_local_simulation(qc, shots=shots)
    best = max(counts, key=counts.get)

    output_regs = parse_all_registers_from_bitstring(
        bitstring=best,
        global_wire_order=global_wire_order,
        N_regs=result["N_regs"],
        M_regs=result["M_regs"],
        c0=result["c0"],
    )

    lam = result["lambda"]
    p1_detected, p1_mismatches = get_p1_detection(output_regs, inputs, lam)

    # if verbose:
        #print("--------------------------------------------------")
        #print("P1-targeted attack experiment")
        #print(f"Input integers                  = {inputs}")
        #print(f"Qubits attackable by P1           = {p1_qubits}")
        #print(f"Number of flips                  = {num_flips}")
        #print(f"Injection time                  = {when}")
        #print(f"Flipped qubits              = {flipped_qubits}")
        #print(f"Measured bitstring             = {best}")
        #print_registers("Output registers:", output_regs)
        #print(f"P1 detected attack         = {p1_detected}")
        #print(f"P1 mismatched registers    = {p1_mismatches}")

    return {
        "p1_detected": p1_detected,
        "p1_mismatches": p1_mismatches,
        "num_flips": num_flips,
        "when": when,
        "flipped_qubits": flipped_qubits,
        "bitstring": best,
    }


def run_p1_targeted_attack_experiments(
    n: int,
    m: int,
    inputs: List[int],
    num_trials: int = 10,
    max_flips_on_p1: Optional[int] = None,
    shots: int = 1,
    circuit_seed: int = 42,
    random_seed: Optional[int] = 1234,
    random_initial_key: bool = False,
    verbose_each_trial: bool = True,
) -> Dict[str, float]:
    """
    Randomly flip bits only in P1's two registers N1 and M1,
    and check only whether P1's verifiable register M1 is abnormal.
    Statistics:
    - P1 detection rate
    - average number of flipped bits
    """
    if random_seed is not None:
        random.seed(random_seed)

    _, _, result = build_vqac_quantum_circuit(
        n=n,
        m=m,
        inputs=inputs,
        seed=circuit_seed,
        random_initial_key=random_initial_key,
    )
    p1_qubits = get_p1_register_qubits(result)

    if max_flips_on_p1 is None:
        max_flips_on_p1 = len(p1_qubits)

    p1_detected_trials = 0
    total_flips = 0

    print("\n================ P1-Targeted Random Bit-Flip Attack Statistics =================")
    print(f"Number of clients n = {n}")
    print(f"Parameter m = {m}")
    print(f"Input integers = {inputs}")
    print(f"P1 attack subset qubits = {p1_qubits}")
    print(f"Number of trials num_trials = {num_trials}")
    print(f"Maximum flips max_flips_on_p1 = {max_flips_on_p1}")
    print(f"shots = {shots}")
    print(f"Circuit seed circuit_seed = {circuit_seed}")
    print(f"Attack random seed random_seed = {random_seed}")
    print(f"random_initial_key = {random_initial_key}")
    print("================================================================\n")

    for t in range(num_trials):
        if verbose_each_trial:
            print(f"[Trial {t + 1}/{num_trials}]")

        trial_result = run_single_p1_targeted_attack_experiment(
            n=n,
            m=m,
            inputs=inputs,
            max_flips_on_p1=max_flips_on_p1,
            shots=shots,
            circuit_seed=circuit_seed,
            random_initial_key=random_initial_key,
            verbose=verbose_each_trial,
        )

        if trial_result["p1_detected"]:
            p1_detected_trials += 1
        total_flips += trial_result["num_flips"]

        if verbose_each_trial:
            print()

    p1_detection_rate = p1_detected_trials / num_trials if num_trials > 0 else 0.0
    avg_flips = total_flips / num_trials if num_trials > 0 else 0.0

    print("================ P1-Targeted Experiment Summary ================")
    print(f"Total trials                = {num_trials}")
    print(f"Trials where P1 detected the attack = {p1_detected_trials}")
    print(f"P1 detection rate                  = {p1_detection_rate:.4f}")
    print(f"Average flipped bits               = {avg_flips:.4f}")
    print("===================================================\n")

    return {
        "p1_detection_rate": p1_detection_rate,
        "avg_flips": avg_flips,
    }

def run_full_experiment_table(
    n_list=[2, 3],
    m_list=[2, 3, 4],
    num_trials=100,
    shots=1,
    circuit_seed=42,
    random_seed=1234,
    random_initial_key=False,
):
    """
    Automatically run:
    n ∈ {2,3}, m ∈ {2,3,4}

    Output:
    n & m & Global AF & Global DR & Targeted AF & Targeted DR
    """

    print("\n================ Automatic Experiment Start ================\n")

    results = []

    for n in n_list:
        for m in m_list:
            print(f"\n===== Running (n={n}, m={m}) =====")

            inputs = list(range(1, n + 1))  # Fixed inputs [1,2] or [1,2,3]

            # ---------- Global ----------
            global_res = run_bit_flip_attack_experiments(
                n=n,
                m=m,
                inputs=inputs,
                num_trials=num_trials,
                max_flips=None,  # Use the full global qubit set
                shots=shots,
                circuit_seed=circuit_seed,
                random_seed=random_seed,
                random_initial_key=random_initial_key,
                verbose_each_trial=False,
            )

            # ---------- Targeted ----------
            targeted_res = run_p1_targeted_attack_experiments(
                n=n,
                m=m,
                inputs=inputs,
                num_trials=num_trials,
                max_flips_on_p1=None,
                shots=shots,
                circuit_seed=circuit_seed,
                random_seed=random_seed,
                random_initial_key=random_initial_key,
                verbose_each_trial=False,
            )

            results.append({
                "n": n,
                "m": m,
                "global_af": global_res["avg_flips"],
                "global_dr": global_res["detection_rate"],
                "targeted_af": targeted_res["avg_flips"],
                "targeted_dr": targeted_res["p1_detection_rate"],
            })

    print("\n================ Experiments Complete ================\n")

    # =========================
    # Output LaTeX table
    # =========================
    print("\n===== LaTeX Table (copy directly) =====\n")

    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Detection performance under global and targeted bit-flip attacks}")
    print("\\begin{tabular}{c c c c c c}")
    print("\\toprule")
    print("$n$ & $m$ & Global AF & Global DR & Targeted AF & Targeted DR \\\\")
    print("\\midrule")

    for r in results:
        print(f"{r['n']} & {r['m']} & "
              f"{r['global_af']:.2f} & {r['global_dr']:.2f} & "
              f"{r['targeted_af']:.2f} & {r['targeted_dr']:.2f} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    return results



if __name__ == "__main__":
    run_full_experiment_table(
        n_list=[4],
        m_list=[2, 3, 4],
        num_trials=100,   # Change here to control experiment intensity
    )