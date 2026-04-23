# -*- coding: utf-8 -*-
from typing import Dict, List
import random

from vqac_builder import Gate, build_vqac_nm_circuit, lambda_len, make_registers


def int_to_bits_msb(x: int, length: int) -> List[int]:
    if x < 0 or x >= (1 << length):
        raise ValueError(f"x={x} is out of range for a {length}-bit representation")
    s = bin(x)[2:].zfill(length)
    return [int(ch) for ch in s]


def bits_msb_to_int(bits: List[int]) -> int:
    return int("".join(str(b) for b in bits), 2)


def pretty_bits(bits: List[int]) -> str:
    return "".join(str(b) for b in bits)


def zero_bits(length: int) -> List[int]:
    return [0] * length


def build_initial_state_dict(n: int, m: int, inputs: List[int]):
    if len(inputs) != n:
        raise ValueError("inputs must have length n")

    lam = lambda_len(n, m)
    total_sum = sum(inputs)
    if total_sum >= (1 << lam):
        raise ValueError(f"sum(inputs)={total_sum} exceeds λ={lam}")

    N_regs, M_regs, c0 = make_registers(n, m)

    state: Dict[str, int] = {}
    for i in range(n):
        for q, bit in zip(N_regs[i], zero_bits(lam)):
            state[q] = bit
        for q, bit in zip(M_regs[i], int_to_bits_msb(inputs[i], lam)):
            state[q] = bit

    state[c0] = 0
    return state, N_regs, M_regs, c0


def extract_reg_value(state: Dict[str, int], reg: List[str]) -> List[int]:
    return [state[q] for q in reg]


def apply_gate_classically(state: Dict[str, int], gate: Gate):
    if gate.name == "CNOT":
        control, target = gate.qubits
        state[target] ^= state[control]
    elif gate.name == "TOFFOLI":
        c1, c2, target = gate.qubits
        state[target] ^= (state[c1] & state[c2])
    else:
        raise NotImplementedError(f"This verifier currently supports only CNOT / TOFFOLI, received {gate.name}")


def apply_circuit_classically(state: Dict[str, int], gates: List[Gate]):
    for gate in gates:
        apply_gate_classically(state, gate)


def check_vqac_nm_semantics(n: int, m: int, inputs: List[int], verbose: bool = True) -> bool:
    lam = lambda_len(n, m)
    total_sum = sum(inputs)

    # The current builder returns a builder object.
    builder, N_regs, M_regs, c0 = build_vqac_nm_circuit(n, m)
    gates = builder.gates

    state, _, _, _ = build_initial_state_dict(n, m, inputs)
    apply_circuit_classically(state, gates)

    N_out = [extract_reg_value(state, reg) for reg in N_regs]
    M_out = [extract_reg_value(state, reg) for reg in M_regs]

    expect_sum = int_to_bits_msb(total_sum, lam)
    expect_inputs = [int_to_bits_msb(v, lam) for v in inputs]

    ok = True

    # =========================
    # Basic checks for all n
    # =========================

    if N_out[0] != expect_sum:
        ok = False
        if verbose:
            print("[Mismatch] N1 != sum")
            print(f"  Expected N1 = {pretty_bits(expect_sum)} ({total_sum})")
            print(f"  Actual   N1 = {pretty_bits(N_out[0])} ({bits_msb_to_int(N_out[0])})")

    if M_out[0] != expect_inputs[0]:
        ok = False
        if verbose:
            print("[Mismatch] M1 != p1")
            print(f"  Expected M1 = {pretty_bits(expect_inputs[0])} ({inputs[0]})")
            print(f"  Actual   M1 = {pretty_bits(M_out[0])} ({bits_msb_to_int(M_out[0])})")

    if n >= 2:
        if N_out[1] != expect_inputs[1]:
            ok = False
            if verbose:
                print("[Mismatch] N2 != p2")
                print(f"  Expected N2 = {pretty_bits(expect_inputs[1])} ({inputs[1]})")
                print(f"  Actual   N2 = {pretty_bits(N_out[1])} ({bits_msb_to_int(N_out[1])})")

    # =========================
    # For n=2, the current implementation does not perform the final copy step
    # so M2 == sum is not treated as a required property here.
    # =========================
    if n >= 2:
        if M_out[1] != expect_sum:
            ok = False
            if verbose:
                print("[Mismatch] M2 != sum")
                print(f"  Expected M2 = {pretty_bits(expect_sum)} ({total_sum})")
                print(f"  Actual   M2 = {pretty_bits(M_out[1])} ({bits_msb_to_int(M_out[1])})")

    # =========================
    # For n>2, continue checking the post-copy semantics.
    # =========================
    if n > 2:
        if M_out[1] != expect_sum:
            ok = False
            if verbose:
                print("[Mismatch] M2 != sum")
                print(f"  Expected M2 = {pretty_bits(expect_sum)} ({total_sum})")
                print(f"  Actual   M2 = {pretty_bits(M_out[1])} ({bits_msb_to_int(M_out[1])})")

        for i in range(2, n):
            if N_out[i] != expect_sum:
                ok = False
                if verbose:
                    print(f"[Mismatch] N{i+1} != sum")
                    print(f"  Expected N{i+1} = {pretty_bits(expect_sum)} ({total_sum})")
                    print(f"  Actual   N{i+1} = {pretty_bits(N_out[i])} ({bits_msb_to_int(N_out[i])})")

            if M_out[i] != expect_inputs[i]:
                ok = False
                if verbose:
                    print(f"[Mismatch] M{i+1} != p{i+1}")
                    print(f"  Expected M{i+1} = {pretty_bits(expect_inputs[i])} ({inputs[i]})")
                    print(f"  Actual   M{i+1} = {pretty_bits(M_out[i])} ({bits_msb_to_int(M_out[i])})")

    if verbose:
        print(f"===== VQAC({n},{m}) Input =====")
        for i in range(n):
            print(f"N{i+1} = {pretty_bits(zero_bits(lam))} (0)")
            print(f"M{i+1} = {pretty_bits(expect_inputs[i])} ({inputs[i]})")
        print("c0 = 0")

        print(f"\n===== VQAC({n},{m}) Output =====")
        for i in range(n):
            print(f"N{i+1} = {pretty_bits(N_out[i])} ({bits_msb_to_int(N_out[i])})")
            print(f"M{i+1} = {pretty_bits(M_out[i])} ({bits_msb_to_int(M_out[i])})")

        print(f"\nTotal gate count = {len(gates)}")
        print(f"Result: {'PASS' if ok else 'FAIL'}\n")

    return ok


def run_random_tests(n: int, m: int, num_tests: int = 100):
    lam = lambda_len(n, m)
    max_input = (1 << m) - 1

    passed = 0
    attempted = 0

    for _ in range(num_tests):
        inputs = [random.randint(0, max_input) for _ in range(n)]
        if sum(inputs) >= (1 << lam):
            continue

        attempted += 1
        if check_vqac_nm_semantics(n, m, inputs, verbose=False):
            passed += 1

    print(f"[Random tests VQAC({n},{m})] Passed {passed}/{attempted}")


if __name__ == "__main__":
    print("===== Fixed Examples =====\n")

    # Added n=2 verification
    check_vqac_nm_semantics(n=2, m=2, inputs=[1, 2], verbose=True)

    # Existing examples
    check_vqac_nm_semantics(n=3, m=2, inputs=[1, 2, 3], verbose=True)
    check_vqac_nm_semantics(n=5, m=2, inputs=[1, 2, 1, 3, 2], verbose=True)

    print("===== Random Examples =====\n")
    run_random_tests(n=2, m=2, num_tests=500)
    run_random_tests(n=3, m=2, num_tests=500)
    run_random_tests(n=9, m=2, num_tests=500)