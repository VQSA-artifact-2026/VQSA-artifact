# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Tuple
from collections import defaultdict
import math


@dataclass
class Gate:
    name: str
    qubits: Tuple[str, ...]


class CircuitBuilder:
    def __init__(self):
        self.gates: List[Gate] = []

    def add_gate(self, name: str, *qubits: str):
        self.gates.append(Gate(name=name, qubits=tuple(qubits)))

    def cnot(self, control: str, target: str):
        self.add_gate("CNOT", control, target)

    def toffoli(self, control1: str, control2: str, target: str):
        self.add_gate("TOFFOLI", control1, control2, target)

    def _format_gate(self, idx: int, gate: Gate) -> str:
        """
        Format a gate as a human-readable string.
        """
        if gate.name == "CNOT":
            return f"{idx:03d}. CNOT({gate.qubits[0]}, {gate.qubits[1]})"
        elif gate.name == "TOFFOLI":
            return f"{idx:03d}. TOFFOLI({gate.qubits[0]}, {gate.qubits[1]}, {gate.qubits[2]})"
        else:
            return f"{idx:03d}. {gate.name}{gate.qubits}"

    def print_global_gate_sequence(self):
        """
        Print the full global gate execution order.
        """
        print("===== Global Gate Execution Order =====")
        for idx, gate in enumerate(self.gates, start=1):
            print(self._format_gate(idx, gate))
        print()

    def print_per_qubit_schedule(self):
        """
        Print the sequence of gates involving each qubit.
        """
        per_qubit = defaultdict(list)

        for idx, gate in enumerate(self.gates, start=1):
            if gate.name == "CNOT":
                c, t = gate.qubits
                per_qubit[c].append(f"{idx:03d}: CNOT(control -> {t})")
                per_qubit[t].append(f"{idx:03d}: CNOT(target <- {c})")
            elif gate.name == "TOFFOLI":
                c1, c2, t = gate.qubits
                per_qubit[c1].append(f"{idx:03d}: TOFFOLI(control1 -> {c2}, {t})")
                per_qubit[c2].append(f"{idx:03d}: TOFFOLI(control2 -> {c1}, {t})")
                per_qubit[t].append(f"{idx:03d}: TOFFOLI(target <- {c1}, {c2})")

        print("===== Per-Qubit Gate Schedule =====")
        for qubit in sorted(per_qubit.keys()):
            print(f"{qubit}:")
            for item in per_qubit[qubit]:
                print(f"  {item}")
            print()

    def get_client_gate_sequences(self, N_regs: List[List[str]], M_regs: List[List[str]]):
        """
        Return the gate sequence associated with each client Pi (Interpretation A):
        a gate is included in Pi's schedule if it touches any qubit in
        Pi's Ni[*] or Mi[*] registers.
        """
        if len(N_regs) != len(M_regs):
            raise ValueError("N_regs and M_regs must have the same length")

        client_schedules = {}

        for i in range(len(N_regs)):
            client_name = f"P{i+1}"
            client_qubits = set(N_regs[i] + M_regs[i])
            involved_gates = []

            for idx, gate in enumerate(self.gates, start=1):
                # Interpretation A: include the gate if it intersects with any qubit owned by the current client.
                if any(q in client_qubits for q in gate.qubits):
                    involved_gates.append((idx, gate))

            client_schedules[client_name] = {
                "qubits": list(N_regs[i] + M_regs[i]),
                "gates": involved_gates
            }

        return client_schedules

    def print_per_client_schedule(self, N_regs: List[List[str]], M_regs: List[List[str]]):
        """
        Print the gate sequence corresponding to each client Pi (Interpretation A).
        """
        client_schedules = self.get_client_gate_sequences(N_regs, M_regs)

        print("===== Per-Client Gate Schedule =====")
        for client_name, info in client_schedules.items():
            print(f"{client_name}:")
            print("  Owned qubits:")
            print(f"  {', '.join(info['qubits'])}")

            print("  Gate sequence:")
            if not info["gates"]:
                print("    (none)")
            else:
                for idx, gate in info["gates"]:
                    print(f"    {self._format_gate(idx, gate)}")
            print()


def lambda_len(n: int, m: int) -> int:
    if n < 2:
        raise ValueError("n must be at least 2")
    return m + math.ceil(math.log2(n))


def make_registers(n: int, m: int):
    lam = lambda_len(n, m)
    N_regs = [[f"N{i+1}[{k}]" for k in range(lam)] for i in range(n)]
    M_regs = [[f"M{i+1}[{k}]" for k in range(lam)] for i in range(n)]
    c0 = "c0"
    return N_regs, M_regs, c0


def reg_to_str(reg: List[str]) -> str:
    return "[" + ", ".join(reg) + "]"


def GMAJ(builder: CircuitBuilder, u: str, v: str, x: str, y: str, z: str):
    builder.cnot(z, v)
    builder.cnot(z, x)
    builder.cnot(z, u)
    builder.toffoli(u, x, z)
    builder.cnot(x, v)


def GUMA(builder: CircuitBuilder, u: str, v: str, x: str, y: str, z: str):
    builder.cnot(x, y)
    builder.toffoli(u, x, z)
    builder.cnot(z, u)
    builder.cnot(u, x)
    builder.cnot(u, y)


def _right_aligned_view(reg: List[str], width: int) -> List[str]:
    if len(reg) < width:
        raise ValueError(f"Register length {len(reg)} is smaller than required width {width}")
    start = len(reg) - width
    return reg[start:]


def apply_vqac2_gate_template(
    builder: CircuitBuilder,
    width: int,
    N_a: List[str],
    M_a: List[str],
    N_b: List[str],
    M_b: List[str],
    carry_qubit: str
):
    if width < 2:
        raise ValueError("The width of VQAC(2, width) must be at least 2")
    if not (len(N_a) >= width and len(M_a) >= width and len(N_b) >= width and len(M_b) >= width):
        raise ValueError(f"apply_vqac2_gate_template requires register length at least {width}")

    N_a_w = _right_aligned_view(N_a, width)
    M_a_w = _right_aligned_view(M_a, width)
    N_b_w = _right_aligned_view(N_b, width)
    M_b_w = _right_aligned_view(M_b, width)

    last = width - 1

    for i in range(0, width - 1):
        if i == 0:
            GMAJ(builder, carry_qubit, N_b_w[last], M_b_w[last], N_a_w[last], M_a_w[last])
        else:
            GMAJ(builder, M_a_w[last + 1 - i], N_b_w[last - i], M_b_w[last - i], N_a_w[last - i], M_a_w[last - i])

    builder.cnot(M_a_w[1], N_a_w[0])
    builder.cnot(M_a_w[1], M_b_w[0])

    for i in range(width - 2, -1, -1):
        if i == 0:
            GUMA(builder, carry_qubit, N_b_w[last], M_b_w[last], N_a_w[last], M_a_w[last])
        else:
            GUMA(builder, M_a_w[last + 1 - i], N_b_w[last - i], M_b_w[last - i], N_a_w[last - i], M_a_w[last - i])


def VQAC_n(builder: CircuitBuilder, n: int, m: int, N_regs: List[List[str]], M_regs: List[List[str]], c0: str):
    lam = lambda_len(n, m)

    if len(N_regs) != n or len(M_regs) != n:
        raise ValueError("The numbers of N_regs and M_regs must both equal n")

    for i in range(n):
        if len(N_regs[i]) != lam or len(M_regs[i]) != lam:
            raise ValueError(f"Register set {i+1} must have length λ={lam}")

    # Main computation stage
    for j in range(1, n):
        if j == 1:
            apply_vqac2_gate_template(builder, lam, N_regs[0], M_regs[0], N_regs[1], M_regs[1], c0)
        elif j == 2:
            apply_vqac2_gate_template(builder, lam, N_regs[2], M_regs[2], M_regs[1], N_regs[0], c0)
        else:
            apply_vqac2_gate_template(builder, lam, N_regs[j], M_regs[j], N_regs[j - 1], N_regs[0], c0)

    # Perform the final copy step only when n > 2
    if n > 2:
        for i in range(0, lam):
            builder.cnot(N_regs[0][i], M_regs[1][i])
            for j in range(3, n):
                builder.cnot(N_regs[0][i], N_regs[j - 1][i])


def build_vqac_nm_circuit(n: int, m: int):
    builder = CircuitBuilder()
    N_regs, M_regs, c0 = make_registers(n, m)
    VQAC_n(builder, n, m, N_regs, M_regs, c0)
    return builder, N_regs, M_regs, c0


if __name__ == "__main__":
    n, m = 2, 2
    builder, N_regs, M_regs, c0 = build_vqac_nm_circuit(n, m)

    print(f"===== VQAC({n},{m}) Builder Summary (Unified λ-bit Width) =====")
    print(f"lambda = {lambda_len(n, m)}")
    print(f"Total gate count = {len(builder.gates)}")
    for i in range(n):
        print(f"N{i+1} = {reg_to_str(N_regs[i])}")
        print(f"M{i+1} = {reg_to_str(M_regs[i])}")
    print(f"c0 = {c0}")
    print()

    builder.print_global_gate_sequence()
    builder.print_per_qubit_schedule()
    builder.print_per_client_schedule(N_regs, M_regs)