
# -*- coding: utf-8 -*-
"""
Generate client encryption key sequences E_i and computation key sequences G_i
(chain-based version with corrected multi-client block ordering and gate composition).

Key fixes:
1. Multi-client blocks are no longer composed using tensor(); instead, they are
   explicitly composed using a shared wire ordering.
2. If theoretically G = K_after · A · K_before^\dagger, then the circuit append
   order must be:
       K_before^\dagger  ->  A  ->  K_after
   because a quantum circuit executes left-to-right, while the corresponding
   matrix multiplication is applied right-to-left.
"""

from __future__ import annotations
import importlib.util
import os
from dataclasses import dataclass
from typing import Dict, List, Set, Any, Optional

from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Operator
from qiskit.quantum_info.random import random_clifford


def _candidate_builder_paths() -> List[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    return [
        os.path.join(here, "vqac_builder.py"),
        os.path.join(cwd, "vqac_builder.py"),
        "/mnt/data/vqac_builder.py",
    ]


def _load_builder_module():
    import sys
    for path in _candidate_builder_paths():
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("vqac_builder", path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules["vqac_builder"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("vqac_builder.py not found. Please place it in the same directory as this file.")


vqac_builder = _load_builder_module()
build_vqac_nm_circuit = vqac_builder.build_vqac_nm_circuit
lambda_len = vqac_builder.lambda_len
Gate = vqac_builder.Gate


@dataclass
class EEntry:
    name: str
    local_slot: int
    global_gate_index: Optional[int]
    gate_text: str
    num_qubits: int
    clifford: Clifford
    circuit: QuantumCircuit
    wire_order: List[str]


@dataclass
class GEntry:
    name: str
    global_gate_index: int
    gate_text: str
    num_qubits: int
    circuit: QuantumCircuit
    wire_order: List[str]
    participant_clients: List[str]
    participant_local_slots: Dict[str, int]
    gate_kind: str
    operator: Optional[Operator] = None


def gate_to_str(gate: Gate) -> str:
    if gate.name == "CNOT":
        return f"CNOT({gate.qubits[0]}, {gate.qubits[1]})"
    if gate.name == "TOFFOLI":
        return f"TOFFOLI({gate.qubits[0]}, {gate.qubits[1]}, {gate.qubits[2]})"
    return f"{gate.name}{gate.qubits}"


def build_client_wire_order(N_reg: List[str], M_reg: List[str]) -> List[str]:
    return list(N_reg) + list(M_reg)


def inverse_clifford_circuit(clf: Clifford) -> QuantumCircuit:
    return clf.to_circuit().inverse()


def identity_clifford(num_qubits: int) -> Clifford:
    return Clifford(QuantumCircuit(num_qubits))


def append_embedded_gate(qc: QuantumCircuit, gate: Gate, wire_order: List[str]):
    pos = {q: i for i, q in enumerate(wire_order)}
    missing = [q for q in gate.qubits if q not in pos]
    if missing:
        raise ValueError(f"Gate {gate_to_str(gate)} cannot be embedded into wire order {wire_order}; missing qubits: {missing}")

    if gate.name == "CNOT":
        c, t = gate.qubits
        qc.cx(pos[c], pos[t])
    elif gate.name == "TOFFOLI":
        c1, c2, t = gate.qubits
        qc.ccx(pos[c1], pos[c2], pos[t])
    else:
        raise ValueError(f"Unsupported gate type: {gate.name}")


def get_client_gate_sequences(builder, N_regs, M_regs):
    return builder.get_client_gate_sequences(N_regs, M_regs)


def make_qubit_owner_map(N_regs: List[List[str]], M_regs: List[List[str]], c0: str) -> Dict[str, str]:
    owner = {c0: "PUBLIC"}
    for i in range(len(N_regs)):
        p = f"P{i+1}"
        for q in N_regs[i] + M_regs[i]:
            owner[q] = p
    return owner


def clients_in_gate(gate: Gate, qubit_owner: Dict[str, str]) -> List[str]:
    s: Set[str] = set()
    for q in gate.qubits:
        owner = qubit_owner[q]
        if owner != "PUBLIC":
            s.add(owner)
    return sorted(s, key=lambda x: int(x[1:]))


def make_shared_wire_order(participants: List[str], client_wire_map: Dict[str, List[str]], include_c0: bool) -> List[str]:
    wire_order: List[str] = []
    for p in participants:
        wire_order.extend(client_wire_map[p])
    if include_c0:
        wire_order.append("c0")
    return wire_order


def block_indices_for_participants(participants: List[str], client_wire_map: Dict[str, List[str]]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    cursor = 0
    for p in participants:
        width = len(client_wire_map[p])
        out[p] = list(range(cursor, cursor + width))
        cursor += width
    return out


def compose_client_blocks(total_qubits: int, participants: List[str], client_wire_map: Dict[str, List[str]], circuits_by_client: Dict[str, QuantumCircuit]) -> QuantumCircuit:
    qc = QuantumCircuit(total_qubits)
    block_map = block_indices_for_participants(participants, client_wire_map)
    for p in participants:
        circ = circuits_by_client[p]
        expected = len(block_map[p])
        if circ.num_qubits != expected:
            raise ValueError(f"Qubit count mismatch for {p}: circ={circ.num_qubits}, expected={expected}")
        qc.compose(circ, qubits=block_map[p], inplace=True)
    return qc


def generate_E_and_G(
    n: int,
    m: int,
    seed: Optional[int] = None,
    build_operator_for_G: bool = False,
    random_initial_key: bool = False,
):
    builder, N_regs, M_regs, c0 = build_vqac_nm_circuit(n, m)
    lam = lambda_len(n, m)

    client_sched = get_client_gate_sequences(builder, N_regs, M_regs)
    qubit_owner = make_qubit_owner_map(N_regs, M_regs, c0)
    client_wire_map: Dict[str, List[str]] = {
        f"P{i+1}": build_client_wire_order(N_regs[i], M_regs[i]) for i in range(n)
    }

    E: Dict[str, List[EEntry]] = {f"P{i+1}": [] for i in range(n)}
    G: Dict[str, List[GEntry]] = {f"P{i+1}": [] for i in range(n)}
    local_slot_of_global: Dict[str, Dict[int, int]] = {f"P{i+1}": {} for i in range(n)}

    rng_seed_base = 0 if seed is None else seed

    for i in range(n):
        p = f"P{i+1}"
        wire_order = client_wire_map[p]
        clf0 = random_clifford(2 * lam, seed=rng_seed_base + 100000 * i) if random_initial_key else identity_clifford(2 * lam)
        E[p].append(
            EEntry(
                name=f"E_{p}[0]",
                local_slot=0,
                global_gate_index=None,
                gate_text="INIT",
                num_qubits=2 * lam,
                clifford=clf0,
                circuit=clf0.to_circuit(),
                wire_order=list(wire_order),
            )
        )

    for i in range(n):
        p = f"P{i+1}"
        wire_order = client_wire_map[p]
        for t, (global_idx, gate) in enumerate(client_sched[p]["gates"], start=1):
            clf = random_clifford(2 * lam, seed=rng_seed_base + 100000 * i + t)
            E[p].append(
                EEntry(
                    name=f"E_{p}[{t}]",
                    local_slot=t,
                    global_gate_index=global_idx,
                    gate_text=gate_to_str(gate),
                    num_qubits=2 * lam,
                    clifford=clf,
                    circuit=clf.to_circuit(),
                    wire_order=list(wire_order),
                )
            )
            local_slot_of_global[p][global_idx] = t

    processed_global_gate: Set[int] = set()
    global_G_by_index: Dict[int, GEntry] = {}

    for global_idx, gate in enumerate(builder.gates, start=1):
        participants = clients_in_gate(gate, qubit_owner)
        include_c0 = ("c0" in gate.qubits)

        if not participants:
            continue
        if global_idx in processed_global_gate:
            continue
        processed_global_gate.add(global_idx)

        local_slots = {p: local_slot_of_global[p][global_idx] for p in participants}
        total_qubits = sum(len(client_wire_map[p]) for p in participants) + (1 if include_c0 else 0)
        wire_order = make_shared_wire_order(participants, client_wire_map, include_c0)

        # K_after
        after_circuits = {p: E[p][local_slots[p]].circuit for p in participants}
        front_full = compose_client_blocks(total_qubits, participants, client_wire_map, after_circuits)

        # A
        middle = QuantumCircuit(total_qubits)
        append_embedded_gate(middle, gate, wire_order)

        # K_before^\dagger
        before_inv_circuits = {p: inverse_clifford_circuit(E[p][local_slots[p] - 1].clifford) for p in participants}
        back_full = compose_client_blocks(total_qubits, participants, client_wire_map, before_inv_circuits)

        # Critical: circuit append order must be back -> middle -> front
        g_circuit = QuantumCircuit(total_qubits)
        g_circuit.compose(back_full, inplace=True)
        g_circuit.compose(middle, inplace=True)
        g_circuit.compose(front_full, inplace=True)

        gate_kind = "non-Clifford" if gate.name == "TOFFOLI" else "Clifford"
        g_operator = Operator(g_circuit) if build_operator_for_G else None

        shared_entry = GEntry(
            name=f"G_global[{global_idx}]",
            global_gate_index=global_idx,
            gate_text=gate_to_str(gate),
            num_qubits=total_qubits,
            circuit=g_circuit,
            wire_order=list(wire_order),
            participant_clients=list(participants),
            participant_local_slots=dict(local_slots),
            gate_kind=gate_kind,
            operator=g_operator,
        )
        global_G_by_index[global_idx] = shared_entry
        for p in participants:
            G[p].append(shared_entry)

    return {
        "n": n,
        "m": m,
        "lambda": lam,
        "builder": builder,
        "N_regs": N_regs,
        "M_regs": M_regs,
        "c0": c0,
        "E": E,
        "G": G,
        "global_G_by_index": global_G_by_index,
        "client_schedules": client_sched,
        "client_wire_map": client_wire_map,
        "random_initial_key": random_initial_key,
    }


def print_summary(result: Dict[str, Any], show_circuit_depth: bool = False):
    n = result["n"]
    m = result["m"]
    lam = result["lambda"]
    print("===== Chain-Based E/G Summary =====")
    print(f"VQAC({n},{m}), lambda = {lam}")
    print(f"random_initial_key = {result.get('random_initial_key', False)}")
    print()

    for p in sorted(result["E"].keys(), key=lambda x: int(x[1:])):
        print(f"{p}:")
        print(f"  E length (including E[0]) = {len(result['E'][p])}")
        for e in result["E"][p][:6]:
            depth = e.circuit.depth() if show_circuit_depth else "-"
            print(f"    {e.name}: local_slot={e.local_slot}, global#{e.global_gate_index}, gate={e.gate_text}, {e.num_qubits}q, depth={depth}")
        if len(result["E"][p]) > 6:
            print("    ...")

    print("Shared G entries (by global gate index):")
    for global_idx in sorted(result["global_G_by_index"].keys())[:8]:
        g = result["global_G_by_index"][global_idx]
        depth = g.circuit.depth() if show_circuit_depth else "-"
        print(f"  global#{global_idx}: {g.name}, gate={g.gate_text}, participants={g.participant_clients}, slots={g.participant_local_slots}, {g.num_qubits}q, kind={g.gate_kind}, depth={depth}")
    if len(result["global_G_by_index"]) > 8:
        print("  ...")


if __name__ == "__main__":
    result = generate_E_and_G(n=2, m=2, seed=42, build_operator_for_G=False, random_initial_key=False)
    print_summary(result, show_circuit_depth=False)
