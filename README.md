# VQSA Artifact Package

This repository provides an anonymous artifact package for the paper on **Verifiable Quantum Secure Aggregation (VQSA)**.

The package is organized into two parts:

- **`core/`**: minimal protocol implementation and correctness verification
- **`experiments/`**: scripts for reproducing the main experimental results reported in the paper

The artifact is designed so that reviewers can first verify the **core correctness claim** with a minimal setup, and then optionally run the additional experimental scripts.

---

## Repository Structure

```text
.
├── README.md
├── run_minimal_repro.py
├── core
│   ├── vqac_builder.py
│   ├── eg_generator.py
│   ├── chain_verifier.py
│   └── semantic_verifier.py
└── experiments
    ├── bitflip_attack_experiment.py
    ├── preprocessing_inconsistency_experiment.py
    └── task_consistency_experiment.py
```

---

## Core Reproduction

The `core/` directory contains the minimal files required to verify the main protocol-level correctness claim.

### Files in `core/`

- **`vqac_builder.py`**  
  Constructs the plaintext `VQSA(n,m)` quantum circuit.

- **`eg_generator.py`**  
  Generates the client-side encryption operators `E_i` and server-side computation operators `G`.

- **`chain_verifier.py`**  
  Verifies that the chained encrypted execution is functionally equivalent to the plaintext circuit.

- **`semantic_verifier.py`**  
  Performs a lightweight classical semantic check of the circuit behavior.

---

## Quick Start

To verify the core correctness claim, run:

```bash
python run_minimal_repro.py
```

This script performs:

1. plaintext semantic verification
2. end-to-end equivalence verification between plaintext and encrypted execution

If successful, it prints:

```text
PASS
PASS
Minimal reproduction successful.
```

---

## Experimental Scripts

The `experiments/` directory contains additional scripts used to reproduce the experimental results in the paper.

### Files in `experiments/`

- **`bitflip_attack_experiment.py`**  
  Reproduces the bit-flip attack experiments and evaluates detection performance.

- **`preprocessing_inconsistency_experiment.py`**  
  Reproduces the preprocessing / model inconsistency attack experiments.

- **`task_consistency_experiment.py`**  
  Reproduces the task-level quantization consistency experiments.

These scripts are included for completeness, but they are **not required** for minimal protocol correctness verification.

---

## Dependencies

### Minimal dependencies

For the core reproduction only:

```bash
pip install qiskit qiskit-aer numpy
```

### Full dependencies

To run both the core verification and the experimental scripts:

```bash
pip install qiskit qiskit-aer numpy torch torchvision matplotlib pandas
```

---

## Scope

This artifact supports two levels of evaluation:

1. **Minimal verification**  
   Reviewers can validate the central correctness claim of the protocol using the files in `core/` together with `run_minimal_repro.py`.

2. **Additional empirical evaluation**  
   Reviewers can optionally run the scripts in `experiments/` to reproduce the paper’s security and consistency experiments.

---

## Notes

- The core verification uses **statevector simulation**, so it is intended for small instances such as `(n=2, m=2)`.
- The experimental scripts may require additional runtime compared with the minimal reproduction.
- All provided scripts use fixed random seeds for reproducibility.

---

## Contact

For questions, please refer to the paper or use the submission system for anonymous communication.
