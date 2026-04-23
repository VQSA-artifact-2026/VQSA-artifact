from __future__ import annotations

import sys
import traceback

from core.semantic_verifier import check_vqac_nm_semantics
from core.chain_verifier import verify_vqac_nm_chain


def run_plain_check() -> None:
    print("[1/2] Running plaintext semantic check for VQAC(2,2) ...")
    ok = check_vqac_nm_semantics(
        n=2,
        m=2,
        inputs=[1, 2],
        verbose=False,
    )
    if not ok:
        raise RuntimeError("Plaintext semantic check failed.")
    print("PASS\n")


def run_chain_check() -> None:
    print("[2/2] Running chained E/G equivalence check for VQAC(2,2) ...")
    result = verify_vqac_nm_chain(
        n=2,
        m=2,
        seed=42,
        num_random_tests=8,
        random_initial_key=False,
        atol=1e-8,
        verbose=True,
    )
    if result["passed"] != result["total"]:
        raise RuntimeError(
            f"Chain equivalence check failed: passed {result['passed']}/{result['total']} tests."
        )
    print("PASS\n")


def main() -> int:
    print("===== Minimal Reproduction for VQAC =====\n")
    try:
        run_plain_check()
        run_chain_check()
    except Exception as exc:
        print("Minimal reproduction FAILED.\n")
        print(f"Error: {exc}\n")
        traceback.print_exc()
        return 1

    print("Minimal reproduction successful.")
    return 0


if __name__ == "__main__":
    sys.exit(main())