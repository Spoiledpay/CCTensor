"""Roda todas as suites de teste do CCT e resume o resultado.

Cada arquivo roda em um processo separado (isolamento de estado e de
`sys.path`). Codigo de saida != 0 em qualquer suite reprova o conjunto.

Uso:
    python cct/tests/run_all.py
    python cct/tests/run_all.py --quick    # pula a suite mais lenta (bit_exact)
"""

import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

SUITES = [
    ("test_entropy.py", "Huffman: Kraft, codigos longos, regressao BF16"),
    ("test_native_l3.py", "decoder nativo C vs numpy"),
    ("test_integrate.py", "from_dense / cctify_model / uncctify"),
    ("test_l3_fase4.py", "L3 bit-exact + Fase 4 + ECD"),
    ("test_bit_exact.py", "codecs, CCTTensor, CCT-GEMM, cache"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="pula test_bit_exact.py (o mais lento)")
    args = ap.parse_args()

    print("CCT — suite completa")
    failures = []
    for name, desc in SUITES:
        if args.quick and name == "test_bit_exact.py":
            print(f"\n=== {name} — pulado (--quick) ===")
            continue
        path = os.path.join(_HERE, name)
        print(f"\n=== {name} — {desc} ===")
        proc = subprocess.run([sys.executable, path])
        if proc.returncode != 0:
            failures.append(name)

    print("\n" + "=" * 60)
    if failures:
        print(f"FALHAS ({len(failures)}): {', '.join(failures)}")
        return 1
    print(f"TODAS AS {len(SUITES) - (1 if args.quick else 0)} SUITES PASSARAM")
    return 0


if __name__ == "__main__":
    sys.exit(main())
