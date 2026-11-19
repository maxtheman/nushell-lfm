#!/usr/bin/env python3
"""How far the target's base weights can be squeezed: score it at fewer bits per weight, by execution, on every
held-out request, first with the existing adapter and then with an adapter retrained on the low-bit weights.

    .venv/bin/python scripts/run_low_bit.py --bits 4 3 2            # existing adapter
    .venv/bin/python scripts/run_low_bit.py --bits 2 ternary --retrain --rank 32   # adapter retrained on the low-bit weights

Results are kept in runs/target/low_bit.json; finished runs are not repeated.
"""

import argparse
import functools
import sys
from pathlib import Path

print = functools.partial(print, flush=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits", nargs="+", default=["4", "3", "2"], help="bits per weight, or 'ternary'")
    parser.add_argument("--rank", type=int, help="with --retrain: adapter rank (default: the final adapter's)")
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    import notebook

    _, defs = notebook.app.run()
    for bits in args.bits:
        saved = defs["score_low_bit"](defs["dataset"], bits if bits == "ternary" else int(bits), args.retrain, print, args.rank)
    print()
    for name, run in saved["runs"].items():
        print(f"{name:<40} well-formed {run['well_formed']:.0%}  runs {run['runs']:.0%}  correct {run['correct']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
