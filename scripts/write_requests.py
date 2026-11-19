#!/usr/bin/env python3
"""Have the base language model write the requests for every chain (cached in runs/generated/).

    mise run requests
"""

import functools
import sys
from pathlib import Path

print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import notebook

_, defs = notebook.app.run()
k = defs["ival"](defs["k_variants_slider"])
stats = defs["write_requests_with_model"](
    defs["verified_chains"], k, defs["model_path"](defs["MODEL_FOLDERS"]["target"]), None, log=print
)
print({key: value for key, value in stats.items() if key != "rejected_examples"})
for example in stats["rejected_examples"]:
    print("  rejected:", example[:160])
