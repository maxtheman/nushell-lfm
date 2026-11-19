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

from nushell_lfm.core import MODEL_FOLDERS, model_path
from nushell_lfm.dataset import write_requests_with_model
from nushell_lfm.pipeline import DEFAULTS, build

stats = write_requests_with_model(build()["chains"], DEFAULTS["requests_per_chain"], model_path(MODEL_FOLDERS["target"]), None, log=print)
print({key: value for key, value in stats.items() if key != "rejected_examples"})
for example in stats["rejected_examples"]:
    print("  rejected:", example[:160])
