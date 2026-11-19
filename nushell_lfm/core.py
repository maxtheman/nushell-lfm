"""Where things are, and the few helpers every stage shares."""

import asyncio
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
# Checkpoints are looked up by folder name under MODELS_DIR (default: ./models).
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models")).expanduser()
MODEL_FOLDERS = {
    "grader": "d1-2b-mlx-8bit",
    "target": "lfm3-1.7b-mlx-8bit",
}
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
nu_path = shutil.which("nu")

# The modules whose functions the notebook shows running line by line.
_SOURCE_MODULES = ("environments", "catalog", "chains", "breaks", "dataset", "grader", "target", "export")


def show_path(path) -> str:
    """A path as the reader should see it: relative to the project or to $MODELS_DIR."""
    path = Path(path)
    for base, label in ((ROOT, "."), (MODELS_DIR, "$MODELS_DIR")):
        try:
            return f"{label}/{path.relative_to(base)}"
        except ValueError:
            continue
    return path.name


def function_source(name: str) -> str:
    """The source of one of the pipeline's functions, for showing it running (LiveEdit)."""
    import importlib
    import inspect
    import textwrap

    for module in _SOURCE_MODULES:
        found = getattr(importlib.import_module(f"nushell_lfm.{module}"), name, None)
        if found is not None:
            return textwrap.dedent(inspect.getsource(found))
    raise KeyError(f"no function named {name} in the pipeline")


def model_path(value: str) -> str:
    """A checkpoint given as a folder name under MODELS_DIR, or as a path of its own."""
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else MODELS_DIR / path)


# MLX GPU streams are thread-local: arrays made on one thread cannot be evaluated on another.
# Every MLX stage therefore runs on this one worker thread, which also keeps the page responsive.
_MLX_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


async def mlx_thread(fn, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(_MLX_EXECUTOR, lambda: fn(*args, **kwargs))
