import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full", app_title="Teaching a small model to drive Nushell")


@app.cell
def _():
    import asyncio
    import gc
    import hashlib
    import json
    import math
    import os
    import random
    import re
    import shutil
    import subprocess
    import tempfile
    import time
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parent
    # Checkpoints are looked up by folder name under MODELS_DIR (default: ./models).
    MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models")).expanduser()
    MODEL_FOLDERS = {
        "grader": "d1-2b-mlx-8bit",
        "target": "lfm3-1.7b-mlx-8bit",
    }
    RUNS_DIR = ROOT / "runs"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

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
        """The source of a function defined in this notebook, for showing it running (LiveEdit)."""
        import ast
        import textwrap

        text = Path(__file__).read_text(encoding="utf-8")
        node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == name)
        return textwrap.dedent(ast.get_source_segment(text, node, padded=True))

    def model_path(value: str) -> str:
        """A checkpoint given as a folder name under MODELS_DIR, or as a path of its own."""
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else MODELS_DIR / path)

    # MLX GPU streams are thread-local: arrays made on one thread cannot be evaluated on another.
    # Every MLX stage therefore runs on this one worker thread, which also keeps the page responsive.
    _MLX_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")

    async def mlx_thread(fn, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(_MLX_EXECUTOR, lambda: fn(*args, **kwargs))

    return (
        asyncio,
        Counter,
        MODELS_DIR,
        MODEL_FOLDERS,
        Path,
        ROOT,
        RUNS_DIR,
        ThreadPoolExecutor,
        alt,
        function_source,
        gc,
        hashlib,
        json,
        math,
        mlx_thread,
        mo,
        model_path,
        np,
        os,
        pd,
        random,
        re,
        show_path,
        shutil,
        subprocess,
        tempfile,
        time,
    )


@app.cell
def _(MODELS_DIR, MODEL_FOLDERS, ROOT, mo, row, shutil):
    nu_path = shutil.which("nu")
    _ready = [("Nushell", bool(nu_path))] + [(f"{role} checkpoint", (MODELS_DIR / folder / "config.json").exists()) for role, folder in MODEL_FOLDERS.items()]
    _missing = [name for name, ok in _ready if not ok]
    _gif = ROOT / "assets" / "usage.gif"
    _intro = mo.md(
        """
        # Teaching a small model to drive Nushell

        Two small models are fine-tuned on a Mac, offline, on data this notebook generates from scratch: a **target**
        (LFM3 1.7B) that turns a plain request into a `run_nushell` tool call, and a **grader** (D1 2B) that says whether a
        call is broken, how, and where. Project environments are generated from seeds, command chains are sampled from
        Nushell's own command signatures, broken variants are kept only when running them confirms the label, and the
        requests are written by the base model. Every score comes from executing commands in environments the models never saw.
        """
    )
    _side = (
        mo.image(src=str(_gif), rounded=True, width="100%")
        if _gif.exists()
        else mo.Html('<div style="aspect-ratio: 16/10; border: 1px dashed currentColor; border-radius: 10px; display: grid; place-items: center; opacity: 0.5;">GIF of usage</div>')
    )
    mo.vstack(
        [row((_intro, 3, 360), (_side, 2, 300))]
        + ([mo.callout(mo.md("Missing: " + ", ".join(_missing) + ". Run `mise run setup` and see the README for the checkpoints."), kind="warn")] if _missing else [])
    )
    return (nu_path,)


@app.cell
def _(mo):
    from nushell_lfm.widgets import StepBar

    # code blocks wrap; nothing on the page scrolls sideways inside a box
    _wrap = mo.Html("<style>.markdown pre, .markdown pre code, .codehilite pre { white-space: pre-wrap !important; overflow-wrap: anywhere; overflow-x: visible !important; }</style>")
    _bar = mo.ui.anywidget(
        StepBar(
            steps=[
                ["step-env", "1. Env"],
                ["step-data", "2. Data gen & rephrase"],
                ["step-grader", "3. Train grader"],
                ["step-target", "4. Train LFM"],
                ["step-nu", "5. Load Nu"],
                ["step-providers", "6. Other models"],
            ]
        )
    )
    mo.vstack([_wrap, _bar], gap=0)
    return


@app.cell
async def _(
    FAILURE_LABELS,
    GRADER_DIR,
    MODEL_FOLDERS,
    ROOT,
    TARGET_DIR,
    attach_lora,
    dataset,
    encode_grader_row,
    grader_readouts,
    mlx_thread,
    mo,
    model_path,
    parse_tool_call,
    set_lora,
    target_prompt,
    target_ran,
):
    import activation_view

    # What the 3D view draws is measured: both models over held-out requests and their reference calls, each with
    # and without its adapter, and guided decoding token by token. After a real training run it is captured again
    # from the new adapters.
    activation_error = None
    _file = ROOT / "runs" / "activation" / "examples.json"
    if target_ran in ("standard", "protocol"):
        _defs = {
            "FAILURE_LABELS": FAILURE_LABELS, "GRADER_DIR": GRADER_DIR, "MODEL_FOLDERS": MODEL_FOLDERS, "TARGET_DIR": TARGET_DIR,
            "attach_lora": attach_lora, "dataset": dataset, "encode_grader_row": encode_grader_row, "grader_readouts": grader_readouts,
            "model_path": model_path, "parse_tool_call": parse_tool_call, "set_lora": set_lora, "target_prompt": target_prompt,
        }

        def _capture():
            examples = activation_view.capture_activations(
                _defs, activation_view.pick_rows(_defs, 4), max_examples=4, stats_rows=activation_view.scored_rows(_defs, 40)
            )
            activation_view.save_payload(_file, examples, activation_view.model_layout(_defs))

        with mo.status.spinner(title="Capturing activations from the new adapters…"):
            try:
                await mlx_thread(_capture)
            except Exception as _exc:
                activation_error = f"{type(_exc).__name__}: {_exc}"
    activation_payload = activation_view.load_payload(_file)
    guidance_tokens = activation_view.load_guidance_totals(TARGET_DIR)
    return activation_error, activation_payload, activation_view, guidance_tokens


@app.cell
def _(activation_error, activation_payload, activation_view, mo, space):
    _views = [space(32, "step-model"), mo.md("## LFM3")]
    if activation_error:
        _views.append(mo.callout(mo.md(f"Capture failed: `{activation_error}`"), kind="danger"))
    if activation_payload:
        _views.append(
            mo.ui.anywidget(
                activation_view.ActivationWidget(
                    examples=activation_payload["examples"], layer_types=activation_payload["layer_types"], conv_window=activation_payload["conv_window"]
                )
            )
        )
    mo.vstack(_views)
    return


@app.cell
def _(mo):
    import inspect as _inspect
    from pathlib import Path as _Path

    from wigglystuff import Fader, Knob, TangleLatex
    from wigglystuff import LiveEdit as _LiveEdit

    class LiveEdit(_LiveEdit):
        """wigglystuff's live code view, made to fill its column: long lines wrap instead of scrolling sideways,
        and in a narrow column the values go under the code instead of beside it."""

        _css = (_Path(_inspect.getfile(_LiveEdit)).parent / "static" / "liveedit.css").read_text(encoding="utf-8") + """
        .liveedit-root { width: 100% !important; max-width: 100%; box-sizing: border-box; container-type: inline-size; }
        .liveedit-card { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
        .liveedit-line, .liveedit-kv { white-space: pre-wrap; overflow-wrap: anywhere; }
        .liveedit-loopblock { width: auto; max-width: 100%; overflow-x: auto; }
        /* nothing here scrolls: the card is as tall as its code and its values */
        .liveedit-card { height: auto !important; overflow: visible; }
        .liveedit-code, .liveedit-trace, .liveedit-loopblock { max-height: none !important; overflow: visible !important; }
        @container (max-width: 560px) {
          .liveedit-card { grid-template-columns: minmax(0, 1fr); grid-template-rows: auto auto; }
          .liveedit-code { border-right: 0; border-bottom: 1px solid var(--liveedit-border); }
        }
        """

    def heading(text, section=None):
        return mo.md(f"## {text}")

    def fader(label, low, high, step, value, length=150, orientation="vertical"):
        return mo.ui.anywidget(Fader(min_value=low, max_value=high, step=step, value=value, label=label, length=length, orientation=orientation))

    def knob(label, low, high, step, value):
        return mo.ui.anywidget(Knob(min_value=low, max_value=high, step=step, value=value, label=label, size=72))

    def row(*cells, gap="28px 40px", align="flex-start"):
        """Columns that share a row while there is room and stack when there is not.
        Each cell is (item, share of the spare width, width in pixels it needs)."""
        parts = "".join(
            f'<div style="flex: {grow} 1 {need}px; min-width: 0; max-width: 100%;">{mo.as_html(item).text}</div>' for item, grow, need in cells
        )
        return mo.Html(f'<div style="display: flex; flex-wrap: wrap; gap: {gap}; align-items: {align}; width: 100%;">{parts}</div>')

    def space(px=40, anchor=None):
        """Vertical room between sections; with an anchor it is also what the step bar scrolls to and watches."""
        name = f' id="{anchor}"' if anchor else ""
        return mo.Html(f'<div{name} style="height: {px}px;"></div>')

    def ival(control) -> int:
        return int(round(control.widget.value))

    def fval(control) -> float:
        return float(control.widget.value)

    return LiveEdit, TangleLatex, fader, fval, heading, ival, knob, row, space


@app.cell
def _(fader):
    env_count_slider = fader("Environments", 2, 24, 1, 20, 260, "horizontal")
    return (env_count_slider,)


@app.cell
def _(
    Path,
    RUNS_DIR,
    ThreadPoolExecutor,
    hashlib,
    json,
    nu_path,
    os,
    random,
    shutil,
    subprocess,
    tempfile,
):
    # -------------------------------------------------------------------------
    # Step 1: generated environments and the sandboxed Nu runner
    #
    # Every environment is a small project with the same kinds of things in it (source files, logs, a
    # JSON table, a TOML config, docs, a script) and different names and contents for all of them.
    # Faker supplies names, words and sentences; polyfactory builds the config and table records from
    # dataclasses. Everything is seeded, so a given environment number always produces the same project.
    # -------------------------------------------------------------------------
    from dataclasses import asdict, dataclass

    from faker import Faker
    from polyfactory.factories import DataclassFactory

    WORKSPACES_DIR = RUNS_DIR / "workspaces"
    _DIRS = {
        "src": ["src", "app", "lib", "services", "pkg"],
        "cfg": ["config", "conf", "settings", "etc"],
        "logs": ["logs", "log", "journal"],
        "docs": ["docs", "doc", "handbook", "notes"],
        "scripts": ["scripts", "tools", "bin", "ops"],
    }
    _LANGUAGES = [("py", "#"), ("js", "//"), ("go", "//"), ("rb", "#"), ("rs", "//")]
    _COMPONENTS = ["billing", "search", "cache", "queue", "auth", "storage", "notify", "gateway", "scheduler", "ingest", "catalog", "payments", "session", "metrics", "export", "sync"]
    _TOPICS = ["timeout", "retry", "token", "migration", "health", "latency", "quota", "backoff", "checksum", "replica"]
    _MODULES = ["handler", "client", "models", "worker", "routes", "store", "utils", "tasks", "schema", "policy"]
    _TABLES = [("flags", "flag"), ("features", "feature"), ("toggles", "toggle"), ("experiments", "experiment"), ("rollouts", "name")]
    _OWNER_COLUMNS = ["owner", "team", "lead", "maintainer"]
    _NUMBER_COLUMNS = ["rollout", "percent", "weight", "share"]
    _BOOL_COLUMNS = ["enabled", "active", "live"]
    _CONFIG_NAMES = ["service", "app", "server", "settings"]
    _SERVICE_SECTIONS = ["service", "server", "app"]
    _STORE_SECTIONS = ["database", "cache", "storage", "broker"]
    OPTIONAL_KEYS = {"retry_limit": 3, "pool_mode": "'pooled'", "ssl_mode": "'prefer'", "idle_timeout": 30, "region": "'local'", "batch_size": 100}
    _LEVELS = ["INFO", "WARN", "ERROR"]
    _EVENTS = {
        "INFO": ["started on port {n}", "{topic} check ok", "reconnected attempt={k}", "drained queue={word}", "{topic} refreshed for {word}"],
        "WARN": ["high {topic} {n}ms", "queue depth {n}", "{topic} near limit user={word}", "slow response {n}ms"],
        "ERROR": ["connection refused host={host}", "{topic} failed job={n}", "invalid signature user={word}", "{topic} exceeded after {k} attempts"],
    }

    @dataclass
    class ServiceSettings:
        name: str
        port: int
        debug: bool

    @dataclass
    class StoreSettings:
        host: str
        port: int
        max_connections: int
        timeout_ms: int

    @dataclass
    class TableRow:
        key: str
        enabled: bool
        owner: str
        amount: int

    def environment_layout(seed: int) -> dict:
        """The names of things in one environment: folders, language, components, files, columns, owners."""
        rng = random.Random(f"layout-{seed}")
        fake = Faker()
        fake.seed_instance(seed)
        ext, comment = rng.choice(_LANGUAGES)
        components = rng.sample(_COMPONENTS, 3)
        table, key_column = rng.choice(_TABLES)
        owners = []
        while len(owners) < 3:
            name = fake.first_name().lower()
            if name.isalpha() and name not in owners:
                owners.append(name)
        return {
            "seed": seed,
            "id": f"env-{seed:02d}",
            "dirs": {role: rng.choice(names) for role, names in _DIRS.items()},
            "ext": ext,
            "comment": comment,
            "components": components,
            "modules": {c: rng.sample(_MODULES, rng.choice([1, 2])) for c in components},
            "topics": rng.sample(_TOPICS, 4),
            "log_names": [f"{name}.log" for name in rng.sample(["app", "worker", "api", "jobs", "access", "audit"], 2)],
            "table": f"{table}.json",
            "columns": {"key": key_column, "owner": rng.choice(_OWNER_COLUMNS), "number": rng.choice(_NUMBER_COLUMNS), "flag": rng.choice(_BOOL_COLUMNS)},
            "owners": owners,
            "config": f"{rng.choice(_CONFIG_NAMES)}.toml",
            "service_section": rng.choice(_SERVICE_SECTIONS),
            "store_section": rng.choice(_STORE_SECTIONS),
            "optional_present": rng.sample(sorted(OPTIONAL_KEYS), 2),
            "doc": f"{rng.choice(['runbook', 'guide', 'operations', 'oncall'])}.md",
            "script": f"{rng.choice(['deploy', 'release', 'backup', 'rotate'])}.nu",
        }

    def environment_files(layout: dict, content_seed: int = 0, extra: bool = False) -> dict:
        """The file contents for a layout. Another `content_seed` gives the same project with different data,
        and `extra` adds a file to every folder: together they make the second copy of an environment that
        tells a hard-coded answer apart from a computed one."""
        rng = random.Random(f"files-{layout['seed']}-{content_seed}")
        fake = Faker()
        fake.seed_instance(layout["seed"] * 1000 + content_seed)
        dirs, ext, comment = layout["dirs"], layout["ext"], layout["comment"]
        components, topics, columns = layout["components"], layout["topics"], layout["columns"]

        class Service(DataclassFactory[ServiceSettings]):
            name = lambda: f"{fake.domain_word()}-{rng.choice(['api', 'svc', 'edge', 'core'])}"
            port = lambda: rng.choice([8080, 8443, 9000, 9090, 3000, 5000])
            debug = lambda: rng.random() < 0.3

        class Store(DataclassFactory[StoreSettings]):
            host = lambda: f"{fake.domain_word()}.internal"
            port = lambda: rng.choice([5432, 6379, 3306, 9042, 27017])
            max_connections = lambda: rng.choice([8, 16, 32, 64, 128])
            timeout_ms = lambda: rng.choice([500, 900, 1500, 2500, 5000])

        class Row(DataclassFactory[TableRow]):
            key = lambda: "_".join(fake.words(2))
            enabled = lambda: rng.random() < 0.65
            owner = lambda: rng.choice(layout["owners"])
            amount = lambda: rng.choice(range(0, 101, 5))

        def toml(sections: dict) -> str:
            show = lambda v: str(v).lower() if isinstance(v, bool) else json.dumps(v)
            return "\n".join(f"[{name}]\n" + "".join(f"{k} = {show(v)}\n" for k, v in values.items()) for name, values in sections.items())

        store = asdict(Store.build())
        for key in layout["optional_present"]:
            store[key] = rng.choice([2, 5, 45, 250]) if isinstance(OPTIONAL_KEYS[key], int) else rng.choice(["session", "require", "west", "strict"])
        rows = Row.batch(rng.choice([6, 7, 8]))
        for owner, row in zip(layout["owners"], rows):  # every owner has at least one row
            row.owner = owner
        rows[0].enabled, rows[1].enabled = True, False
        config_path = f"{dirs['cfg']}/{layout['config']}"
        files = {
            "README.md": f"# {fake.company()}\n\n{components[0].title()} and {components[1]} service. {fake.sentence()}\nSee `{config_path}` for the {topics[0]} and {topics[1]} settings.\n",
            config_path: toml({layout["service_section"]: asdict(Service.build()), layout["store_section"]: store}),
            f"{dirs['cfg']}/{layout['table']}": json.dumps(
                [{columns["key"]: r.key, columns["flag"]: r.enabled, columns["owner"]: r.owner, columns["number"]: r.amount} for r in rows], indent=2
            )
            + "\n",
            f"{dirs['docs']}/{layout['doc']}": (
                f"# {layout['doc'][:-3].title()}\n\n## {components[0].title()} {topics[0]} errors\n"
                f"Check `{dirs['logs']}/{layout['log_names'][0]}` for `ERROR` lines about {components[0]} and the {topics[1]}.\n{fake.sentence()}\n"
            ),
            f"{dirs['scripts']}/{layout['script']}": (
                f"# {layout['script'][:-3].title()} helper\nlet cfg = (open {config_path})\n"
                f"print $\"{layout['script'][:-3]} ($cfg.{layout['service_section']}.name) on port ($cfg.{layout['service_section']}.port)\"\n"
            ),
        }
        for component in components:
            for module in layout["modules"][component]:
                lines = [f"{comment} {component.title()} {module}: {fake.sentence()}", "", f"{comment} handles {rng.choice(topics)} for the {rng.choice(components)} path"]
                if rng.random() < 0.6:
                    lines.append(f"{comment} {rng.choice(['TODO', 'FIXME'])}: {fake.sentence(nb_words=5).rstrip('.').lower()} on {rng.choice(topics)}")
                lines += [f"{module.upper()}_LIMIT = {rng.choice([16, 64, 300, 3600])}", ""]
                files[f"{dirs['src']}/{component}/{module}.{ext}"] = "\n".join(lines)
        for name in layout["log_names"]:
            minute, entries = rng.randrange(0, 20), []
            levels = _LEVELS + [rng.choice(_LEVELS) for _ in range(rng.choice([2, 3, 4, 5]))]  # every level at least once
            rng.shuffle(levels)
            for level in levels:
                minute += rng.randrange(1, 4)
                event = rng.choice(_EVENTS[level]).format(n=rng.choice([120, 410, 4921, 8080, 37]), k=rng.randrange(2, 6), topic=rng.choice(topics), word=fake.word(), host=store["host"])
                entries.append(f"2026-03-02T08:{minute:02d}:{rng.randrange(60):02d}Z {level:<5} {rng.choice(components)} {event}")
            files[f"{dirs['logs']}/{name}"] = "\n".join(entries) + "\n"
        if extra:
            files[f"{dirs['src']}/{components[0]}/extra.{ext}"] = f"{comment} extra {topics[0]} {topics[1]} helper for {components[1]}\n{comment} TODO: remove after the {topics[2]} change\n"
            files[f"{dirs['logs']}/extra.log"] = (
                f"2026-03-02T09:00:00Z ERROR {components[1]} {topics[0]} failed job=1\n"
                f"2026-03-02T09:01:00Z WARN  {components[0]} slow response 99ms\n"
                f"2026-03-02T09:02:00Z INFO  {components[2]} {topics[3]} check ok\n"
            )
            files[f"{dirs['cfg']}/extra.toml"] = "[extra]\nvalue = 1\n"
            files[f"{dirs['docs']}/extra.md"] = f"# Extra\n\nNotes on {topics[2]}, {topics[3]} and {components[2]}.\n"
            files[f"{dirs['scripts']}/extra.nu"] = "# extra\nprint 'extra'\n"
        return files

    def _materialize(base_dir: Path, files: dict) -> None:
        """Write an environment to disk. Generation is deterministic, so a folder that already holds exactly
        these files is left alone; otherwise it is built beside its final place and swapped in, so another
        process reading or writing the same environment never sees it half-made."""
        digest = hashlib.sha1(json.dumps(files, sort_keys=True).encode()).hexdigest()
        marker = base_dir.with_suffix(".sha1")
        if base_dir.is_dir() and marker.exists() and marker.read_text() == digest:
            return
        building = Path(tempfile.mkdtemp(prefix=base_dir.name + "-", dir=base_dir.parent if base_dir.parent.exists() else None))
        if building.parent != base_dir.parent:
            base_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(building)
            building = Path(tempfile.mkdtemp(prefix=base_dir.name + "-", dir=base_dir.parent))
        for rel_path, content in files.items():
            target = building / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        # one fixed time on everything, so `ls` reports the same `modified` on every machine and every run
        for path in [building, *building.rglob("*")]:
            os.utime(path, (FIXED_TIME, FIXED_TIME))
        os.chmod(building, 0o755)
        retired = base_dir.with_name(base_dir.name + f".old-{os.getpid()}")
        try:
            if base_dir.exists():
                os.replace(base_dir, retired)
            os.replace(building, base_dir)
            marker.write_text(digest)
        except OSError:  # another process got there first with the same contents
            shutil.rmtree(building, ignore_errors=True)
        shutil.rmtree(retired, ignore_errors=True)

    FIXED_TIME = 1772438400  # 2026-03-02T08:00:00Z, the same instant the WebAssembly build reports
    _digests = {}

    def build_environments(count: int) -> list[dict]:
        """Generate `count` environments and write each to disk twice: as it is, and as its second copy."""
        environments = []
        for seed in range(1, count + 1):
            layout = environment_layout(seed)
            files, variant = environment_files(layout), environment_files(layout, 1, extra=True)
            _materialize(WORKSPACES_DIR / layout["id"] / "base", files)
            _materialize(WORKSPACES_DIR / layout["id"] / "variant", variant)
            _digests[layout["id"]] = hashlib.sha1(json.dumps([files, variant], sort_keys=True).encode()).hexdigest()[:12]
            dirs = layout["dirs"]
            title = f"{layout['id']} · {layout['ext']} · {dirs['src']}/ {dirs['cfg']}/ {dirs['logs']}/"
            environments.append({"id": layout["id"], "title": title, "layout": layout, "files": files})
        return environments

    def render_tree(paths: list[str]) -> str:
        lines, previous = [], []
        for path in sorted(paths):
            parts = path.split("/")
            for depth, name in enumerate(parts[:-1]):
                if previous[: depth + 1] != parts[: depth + 1]:
                    lines.append("  " * depth + name + "/")
            previous = parts[:-1]
            lines.append("  " * (len(parts) - 1) + parts[-1])
        return "\n".join(lines)

    # macOS can confine a process with a seatbelt profile. When that works, Nu gets no network and may
    # write only inside its own throwaway copy of the workspace. Elsewhere it runs unconfined in the copy.
    _PROFILE = (
        '(version 1)(allow default)(deny network*)(deny file-write*)'
        '(allow file-write* (subpath "{root}") (literal "/dev/null"))'
    )

    def _seatbelt_works() -> bool:
        if not shutil.which("sandbox-exec"):
            return False
        try:
            probe = subprocess.run(
                ["sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"], capture_output=True, timeout=5
            )
            return probe.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    NU_CONFINED = _seatbelt_works()
    _nu_version = subprocess.run([nu_path, "--version"], capture_output=True, text=True).stdout.strip() if nu_path else "missing"
    _cache_file = RUNS_DIR / "nu-cache.json"
    _cache = {}
    if _cache_file.exists():
        try:
            _cache = json.loads(_cache_file.read_text(encoding="utf-8"))
        except ValueError:
            _cache = {}

    def _snapshot(directory: Path) -> dict:
        return {
            str(p.relative_to(directory)): hashlib.sha1(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*"))
            if p.is_file()
        }

    def run_nu(code: str, env: str, variant: bool = False, timeout: float = 8.0, cache: bool = True) -> dict:
        """Run Nushell source in a fresh copy of an environment. Returns the exit status, both streams,
        whether the failure was a parse error, and which files the command added or changed."""
        key = hashlib.sha1(f"{_nu_version}\n{_digests.get(env)}\n{variant}\n{code}".encode()).hexdigest()
        if cache and key in _cache:
            return _cache[key]
        if not nu_path:
            return {"ok": False, "returncode": -1, "stdout": "", "stderr": "nu is not on PATH", "parse_error": False, "changed": []}
        scratch = Path(tempfile.mkdtemp(prefix="nu-run-")).resolve()
        workspace = scratch / "workspace"
        try:
            shutil.copytree(WORKSPACES_DIR / env / ("variant" if variant else "base"), workspace)
            before = _snapshot(workspace)
            command = [nu_path, "--no-config-file", "-c", code]
            if NU_CONFINED:
                command = ["sandbox-exec", "-p", _PROFILE.format(root=scratch)] + command
            environ = {"PATH": os.environ.get("PATH", ""), "HOME": str(scratch), "TMPDIR": str(scratch), "NO_COLOR": "1"}
            try:
                proc = subprocess.run(command, cwd=str(workspace), capture_output=True, text=True, timeout=timeout, env=environ)
                after = _snapshot(workspace)
                result = {
                    "ok": proc.returncode == 0,
                    "returncode": proc.returncode,
                    # the copy lives at a random path; show it the way the browser build does
                    "stdout": proc.stdout.strip().replace(str(workspace), "/workspace"),
                    "stderr": proc.stderr.strip().replace(str(workspace), "/workspace"),
                    "parse_error": proc.returncode != 0 and "nu::parser" in proc.stderr,
                    "changed": sorted(k for k in after if before.get(k) != after[k]),
                }
            except subprocess.TimeoutExpired:
                result = {"ok": False, "returncode": -2, "stdout": "", "stderr": f"timeout after {timeout}s", "parse_error": False, "changed": []}
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        if cache:
            _cache[key] = result
        return result

    def run_nu_many(jobs: list[tuple]) -> list[dict]:
        """`(code, env)` or `(code, env, variant)` jobs, run eight at a time; results in order."""
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(lambda job: run_nu(*job), jobs))

    def save_nu_cache():
        _cache_file.write_text(json.dumps(_cache), encoding="utf-8")

    def same_output(a: str, b: str) -> bool:
        """Equal as JSON values when both parse (key order ignored, list order kept), else as text."""
        try:
            return json.dumps(json.loads(a), sort_keys=True) == json.dumps(json.loads(b), sort_keys=True)
        except ValueError:
            return a.strip() == b.strip()

    # what environment_layout reads, so the notebook can show it running line by line
    GENERATOR_SCOPE = {
        "_DIRS": _DIRS, "_LANGUAGES": _LANGUAGES, "_COMPONENTS": _COMPONENTS, "_TOPICS": _TOPICS, "_MODULES": _MODULES, "_TABLES": _TABLES,
        "_OWNER_COLUMNS": _OWNER_COLUMNS, "_NUMBER_COLUMNS": _NUMBER_COLUMNS, "_BOOL_COLUMNS": _BOOL_COLUMNS, "_CONFIG_NAMES": _CONFIG_NAMES,
        "_SERVICE_SECTIONS": _SERVICE_SECTIONS, "_STORE_SECTIONS": _STORE_SECTIONS, "OPTIONAL_KEYS": OPTIONAL_KEYS, "random": random, "Faker": Faker,
    }

    return (
        GENERATOR_SCOPE,
        NU_CONFINED,
        OPTIONAL_KEYS,
        build_environments,
        render_tree,
        run_nu,
        run_nu_many,
        same_output,
        save_nu_cache,
    )


@app.cell
def _(build_environments, env_count_slider, ival, mo):
    environments = build_environments(ival(env_count_slider))
    env_picker = mo.ui.dropdown(options={e["title"]: e["id"] for e in environments}, value=environments[0]["title"], label="Environment")
    return env_picker, environments


@app.cell
def _(DEMO_PORT, ROOT, env_picker, environments, mo):

    shown_env = next(e for e in environments if e["id"] == env_picker.value)
    _layout = shown_env["layout"]
    return (shown_env,)


@app.cell
def _(GENERATOR_SCOPE, LiveEdit, env_count_slider, env_picker, function_source, mo, render_tree, row, shown_env, space):
    # The code shown is the generator itself, running for the selected seed: each line carries the value it produced.
    _code = LiveEdit(function_source("environment_layout"), args=(shown_env["layout"]["seed"],), function_name="environment_layout", globalns=GENERATOR_SCOPE)
    mo.vstack(
        [
            space(56, "step-env"),
            mo.md("## Target environment generation"),
            mo.hstack([env_count_slider, env_picker], justify="start", align="end", gap=3, wrap=True),
            row((mo.ui.anywidget(_code), 1, 620), (mo.plain_text(render_tree(list(shown_env["files"]))), 0, 190)),
        ],
        gap=1.5,
    )
    return


@app.cell
def _(ROOT, json):
    # -------------------------------------------------------------------------
    # Step 2: command catalog. Every built-in Nushell command with its input and output types and its
    # parameters, exported from the installed nu by `mise run cache-nu-docs` (help commands).
    # -------------------------------------------------------------------------
    NU_COMMANDS = json.loads((ROOT / "assets" / "nu_commands.json").read_text(encoding="utf-8"))

    def signature(command: dict) -> str:
        pairs = "; ".join(f"{io['input']} → {io['output']}" for io in command["input_output"][:4])
        params = " ".join(("<{}>" if p["required"] else "[{}]").format(p["name"]) for p in command["params"] if not p["name"].startswith("--"))
        return f"{command['name']} {params}".strip() + f"   ({pairs})"

    COMMAND_CATALOG = [{"name": c["name"], "category": c["category"], "usage": c["description"], "syntax": signature(c)} for c in NU_COMMANDS]
    ALL_CATEGORIES = ["search", "text_manipulation", "filesystem", "tables_records", "pipelines_recovery"]
    CATEGORY_NOTES = {
        "search": "finding files and lines across the project",
        "text_manipulation": "reading, splitting, parsing and reshaping text and logs",
        "filesystem": "listing, sizing and naming paths",
        "tables_records": "filtering and summarising JSON and TOML data",
        "pipelines_recovery": "offering choices or a default when a path or key is missing",
    }
    return ALL_CATEGORIES, CATEGORY_NOTES, COMMAND_CATALOG, NU_COMMANDS


@app.cell
def _(ALL_CATEGORIES, fader, mo):
    category_boxes = mo.ui.array([mo.ui.checkbox(value=True, label=name.replace("_", " ")) for name in ALL_CATEGORIES])
    chains_total_slider = fader("Chains", 20, 400, 20, 400, 120)
    ported_share_slider = fader("Unix-steered", 0.0, 0.3, 0.05, 0.25, 120)
    length_bias_slider = fader("Longer chains", 0.0, 1.0, 0.1, 0.5, 120)
    breaks_per_chain_slider = fader("Breaks / chain", 1, 10, 1, 4, 120)
    k_variants_slider = fader("Requests / chain", 1, 6, 1, 3, 120)
    return breaks_per_chain_slider, category_boxes, chains_total_slider, k_variants_slider, length_bias_slider, ported_share_slider


@app.cell
def _(NU_COMMANDS, OPTIONAL_KEYS, ROOT, json, random, run_nu_many, same_output, save_nu_cache):
    # -------------------------------------------------------------------------
    # Step 3: chain generator.
    #
    # No chain is written by hand. A chain is a walk on a graph of data kinds: a source, then stages whose
    # input matches what the previous step produced. Two sorts of operator make up the edges:
    #   - recipes, for commands that need arguments (a file, a column, a term), which draw those arguments
    #     from the environment the chain is for;
    #   - every other Nushell command that takes no argument at all, added automatically from Nu's own
    #     signatures (assets/nu_commands.json): its input and output types say where it fits.
    # Operators are drawn in proportion to how often their commands occur in real shell usage
    # (data/command_usage.json, produced by scripts/fetch_command_usage.py), so one-liners and long
    # pipelines appear in realistic proportions. Execution then decides which sampled chains are kept,
    # and the same sampled structure is later mutated to produce the broken variants.
    # -------------------------------------------------------------------------
    _LOG_RX = r"'(?<ts>\S+)\s+(?<level>\S+)\s+(?<msg>.*)'"
    _REL = "each {|p| $p | path relative-to (pwd)}"

    def _priors() -> tuple[dict, dict, dict]:
        """Two sets of command weights, and where they came from.

        NATIVE: how often each command heads a pipeline stage in the example code of the Nushell book and
        cookbook (data/nushell_examples_usage.json). No translation is involved; this is the main prior.

        PORTED: Unix command frequencies from public corpora (data/command_usage.json), carried over through
        data/unix_to_nushell.json: the Nushell book's own Bash table plus a short supplementary list kept apart
        in that file. It decides how the dataset is divided among the categories, and steers a capped minority
        of chains; it never decides what a Nushell pipeline looks like."""
        known = {c["name"] for c in NU_COMMANDS}
        native, ported, info = {}, {}, {"native": None, "ported": None, "lengths": {}}
        examples_file = ROOT / "data" / "nushell_examples_usage.json"
        if examples_file.exists():
            examples = json.loads(examples_file.read_text(encoding="utf-8"))
            native = {k: v for k, v in examples["share"].items() if k in known}
            info["native"] = {"pages": examples["source"]["pages"], "command_lines": examples["source"]["command_lines"], "commands": len(native)}
            info["lengths"]["native"] = {int(k): v for k, v in examples["pipeline_lengths"].items()}
        usage_file, map_file = ROOT / "data" / "command_usage.json", ROOT / "data" / "unix_to_nushell.json"
        if usage_file.exists() and map_file.exists():
            usage, mapping = json.loads(usage_file.read_text(encoding="utf-8")), json.loads(map_file.read_text(encoding="utf-8"))
            book = mapping["map"]
            extra = {unix: entry["nu"] for unix, entry in mapping.get("suggested_not_in_source", {}).items() if unix not in book}
            mass = {"book": 0.0, "supplementary": 0.0}
            for origin, table in (("book", book), ("supplementary", extra)):
                for unix, targets in table.items():
                    share = usage["combined"].get(unix, 0.0)
                    targets = [t for t in targets if t in known]
                    for target in targets:  # a Unix command's usage is shared among its Nushell equivalents
                        ported[target] = ported.get(target, 0.0) + share / len(targets)
                    mass[origin] += share if targets else 0.0
            info["ported"] = {"sources": [s["id"] for s in usage["sources"]], "commands": len(ported), "usage_covered": {k: round(v, 3) for k, v in mass.items()}}
            info["lengths"]["ported"] = {int(k): v for k, v in usage.get("pipeline_lengths_combined", {}).items()}
        return native, ported, info

    NATIVE, PORTED, PRIOR_INFO = _priors()
    PORTED_LIMIT = 0.3  # at most this share of chains may be steered by the translated Unix weights

    def _op(name, src, dst, commands, make, code, say, alt=None, wrong=None, fields=None, binds=False, auto=False):
        return {"name": name, "src": src, "dst": dst, "commands": commands, "make": make, "code": code, "say": say,
                "alt": alt, "wrong": wrong or (lambda p: []), "fields": fields, "binds": binds, "auto": auto}

    def _pick(rng, items):
        return rng.choice(list(items)) if items else None

    def _files_of(env, kind):
        layout, dirs = env["layout"], env["layout"]["dirs"]
        if kind == "log":
            return [f"{dirs['logs']}/{n}" for n in layout["log_names"]]
        if kind == "text":
            return sorted(p for p in env["files"] if not p.endswith((".log", ".json")))
        return sorted(env["files"])

    def _folders(env):
        dirs = env["layout"]["dirs"]
        return [(dirs[r], f"{dirs[r]}/**/*") for r in ("src", "cfg", "logs", "docs", "scripts")] + [("the whole project", "**/*")]

    def _terms(env):
        return env["layout"]["components"] + env["layout"]["topics"]

    def _n(rng):
        return rng.choice([1, 2, 3, 5])

    def _c(env, role):
        return env["layout"]["columns"][role]

    # ---- sources: (make returns the parameters, or None when the environment offers nothing suitable) ----
    SOURCES = [
        _op("where-am-i", None, "scalar", ["pwd"], lambda e, r: {}, lambda p: "pwd",
            lambda p: ("show the current directory", "cwd", "where are we"), alt=lambda p: "$env.PWD"),
        _op("list-folder", None, "files", ["ls"],
            lambda e, r: _pick(r, [{"dir": "", "label": "this folder"}] + [{"dir": d, "label": d} for d, _ in _folders(e)[:5]]),
            lambda p: f"ls {p['dir']}".strip(),
            lambda p: (f"list what is in {p['label']}", f"ls {p['label']}", f"what's in {p['label']}")),
        _op("list-tree", None, "files", ["ls"],
            lambda e, r: (lambda f: {"dir": f[0], "g": f[1]})(_pick(r, _folders(e))),
            lambda p: f"ls {p['g']}",
            lambda p: (f"list everything under {p['dir']}, at any depth", f"all of {p['dir']}, recursive", f"everything inside {p['dir']}, subfolders too")),
        _op("find-paths", None, "paths", ["glob", "each", "path relative-to"],
            lambda e, r: _pick(r, [{"g": f"**/*.{x}", "label": f".{x} files anywhere"} for x in (e["layout"]["ext"], "log", "md", "toml")]
                               + [{"g": f"{d}/**/*", "label": f"paths under {d}"} for d, _ in _folders(e)[:5]]),
            lambda p: f"glob {p['g']} | {_REL}",
            lambda p: (f"find the {p['label']}", f"glob {p['label']}", f"track down the {p['label']}"),
            alt=lambda p: f"ls {p['g']} | get name"),
        _op("read-file", None, "text", ["open"],
            lambda e, r: {"path": _pick(r, _files_of(e, "text"))},
            lambda p: f"open --raw {p['path']}",
            lambda p: (f"show the contents of {p['path']}", f"cat {p['path']}", f"what does {p['path']} say")),
        _op("read-log", None, "loglines", ["open", "lines"],
            lambda e, r: {"path": _pick(r, _files_of(e, "log"))},
            lambda p: f"open --raw {p['path']} | lines",
            lambda p: (f"read the lines of {p['path']}", f"lines of {p['path']}", f"go through {p['path']} line by line")),
        _op("read-table", None, "table", ["open"],
            lambda e, r: {"path": f"{e['layout']['dirs']['cfg']}/{e['layout']['table']}"},
            lambda p: f"open {p['path']}",
            lambda p: (f"load {p['path']}", f"open {p['path']}", f"take the rows in {p['path']}")),
        _op("read-config", None, "config", ["open"],
            lambda e, r: {"path": f"{e['layout']['dirs']['cfg']}/{e['layout']['config']}"},
            lambda p: f"open {p['path']}",
            lambda p: (f"load {p['path']}", f"open {p['path']}", f"look in {p['path']}")),
        _op("try-open", None, "result", ["try", "open", "glob"],
            lambda e, r: (lambda d, x: _pick(r, [
                {"path": f"{d['cfg']}/defaults.toml", "g": f"{d['cfg']}/*"}, {"path": f"{d['logs']}/server.log", "g": f"{d['logs']}/*"},
                {"path": f"{d['docs']}/setup.md", "g": f"{d['docs']}/*"}, {"path": f"{d['src']}/main.{x}", "g": f"{d['src']}/**/*.{x}"}]))(e["layout"]["dirs"], e["layout"]["ext"]),
            lambda p: f"try {{ open {p['path']} }} catch {{ {{status: 'needs_user_choice', requested: '{p['path']}', candidates: (glob {p['g']} | {_REL} | sort)}} }}",
            lambda p: (f"open {p['path']}; if it is missing, return status needs_user_choice with the requested path and the candidates matching {p['g']}",
                       f"open {p['path']} or list {p['g']} as candidates (status, requested, candidates)",
                       f"read {p['path']} — not sure that's the right name, so if it's missing show me what's in {p['g']} to pick from (status, requested, candidates)"),
            alt=lambda p: f"if ('{p['path']}' | path exists) {{ open {p['path']} }} else {{ {{status: 'needs_user_choice', requested: '{p['path']}', candidates: (glob {p['g']} | {_REL} | sort)}} }}"),
    ]

    # ---- stages ----
    def _first_last(src):
        return [
            _op(f"first-{src}", src, src, ["first"], lambda e, r: {"n": _n(r)}, lambda p: f"first {p['n']}",
                lambda p: (f"keep the first {p['n']}", f"first {p['n']}", f"just the first {p['n']}"),
                alt=lambda p: f"take {p['n']}", wrong=lambda p: [(f"last {p['n']}", "Takes the last rows instead of the first.")]),
            _op(f"last-{src}", src, src, ["last"], lambda e, r: {"n": _n(r)}, lambda p: f"last {p['n']}",
                lambda p: (f"keep the last {p['n']}", f"last {p['n']}", f"only the final {p['n']}"),
                wrong=lambda p: [(f"first {p['n']}", "Takes the first rows instead of the last.")]),
            _op(f"count-{src}", src, "scalar", ["length"], lambda e, r: {}, lambda p: "length",
                lambda p: ("count them", "count", "how many is that"),
                wrong=lambda p: [("first", "Returns the first item instead of the count.")]),
        ]

    STAGES = [
        # files (rows of name, type, size)
        _op("only-files", "files", "files", ["where"], lambda e, r: {}, lambda p: "where type == file",
            lambda p: ("keep only the files", "files only", "skip the folders"),
            wrong=lambda p: [("where type == dir", "Keeps the folders instead of the files.")]),
        _op("largest-first", "files", "files", ["sort-by"], lambda e, r: {}, lambda p: "sort-by name | sort-by size --reverse",
            lambda p: ("sort by size, largest first", "by size desc", "biggest on top"),
            wrong=lambda p: [("sort-by name | sort-by size", "Sorts ascending; the request wanted the largest first.")]),
        _op("by-name", "files", "files", ["sort-by"], lambda e, r: {}, lambda p: "sort-by name",
            lambda p: ("sort by name", "by name", "in alphabetical order"),
            wrong=lambda p: [("sort-by name --reverse", "Sorts in reverse order.")]),
        _op("name-has", "files", "files", ["where"], lambda e, r: {"term": _pick(r, _terms(e) + [e["layout"]["ext"], "log"])},
            lambda p: f"where name =~ '{p['term']}'",
            lambda p: (f"keep those whose name contains '{p['term']}'", f"name ~ {p['term']}", f"only the ones with '{p['term']}' in the name"),
            wrong=lambda p: [(f"where name !~ '{p['term']}'", "Keeps the names that do not match.")]),
        _op("name-size", "files", "rows", ["select"], lambda e, r: {}, lambda p: "select name size",
            lambda p: ("return name and size", "name+size", "just the name and size"), fields=lambda p: ["name", "size"],
            alt=lambda p: "reject type modified", wrong=lambda p: [("select name type", "Returns the type column instead of the size.")]),
        _op("names", "files", "paths", ["get"], lambda e, r: {}, lambda p: "get name",
            lambda p: ("return their names", "names", "just the names"),
            wrong=lambda p: [("get size", "Returns sizes instead of names.")]),
        _op("total-size", "files", "scalar", ["get", "math"], lambda e, r: {}, lambda p: "get size | math sum",
            lambda p: ("add up their sizes", "total size", "how big is that altogether"),
            wrong=lambda p: [("get size | math max", "Returns the largest size instead of the total."), ("length", "Counts the entries instead of adding their sizes.")]),
        _op("extension-counts", "files", "rows", ["get", "path parse", "uniq", "sort-by"], lambda e, r: {},
            lambda p: "get name | path parse | get extension | uniq -c | sort-by value | sort-by count --reverse",
            lambda p: ("count them by file extension, most common first, as value and count", "ext counts (value, count)", "tally them by extension, biggest group first (value, count)"),
            fields=lambda p: ["value", "count"],
            wrong=lambda p: [("get name | path parse | get extension | uniq | sort", "Lists the extensions without counting them.")]),
        # paths (list of relative paths)
        _op("base-names", "paths", "list", ["path basename"], lambda e, r: {}, lambda p: "path basename",
            lambda p: ("reduce each to its file name", "basenames", "drop the folder part"),
            wrong=lambda p: [("path dirname", "Returns directories instead of file names.")]),
        _op("parent-folders", "paths", "list", ["path dirname", "uniq"], lambda e, r: {}, lambda p: "path dirname | uniq | sort",
            lambda p: ("list the distinct folders they are in, sorted", "unique dirs, sorted", "which folders are those in"),
            wrong=lambda p: [("path basename | uniq | sort", "Returns file names instead of folders."), ("path dirname | sort", "Keeps duplicates.")]),
        _op("path-has", "paths", "paths", ["where"], lambda e, r: {"term": _pick(r, _terms(e))},
            lambda p: f"where ($it =~ '(?i){p['term']}')",
            lambda p: (f"keep the paths mentioning '{p['term']}', any case", f"path ~ {p['term']} (any case)", f"only the ones with {p['term']} somewhere in the path"),
            wrong=lambda p: [(f"where ($it !~ '(?i){p['term']}')", "Keeps the paths that do not match.")]),
        _op("sorted", "paths", "paths", ["sort"], lambda e, r: {}, lambda p: "sort",
            lambda p: ("sort them", "sorted", "in order"), wrong=lambda p: [("sort --reverse", "Sorts in reverse order.")]),
        _op("hit-counts", "paths", "rows", ["par-each", "open", "lines", "where", "str contains", "sort-by"], lambda e, r: {"term": _pick(r, _terms(e))},
            lambda p: f"where ($it | path type) == file | par-each {{|f| {{file: $f, hits: (open --raw $f | lines | where ($it | str contains -i '{p['term']}') | length)}} }} | where hits > 0 | sort-by file | sort-by hits --reverse",
            lambda p: (f"count the lines mentioning '{p['term']}' in each file and rank the files with any, most hits first, as file and hits", f"grep -c {p['term']} per file, >0, by hits desc (file, hits)", f"which of them talk about {p['term']} the most? file + number of matching lines"),
            fields=lambda p: ["file", "hits"],
            alt=lambda p: f"where ($it | path type) == file | each {{|f| {{file: $f, hits: (open --raw $f | lines | find -n -i '{p['term']}' | length)}} }} | where hits > 0 | sort-by file | sort-by hits --reverse",
            wrong=lambda p: [(f"where ($it | path type) == file | par-each {{|f| {{file: $f, hits: (open --raw $f | lines | where ($it | str contains -i '{p['term']}') | length)}} }} | where hits == 0 | sort-by file", "Keeps the files without a match.")]),
        _op("matching-lines", "paths", "rows", ["each", "open", "lines", "where", "str contains", "str trim", "flatten"], lambda e, r: {"term": _pick(r, _terms(e))},
            lambda p: f"where ($it | path type) == file | sort | each {{|f| open --raw $f | lines | where ($it | str contains -i '{p['term']}') | each {{|l| {{file: $f, line: ($l | str trim)}} }} }} | flatten",
            lambda p: (f"show every line mentioning '{p['term']}' with its file, as file and line", f"grep {p['term']} (file, line)", f"pull out the lines that mention {p['term']}, with the file each is from"),
            fields=lambda p: ["file", "line"],
            alt=lambda p: f"where ($it | path type) == file | sort | each {{|f| open --raw $f | lines | find -n -i '{p['term']}' | each {{|l| {{file: $f, line: ($l | str trim)}} }} }} | flatten",
            wrong=lambda p: [(f"where ($it | path type) == file | sort | each {{|f| open --raw $f | lines | where ($it | str starts-with -i '{p['term']}') | each {{|l| {{file: $f, line: ($l | str trim)}} }} }} | flatten", "Matches only at the start of a line.")]),
        _op("resolve", "paths", "result", ["is-empty", "length"], lambda e, r: {}, 
            lambda p: "if ($matches | is-empty) { {status: 'not_found'} } else if (($matches | length) > 1) { {status: 'choose_one', matches: $matches} } else { {status: 'resolved', file: ($matches | first)} }",
            lambda p: ("report status not_found, resolved (with file) or choose_one (with matches)", "→ not_found / resolved+file / choose_one+matches", "if there's exactly one use it, if several let me choose, if none say so (status, file or matches)"),
            alt=lambda p: "match ($matches | length) { 0 => {status: 'not_found'}, 1 => {status: 'resolved', file: $matches.0}, _ => {status: 'choose_one', matches: $matches} }",
            wrong=lambda p: [("if ($matches | is-not-empty) { {status: 'not_found'} } else if (($matches | length) > 1) { {status: 'choose_one', matches: $matches} } else { {status: 'resolved', file: ($matches | first)} }", "The emptiness test is inverted.")],
            binds="matches"),
        # text and lines
        _op("split-lines", "text", "lines", ["lines"], lambda e, r: {}, lambda p: "lines",
            lambda p: ("split it into lines", "lines", "line by line")),
        *[_op(f"containing-{t}", t, t, ["where", "str contains"], lambda e, r, t=t: {"term": _pick(r, (["ERROR", "WARN", "INFO"] if t == "loglines" else []) + _terms(e))},
              lambda p: f"where ($it | str contains '{p['term']}')",
              lambda p: (f"keep the lines containing '{p['term']}'", f"grep {p['term']}", f"only the lines that mention {p['term']}"),
              alt=lambda p: f"find -n '{p['term']}'",
              wrong=lambda p: [(f"where not ($it | str contains '{p['term']}')", "Keeps the lines that do not contain it."), (f"where ($it | str ends-with '{p['term']}')", "Matches only at the end of a line.")])
          for t in ("lines", "loglines")],
        *[_op(f"lowercase-{t}", t, "lines", ["str downcase", "str trim"], lambda e, r: {}, lambda p: "str downcase | str trim",
              lambda p: ("lowercase and trim each line", "lower+trim", "all lower case, no stray spaces"),
              wrong=lambda p: [("str upcase | str trim", "Upper-cases instead of lower-casing.")]) for t in ("lines", "loglines")],
        _op("join-lines", "lines", "scalar", ["str join"], lambda e, r: {"sep": _pick(r, [" | ", "; ", ", "])}, lambda p: f"str join '{p['sep']}'",
            lambda p: (f"join them with '{p['sep']}'", f"join '{p['sep']}'", f"glue them together with '{p['sep']}' between"),
            wrong=lambda p: [("str join", "Joins without the requested separator.")]),
        _op("parse-log", "loglines", "logrows", ["parse"], lambda e, r: {}, lambda p: f"parse -r {_LOG_RX}",
            lambda p: ("parse each line into ts, level and msg", "parse → ts, level, msg", "split each entry into time, level and message (ts, level, msg)"), fields=lambda p: ["ts", "level", "msg"]),
        _op("level-is", "logrows", "logrows", ["where"], lambda e, r: {"lvl": _pick(r, ["ERROR", "WARN", "INFO"])}, lambda p: f"where level == '{p['lvl']}'",
            lambda p: (f"keep the {p['lvl']} rows", f"level={p['lvl']}", f"only the {p['lvl']} ones"),
            wrong=lambda p: [(f"where level != '{p['lvl']}'", "Keeps every level except the one asked for.")]),
        _op("when-and-what", "logrows", "rows", ["select"], lambda e, r: {}, lambda p: "select ts msg",
            lambda p: ("return ts and msg", "ts+msg", "when and what (ts, msg)"), fields=lambda p: ["ts", "msg"],
            alt=lambda p: "reject level", wrong=lambda p: [("select ts level", "Returns the level column instead of the message.")]),
        _op("messages", "logrows", "list", ["get"], lambda e, r: {}, lambda p: "get msg",
            lambda p: ("return the message text", "msgs", "just the messages"), wrong=lambda p: [("get ts", "Returns timestamps instead of messages.")]),
        _op("level-counts", "logrows", "rows", ["get", "uniq", "sort-by"], lambda e, r: {}, lambda p: "get level | uniq -c | sort-by value",
            lambda p: ("count the rows per level, as value and count, sorted by level", "level histogram (value, count)", "how many of each level (value, count)"), fields=lambda p: ["value", "count"],
            alt=lambda p: "group-by level | transpose value rows | each {|g| {value: $g.value, count: ($g.rows | length)} } | sort-by value",
            wrong=lambda p: [("get level | uniq | sort", "Lists the levels without counting them.")]),
        # table (rows from the JSON file)
        _op("owned-by", "table", "table", ["where"], lambda e, r: {"col": _c(e, "owner"), "who": _pick(r, e["layout"]["owners"])},
            lambda p: f"where {p['col']} == '{p['who']}'",
            lambda p: (f"keep the rows where {p['col']} is {p['who']}", f"{p['col']}={p['who']}", f"only {p['who']}'s rows"),
            wrong=lambda p: [(f"where {p['col']} != '{p['who']}'", "Keeps everything except what was asked for.")]),
        _op("at-least", "table", "table", ["where"], lambda e, r: {"col": _c(e, "number"), "t": _pick(r, [20, 40, 50, 60, 80])},
            lambda p: f"where {p['col']} >= {p['t']}",
            lambda p: (f"keep the rows with {p['col']} of at least {p['t']}", f"{p['col']}>={p['t']}", f"only where {p['col']} is {p['t']} or more"),
            wrong=lambda p: [(f"where {p['col']} < {p['t']}", "The comparison is inverted."), (f"where {p['col']} > {p['t']}", "Excludes rows exactly at the threshold.")]),
        _op("switched-on", "table", "table", ["where"], lambda e, r: {"col": _c(e, "flag"), "on": _pick(r, [True, True, False])},
            lambda p: f"where {p['col']} == {str(p['on']).lower()}",
            lambda p: (f"keep the rows where {p['col']} is {str(p['on']).lower()}", f"{p['col']}={str(p['on']).lower()}", f"only the ones that {'are' if p['on'] else 'are not'} {p['col']}"),
            alt=lambda p: f"where {'' if p['on'] else 'not '}{p['col']}",
            wrong=lambda p: [(f"where {p['col']} == {str(not p['on']).lower()}", "Keeps the opposite rows.")]),
        _op("highest-first", "table", "table", ["sort-by"], lambda e, r: {"key": _c(e, "key"), "col": _c(e, "number")},
            lambda p: f"sort-by {p['key']} | sort-by {p['col']} --reverse",
            lambda p: (f"sort by {p['col']}, highest first", f"by {p['col']} desc", f"highest {p['col']} on top"),
            wrong=lambda p: [(f"sort-by {p['key']} | sort-by {p['col']}", "Sorts ascending; the request wanted the highest first.")]),
        _op("pick-columns", "table", "rows", ["select"], lambda e, r: {"cols": [_c(e, "key")] + sorted(r.sample([_c(e, "owner"), _c(e, "number"), _c(e, "flag")], 2))},
            lambda p: "select " + " ".join(p["cols"]),
            lambda p: ("return " + ", ".join(p["cols"]), "+".join(p["cols"]), "just " + ", ".join(p["cols"])), fields=lambda p: p["cols"],
            wrong=lambda p: [("select " + " ".join(p["cols"][:-1]), f"Drops the {p['cols'][-1]} column.")]),
        _op("column-values", "table", "numbers", ["get"], lambda e, r: {"col": _c(e, "number")}, lambda p: f"get {p['col']}",
            lambda p: (f"take the {p['col']} values", p["col"], f"the {p['col']} numbers")),
        _op("count-per-owner", "table", "rows", ["get", "uniq", "sort-by"], lambda e, r: {"col": _c(e, "owner")},
            lambda p: f"get {p['col']} | uniq -c | rename {p['col']} count | sort-by {p['col']}",
            lambda p: (f"count the rows per {p['col']}, as {p['col']} and count, sorted by {p['col']}", f"count by {p['col']}", f"how many each {p['col']} has ({p['col']}, count)"), fields=lambda p: [p["col"], "count"],
            alt=lambda p: f"group-by {p['col']} | transpose {p['col']} items | insert count {{|r| $r.items | length}} | select {p['col']} count | sort-by {p['col']}",
            wrong=lambda p: [(f"get {p['col']} | uniq | sort", "Lists the values without counting them.")]),
        _op("sum-per-owner", "table", "rows", ["group-by", "transpose", "each", "math", "sort-by"], lambda e, r: {"col": _c(e, "owner"), "num": _c(e, "number")},
            lambda p: f"group-by {p['col']} | transpose {p['col']} rows | each {{|g| {{{p['col']}: $g.{p['col']}, total: ($g.rows.{p['num']} | math sum)}} }} | sort-by {p['col']}",
            lambda p: (f"sum {p['num']} per {p['col']}, as {p['col']} and total, sorted by {p['col']}", f"sum {p['num']} by {p['col']}", f"add up each {p['col']}'s {p['num']} ({p['col']}, total)"), fields=lambda p: [p["col"], "total"],
            wrong=lambda p: [(f"group-by {p['col']} | transpose {p['col']} rows | each {{|g| {{{p['col']}: $g.{p['col']}, total: ($g.rows | length)}} }} | sort-by {p['col']}", "Counts the rows instead of summing them.")]),
        _op("tag-reviewer", "table", "table", ["upsert"], lambda e, r: {"who": _pick(r, e["layout"]["owners"])}, lambda p: f"upsert reviewer '{p['who']}'",
            lambda p: (f"add a reviewer column set to '{p['who']}'", f"reviewer={p['who']}", f"mark {p['who']} as the reviewer on each"),
            alt=lambda p: f"insert reviewer '{p['who']}'"),
        *[_op(f"math-{f}", "numbers", "scalar", ["math"], lambda e, r: {}, lambda p, f=f: f"math {f}",
              lambda p, w=w: (f"take the {w}", w, f"what's the {w}"),
              wrong=lambda p, g=g, w=w: [(f"math {g}", f"Does not compute the {w}.")])
          for f, w, g in (("sum", "sum", "avg"), ("avg", "average", "sum"), ("max", "maximum", "min"), ("min", "minimum", "max"))],
        # config (record of sections) and record (one section)
        _op("section", "config", "record", ["get"], lambda e, r: {"sec": _pick(r, [e["layout"]["service_section"], e["layout"]["store_section"]])},
            lambda p: f"get {p['sec']}", lambda p: (f"take the {p['sec']} section", p["sec"], f"the {p['sec']} part")),
        _op("setting", "config", "scalar", ["get"],
            lambda e, r: _pick(r, [{"sec": e["layout"]["service_section"], "key": k} for k in ("name", "port", "debug")] + [{"sec": e["layout"]["store_section"], "key": k} for k in ("host", "port", "timeout_ms", "max_connections")]),
            lambda p: f"get {p['sec']}.{p['key']}",
            lambda p: (f"read {p['key']} from the {p['sec']} section", f"{p['sec']}.{p['key']}", f"what's the {p['sec']} {p['key']}"),
            wrong=lambda p: [(f"get {p['sec']}", "Returns the whole section instead of the one setting.")]),
        _op("key", "record", "scalar", ["get"], lambda e, r: {"key": _pick(r, ["host", "port", "name", "timeout_ms", "max_connections"])}, lambda p: f"get -o {p['key']}",
            lambda p: (f"read its {p['key']}, or nothing if it has none", f".{p['key']}?", f"its {p['key']}, if there is one")),
        _op("key-or-default", "record", "scalar", ["get", "default"], lambda e, r: (lambda k: {"key": k, "d": OPTIONAL_KEYS[k]})(_pick(r, sorted(OPTIONAL_KEYS))),
            lambda p: f"get -o {p['key']} | default {p['d']}",
            lambda p: (f"read its {p['key']}, falling back to {p['d']} when it is not set", f"{p['key']} default {p['d']}", f"the {p['key']} — may not be there, use {p['d']} then"),
            alt=lambda p: f"get {p['key']}? | default {p['d']}",
            wrong=lambda p: [(f"get -o {p['key']} | default (-1)", "Falls back to -1 instead of the requested default.")]),
        _op("keys", "record", "list", ["columns", "sort"], lambda e, r: {}, lambda p: "columns | sort",
            lambda p: ("list its keys, sorted", "keys", "which settings does it have"),
            alt=lambda p: "transpose key value | get key | sort", wrong=lambda p: [("values", "Lists the values instead of the keys.")]),
        *[s for t in ("files", "paths", "lines", "loglines", "logrows", "table", "rows", "list") for s in _first_last(t)],
    ]
    _JSON_TYPES = {"files", "paths", "rows", "table", "logrows", "record", "result", "list", "lines", "loglines", "config", "numbers", "value"}

    # ---- every other command, from Nu's signatures ----
    # Each kind of data the recipes produce is one of Nu's types; a command fits after a step when one of its
    # declared input types accepts that type, and its declared output type says what kind comes next.
    KIND_TYPE = {
        "files": "table", "rows": "table", "table": "table", "logrows": "table", "paths": "list<string>", "lines": "list<string>",
        "loglines": "list<string>", "list": "list<string>", "numbers": "list<number>", "text": "string", "record": "record", "config": "record",
    }
    _ACCEPTS = {
        "table": ("table", "list<any>", "any"), "list<string>": ("list<string>", "list<any>", "any"),
        "list<number>": ("list<number>", "list<any>", "any"), "string": ("string", "any"), "record": ("record", "any"),
    }
    _PRODUCES = {
        "table": "rows", "list<string>": "list", "list<number>": "numbers", "list<int>": "numbers", "list<float>": "numbers",
        "string": "text", "record": "record", "int": "scalar", "float": "scalar", "number": "scalar", "bool": "scalar", "filesize": "scalar",
    }
    _AUTO_CATEGORIES = {"filters", "strings", "math", "path", "conversions", "hash"}
    _SKIPPED = {"print": "writes to the terminal", "shuffle": "random order", "path self": "only valid in scripts", "path expand": "depends on where the copy lives"}
    _recipe_commands = {c for o in SOURCES + STAGES for c in o["commands"]}
    _names = {c["name"] for c in NU_COMMANDS}

    def _automatic_stages() -> list[dict]:
        stages = []
        for command in NU_COMMANDS:
            name = command["name"]
            is_namespace = any(other.startswith(name + " ") for other in _names)  # `str`, `math`, `path`: help text only
            if (command["category"] not in _AUTO_CATEGORIES or name in _recipe_commands or name in _SKIPPED or is_namespace
                    or any(not p["name"].startswith("--") for p in command["params"])):  # takes an argument: needs a recipe
                continue
            said = command["description"].rstrip(".")
            said = said[0].lower() + said[1:]
            for nu_type, accepted in _ACCEPTS.items():
                match = next((io for want in accepted for io in command["input_output"] if io["input"] == want), None)
                if match is None:
                    continue
                produced = _PRODUCES.get(match["output"], "value")
                if match["output"] in ("any", "list<any>") and match["input"] in ("any", "list<any>"):
                    produced = {"table": "rows", "list<string>": "list", "list<number>": "numbers", "string": "text", "record": "record"}[nu_type]
                stages.append(
                    _op(f"{name} ({nu_type})", nu_type, produced, [name], lambda e, r: {}, lambda p, n=name: n,
                        lambda p, s=said, n=name: (s, n, s), auto=True)
                )
        return stages

    AUTO_STAGES = _automatic_stages()
    STAGES = STAGES + AUTO_STAGES

    import networkx as nx

    KIND_GRAPH = nx.MultiDiGraph()  # nodes: kinds of data; edges: the operators that turn one into another
    for _o in STAGES:
        for _kind in ([k for k, t in KIND_TYPE.items() if t == _o["src"]] if _o["auto"] else [_o["src"]]):
            KIND_GRAPH.add_edge(_kind, _o["dst"], key=_o["name"], op=_o)
    for _o in SOURCES:
        KIND_GRAPH.add_edge("start", _o["dst"], key=_o["name"], op=_o)

    def _weight(op: dict, prior: dict) -> float:
        """An operator's sampling weight under a prior: the mean weight of its commands, relative to the average
        command, with a small floor so that commands the prior never saw can still appear occasionally."""
        if not prior:
            return 1.0
        mean = sum(prior.values()) / len(prior)
        seen = sum(prior.get(c, 0.0) for c in op["commands"]) / len(op["commands"]) / mean
        # a recipe keeps a small floor; an automatically added command appears only if the prior has seen it
        return seen if op["auto"] else max(seen, 0.02)

    def _choose(rng, ops, prior):
        weights = [_weight(o, prior) for o in ops]
        return rng.choices(ops, weights=weights)[0] if any(weights) else None

    def length_weights(origin: str, bias: float) -> list[float]:
        """How likely a chain of 1..6 steps is. At bias 0 this is the measured distribution: pipeline lengths in
        Nushell's documentation for native chains, in real command lines for ported ones. At bias 1 every length
        is equally likely. In between the two are mixed."""
        measured = PRIOR_INFO["lengths"].get(origin) or {1: 0.4, 2: 0.35, 3: 0.15, 4: 0.07, 5: 0.03}
        total = sum(measured.get(n, 0.0) for n in range(1, 7))
        return [(1 - bias) * measured.get(n, 0.0) / total + bias / 6 for n in range(1, 7)]

    def sample_chain(env: dict, rng, prior: dict, lengths: list[float]):
        """One chain: a source, then stages whose input type matches what the previous step produced, up to a
        length drawn from `lengths` (or until nothing fits)."""
        target = rng.choices(range(1, 7), weights=lengths)[0]
        source = _choose(rng, [d["op"] for _, _, d in KIND_GRAPH.out_edges("start", data=True)], prior)
        params = source["make"](env, rng)
        if params is None or None in params.values():
            return None
        steps, kind = [(source, params)], source["dst"]
        while len(steps) < max(target, 2 if kind == "config" else 1):
            used = {s[0]["name"] for s in steps}
            options = [d["op"] for _, _, d in KIND_GRAPH.out_edges(kind, data=True) if d["op"]["name"] not in used] if kind in KIND_GRAPH else []
            if not options:
                break
            op = _choose(rng, options, prior)
            if op is None:
                break
            params = op["make"](env, rng)
            if params is None or None in params.values():
                break
            steps.append((op, params))
            kind = op["dst"]
        as_json = kind in _JSON_TYPES and (kind in ("rows", "result", "record", "config") or rng.random() < 0.6)
        return {"steps": steps, "json": as_json, "kind": kind}

    def render(spec: dict, alt: bool = False, replace: dict | None = None) -> tuple[str, list[str]]:
        """Nushell source for a chain and the text of each step. `replace` maps a step index to other text."""
        parts = []
        for i, (op, params) in enumerate(spec["steps"]):
            text = (replace or {}).get(i)
            if text is None:
                text = op["alt"](params) if alt and op["alt"] else op["code"](params)
            parts.append(text)
        tail = " | to json --raw" if spec["json"] else ""
        bound = next((i for i, (op, _) in enumerate(spec["steps"]) if op["binds"]), None)
        if bound is not None:  # a step that reads the pipeline so far through a variable
            name = spec["steps"][bound][0]["binds"]
            rest = "".join(f" | {p}" for p in parts[bound + 1 :])
            return f"let {name} = ({' | '.join(parts[:bound])} | sort); let result = ({parts[bound]}); $result{rest}{tail}", parts
        if len(parts) >= 4:  # longer chains bind the first half, the way people write them
            cut = len(parts) // 2
            return f"let found = ({' | '.join(parts[:cut])}); $found | {' | '.join(parts[cut:])}{tail}", parts
        return " | ".join(parts) + tail, parts

    def describe(spec: dict) -> list[str]:
        """The request in three registers, composed from what each step says about itself."""
        said = [op["say"](params) for op, params in spec["steps"]]
        fields = next((op["fields"](params) for op, params in reversed(spec["steps"]) if op["fields"]), None)
        ending = ("as JSON" + (f" with {', '.join(fields)}" if fields and spec["kind"] == "rows" else "")) if spec["json"] else ""
        plain = "; then ".join(s[0] for s in said)
        plain = plain[0].upper() + plain[1:] + (f". Return it {ending}." if ending else ".")
        short = ", ".join(s[1] for s in said) + (", json" if ending else "")
        casual = " and ".join(s[2] for s in said) + (" — json please" if ending else "") + "?"
        return [plain, short, casual]

    def categorize(spec: dict) -> str:
        """Which catalog category a chain belongs to, from the operators it uses."""
        names = [op["name"] for op, _ in spec["steps"]]
        first = spec["steps"][0][0]["dst"]
        if names[0] == "try-open" or "resolve" in names or "key-or-default" in names:
            return "pipelines_recovery"
        if any(n in ("hit-counts", "matching-lines", "path-has", "name-has") or n.startswith("containing-") for n in names) or names[0] == "find-paths":
            return "search"
        if first in ("table", "config"):
            return "tables_records"
        if first in ("text", "loglines"):
            return "text_manipulation"
        return "filesystem"

    # What each common Unix command is for, in this dataset's categories. Only used to turn measured usage
    # into category shares; commands that are not about files or text (git, editors, sudo, package managers)
    # are left out.
    UNIX_CATEGORY = {
        "search": ["find", "grep", "egrep", "fgrep", "rg", "ag", "ack", "locate", "which", "whereis"],
        "text_manipulation": ["cat", "sed", "awk", "cut", "tr", "sort", "uniq", "wc", "head", "tail", "echo", "less", "more", "tac", "rev", "paste", "printf", "nl", "diff", "strings", "fold"],
        "filesystem": ["ls", "ll", "la", "l", "cd", "pwd", "du", "df", "stat", "file", "tree", "basename", "dirname", "readlink", "realpath", "mkdir", "rm", "mv", "cp", "touch", "ln", "chmod", "chown"],
        "tables_records": ["jq", "column", "join", "bc", "expr", "seq", "yq", "csvcut", "sqlite3"],
        "pipelines_recovery": ["xargs", "test", "[", "read", "tee", "parallel", "true", "false", "timeout", "watch"],
    }

    def category_shares(categories: list[str], floor: float = 0.12) -> dict:
        """How the dataset is divided among the categories: each category's share of measured Unix usage
        (data/command_usage.json, grouped by UNIX_CATEGORY), so that what people do most in a shell gets the
        most data. A category is never given less than `floor`: structured data and recovery are what Nushell
        adds, and Unix usage barely registers them."""
        usage_file = ROOT / "data" / "command_usage.json"
        if not usage_file.exists():
            return {c: 1 / len(categories) for c in categories}
        combined = json.loads(usage_file.read_text(encoding="utf-8"))["combined"]
        mass = {c: sum(combined.get(u, 0.0) for u in UNIX_CATEGORY[c]) for c in categories}
        total = sum(mass.values()) or 1
        raised = {c: max(mass[c] / total, floor) for c in categories}
        return {c: v / sum(raised.values()) for c, v in raised.items()}

    _PROBE = {}

    def _chain(env: dict, spec: dict, index: int) -> dict:
        code, parts = render(spec)
        alt = render(spec, alt=True)[0]
        commands = list(dict.fromkeys(c for op, _ in spec["steps"] for c in op["commands"])) + (["to json"] if spec["json"] else [])
        asks = describe(spec)
        return {
            "chain_id": f"{env['id']}/{index:03d}", "env": env["id"], "layout": env["layout"], "files": env["files"], "spec": spec, "parts": parts,
            "family": "+".join(op["name"] for op, _ in spec["steps"]), "category": categorize(spec), "commands": commands,
            "intent": asks[0], "asks": asks, "code": code, "alt_code": alt,
        }

    def build_chains(environments: list[dict], categories: list[str], total: int, ported_share: float = 0.25, length_bias: float = 0.5, seed: int = 73) -> tuple[list[dict], list[dict], dict]:
        """Sample chains, keep those that survive execution, and fill each category's quota.

        Quotas come from `category_shares`. Within a category at most `ported_share` (never more than
        PORTED_LIMIT) of the chains are sampled under the translated Unix prior; the rest are sampled under the
        Nushell-native prior. Chains are spread evenly over the environments."""
        ported_share = min(ported_share, PORTED_LIMIT)
        shares = category_shares(categories)
        quota = {c: round(total * shares[c]) for c in categories}
        candidates = []
        for env in environments:
            rng, seen = random.Random(f"chains-{seed}-{env['id']}"), set()
            for origin, prior, share in (("native", NATIVE, 1 - ported_share), ("ported", PORTED, ported_share)):
                wanted = {c: -(-quota[c] * share * 2.5 // len(environments)) for c in categories}  # oversample; some will fail
                got, tries = {c: 0 for c in categories}, 0
                while any(got[c] < wanted[c] for c in categories) and tries < 4000:
                    tries += 1
                    spec = sample_chain(env, rng, prior, length_weights(origin, length_bias))
                    if spec is None:
                        continue
                    chain = {**_chain(env, spec, len(seen) + 1), "origin": origin}
                    category = chain["category"]
                    if chain["code"] in seen or category not in wanted or got[category] >= wanted[category]:
                        continue
                    seen.add(chain["code"])
                    got[category] += 1
                    candidates.append(chain)
        # an automatic stage must change the result, or it is just noise in the chain
        def without_automatic(chain):
            """The chain with each automatically added stage left out in turn."""
            steps = chain["spec"]["steps"]
            return [render({**chain["spec"], "steps": steps[:i] + steps[i + 1 :]})[0] for i, (o, _) in enumerate(steps) if o["auto"]]

        shorter = [without_automatic(c) for c in candidates]
        _jobs = [(code, c["env"]) for codes, c in zip(shorter, candidates) for code in codes]
        shorter_runs = dict(zip([j[0] for j in _jobs], run_nu_many(_jobs)))
        runs = run_nu_many([(c["code"], c["env"]) for c in candidates])
        variants = run_nu_many([(c["code"], c["env"], True) for c in candidates])
        alts = run_nu_many([(c["alt_code"], c["env"]) for c in candidates])
        # par-each returns in any order, so run each chain a second time to catch an unstable result
        repeats = run_nu_many([(c["code"] + " ", c["env"]) for c in candidates])
        verified, dropped = [], []
        for chain, base, variant, alt, again, short in zip(candidates, runs, variants, alts, repeats, shorter):
            reason = None
            if any(shorter_runs[code]["ok"] and shorter_runs[code]["stdout"] == base["stdout"] for code in short):
                reason = "an automatic stage changes nothing"
            elif not base["ok"]:
                reason = "does not run: " + base["stderr"][:120]
            elif base["stdout"] in ("", "[]", "{}", "null", '""'):
                reason = "returns nothing"
            elif again["stdout"] != base["stdout"]:
                reason = "output changes between runs"
            elif len(base["stdout"]) > 4000:
                reason = "output too long to be a useful reference"
            if reason:
                dropped.append({"chain_id": chain["chain_id"], "family": chain["family"], "reason": reason})
                continue
            verified.append(
                {
                    **chain,
                    "expected": base["stdout"],
                    "expected_variant": variant["stdout"] if variant["ok"] else None,
                    "alt_ok": chain["alt_code"] != chain["code"] and alt["ok"] and same_output(alt["stdout"], base["stdout"]),
                }
            )
        save_nu_cache()
        # fill each category's quota, going round the environments, native chains first and ported ones up to their cap
        kept = []
        for category in categories:
            pools = {origin: {e["id"]: [c for c in verified if c["category"] == category and c["origin"] == origin and c["env"] == e["id"]] for e in environments} for origin in ("native", "ported")}
            limits = {"ported": int(quota[category] * ported_share), "native": quota[category]}
            chosen = []
            for origin in ("ported", "native"):
                room = limits[origin] if origin == "ported" else quota[category] - len(chosen)
                while room > 0 and any(pools[origin].values()):
                    for env_id in list(pools[origin]):
                        if room > 0 and pools[origin][env_id]:
                            chosen.append(pools[origin][env_id].pop(0))
                            room -= 1
            kept.extend(chosen)
        summary = {
            "category_shares": {c: round(v, 3) for c, v in shares.items()},
            "quota": quota,
            "ported_share": round(sum(c["origin"] == "ported" for c in kept) / max(1, len(kept)), 3),
        }
        return kept, dropped, summary

    def structural_breaks(chain: dict, failure_type: str, index: int) -> list[tuple]:
        """Broken variants made by changing the sampled structure, as (code, span, note). Whether each one
        deserves its label is decided afterwards by running it."""
        spec, parts, out = chain["spec"], chain["parts"], []
        steps = spec["steps"]
        rng = random.Random(f"{chain['chain_id']}-{failure_type}")
        order = list(range(len(steps)))
        rng.shuffle(order)
        if failure_type == "wrong_operation_or_argument":
            for i in order:
                for text, note in steps[i][0]["wrong"](steps[i][1]):
                    out.append((render(spec, replace={i: text})[0], text, note))
        elif failure_type == "request_misunderstood":
            env = {"layout": chain["layout"], "files": chain["files"], "id": chain["env"]}
            for i in order:
                op, params = steps[i]
                if not params:
                    continue
                for _ in range(6):  # the same step with another value: the wrong name, number or term
                    other = op["make"](env, rng)
                    if other and None not in other.values() and op["code"](other) != op["code"](params) and (i > 0 or failure_type != "request_misunderstood" or op["dst"] in ("scalar", "result", "config")):
                        text = op["code"](other)
                        out.append((render(spec, replace={i: text})[0], text, f"Uses `{text}` where the request asked for `{parts[i]}`."))
                        break
        elif failure_type == "wrong_data_source":
            env = {"layout": chain["layout"], "files": chain["files"], "id": chain["env"]}
            op, params = steps[0]
            for _ in range(8):  # the same kind of source, pointed somewhere else
                other = op["make"](env, rng)
                if other and None not in other.values() and op["code"](other) != op["code"](params):
                    text = op["code"](other)
                    out.append((render(spec, replace={0: text})[0], text, f"Reads from `{text}`; the request named `{parts[0]}`."))
            columns, owners = chain["layout"]["columns"], chain["layout"]["owners"]
            if op["name"] == "read-table":
                text = f"[[{columns['key']} {columns['flag']} {columns['owner']} {columns['number']}]; [demo_a true {owners[0]} 10] [demo_b false {owners[1]} 90] [demo_c true {owners[2]} 55]]"
                out.append((render(spec, replace={0: text})[0], text, f"Uses a made-up example table and never opens {params['path']}."))
            if op["name"] == "read-config":
                text = f"{{{chain['layout']['service_section']}: {{name: 'example', port: 8000, debug: true}}, {chain['layout']['store_section']}: {{host: 'localhost', port: 5432, max_connections: 5, timeout_ms: 100}}}}"
                out.append((render(spec, replace={0: text})[0], text, f"Uses a made-up record and never opens {params['path']}."))
        elif failure_type == "broken_chain_or_function":
            for i in order:
                if 0 < i < len(steps) and not steps[i][0]["binds"]:
                    shorter = {**spec, "steps": steps[:i] + steps[i + 1 :]}
                    code, new_parts = render(shorter)
                    span = new_parts[i - 1] if i - 1 < len(new_parts) else new_parts[-1]
                    out.append((code, span, f"The `{parts[i][:60]}` step is missing."))
            for i in order:
                if 0 < i < len(steps) - 1 and not steps[i][0]["binds"] and not steps[i + 1][0]["binds"]:
                    swapped = {**spec, "steps": steps[:i] + [steps[i + 1], steps[i]] + steps[i + 2 :]}
                    code, new_parts = render(swapped)
                    out.append((code, new_parts[i], f"`{parts[i + 1][:40]}` runs before `{parts[i][:40]}`; the order is reversed."))
        return out

    return AUTO_STAGES, KIND_GRAPH, NATIVE, PORTED, PORTED_LIMIT, PRIOR_INFO, SOURCES, STAGES, build_chains, structural_breaks


@app.cell
def _(ALL_CATEGORIES, build_chains, category_boxes, chains_total_slider, environments, fval, ival, length_bias_slider, ported_share_slider):
    selected_categories = [name for name, on in zip(ALL_CATEGORIES, category_boxes.value) if on] or list(ALL_CATEGORIES)
    verified_chains, dropped_chains, chain_summary = build_chains(
        environments, selected_categories, ival(chains_total_slider), fval(ported_share_slider), fval(length_bias_slider)
    )
    return chain_summary, selected_categories, verified_chains


@app.cell
def _(COMMAND_CATALOG, Counter, mo, selected_categories, verified_chains):
    # One tab per category, and one for the full list the chains are sampled from. The tab picks what the table shows.
    _mix = Counter(c["category"] for c in verified_chains)
    CATALOG_TAB = f"all commands ({len(COMMAND_CATALOG)})"
    chain_tab_names = {f"{name.replace('_', ' ')} ({_mix.get(name, 0)})": name for name in selected_categories}
    chain_tabs = mo.ui.tabs({**{label: "" for label in chain_tab_names}, CATALOG_TAB: ""})
    return CATALOG_TAB, chain_tab_names, chain_tabs


@app.cell
def _(CATALOG_TAB, COMMAND_CATALOG, NATIVE, PORTED, SOURCES, STAGES, chain_tab_names, chain_tabs, mo, pd, verified_chains):
    if chain_tabs.value == CATALOG_TAB:
        _reachable = {c for o in SOURCES + STAGES for c in o["commands"]}
        tab_chains = []
        chain_table = mo.ui.table(
            pd.DataFrame(
                [
                    {**c, "in the generator": c["name"] in _reachable, "native weight": round(NATIVE.get(c["name"], 0.0), 4), "ported weight": round(PORTED.get(c["name"], 0.0), 4)}
                    for c in COMMAND_CATALOG
                ]
            ),
            selection=None, page_size=8,
        )
    else:
        tab_chains = [c for c in verified_chains if c["category"] == chain_tab_names.get(chain_tabs.value)]
        chain_table = mo.ui.table(
            pd.DataFrame([{"request": c["asks"][0], "command": c["code"], "steps": len(c["spec"]["steps"]), "prior": c["origin"]} for c in tab_chains]),
            selection="single", page_size=8,
        )
    return chain_table, tab_chains


@app.cell
def _(chain_table, tab_chains, verified_chains):
    # The chain in focus: the row clicked in the table, else the first chain with three steps or more.
    _picked = chain_table.value
    _row = int(_picked.index[0]) if tab_chains and _picked is not None and len(_picked) else None
    selected_chain = tab_chains[_row] if _row is not None and _row < len(tab_chains) else next(c for c in verified_chains if len(c["spec"]["steps"]) >= 3)
    return (selected_chain,)


@app.cell
def _(
    breaks_per_chain_slider,
    category_boxes,
    chain_table,
    chain_tabs,
    chains_total_slider,
    json,
    k_variants_slider,
    length_bias_slider,
    mo,
    ported_share_slider,
    request_btn,
    row,
    selected_chain,
    space,
):
    def _value(text):
        try:
            return mo.json(json.loads(text))
        except ValueError:
            return mo.md(f"```text\n{text[:2000]}\n```")

    mo.vstack(
        [
            space(56, "step-data"),
            row(
                (
                    mo.vstack(
                        [
                            mo.md("## Data faders"),
                            mo.hstack([chains_total_slider, ported_share_slider, length_bias_slider, breaks_per_chain_slider, k_variants_slider], justify="start", gap=0.5, wrap=False),
                            mo.hstack(list(category_boxes), justify="start", gap=0.75, wrap=True),
                            request_btn,
                        ],
                        gap=1.5,
                    ),
                    0, 470,
                ),
                (mo.vstack([mo.md("## Chains"), chain_tabs, chain_table], gap=0.75), 1, 440),
            ),
            # the chain in focus, in full
            mo.md("\n".join(f"> {ask}" for ask in selected_chain["asks"][:1])),
            mo.md(f"```nu\n{selected_chain['code']}\n```"),
            _value(selected_chain["expected"]),
        ],
        gap=1.25,
    )
    return


@app.cell
def _(OPTIONAL_KEYS, ThreadPoolExecutor, json, re, run_nu, same_output, save_nu_cache, structural_breaks):
    # -------------------------------------------------------------------------
    # Step 4: breaking procedure. Every break is proposed by a rule and kept only if execution agrees
    # with its label, so the labels describe what the command does, not what the rule intended.
    # -------------------------------------------------------------------------
    # The ten failure types of the training sketch, each with the examples it lists. Every generator below
    # produces one of those examples; execution decides whether a candidate has earned its label.
    FAILURE_TYPES = {
        "valid": "The tool call is well formed and does what was asked.",
        "request_misunderstood": "Request misunderstood: a misheard file name, a missing negation, or a wrong number.",
        "wrong_tool_or_malformed_call": "Wrong tool or malformed call: another tool, prose, or unreadable call syntax.",
        "invalid_nushell": "Invalid Nushell: an unclosed delimiter, an invalid function signature, or syntax from another language.",
        "wrong_data_source": "Wrong data source: the wrong file or column, or example data in place of the real file.",
        "wrong_operation_or_argument": "Wrong operation or argument: count instead of sum, wrong sort column or direction, or wrong filter.",
        "broken_chain_or_function": "Broken chain or function: missing or reordered steps, an uncalled function, or a wrong intermediate value.",
        "wrong_output_or_side_effect": "Wrong output or side effect: wrong output shape, extra text, or a write to the wrong path.",
        "runtime_or_environment_failure": "Runtime or environment failure: a timeout, an unsupported feature, a version difference, or a path the project does not have.",
        "grader_false_pass": "Grader false pass: a hard-coded result that matches this one workspace without computing anything.",
        "grader_false_fail": "Grader false fail: an equivalent answer that differs only in formatting, order, type, or an unspecified tie-break.",
    }
    FAILURE_LABELS = list(FAILURE_TYPES)
    BREAK_LABELS = FAILURE_LABELS[1:]
    NOT_BROKEN = {"valid", "grader_false_fail"}  # a false fail is a correct answer that a strict comparison would reject
    VERIFIED_BY = {
        "request_misunderstood": "runs; output differs from the reference",
        "wrong_tool_or_malformed_call": "the call does not parse as a run_nushell tool call",
        "invalid_nushell": "Nu reports a parse error",
        "wrong_data_source": "runs; output differs from the reference",
        "wrong_operation_or_argument": "runs; output differs from the reference",
        "broken_chain_or_function": "runs; output differs from the reference",
        "wrong_output_or_side_effect": "runs; output differs or a file was written",
        "runtime_or_environment_failure": "parses; then fails or times out",
        "grader_false_pass": "matches the reference here; the reference changes in a second copy of the environment",
        "grader_false_fail": "output is not identical to the reference but is the same answer once formatting, order and type are set aside",
    }

    def wrap_tool_call(code: str) -> str:
        return f"<function=run_nushell>\n<parameter=code>\n{code}\n</parameter>\n</function>\n</tool_call>"

    _CALL = re.compile(r"\A<function=run_nushell>\n<parameter=code>\n(.*)\n</parameter>\n</function>\n</tool_call>\s*\Z", re.S)

    def parse_tool_call(completion: str):
        """The Nushell source of a well-formed run_nushell call, else None."""
        found = _CALL.match(completion)
        return found.group(1) if found else None

    def split_top(code: str, sep: str) -> list[str]:
        """Split at `sep` outside quotes and brackets, so pipes inside closures stay put."""
        parts, depth, quote, start = [], 0, None, 0
        for i, ch in enumerate(code):
            if quote:
                quote = None if ch == quote else quote
            elif ch in "'\"`":
                quote = ch
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == sep and depth == 0:
                parts.append(code[start:i])
                start = i + 1
        parts.append(code[start:])
        return parts

    def _pipelines(code: str):
        """Each statement that is a pipeline of three or more stages, longest first, with what is needed to rebuild
        it. A `let name = ( ... )` statement is opened so its inner pipeline can be edited too."""
        statements = split_top(code, ";")
        found = []
        for idx, statement in enumerate(statements):
            lead = statement[: len(statement) - len(statement.lstrip())]
            body, prefix, suffix = statement.strip(), "", ""
            bound = re.match(r"(let \w+ = \()(.*)(\))\Z", body, re.S)
            if bound and len(split_top(bound.group(2), ")")) == 1:
                prefix, body, suffix = bound.groups()
            stages = [s.strip() for s in split_top(body, "|")]
            if len(stages) >= 3:
                found.append((idx, lead, prefix, stages, suffix))

        def rebuild(idx, lead, prefix, stages, suffix):
            out = list(statements)
            out[idx] = f"{lead}{prefix}{' | '.join(stages)}{suffix}"
            return ";".join(out)

        return sorted(found, key=lambda f: -len(f[3])), rebuild

    def _sub(code, pattern, repl, span, note, last=False, every=False):
        """One regex edit as a candidate `(code, span, note)`, or nothing when the pattern is absent."""
        matches = list(re.finditer(pattern, code))
        if not matches:
            return []
        if every:
            return [(re.sub(pattern, repl, code), matches[0].expand(span), matches[0].expand(note))]
        m = matches[-1] if last else matches[0]
        return [(code[: m.start()] + m.expand(repl) + code[m.end() :], m.expand(span), m.expand(note))]

    _NUMBERS = {0: 50, 50: 5, 80: 8, 1: 3, 2: 5, 3: 1, 5: 2, 10: 100, 30: 3, 100: 7, 1000: 100}

    def _vocabulary(chain):
        layout = chain["layout"]
        return layout["components"] + layout["topics"]

    def _request_misunderstood(chain):
        code, layout, out = chain["code"], chain["layout"], []
        for level, other in (("ERROR", "WARN"), ("WARN", "INFO"), ("INFO", "ERROR")):
            out += _sub(code, rf"(?<![A-Za-z]){level}(?![A-Za-z])", other, other, f"Uses {other} where the request said {level}.", every=True)
        owners = layout["owners"]
        for i, owner in enumerate(owners):
            other = owners[(i + 1) % len(owners)]
            out += _sub(code, rf"'{owner}'", f"'{other}'", f"'{other}'", f"Uses '{other}' where the request said '{owner}'.", every=True)
        words = _vocabulary(chain)
        for i, term in enumerate(words):
            other = words[(i + 3) % len(words)]
            out += _sub(code, rf"'{term}'", f"'{other}'", f"'{other}'", f"Searches for '{other}' where the request said '{term}'.", every=True)
            out += _sub(code, rf"\*{term}\*", f"*{other}*", f"*{other}*", f"Looks for names with '{other}' where the request said '{term}'.")
        for key in list(OPTIONAL_KEYS) + ["timeout_ms", "max_connections"]:
            other = key.replace("_", "") if "_" in key else key + "s"
            out += _sub(code, rf"\b{key}\b", other, other, f"Reads `{other}` where the request named `{key}`.", every=True)
        for m in re.finditer(r"(first|take|>=|default) (\d+)\b", code):
            n = int(m.group(2))
            wrong = _NUMBERS.get(n, n * 10 + 1)
            out.append((code[: m.start(2)] + str(wrong) + code[m.end(2) :], f"{m.group(1)} {wrong}", f"Uses {wrong} where the request said {n}."))
        out += _sub(code, r"== true", "== false", "== false", "Keeps the rows where the condition is false; the request asked for the true ones.")
        out += _sub(code, r"== false", "== true", "== true", "Keeps the rows where the condition is true; the request asked for the false ones.")
        out += _sub(code, r"'TODO\|FIXME'", "'NOTE|HACK'", "'NOTE|HACK'", "Looks for other markers than the ones requested.")
        out += _sub(code, r"'(TODO|FIXME)'", "'XXX'", "'XXX'", r"Looks for XXX where the request said \1.")
        out += _sub(code, r"'([\w/]+)(\w)\.(toml|json|yaml|py|js|go|rb|rs|log|md)'", r"'\1x.\3'", r"'\1x.\3'", "The file name is not the one requested.")
        out += _sub(code, r"\|(\w+)'\)", r"|\1s_old')", r"|\1s_old'", "One of the two name patterns is not what was requested.")
        return out

    def _wrong_data_source(chain):
        code, layout, out = chain["code"], chain["layout"], []
        dirs, columns = layout["dirs"], layout["columns"]
        logs = [f"{dirs['logs']}/{name}" for name in layout["log_names"]]
        for log, other in ((logs[0], logs[1]), (logs[1], logs[0])):
            out += _sub(code, rf"(open --raw) {re.escape(log)}", rf"\1 {other}", f"open --raw {other}", f"Reads {other}; the request was about {log}.")
        table_path = f"{dirs['cfg']}/{layout['table']}"
        table = f"[[{columns['key']} {columns['flag']} {columns['owner']} {columns['number']}]; [demo_a true {layout['owners'][0]} 10] [demo_b false {layout['owners'][1]} 90] [demo_c true {layout['owners'][2]} 55]]"
        out += _sub(code, rf"open {re.escape(table_path)}", table, table, f"Uses a made-up example table and never opens {table_path}.")
        config_path = f"{dirs['cfg']}/{layout['config']}"
        record = f"{{{layout['service_section']}: {{name: 'example', port: 8000}}, {layout['store_section']}: {{host: 'localhost', port: 5432, max_connections: 5}}}}"
        out += _sub(code, rf"open {re.escape(config_path)}", record, record, f"Uses a made-up record and never opens {config_path}.")
        roles = ["src", "cfg", "logs", "docs", "scripts"]
        for i, role in enumerate(roles):  # the same pattern, pointed at another folder
            here, there = re.escape(dirs[role]), dirs[roles[(i + 2) % len(roles)]]
            out += _sub(code, rf"(glob|ls) {here}/(\S*?)(?=[ )])", rf"\1 {there}/**/*", rf"\1 {there}/**/*", f"Looks in {there}; the request named {dirs[role]}.")
        out += _sub(code, r"(glob|ls) \*\*/\*\.\{[\w,]+\}", rf"\1 {dirs['docs']}/**/*", rf"\1 {dirs['docs']}/**/*", "Looks in a different place than the request named.")
        out += _sub(code, r"(glob|ls) \*\*/\*\.(\w+)", rf"\1 {dirs['src']}/**/*", rf"\1 {dirs['src']}/**/*", "Looks in a different place than the request named.")
        out += _sub(code, r"(glob|ls) \*\*/\*(?=[ )])", rf"\1 {dirs['src']}/**/*", rf"\1 {dirs['src']}/**/*", "Looks in a different place than the request named.")
        return out

    def _wrong_operation(chain):
        code, columns, out = chain["code"], chain["layout"]["columns"], []
        k, o, n, b = columns["key"], columns["owner"], columns["number"], columns["flag"]
        out += _sub(code, r"sort-by (\w+) --reverse", r"sort-by \1", r"sort-by \1", "Sorts ascending; the request wanted the largest first.", last=True)
        out += _sub(code, r"first (\d+)", r"last \1", r"last \1", "Takes the last rows instead of the first.")
        out += _sub(code, r">= (\d+)", r"< \1", r"< \1", "The comparison is inverted.")
        out += _sub(code, r"hits > 0", "hits == 0", "hits == 0", "Keeps the files without a match.")
        out += _sub(code, r"== '(\w+)'", r"!= '\1'", r"!= '\1'", "Keeps everything except what was asked for.", last=True)
        for a, c, why in (
            ("math avg", "math sum", "Sums where an average was asked for."),
            ("math sum", "length", "Counts the rows instead of summing them."),
            ("math max", "math min", "Takes the minimum instead of the maximum."),
            ("str downcase", "str upcase", "Upper-cases instead of lower-casing."),
            ("path basename", "path dirname", "Returns directories instead of file names."),
            ("path dirname", "path basename", "Returns file names instead of directories."),
            ("is-empty", "is-not-empty", "The emptiness test is inverted."),
            ("select ts msg", "select ts level", "Returns the level column instead of the message."),
            ("get msg", "get ts", "Returns timestamps instead of messages."),
            ("select name size", "select name type", "Returns the type column instead of the size."),
            (f"select {k} {b} {n}", f"select {k} {o}", "Returns other columns than requested."),
            (f"select {k} {n} reviewer", f"select {k} reviewer", f"Drops the {n} column."),
            (f"select {k} {o} {n}", f"select {k} {b}", "Returns other columns than requested."),
            ("$p.stem", "$p.parent", "Reports the parent directory as the stem."),
            ("str contains -i", "str starts-with -i", "Matches only at the start of a line."),
            ("str contains '", "str ends-with '", "Matches only at the end of a line."),
            ("| length)", "| str join | str length)", "Counts characters instead of lines."),
            ("uniq | sort", "sort", "Keeps duplicates."),
            ("| columns", "| values", "Lists the values instead of the keys."),
            ("status: 'choose_one'", "status: 'resolved'", "Reports an ambiguous match as resolved."),
            ("default ", "default (-1) | default ", "Falls back to -1 instead of the requested default."),
            ("candidates: $choices", "candidates: ($choices | length)", "Returns how many candidates there are, not which."),
            ("=~ '(?i)", "!~ '(?i)", "Keeps the paths that do not match."),
            ("=~ '", "!~ '", "Keeps the lines that do not match."),
            ("append $by_b", "append $by_a", "Combines the first search with itself."),
            ("insert count {|r| $r.items | length}", f"insert count {{|r| $r.items | get {n} | math sum}}", f"Sums {n} where a count was asked for."),
        ):
            if a in code:
                at = code.index(a)
                out.append((code[:at] + c + code[at + len(a) :], c, why))
        return out

    def _broken_chain(chain):
        found, rebuild = _pipelines(chain["code"])
        drops, swaps = [], []
        for idx, lead, prefix, stages, suffix in found:
            for i in range(1, len(stages) - 1):
                kept = stages[:i] + stages[i + 1 :]
                span = f"{stages[i - 1][-28:]} | {stages[i + 1][:28]}"
                drops.append((rebuild(idx, lead, prefix, kept, suffix), span, f"The `{stages[i][:60]}` step is missing."))
            for i in range(1, len(stages) - 2):
                moved = stages[:i] + [stages[i + 1], stages[i]] + stages[i + 2 :]
                span = f"{stages[i + 1][-28:]} | {stages[i][:28]}"
                swaps.append((rebuild(idx, lead, prefix, moved, suffix), span, f"`{stages[i + 1][:40]}` runs before `{stages[i][:40]}`; the order is reversed."))
        mixed = []
        for pair in zip(drops, swaps):
            mixed.extend(pair)
        return mixed + drops[len(swaps) :] + swaps[len(drops) :]

    def _wrong_output(chain):
        code, out, tail = chain["code"], [], " | to json --raw"
        if code.endswith(tail):
            body = code[: -len(tail)]
            out.append((body, body[-40:], "Returns Nu's table rendering; the request needs JSON."))
            out.append((body + " | to yaml", "| to yaml", "Returns YAML where JSON was needed."))
            out.append((code + " | save -f results.json", "| save -f results.json", "Writes results.json, which nobody asked for, and returns nothing."))
        out.append((f"print 'Here are the results:'; {code}", "print 'Here are the results:'", "Prints extra text before the answer."))
        return out

    def _runtime_failure(chain):
        code, layout, out = chain["code"], chain["layout"], []
        columns, store, service = layout["columns"], layout["store_section"], layout["service_section"]
        o, n, b = columns["owner"], columns["number"], columns["flag"]
        out += _sub(code, r"(open(?: --raw)?) ([\w/]+)\.(toml|json|log|md)", r"\1 \2_old.\3", r"\1 \2_old.\3", "Opens a path that does not exist.")
        for a, c in (
            (f"get {store}", f"get {store}s_v2"), ("$store.host", "$store.hostname"), (f"$cfg.{service}.name", f"$cfg.{service}_v2.name"),
            ("$r.name", "$r.path"), ("get name", "get path"), ("get level", "get severity"), ("level ==", "severity =="),
            (f"get {n}", f"get {n}_pct"), (f"{o} ==", f"{o}_id =="), (f"{n} >=", f"{n}_pct >="), (f"where {b}", f"where {b}_now"),
            (f"where not {b}", f"where not {b}_now"), (f"group-by {o}", f"group-by {o}_id"), ("get msg", "get message"),
            ("select ts msg", "select time msg"), ("select name size", "select name bytes"), ("where type == file", "where kind == file"),
            ("$g.value", "$g.label"), ("$matches | first", "$matches | get 5"), ("$matches.0", "$matches.9"),
            ("$it | str contains", "$it.text | str contains"), ("path relative-to (pwd)", "path relative-to '/nonexistent'"),
        ):
            if a in code:
                at = code.index(a)
                out.append((code[:at] + c + code[at + len(a) :], c, f"`{c}` does not exist at run time."))
        return out

    def _invalid_nushell(chain):
        code, out = chain["code"], []
        for closer, name in ((")", "parenthesis"), ("}", "brace")):
            if closer in code:
                at = code.rindex(closer)
                broken = code[:at] + code[at + 1 :]
                out.append((broken, broken[max(0, at - 24) : at + 6].strip(), f"A closing {name} is missing."))
        out += _sub(code, r"; ", " && ", "&&", "Chains statements with &&, which Nu does not have.")
        out += _sub(code, r"let (\w+) = \((glob|open|ls)", r"\1=$(\2", r"\1=$(\2", "Bash-style assignment and command substitution.")
        out += _sub(code, r"\{\|(\w+)\| ", r"lambda \1: {", r"lambda \1:", "Python lambda syntax in place of a Nu closure.")
        out += _sub(code, r" \| where ", " | grep -v '^$' | awk '{print $1}' | where (", "awk '{print $1}' | where (", "Pipes through grep and awk with an unclosed parenthesis.")
        out += _sub(code, r"if \((.+?)\) \{", r"if [ \1 ]; then {", r"if [ \1 ]; then", "Bash test syntax.")
        return out

    def _equivalent_answers(chain):
        """Correct answers a strict comparison would reject: other formatting, column order, type, or tie order."""
        code, out, tail = chain["code"], [], " | to json --raw"
        if code.endswith(tail):
            body = code[: -len(tail)]
            out.append((body + " | to json", "| to json", "The same answer as indented JSON: formatting only."))
            out.append((body + " | to json --raw | str trim", "| str trim", "The same answer through an extra no-op step: formatting only."))
        for m in re.finditer(r"select (\w+) (\w+)(?: (\w+))?", code):
            columns = [c for c in m.groups() if c]
            swapped = "select " + " ".join(reversed(columns))
            out.append((code[: m.start()] + swapped + code[m.end() :], swapped, "The same columns in another order."))
        out += _sub(code, r"sort-by (\w+) \| sort-by (\w+)( --reverse)?", r"sort-by \2\3", r"sort-by \2\3", "The same ranking with ties left in a different order: the request does not say how to break them.")
        if not code.endswith(tail):
            out.append((code + " | into string", "| into string", "The same value as text instead of a number."))
        return out

    def _loosely_equal(a: str, b: str) -> bool:
        """The same answer once formatting, key and row order, and number-versus-text are set aside."""
        def canon(value):
            if isinstance(value, dict):
                return {k: canon(v) for k, v in sorted(value.items())}
            if isinstance(value, list):
                return sorted((canon(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
            return str(value).lower() if isinstance(value, (bool, int, float)) else value
        try:
            return canon(json.loads(a)) == canon(json.loads(b))
        except ValueError:
            return " ".join(a.split()).lower() == " ".join(b.split()).lower()

    def _from_the_sketch(chain, failure_type):
        """Examples the failure table names that are not edits of a single step."""
        code, layout, out = chain["code"], chain["layout"], []
        columns = layout["columns"]
        if failure_type == "invalid_nushell":
            out.append((f"def run [dir: strng, --limit(-l) int] {{ {code} }}; run .", "def run [dir: strng, --limit(-l) int]", "The function signature is invalid: an unknown type and a flag declared without a colon."))
        if failure_type == "broken_chain_or_function":
            out.append((f"def run [] {{ {code} }}", "def run []", "Defines the work as a function and never calls it."))
            bound = re.match(r"let (\w+) = \((.+?)\); \$\1 \| ", code)
            if bound:  # the bound value is ignored and the second half runs on something else
                first_step = chain["parts"][0]
                out.append((code.replace(f"${bound.group(1)} | ", f"{first_step} | ", 1), first_step, f"Binds `{bound.group(1)}` and then does not use it: the rest runs on the unfiltered input."))
        if failure_type == "wrong_data_source":
            for a, b in ((columns["owner"], columns["key"]), (columns["number"], columns["flag"]), (columns["key"], columns["owner"])):
                out += _sub(code, rf"\b(get|group-by|sort-by) {a}\b", rf"\1 {b}", rf"\1 {b}", f"Reads the `{b}` column where the request meant `{a}`.")
        if failure_type == "wrong_operation_or_argument":
            for a, b in ((columns["number"], columns["key"]), ("size", "name"), ("hits", "file"), ("count", "value")):
                out += _sub(code, rf"sort-by {a}( --reverse)?(?= |$)", rf"sort-by {b}\1", rf"sort-by {b}", f"Sorts by `{b}` where the request said `{a}`.", last=True)
            out += _sub(code, r"math sum", "length", "length", "Counts the values instead of summing them.")
        if failure_type == "runtime_or_environment_failure":
            out.append((f"sleep 30sec; {code}", "sleep 30sec", "Waits longer than the tool allows and times out."))
            for a, b, why in (
                ("to json --raw", "to-json --raw", "`to-json` is how an older release spelled it; this one has no such command."),
                ("| lines", "| split-row (char nl)", "`split-row` is from an older release; this one has no such command."),
                ("str contains", "str find", "`str find` does not exist in this release."),
                ("sort-by", "sort_by", "`sort_by` is not a command in this release."),
                ("ls ", "ls --tree ", "`ls` has no tree view in this release."),
            ):
                if a in code:
                    at = code.index(a)
                    out.append((code[:at] + b + code[at + len(a) :], b.strip(), why))
        return out

    def _hardcoded(chain):
        literal = chain["expected"]
        if len(literal) > 700 or chain["expected_variant"] in (None, chain["expected"]):
            return []
        quoted = f"'{literal}'" if "'" not in literal else f"r#'{literal}'#"
        note = "The answer is pasted in as a literal; nothing in the workspace is read."
        return [(f"{quoted} | from json | to json --raw", quoted, note), (quoted, quoted, note)]

    def _malformed_calls(chain):
        code = chain["code"]
        good = wrap_tool_call(code)
        return [
            (good.replace("run_nushell", "run_bash"), "<function=run_bash>", "Calls a tool that was not offered."),
            (f"You can do that with this command:\n```nu\n{code}\n```", "You can do that with this command:", "Answers in prose; there is no tool call."),
            (good.replace("<parameter=code>", "<parameter=command>"), "<parameter=command>", "The parameter is named `command`; the tool takes `code`."),
            ('{"name": "run_nushell", "arguments": {"code": ' + repr(code)[:400] + "}}\n</tool_call>", '{"name": "run_nushell", "arguments":', "A JSON call where the function-tag format is required."),
            (good.replace("\n</parameter>\n</function>", ""), "</tool_call>", "The parameter and function tags are never closed."),
            (good.replace("</tool_call>", "") + "<tool_call>\n" + good, "<tool_call>\n<function=run_nushell>", "Makes the same call twice; one call was asked for."),
        ]

    _COMMON_FOLDERS = ["src", "lib", "app", "config", "conf", "logs", "log", "docs", "scripts", "bin", "tests", "data", "build"]

    def _assumed_layout(chain):
        """What the target model was seen to do: write a path for a folder or file this project does not have,
        as if every project were laid out the same way. Whether that fails outright or quietly returns the
        wrong thing decides the label."""
        code, dirs, out = chain["code"], chain["layout"]["dirs"], []
        missing = [d for d in _COMMON_FOLDERS if d not in dirs.values()]
        for i, own in enumerate(sorted(set(dirs.values()))):
            other = missing[(len(chain["chain_id"]) + i) % len(missing)]
            out += _sub(code, rf"(?<![\w/]){re.escape(own)}/", f"{other}/", f"{other}/", f"Assumes a `{other}/` folder; this project has `{own}/`.")
        return out

    def _shell_habits(chain):
        """Commands written the way another shell would have them."""
        code, parts, out = chain["code"], chain["parts"], []
        ext = chain["layout"]["ext"]
        if parts[0].startswith(("ls", "glob")):
            out.append((code.replace(parts[0], f"find . -type f -name '*.{ext}'", 1), f"find . -type f -name '*.{ext}'", "Uses `find` with POSIX flags; Nushell's `find` is a different command."))
        if parts[0].startswith("open --raw"):
            path = parts[0].split(" ")[2]
            out.append((code.replace(parts[0], f"cat {path} | grep -v '^$'", 1), f"cat {path} | grep -v '^$'", "Pipes through `cat` and `grep` instead of reading the file with Nushell."))
        return out

    _RULES = {
        "request_misunderstood": _request_misunderstood,
        "wrong_data_source": _wrong_data_source,
        "wrong_operation_or_argument": _wrong_operation,
        "broken_chain_or_function": _broken_chain,
        "wrong_output_or_side_effect": _wrong_output,
        "runtime_or_environment_failure": _runtime_failure,
        "invalid_nushell": _invalid_nushell,
        "grader_false_pass": _hardcoded,
        "grader_false_fail": _equivalent_answers,
    }

    def confirms(failure_type: str, code: str, chain: dict) -> bool:
        """Does running `code` agree with the label?"""
        result = run_nu(code, chain["env"])
        differs = result["ok"] and not same_output(result["stdout"], chain["expected"])
        if failure_type == "invalid_nushell":
            return result["parse_error"]
        if failure_type == "runtime_or_environment_failure":
            return not result["ok"] and not result["parse_error"]
        if failure_type == "grader_false_fail":
            return result["ok"] and result["stdout"].strip() != chain["expected"].strip() and _loosely_equal(result["stdout"], chain["expected"])
        if failure_type == "wrong_output_or_side_effect":
            return result["ok"] and (result["stdout"].strip() != chain["expected"].strip() or bool(result["changed"]))
        if failure_type == "grader_false_pass":
            return result["ok"] and same_output(result["stdout"], chain["expected"])
        return differs

    def confirmed_breaks(chain: dict, index: int) -> dict:
        """For one chain, the first candidate of each failure type that execution confirms."""
        found = {}
        for failure_type in BREAK_LABELS:
            if failure_type == "wrong_tool_or_malformed_call":
                options = _malformed_calls(chain)
                completion, span, note = options[index % len(options)]
                found[failure_type] = {"failure_type": failure_type, "code": chain["code"], "completion": completion, "span_text": span, "note": note}
                continue
            # mutations of the sampled structure first, then edits of the text, then the habits seen in model output
            candidates = structural_breaks(chain, failure_type, index) + _RULES[failure_type](chain) + _from_the_sketch(chain, failure_type)
            if failure_type in ("wrong_data_source", "runtime_or_environment_failure"):
                candidates = _assumed_layout(chain) + candidates
            if failure_type in ("invalid_nushell", "runtime_or_environment_failure"):
                candidates = candidates + _shell_habits(chain)
            shift = index % len(candidates) if candidates else 0
            for code, span, note in (candidates[shift:] + candidates[:shift])[:12]:
                if code != chain["code"] and span in code and confirms(failure_type, code, chain):
                    found[failure_type] = {"failure_type": failure_type, "code": code, "completion": wrap_tool_call(code), "span_text": span, "note": note}
                    break
        return found

    def build_samples(chains: list[dict], breaks_per_chain: int) -> list[dict]:
        """Per chain: the reference call, its verified alternative form, and `breaks_per_chain` confirmed
        breaks. Which types a chain gets is decided after all chains are tried: each takes the types that are
        so far rarest in the dataset, so a type that few chains can produce is not crowded out."""
        with ThreadPoolExecutor(max_workers=8) as pool:
            possible = list(pool.map(lambda pair: confirmed_breaks(pair[1], pair[0]), enumerate(chains)))
        save_nu_cache()
        used = {label: 0 for label in BREAK_LABELS}
        chosen = [[] for _ in chains]
        for i in sorted(range(len(chains)), key=lambda i: len(possible[i])):  # the least flexible chains choose first
            for label in sorted(possible[i], key=lambda label: (used[label], BREAK_LABELS.index(label)))[:breaks_per_chain]:
                chosen[i].append(possible[i][label])
                used[label] += 1
        samples = []
        for chain, breaks in zip(chains, chosen):
            base = {"chain_id": chain["chain_id"], "env": chain["env"], "family": chain["family"], "category": chain["category"]}
            valid = {"failure_type": "valid", "is_broken": False, "span_text": "", "note": FAILURE_TYPES["valid"]}
            samples.append({**base, **valid, "form": "reference", "code": chain["code"], "completion": wrap_tool_call(chain["code"])})
            if chain["alt_ok"]:
                samples.append({**base, **valid, "form": "alternative", "code": chain["alt_code"], "completion": wrap_tool_call(chain["alt_code"])})
            for item in breaks:
                assert item["span_text"] in item["completion"], (chain["chain_id"], item["failure_type"])
                samples.append({**base, **item, "is_broken": item["failure_type"] not in NOT_BROKEN, "form": "break"})
        return samples

    def preview_break(chain, failure_type, index):
        """One break, start to finish: the working command, the first broken variant of this type that
        running it confirms, the part that is wrong, and why."""
        before = chain["code"]
        after, span, reason = confirmed_break(chain, index, failure_type)
        return after is not None

    def confirmed_break(chain, index, failure_type):
        """The call, the broken span and the reason of the first variant of this type that execution confirms."""
        found = confirmed_breaks(chain, index).get(failure_type)
        if not found:
            return None, None, "no variant of this type survives execution for this chain"
        return found["code"], found["span_text"], found["note"]

    BREAK_SCOPE = {"confirmed_break": confirmed_break}

    return (
        BREAK_LABELS,
        BREAK_SCOPE,
        FAILURE_LABELS,
        FAILURE_TYPES,
        VERIFIED_BY,
        build_samples,
        parse_tool_call,
        wrap_tool_call,
    )


@app.cell
def _(BREAK_LABELS, mo):
    break_type_picker = mo.ui.dropdown(options=BREAK_LABELS, value="wrong_operation_or_argument", label="Failure type")
    return (break_type_picker,)


@app.cell
def _(BREAK_SCOPE, LiveEdit, break_type_picker, breaks_per_chain_slider, build_samples, function_source, ival, mo, selected_chain, space, verified_chains):
    samples = build_samples(verified_chains, ival(breaks_per_chain_slider))

    class _Chain(dict):
        """The chain in focus, shown by its id in the live view (its full contents are above)."""

        def __repr__(self):
            return f"<chain {self['chain_id']}>"

    _index = next(i for i, c in enumerate(verified_chains) if c["chain_id"] == selected_chain["chain_id"])
    _chain = _Chain({k: selected_chain[k] for k in ("chain_id", "env", "code", "parts", "expected", "expected_variant", "layout", "files", "spec")})
    # The break generator, running on the chain in focus: the call before and after, the broken span, and why it is kept.
    _live = LiveEdit(
        function_source("preview_break"), args=(_chain, break_type_picker.value, _index),
        function_name="preview_break", globalns=BREAK_SCOPE, visible_columns=["before", "after", "span", "reason"],
    )
    mo.vstack(
        [
            space(40),
            mo.md("## Breaks, before & after"),
            break_type_picker,
            mo.ui.anywidget(_live),
        ],
        gap=1.25,
    )
    return (samples,)


@app.cell
def _(mo):
    # -------------------------------------------------------------------------
    # Step 5: requests and splits
    # -------------------------------------------------------------------------
    request_btn = mo.ui.run_button(label="Generate new data")
    return (request_btn,)


@app.cell
def _(COMMAND_CATALOG, Counter, RUNS_DIR, gc, hashlib, json, random, re):
    REQUESTS_FILE = RUNS_DIR / "generated" / "model_requests.json"
    _CATALOG = {c["name"]: c for c in COMMAND_CATALOG}

    # How people actually ask, from the training sketch. Each of a chain's K requests is written in one of
    # these, starting at a different one for each chain so all six are spread evenly over the dataset.
    REQUEST_STYLES = [
        ("inconsistent formatting", "erratic capitalisation, spacing and punctuation, as if typed in a hurry"),
        ("spelling mistakes or abbreviations", "a couple of misspelled words or shortened forms such as pls, cfg, dir"),
        ("less technical language", "plain words from someone who does not know the technical terms"),
        ("part for the whole", "names only one or two of the things involved while meaning all of them"),
        ("analogy, same domain", "describes what is wanted by likening it to another developer tool or task"),
        ("analogy, other domain", "describes what is wanted by likening it to something outside computing"),
    ]

    def _styles_for(chain: dict, k: int) -> list[tuple]:
        start = sum(map(ord, chain["chain_id"])) % len(REQUEST_STYLES)
        return [(REQUEST_STYLES * 2)[start + i] for i in range(min(k, len(REQUEST_STYLES)))]

    def composed_requests(chain: dict, k: int) -> list[dict]:
        """K requests put together from what each step of the chain says about itself. Used as the grounding
        for the model, and as the fallback for a request the model fails to write acceptably."""
        return [{"text": chain["asks"][i % len(chain["asks"])], "style": style, "by": "composed"} for i, (style, _) in enumerate(_styles_for(chain, k))]

    def _anchors(chain: dict) -> list[str]:
        """The quoted terms, paths and numbers of the plain request: what a rewrite must not lose."""
        found = re.findall(r"'([^']+)'|((?:[\w.-]+/)+[\w.*{},-]+|[\w-]+\.(?:toml|json|log|md|py|js|go|rb|rs|nu))|\b(\d+)\b", chain["asks"][0])
        named = [next(part for part in match if part) for match in found]
        # A folder the command lists or searches, when the plain request names it as a bare word.
        folders = re.findall(r"(?:^|[|;(] ?)(?:ls|glob|cd) ([A-Za-z][\w-]*)(?=[/ ]|$)", chain["code"])
        named += [f for f in folders if re.search(rf"(?<!\w){re.escape(f)}(?!\w)", chain["asks"][0])]
        return list(dict.fromkeys(named))

    def _keeps(text: str, anchors: list[str]) -> bool:
        """Every anchor is still there as a whole word ('doc' is not kept by 'document')."""
        low = text.lower()
        return all(re.search(rf"(?<!\w){re.escape(a.lower())}(?!\w)", low) for a in anchors)

    def write_requests_with_model(chains: list[dict], k: int, model_path: str, limit: int | None = None, seed: int = 73, thinking: int = 80, log=print) -> dict:
        """Have the base language model write each chain's K requests, one call per request and style.

        The model is given the literal description composed from the chain's steps, never the command itself
        (shown the command, it pastes it into the request). It first thinks for up to `thinking` tokens; the
        thinking is then closed and it writes the request. A request is kept only if it is a plausible length,
        contains no code, and still names every term, path and number the command needs; otherwise that one
        request falls back to the composed text. Results are cached by command text. Runs on the MLX thread."""
        import mlx.core as mx
        from mlx_vlm import generate as vlm_generate
        from mlx_vlm import load as vlm_load

        mx.set_cache_limit(512 * 1024**2)
        mx.random.seed(seed)
        stored = load_model_requests(k)
        def _stale(c: dict) -> bool:
            had = stored.get(c["chain_id"], {})
            return had.get("code") != c["code"] or any(r["by"] == "model" and not _keeps(r["text"], _anchors(c)) for r in had["requests"])

        todo = [c for c in chains if _stale(c)][:limit]
        stats = {"chains": len(todo), "written": 0, "fell_back": 0, "rejected_examples": []}
        if not todo:
            return stats
        model, processor = vlm_load(model_path)
        tokenizer = getattr(processor, "tokenizer", processor)

        def generate(prompt: str, max_tokens: int) -> str:
            raw = vlm_generate(model, processor, prompt, max_tokens=max_tokens, temperature=0.7)
            return raw.text if hasattr(raw, "text") else str(raw)

        try:
            for done, chain in enumerate(todo, 1):
                anchors = _anchors(chain)
                fallback, requests, seen = composed_requests(chain, k), [], set()
                for i, (style, what) in enumerate(_styles_for(chain, k)):
                    task = (
                        "Rewrite this request, which someone is making to a terminal assistant, in the style described.\n\n"
                        f"Request: {chain['asks'][0]}\n"
                        f"Style: {style} ({what}).\n"
                        + (f"These must appear in your rewrite exactly as written: {', '.join(anchors)}.\n" if anchors else "")
                        + "Keep the meaning the same. Do not name any command and do not write code. Reply with the rewritten request only, on one line."
                    )
                    messages = [{"role": "user", "content": task}]
                    if thinking:
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
                        thought = generate(prompt, thinking).split("</think>")[0]
                        prompt = prompt + thought.rstrip() + "\n</think>\n"  # close the thinking and ask for the answer
                    else:
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    candidate = generate(prompt, 160).split("</think>")[-1].strip().split("\n")[0].strip().strip('"')
                    low = candidate.lower()
                    ok = (
                        12 <= len(candidate) <= 420 and low not in seen and "|" not in candidate and "```" not in candidate
                        and _keeps(candidate, anchors)
                        # the model sometimes repeats its instructions instead of following them
                        and not re.search(r"rewrite|one line|write code|(?:name|mention) any command|exactly as written", low)
                    )
                    if ok:
                        requests.append({"text": candidate, "style": style, "by": "model"})
                        seen.add(low)
                        stats["written"] += 1
                    else:
                        requests.append(fallback[i])
                        stats["fell_back"] += 1
                        if candidate and len(stats["rejected_examples"]) < 8:
                            stats["rejected_examples"].append(candidate)
                stored[chain["chain_id"]] = {"code": chain["code"], "requests": requests}
                if done % 10 == 0 or done == len(todo):
                    log(f"[requests] {done}/{len(todo)} chains · {stats['written']} written by the model, {stats['fell_back']} fell back")
                    REQUESTS_FILE.parent.mkdir(parents=True, exist_ok=True)
                    REQUESTS_FILE.write_text(json.dumps({"k": k, "seed": seed, "chains": stored, "stats": stats}, indent=1), encoding="utf-8")
        finally:
            del model, processor
            gc.collect()
            mx.clear_cache()
        return stats

    def load_model_requests(k: int) -> dict:
        if not REQUESTS_FILE.exists():
            return {}
        stored = json.loads(REQUESTS_FILE.read_text(encoding="utf-8"))
        return stored["chains"] if stored.get("k") == k else {}

    def assemble_dataset(environments: list[dict], chains: list[dict], samples: list[dict], k: int, model_requests: dict, seed: int = 73) -> dict:
        """Grader rows (request × call) and target rows (request → reference call). One environment in five is
        held out whole: nothing from it, in any form or break, is seen in training, so dev scores are for
        projects the models have never read."""
        rng = random.Random(seed)
        env_ids = sorted({chain["env"] for chain in chains})
        rng.shuffle(env_ids)
        dev_envs = set(env_ids[: max(1, round(len(env_ids) / 5))]) if len(env_ids) > 1 else set()
        dev_chains = {chain["chain_id"] for chain in chains if chain["env"] in dev_envs}
        files_of = {e["id"]: sorted(e["files"]) for e in environments}
        requests = {}
        for chain in chains:
            stored = model_requests.get(chain["chain_id"])
            ready = stored and stored["code"] == chain["code"] and len(stored["requests"]) >= k
            requests[chain["chain_id"]] = stored["requests"][:k] if ready else composed_requests(chain, k)
        written = Counter(r["by"] for rs in requests.values() for r in rs)
        by_id = {c["chain_id"]: c for c in chains}
        grader, target = {"train": [], "dev": []}, {"train": [], "dev": []}
        for sample in samples:
            chain = by_id[sample["chain_id"]]
            split = "dev" if sample["chain_id"] in dev_chains else "train"
            for i, asked in enumerate(requests[sample["chain_id"]]):
                request = asked["text"]
                row_id = f"{sample['chain_id']}/{sample['form']}/{sample['failure_type']}/{i}"
                grader[split].append({"id": row_id, "prompt": request, "style": asked["style"], **sample})
                if sample["form"] == "reference":
                    target[split].append(
                        {
                            "id": row_id,
                            "chain_id": sample["chain_id"],
                            "env": sample["env"],
                            "category": sample["category"],
                            "speech": request,
                            "style": asked["style"],
                            "context": "A project checkout. Paths are relative to the workspace root.",
                            "workspace_files": files_of[sample["env"]],
                            # every other request comes without syntax references, as a bare question would
                            "references": [
                                {"title": name, "text": f"{_CATALOG[name]['usage']}\n\n> {_CATALOG[name]['syntax']}"}
                                for name in chain["commands"][:5]
                                if name in _CATALOG and i % 2 == 0
                            ],
                            "completion": sample["completion"],
                            "expected": chain["expected"],
                        }
                    )
        for rows in (*grader.values(), *target.values()):
            rng.shuffle(rows)
        digest = hashlib.sha1(json.dumps([[r["id"], r["prompt"], r["completion"]] for r in grader["train"] + grader["dev"]]).encode()).hexdigest()[:12]
        return {
            "grader": grader,
            "target": target,
            "dev_chains": sorted(dev_chains),
            "dev_envs": sorted(dev_envs),
            "fingerprint": digest,
            "requests_by": dict(written),
            "environment_seeds": {e["id"]: e["layout"]["seed"] for e in environments},
        }

    def save_dataset(dataset: dict, settings: dict | None = None) -> None:
        """Write the splits and a manifest that is enough to regenerate them: the environment seeds, the
        generator settings, and where the usage weights came from."""
        out = RUNS_DIR / "generated"
        out.mkdir(parents=True, exist_ok=True)
        manifest = {
            "fingerprint": dataset["fingerprint"],
            "environment_seeds": dataset["environment_seeds"],
            "held_out_environments": dataset["dev_envs"],
            "requests_by": dataset["requests_by"],
            "rows": {f"{kind}_{split}": len(dataset[kind][split]) for kind in ("grader", "target") for split in ("train", "dev")},
            "settings": settings or {},
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        for name, rows in (
            ("grader_train", dataset["grader"]["train"]),
            ("grader_dev", dataset["grader"]["dev"]),
            ("target_train", dataset["target"]["train"]),
            ("target_dev", dataset["target"]["dev"]),
        ):
            (out / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    return REQUEST_STYLES, assemble_dataset, load_model_requests, save_dataset, write_requests_with_model


@app.cell
async def _(
    Counter,
    MODEL_FOLDERS,
    PRIOR_INFO,
    assemble_dataset,
    breaks_per_chain_slider,
    chain_summary,
    chains_total_slider,
    environments,
    fval,
    ival,
    k_variants_slider,
    length_bias_slider,
    load_model_requests,
    mlx_thread,
    mo,
    model_path,
    ported_share_slider,
    request_btn,
    samples,
    save_dataset,
    selected_categories,
    space,
    verified_chains,
    write_requests_with_model,
):
    # Requests are written by the base model, K per chain, one per style, and cached; a request that loses a term
    # the command needs falls back to the description composed from the chain's steps. The split is by environment:
    # one in five is held out whole. The splits and a manifest with the environment seeds are saved on every build.
    _error = None
    if request_btn.value:
        with mo.status.spinner(title="Writing requests with the model…"):
            try:
                await mlx_thread(write_requests_with_model, verified_chains, ival(k_variants_slider), model_path(MODEL_FOLDERS["target"]))
            except Exception as _exc:
                _error = f"{type(_exc).__name__}: {_exc}"
    dataset = assemble_dataset(environments, verified_chains, samples, ival(k_variants_slider), load_model_requests(ival(k_variants_slider)))
    save_dataset(
        dataset,
        {
            "chains": ival(chains_total_slider), "ported_share": fval(ported_share_slider), "length_bias": fval(length_bias_slider),
            "breaks_per_chain": ival(breaks_per_chain_slider), "requests_per_chain": ival(k_variants_slider),
            "categories": selected_categories, "priors": PRIOR_INFO, "chain_summary": chain_summary,
        },
    )
    _g = dataset["grader"]
    _kind = Counter("valid" if r["failure_type"] == "valid" else "equivalent" if r["failure_type"] == "grader_false_fail" else "broken" for r in _g["train"] + _g["dev"])
    mo.vstack(
        ([mo.callout(mo.md(f"Request writing failed: `{_error}`"), kind="danger")] if _error else [])
        + [
            space(40),
            mo.md(f"## Final data size = {_kind['valid']:,} + {_kind['broken']:,} + {_kind['equivalent']:,} = {sum(_kind.values()):,}"),
        ]
    )
    return (dataset,)


@app.cell
def _(FAILURE_LABELS, FAILURE_TYPES, RUNS_DIR, gc, json, math, np, random, shutil, time):
    # -------------------------------------------------------------------------
    # Step 6: grader training. Plain functions, so the notebook and scripts/run_smoke_test.py share them.
    # All of them must run on one thread (MLX arrays do not cross threads).
    # -------------------------------------------------------------------------
    GRADER_DIR = RUNS_DIR / "grader"
    PLACEMENTS = {
        "all_linear": ("self_attn.", "feed_forward.", "conv."),
        "attention_only": ("self_attn.",),
        "mlp_only": ("feed_forward.",),
        "conv_only": ("conv.",),
    }

    def attach_lora(model, rank: int, alpha: float, dropout: float, placement: str):
        """Wrap the chosen linear layers of a freshly loaded model in LoRA. Returns the language model, the
        LoRA layers, and the trainable parameter count."""
        import mlx.nn as nn
        from mlx.utils import tree_flatten, tree_unflatten
        from mlx_lm.tuner import trainer
        from mlx_lm.tuner.lora import LoRALinear

        model.freeze()
        lm = model.language_model
        layers = []
        for block in lm.layers:
            replacements = []
            for name, layer in block.named_modules():
                if isinstance(layer, (nn.Linear, nn.QuantizedLinear)) and name.startswith(PLACEMENTS[placement]):
                    wrapped = LoRALinear.from_base(layer, r=rank, scale=alpha / rank, dropout=dropout)
                    replacements.append((name, wrapped))
                    layers.append(wrapped)
            if replacements:
                block.update_modules(tree_unflatten(replacements))
        block_type = type(lm.layers[0])
        if not getattr(block_type, "_checkpointed", False):  # grad_checkpoint patches the class; do it once
            trainer.grad_checkpoint(lm.layers[0])
            block_type._checkpointed = True
        return lm, layers, sum(v.size for _, v in tree_flatten(lm.trainable_parameters()))

    def set_lora(layers, scale: float | None = None, training: bool | None = None):
        """Scale 0 gives exactly the base model. LoRA dropout is on only while `training` is true."""
        for layer in layers:
            if scale is not None:
                layer.scale = scale
            if training is not None:
                layer.train(training)

    def learning_rate(step: int, total: int, base: float) -> float:
        """Linear warm-up over the first tenth of training, then cosine decay to a tenth of the base rate."""
        warm = max(1, round(0.1 * total))
        return base * min(1.0, step / warm) * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - 1) / max(1, total))))

    def balanced(rows: list[dict], n: int, key: str = "failure_type") -> list[dict]:
        """Up to `n` rows taken round-robin across the values of `key`."""
        buckets = {}
        for row in rows:
            buckets.setdefault(row[key], []).append(row)
        out = []
        while len(out) < n and any(buckets.values()):
            for label in sorted(buckets):
                if buckets[label] and len(out) < n:
                    out.append(buckets[label].pop(0))
        return out

    _QUESTIONS = {
        "is_broken": {"type": "noul", "instructions": "Is the assistant's tool call wrong for the user request?"},
        "failure_type": {"type": "choice", "instructions": "Which description fits the tool call?", "criteria": FAILURE_TYPES},
        "fault_span": {"type": "text", "instructions": "Point at the part of the tool call that is wrong, if any."},
    }

    def _encode(engine, row: dict) -> dict:
        """One request and tool call packed with the three questions the way the D1 engine packs them, plus
        the training targets when the row is labelled."""
        from mlx_vlm.models.d1_lfm3_vl.d1 import pack_set
        from mlx_vlm.models.d1_lfm3_vl.pointer import state_positions
        from mlx_vlm.models.d1_lfm3_vl.prompt import as_question, prefix_text, question_header, question_slots, readout_table, suffix_layout

        tokenizer = engine.tokenizer
        head = f"User request: {row['prompt']}\n\nAssistant tool call:\n"
        state = head + row["completion"]
        questions = [as_question(_QUESTIONS[name]) for name in ("is_broken", "failure_type", "fault_span")]
        prefix = prefix_text(tokenizer, state, engine.bos, engine.state_style, engine.system, "")
        layouts = [suffix_layout(tokenizer, q, engine.lead) for q in questions]
        suffixes = [text for text, _ in layouts]
        full_text = prefix + suffixes[0]
        encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        ids, state_len, question, ends = pack_set(
            tokenizer, prefix, suffixes, question_header(tokenizer, state), [offsets for _, offsets in layouts], encoded["input_ids"]
        )
        positions = state_positions(tokenizer, state, prefix, full_text, engine.state_style, encoded=encoded)
        span = (0, 0)  # position 0 is the engine's "nothing to point at"
        if row.get("is_broken") and row.get("span_text"):
            lo = len(head) + row["completion"].find(row["span_text"])
            hi = lo + len(row["span_text"])
            hits = [i for i, off in enumerate(positions.offsets) if off is not None and off[1] > lo and off[0] < hi]
            if hits:
                span = (hits[0], hits[-1])
        return {
            "row": row,
            "ids": ids,
            "state_len": state_len,
            "question": question,
            "ends": ends,
            "positions": positions,
            "mask": np.asarray(positions.pointer_mask, dtype=bool),
            "yes_no": readout_table(tokenizer, question_slots(questions[0], tokenizer)[0]),
            "choice": readout_table(tokenizer, question_slots(questions[1], tokenizer)[0]),
            "y_broken": 0 if row.get("is_broken") else 1,  # the yes/no readout lists yes first
            "y_type": FAILURE_LABELS.index(row.get("failure_type", "valid")),
            "y_span": span,
        }

    def _readouts(model, item: dict):
        """Yes/no logits, failure-type logits, and start/end pointer logits over the state, read the same way
        the engine reads them: option tokens through the tied embedding, pointers by scaled dot product."""
        import mlx.core as mx

        hidden = model.d1_hidden(item["ids"], item["state_len"], item["question"])
        embed = model.language_model.model.embed_tokens

        def options(table, slot):
            tokens, forms, _ = table
            raw = (hidden[slot : slot + 1] @ embed(mx.array(tokens)).T).astype(mx.float32)[0]
            return mx.stack([mx.max(raw[mx.array(form)]) for form in forms])

        queries = hidden[mx.array(item["ends"][2])].astype(mx.float32)
        keys = hidden[: len(item["mask"])].astype(mx.float32)
        pointer = (queries @ keys.T) / math.sqrt(queries.shape[-1])
        pointer = mx.where(mx.array(item["mask"])[None, :], pointer, mx.array(-1e9, dtype=mx.float32))
        return options(item["yes_no"], item["ends"][0][0]), options(item["choice"], item["ends"][1][0]), pointer

    def _grader_loss(model, item: dict):
        import mlx.core as mx
        import mlx.nn as nn

        yes_no, choice, pointer = _readouts(model, item)
        ce = nn.losses.cross_entropy
        span = 0.5 * (ce(pointer[0:1], mx.array([item["y_span"][0]])) + ce(pointer[1:2], mx.array([item["y_span"][1]])))
        return (ce(yes_no[None], mx.array([item["y_broken"]])) + ce(choice[None], mx.array([item["y_type"]])) + 0.5 * span).sum()

    def _predict(model, layers, scale: float, item: dict, weights: list[float]) -> dict:
        """Predictions for one item from the adapter, from the base model, and from each guidance weight w:
        log p_w = log p_base + w · (log p_adapter − log p_base). w = 0 is the base model, w = 1 the adapter."""
        import mlx.core as mx
        import mlx.nn as nn

        set_lora(layers, scale=scale)
        yes_no, choice, pointer = _readouts(model, item)
        set_lora(layers, scale=0.0)
        base_yes_no, base_choice, _ = _readouts(model, item)
        set_lora(layers, scale=scale)
        adapter, base = nn.log_softmax(choice), nn.log_softmax(base_choice)
        mx.eval(adapter, base, yes_no, base_yes_no, pointer)
        start, end = int(mx.argmax(pointer[0]).item()), int(mx.argmax(pointer[1]).item())
        text = None
        if 0 < start <= end:
            try:
                text = item["positions"].read(start, end)["text"]
            except ValueError:
                text = None
        broken_only = mx.where(mx.arange(len(FAILURE_LABELS)) == 0, mx.array(-1e9), adapter)  # label 0 is "valid"
        return {
            "logp": [round(float(v), 4) for v in adapter.tolist()],
            "logp_base": [round(float(v), 4) for v in base.tolist()],
            "types": {str(w): FAILURE_LABELS[int(mx.argmax(base + w * (adapter - base)).item())] for w in weights},
            "type_if_broken": FAILURE_LABELS[int(mx.argmax(broken_only).item())],
            "p_broken": float(mx.softmax(yes_no)[0].item()),
            "p_broken_base": float(mx.softmax(base_yes_no)[0].item()),
            "type_loss": float(-adapter[item["y_type"]].item()),
            "span": (start, end),
            "span_text": text,
        }

    def _score(items: list[dict], predictions: list[dict], weights: list[float]) -> dict:
        rows = [item["row"] for item in items]
        share = lambda flags: sum(flags) / len(flags) if flags else None
        spans = [
            p["span"][0] > 0 and p["span"][0] <= item["y_span"][1] and p["span"][1] >= item["y_span"][0]
            for item, p in zip(items, predictions)
            if item["y_span"] != (0, 0)
        ]
        return {
            "type_accuracy": {str(w): share([p["types"][str(w)] == r["failure_type"] for r, p in zip(rows, predictions)]) for w in weights},
            "broken_accuracy": share([(p["p_broken"] > 0.5) == bool(r["is_broken"]) for r, p in zip(rows, predictions)]),
            "broken_accuracy_base": share([(p["p_broken_base"] > 0.5) == bool(r["is_broken"]) for r, p in zip(rows, predictions)]),
            "false_fail_reference": share([p["p_broken"] > 0.5 for r, p in zip(rows, predictions) if r.get("form") == "reference"]),
            "false_fail_alternative": share([p["p_broken"] > 0.5 for r, p in zip(rows, predictions) if r.get("form") == "alternative"]),
            "span_hit": share(spans),
            "type_loss": share([p["type_loss"] for p in predictions]),
        }

    def grader_subset(rows: list[dict], n: int) -> list[dict]:
        """Half correct calls (reference and alternative forms), half broken ones spread evenly over the
        failure types, so that "is it broken?" cannot be answered by always saying yes."""
        valid = balanced([r for r in rows if not r["is_broken"]], n // 2, key="form")
        broken = balanced([r for r in rows if r["is_broken"]], n - len(valid))
        return valid + broken

    GUIDANCE_GRID = [0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 3.0]

    def guidance_table(records: list[dict]) -> dict:
        """Failure-type accuracy and mean log-probability of the true type on the dev rows, for two ways of
        guiding the trained adapter and a grid of weights w:
          from the base model:   log p = log p_base + w · (log p_adapter − log p_base)   (w = 1 is the adapter)
          autoguidance:          log p = log p_adapter + w · (log p_adapter − log p_weak) (w = 0 is the adapter)
        where the weak model is the same adapter a quarter of the way through its training."""
        def score(combine):
            hits, logps = [], []
            for r in records:
                mixed = np.array(combine(r), dtype=np.float64)
                mixed = mixed - (np.log(np.exp(mixed - mixed.max()).sum()) + mixed.max())
                true = FAILURE_LABELS.index(r["true"])
                hits.append(int(mixed.argmax()) == true)
                logps.append(float(mixed[true]))
            return {"accuracy": sum(hits) / len(hits), "logp": sum(logps) / len(logps)}

        table = {"from_base": {}, "autoguidance": {}}
        for w in GUIDANCE_GRID:
            table["from_base"][str(w)] = score(lambda r: np.array(r["logp_base"]) + w * (np.array(r["logp"]) - np.array(r["logp_base"])))
            if records and "logp_weak" in records[0]:
                table["autoguidance"][str(w)] = score(lambda r: np.array(r["logp"]) + w * (np.array(r["logp"]) - np.array(r["logp_weak"])))
        return table

    def train_grader(cfg: dict, train_rows: list[dict], dev_rows: list[dict], adapter_path, log=print) -> dict:
        """Train one LoRA configuration on a freshly loaded grader and score it on the dev rows."""
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten, tree_map, tree_unflatten
        from mlx_vlm import load as vlm_load

        mx.set_cache_limit(512 * 1024**2)
        mx.random.seed(cfg["seed"])
        started = time.monotonic()
        model, _processor = vlm_load(cfg["model_path"])
        try:
            train_items = [_encode(model.engine, r) for r in train_rows]
            dev_items = [_encode(model.engine, r) for r in dev_rows]
            lm, layers, n_params = attach_lora(model, cfg["rank"], cfg["alpha"], cfg["dropout"], cfg["placement"])
            scale = cfg["alpha"] / cfg["rank"]
            optimizer = optim.AdamW(learning_rate=cfg["lr"], weight_decay=0.0)
            value_and_grad = nn.value_and_grad(lm, lambda _lm, item: _grader_loss(model, item))
            total = math.ceil(len(train_items) / cfg["batch"]) * cfg["epochs"]
            order = random.Random(cfg["seed"])
            history, step = [], 0
            for epoch in range(cfg["epochs"]):
                set_lora(layers, training=True)
                order.shuffle(train_items)
                for offset in range(0, len(train_items), cfg["batch"]):
                    batch = train_items[offset : offset + cfg["batch"]]
                    grads, losses = None, []
                    for item in batch:
                        loss, grad = value_and_grad(lm, item)
                        grads = grad if grads is None else tree_map(lambda a, b: a + b, grads, grad)
                        mx.eval(grads)
                        losses.append(float(loss.item()))
                    step += 1
                    optimizer.learning_rate = learning_rate(step, total, cfg["lr"])
                    clipped, _ = optim.clip_grad_norm(tree_map(lambda g: g / len(batch), grads), cfg["clip"])
                    optimizer.update(lm, clipped)
                    mx.eval(lm.trainable_parameters(), optimizer.state)
                    history.append({"config": cfg["name"], "step": step, "loss": sum(losses) / len(losses)})
                    if step == 1 or step % 10 == 0 or step == total:
                        log(f"[grader {cfg['name']}] step {step}/{total} loss {history[-1]['loss']:.3f}")
                    if step == max(1, total // 4):  # the weak version of this adapter, for autoguidance
                        weak = {k: mx.array(v) for k, v in tree_flatten(lm.trainable_parameters())}
                        mx.eval(weak)
            set_lora(layers, training=False)
            predictions = [_predict(model, layers, scale, item, cfg["guidance_weights"]) for item in dev_items]
            trained = dict(tree_flatten(lm.trainable_parameters()))
            mx.save_safetensors(str(adapter_path), trained)
            lm.update(tree_unflatten(list(weak.items())))
            for item, prediction in zip(dev_items, predictions):
                prediction["logp_weak"] = _predict(model, layers, scale, item, [1.0])["logp"]
            lm.update(tree_unflatten(list(trained.items())))
        finally:
            del model, _processor
            gc.collect()
            mx.clear_cache()
        metrics = _score(dev_items, predictions, cfg["guidance_weights"])
        log(f"[grader {cfg['name']}] dev type accuracy {metrics['type_accuracy']} · broken? {metrics['broken_accuracy']:.0%}")
        return {
            "config": cfg,
            "trainable_parameters": n_params,
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "history": history,
            "metrics": metrics,
            "records": (records := [
                {
                    "id": item["row"]["id"], "true": item["row"]["failure_type"], "form": item["row"].get("form"), **p["types"],
                    "p_broken": p["p_broken"], "logp": p["logp"], "logp_base": p["logp_base"], "logp_weak": p["logp_weak"],
                }
                for item, p in zip(dev_items, predictions)
            ]),
            "guidance": guidance_table(records),
            "seconds": round(time.monotonic() - started, 1),
            "peak_memory_gb": round(mx.get_peak_memory() / 1024**3, 2),
        }

    def run_grader_sweep(configs: list[dict], dataset: dict, n_train: int, n_dev: int, log=print, keep: bool = True) -> dict:
        """Train each configuration from a fresh model, keep the adapter with the lowest dev failure-type
        loss, and write `runs/grader/{adapter.safetensors, adapter_config.json, report.json}`. With
        `keep=False` nothing on disk is replaced: the runs are only measured."""
        GRADER_DIR.mkdir(parents=True, exist_ok=True)
        train_rows = grader_subset(dataset["grader"]["train"], n_train)
        dev_rows = grader_subset(dataset["grader"]["dev"], n_dev)
        runs = []
        for cfg in configs:
            runs.append(train_grader(cfg, train_rows, dev_rows, GRADER_DIR / f"candidate-{cfg['name']}.safetensors", log))
        best = min(runs, key=lambda r: r["metrics"]["type_loss"])
        if keep:
            shutil.copyfile(GRADER_DIR / f"candidate-{best['config']['name']}.safetensors", GRADER_DIR / "adapter.safetensors")
            (GRADER_DIR / "adapter_config.json").write_text(json.dumps(best["config"], indent=2), encoding="utf-8")
        for cfg in configs:
            (GRADER_DIR / f"candidate-{cfg['name']}.safetensors").unlink()
        report = {"created": time.time(), "dataset": dataset["fingerprint"], "best": best["config"]["name"], "runs": runs}
        if keep:
            (GRADER_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def run_grader_experiments(model_path: str, dataset: dict, sizes: dict, log=print) -> dict:
        """The grader's full protocol, each stage choosing by dev failure-type loss on held-out environments:
        1. a sweep over learning rate and rank with the adapter on every linear layer;
        2. an ablation of where the adapter goes (attention, MLP or convolution layers only) at the best setting;
        3. the final adapter: the best setting and placement, trained on the full budget.
        Guidance is scored on the final adapter (see `guidance_table`). Everything is written to
        `runs/grader/experiments.json`."""
        base = {"model_path": model_path, "placement": "all_linear", "dropout": 0.05, "clip": 1.0, "batch": 4, "epochs": 1, "seed": 73, "guidance_weights": [0.0, 1.0, 1.5]}
        slim = lambda report: [{"config": r["config"], "metrics": r["metrics"], "seconds": r["seconds"], "trainable_parameters": r["trainable_parameters"], "train_rows": r["train_rows"]} for r in report["runs"]]
        sweep = run_grader_sweep(
            [{**base, "name": f"all_linear-r{rank}-lr{lr:g}", "rank": rank, "alpha": 2.0 * rank, "lr": lr} for rank in sizes["ranks"] for lr in sizes["learning_rates"]],
            dataset, sizes["n_sweep"], sizes["n_dev"], log, keep=False,
        )
        best = next(r["config"] for r in sweep["runs"] if r["config"]["name"] == sweep["best"])
        ablation = run_grader_sweep(
            [{**best, "name": f"{placement}-r{best['rank']}-lr{best['lr']:g}", "placement": placement} for placement in ("attention_only", "mlp_only", "conv_only")],
            dataset, sizes["n_sweep"], sizes["n_dev"], log, keep=False,
        )
        candidates = [r for r in sweep["runs"] if r["config"]["name"] == sweep["best"]] + ablation["runs"]
        chosen = min(candidates, key=lambda r: r["metrics"]["type_loss"])["config"]
        final = run_grader_sweep([{**chosen, "name": "final-" + chosen["name"], "epochs": sizes["epochs"]}], dataset, sizes["n_full"], sizes.get("n_dev_final", sizes["n_dev"]), log)
        experiments = {
            "created": time.time(), "dataset": dataset["fingerprint"], "sizes": sizes,
            "sweep": slim(sweep), "sweep_best": sweep["best"], "placement_ablation": slim(ablation), "chosen": chosen["name"],
            "final": slim(final)[0], "guidance": final["runs"][0]["guidance"],
        }
        (GRADER_DIR / "experiments.json").write_text(json.dumps(experiments, indent=1), encoding="utf-8")
        return experiments

    def grade_calls(rows: list[dict], model_path: str | None = None) -> list[dict]:
        """Grade `{prompt, completion}` rows with the saved grader adapter: P(broken), failure type, the most
        likely failure type given that the call is known to be broken, and the span."""
        import mlx.core as mx
        from mlx_vlm import load as vlm_load

        cfg = json.loads((GRADER_DIR / "adapter_config.json").read_text(encoding="utf-8"))
        mx.set_cache_limit(512 * 1024**2)
        model, _processor = vlm_load(model_path or cfg["model_path"])
        try:
            lm, layers, _ = attach_lora(model, cfg["rank"], cfg["alpha"], 0.0, cfg["placement"])
            lm.load_weights(str(GRADER_DIR / "adapter.safetensors"), strict=False)
            set_lora(layers, training=False)
            out = []
            for row in rows:
                p = _predict(model, layers, cfg["alpha"] / cfg["rank"], _encode(model.engine, row), [1.0])
                out.append({"p_broken": p["p_broken"], "failure_type": p["types"]["1.0"], "type_if_broken": p["type_if_broken"], "span_text": p["span_text"]})
        finally:
            del model, _processor
            gc.collect()
            mx.clear_cache()
        return out

    encode_grader_row, grader_readouts = _encode, _readouts  # also used by the activation view
    return GRADER_DIR, GUIDANCE_GRID, PLACEMENTS, attach_lora, balanced, encode_grader_row, grade_calls, grader_readouts, guidance_table, learning_rate, run_grader_experiments, run_grader_sweep, set_lora


@app.cell
def _(MODEL_FOLDERS, PLACEMENTS, TangleLatex, fader, knob, mo):
    # Not shown: a folder under $MODELS_DIR, or a path. Change it here to train another checkpoint.
    grader_model_path = mo.ui.text(value=MODEL_FOLDERS["grader"])
    # smoke: 20 rows, 1 epoch (proves the code path) · standard: 400 rows, 2 epochs, the settings shown ·
    # protocol: sweep over rank and learning rate, placement ablation, final adapter
    grader_preset = mo.ui.dropdown(options=["smoke", "standard", "protocol"], value="smoke", label="Budget")
    grader_placement = mo.ui.dropdown(options=list(PLACEMENTS), value="mlp_only", label="Layers")
    grader_lr = fader("Learning rate ×1e-4", 0.5, 8.0, 0.5, 1.0)
    grader_dropout = fader("LoRA dropout", 0.0, 0.2, 0.05, 0.05)
    grader_clip = fader("Gradient clip", 0.5, 5.0, 0.5, 1.0)
    grader_rank = knob("Rank", 4, 32, 4, 8)
    grader_alpha_mult = knob("Alpha ÷ rank", 1.0, 4.0, 0.5, 2.0)
    grader_batch = knob("Batch", 1, 8, 1, 4)
    grader_train_btn = mo.ui.run_button(label="▶")
    grader_tangle = mo.ui.anywidget(
        TangleLatex(
            latex=(
                r"\begin{aligned} \log p &= \log p_{\text{base}} + \tangle{w}\,\bigl(\log p_{\text{adapter}} - \log p_{\text{base}}\bigr) \\[6pt]"
                r" \log p &= \log p_{\text{adapter}} + \tangle{v}\,\bigl(\log p_{\text{adapter}} - \log p_{\text{weak}}\bigr) \end{aligned}"
            ),
            parameters={
                "w": {"value": 1.0, "min_value": 0.0, "max_value": 3.0, "step": 0.05, "digits": 2},
                "v": {"value": 0.0, "min_value": 0.0, "max_value": 3.0, "step": 0.05, "digits": 2},
            },
        )
    )
    return (
        grader_alpha_mult,
        grader_batch,
        grader_clip,
        grader_dropout,
        grader_lr,
        grader_model_path,
        grader_placement,
        grader_preset,
        grader_rank,
        grader_tangle,
        grader_train_btn,
    )


@app.cell
async def _(
    GRADER_DIR,
    dataset,
    fval,
    grader_alpha_mult,
    grader_batch,
    grader_clip,
    grader_dropout,
    grader_lr,
    grader_model_path,
    grader_placement,
    grader_preset,
    grader_rank,
    grader_train_btn,
    ival,
    json,
    mlx_thread,
    mo,
    model_path,
    run_grader_experiments,
    run_grader_sweep,
    shutil,
    time,
):
    grader_error, _log = None, []
    if grader_train_btn.value:
        _base = {
            "name": f"{grader_placement.value}-r{ival(grader_rank)}",
            "model_path": model_path(grader_model_path.value),
            "placement": grader_placement.value,
            "rank": ival(grader_rank),
            "alpha": ival(grader_rank) * fval(grader_alpha_mult),
            "lr": fval(grader_lr) * 1e-4,
            "dropout": fval(grader_dropout),
            "clip": fval(grader_clip),
            "batch": ival(grader_batch),
            "epochs": 1,
            "seed": 73,
            "guidance_weights": [0.0, 1.0, 1.5],
        }
        _smoke = None
        with mo.status.spinner(title="Training the grader…"):
            try:
                if grader_preset.value != "smoke" and GRADER_DIR.exists():  # a real run replaces the saved adapter: keep the old one
                    shutil.copytree(GRADER_DIR, GRADER_DIR.parent / "previous" / f"grader-{int(time.time())}")
                if grader_preset.value == "protocol":
                    _sizes = {"ranks": [8, 16], "learning_rates": [1e-4, 2e-4, 4e-4], "n_sweep": 160, "n_dev": 100, "n_full": 2400, "n_dev_final": 400, "epochs": 2}
                    await mlx_thread(run_grader_experiments, model_path(grader_model_path.value), dataset, _sizes, _log.append)
                elif grader_preset.value == "standard":
                    await mlx_thread(run_grader_sweep, [{**_base, "epochs": 2}], dataset, 400, 120, _log.append)
                else:
                    # the smoke budget only proves the code path; it is shown but never saved over a real adapter
                    _smoke = await mlx_thread(run_grader_sweep, [_base], dataset, 20, 20, _log.append, False)
            except Exception as _exc:
                grader_error = f"{type(_exc).__name__}: {_exc}"
    _file = GRADER_DIR / "report.json"
    grader_report = json.loads(_file.read_text(encoding="utf-8")) if _file.exists() else None
    if grader_train_btn.value and not grader_error and grader_preset.value == "smoke" and _smoke:
        grader_report = _smoke
    _file = GRADER_DIR / "experiments.json"
    grader_experiments = json.loads(_file.read_text(encoding="utf-8")) if _file.exists() else None
    return grader_error, grader_experiments, grader_report


@app.cell
def _(
    FAILURE_LABELS,
    alt,
    dataset,
    grader_alpha_mult,
    grader_batch,
    grader_clip,
    grader_dropout,
    grader_error,
    grader_experiments,
    grader_lr,
    grader_placement,
    grader_preset,
    grader_rank,
    grader_report,
    grader_tangle,
    grader_train_btn,
    math,
    mo,
    np,
    pd,
    row,
    space,
):
    _pct = lambda v: "–" if v is None else f"{v:.0%}"
    _notes, _charts, _guidance = [], [], []
    if grader_error:
        _notes.append(mo.callout(mo.md(f"Training failed: `{grader_error}`"), kind="danger"))
    if grader_report:
        _best = next(r for r in grader_report["runs"] if r["config"]["name"] == grader_report["best"])
        _m = _best["metrics"]
        _broken = [r for r in _best.get("records", []) if r["true"] not in ("valid", "grader_false_fail")]
        if _broken:  # failure-type accuracy on the calls that really are broken, the harder half
            _m = {**_m, "type_accuracy_broken": sum(r["true"] == r["1.0"] for r in _broken) / len(_broken)}
        if grader_report["dataset"] != dataset["fingerprint"]:
            _notes.append(mo.callout(mo.md("Trained on a different dataset than the one built above."), kind="warn"))
        _loss = mo.ui.altair_chart(
            alt.Chart(pd.DataFrame([h for r in grader_report["runs"] for h in r["history"]]))
            .mark_line()
            .encode(x=alt.X("step:Q", title="Update"), y=alt.Y("loss:Q", title=None), color=alt.Color("config:N", title=None, legend=alt.Legend(orient="bottom")))
            .properties(title="Loss", height=200, width="container")
        )
        if grader_experiments:
            # the sweep and the placement ablation of the full protocol sit behind the loss chart
            _e = grader_experiments
            _row = lambda r: {
                "configuration": r["config"]["name"], "LoRA on": r["config"]["placement"], "rank": r["config"]["rank"], "learning rate": r["config"]["lr"],
                "rows": r["train_rows"], "dev type loss": round(r["metrics"]["type_loss"], 3), "type accuracy": _pct(r["metrics"]["type_accuracy"]["1.0"]),
                "broken accuracy": _pct(r["metrics"]["broken_accuracy"]), "parameters": r["trainable_parameters"], "seconds": r["seconds"],
            }
            _loss = mo.ui.tabs(
                {
                    "Loss": _loss,
                    "Sweep": mo.ui.table(pd.DataFrame([_row(r) for r in _e["sweep"]]), selection=None, pagination=False),
                    "LoRA layers": mo.ui.table(pd.DataFrame([_row(r) for r in _e["placement_ablation"]] + [_row(r) for r in _e["sweep"] if r["config"]["name"] == _e["sweep_best"]]), selection=None, pagination=False),
                    "Guidance grid": mo.ui.table(
                        pd.DataFrame([{"guidance": kind.replace("_", " "), "weight": w, "type accuracy": _pct(v["accuracy"]), "log p": round(v["logp"], 3)} for kind, grid in _e["guidance"].items() for w, v in grid.items()]),
                        selection=None, pagination=False,
                    ),
                }
            )
        _cm = pd.DataFrame(_best["records"]).groupby(["true", "1.0"]).size().reset_index(name="rows").rename(columns={"1.0": "predicted"})
        _matrix = mo.ui.altair_chart(
            alt.Chart(_cm)
            .mark_rect()
            .encode(
                x=alt.X("predicted:N", title="Predicted", scale=alt.Scale(domain=FAILURE_LABELS), axis=alt.Axis(labelAngle=-40, labelLimit=120, labelOverlap=True)),
                y=alt.Y("true:N", title="True", scale=alt.Scale(domain=FAILURE_LABELS), axis=alt.Axis(labelLimit=200)),
                color=alt.Color("rows:Q", title="Rows", scale=alt.Scale(scheme="blues")),
                tooltip=["true", "predicted", "rows"],
            )
            .properties(
                title=alt.TitleParams(
                    "Confusion",
                    subtitle=[
                        f"{_best['dev_rows']} held-out rows · broken? {_pct(_m['broken_accuracy'])} (base {_pct(_m['broken_accuracy_base'])})",
                        f"type {_pct(_m['type_accuracy']['1.0'])} (base {_pct(_m['type_accuracy']['0.0'])})"
                        + (f", on broken calls only {_pct(_m['type_accuracy_broken'])}" if "type_accuracy_broken" in _m else "")
                        + f" · span {_pct(_m['span_hit'])}",
                    ],
                    anchor="start",
                ),
                height=280, width="container",
            )
        )
        _charts = [_loss, _matrix]
    mo.vstack(
        [
            space(56, "step-grader"),
            mo.md("## Train D1"),
            *_notes,
            mo.hstack(
                [grader_lr, grader_dropout, grader_clip, grader_rank, grader_alpha_mult, grader_batch, mo.vstack([grader_train_btn, grader_placement, grader_preset], gap=0.75)],
                justify="start", align="end", gap=1.5, wrap=True,
            ),
            *([row(*[(chart, 1, 380) for chart in _charts])] if _charts else []),
        ],
        gap=1.5,
    )
    return


@app.cell
def _(
    Counter,
    GRADER_DIR,
    RUNS_DIR,
    attach_lora,
    gc,
    grade_calls,
    json,
    learning_rate,
    math,
    parse_tool_call,
    random,
    run_nu,
    same_output,
    set_lora,
    time,
):
    # -------------------------------------------------------------------------
    # Step 7: target training, execution scoring, grader-driven P_bad, guided decoding
    # -------------------------------------------------------------------------
    TARGET_DIR = RUNS_DIR / "target"
    TARGET_SYSTEM = (
        "Make one run_nushell tool call for the request. Preserve every requested step in order. "
        "Discover files with glob, ls and open instead of assuming their contents, and bind intermediate results with let. "
        "When a path or key may be missing, recover with try, default or is-empty and return the choices. "
        "Answer a simple question with a simple command. When the request names fields or asks for structured "
        "output, return JSON with | to json --raw. Code contains only Nushell source."
    )
    TARGET_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "run_nushell",
                "description": "Execute Nushell (Nu) source in the workspace root.",
                "parameters": {"type": "object", "required": ["code"], "properties": {"code": {"type": "string"}}},
            },
        }
    ]

    def target_prompt(tokenizer, row: dict) -> str:
        """The chat prompt up to and including the opening `<tool_call>` line; the model writes the rest."""
        refs = "\n\n".join(f"{doc['title']} (nushell)\n{doc['text']}" for doc in row.get("references", []))
        text = f"\nSyntax references:\n{refs}\n\n" if refs else "\n"
        text += f"Request: {row['speech']}\nWorkspace: {row['context']}\nFiles: " + ", ".join(row["workspace_files"])
        messages = [{"role": "system", "content": TARGET_SYSTEM}, {"role": "user", "content": [{"type": "text", "text": text}]}]
        # Thinking is switched off so the assistant turn starts directly with the tool call.
        template = tokenizer.apply_chat_template(messages, tools=TARGET_TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return template + "<tool_call>\n"

    def _encode(tokenizer, row: dict):
        """Input tokens, the completion tokens the loss is taken on, and where they start."""
        prompt = target_prompt(tokenizer, row)
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        tokens = tokenizer.encode(prompt + row["completion"] + "<|im_end|>", add_special_tokens=False)
        start = len(prefix)
        if tokens[:start] != prefix:  # the prompt's final newline can merge with the first completion token
            start -= 1
        if not 0 < start < len(tokens) or tokens[:start] != prefix[:start]:
            raise ValueError(f"completion is misaligned for {row['id']}")
        return tokens[:-1], tokens[start:], start - 1

    def _completion_loss(lm, inputs, expected, start):
        import mlx.core as mx
        import mlx.nn as nn

        backbone = lm.model
        hidden = backbone(mx.array([inputs]))[:, start:]
        logits = backbone.embed_tokens.as_linear(hidden) if lm.config.tie_word_embeddings else lm.lm_head(hidden)
        return nn.losses.cross_entropy(logits.astype(mx.float32), mx.array([expected]), reduction="mean")

    def _mean_loss(lm, tokenizer, rows) -> float | None:
        return sum(float(_completion_loss(lm, *_encode(tokenizer, r)).item()) for r in rows) / len(rows) if rows else None

    def _ternarize(lm, group_size: int = 64) -> None:
        """Replace every linear weight by its ternary version: within each group of `group_size` weights in a row,
        each weight becomes −s, 0 or +s. Weights smaller than 0.7 × the group's mean size go to 0 and s is the mean
        size of the rest (ternary weight networks). The values are kept as ordinary floats, so this measures what
        ternary weights do to the answers; it does not shrink the model in memory. The embedding goes to 4 bits."""
        import mlx.core as mx
        import mlx.nn as nn

        def ternary(w):
            out_dim, in_dim = w.shape
            groups = w.astype(mx.float32).reshape(out_dim, in_dim // group_size, group_size)
            size = mx.abs(groups)
            keep = size > 0.7 * size.mean(axis=-1, keepdims=True)
            scale = (size * keep).sum(axis=-1, keepdims=True) / mx.maximum(keep.sum(axis=-1, keepdims=True), 1)
            return (mx.sign(groups) * keep * scale).reshape(out_dim, in_dim).astype(w.dtype)

        for _, module in lm.named_modules():
            if isinstance(module, nn.Linear) and module.weight.shape[1] % group_size == 0:
                module.weight = ternary(module.weight)
                mx.eval(module.weight)
        nn.quantize(lm, group_size=group_size, bits=4, class_predicate=lambda _, m: isinstance(m, nn.Embedding))

    def _load(cfg: dict, seed: int):
        import mlx.core as mx
        from mlx_vlm import load as vlm_load

        mx.set_cache_limit(512 * 1024**2)
        mx.random.seed(seed)
        model, processor = vlm_load(cfg["model_path"])
        if cfg.get("bits"):  # the same weights stored with fewer bits: back to full precision, then quantised again
            import mlx.nn as nn
            from mlx_lm.utils import dequantize_model

            dequantize_model(model.language_model)
            if cfg["bits"] == "ternary":
                _ternarize(model.language_model, cfg.get("group_size", 64))
            else:
                nn.quantize(model.language_model, group_size=cfg.get("group_size", 64), bits=cfg["bits"])
            mx.eval(model.language_model.parameters())
        lm, layers, n_params = attach_lora(model, cfg["rank"], cfg["alpha"], 0.0, cfg.get("placement", "all_linear"))
        return model, getattr(processor, "tokenizer", processor), lm, layers, n_params

    def _fit(lm, tokenizer, rows: list[dict], cfg: dict, name: str, log, snapshots: dict | None = None, stop_at_weak: bool = False) -> list[dict]:
        """One LoRA on `rows`: per-example gradients averaged over a batch, clipped, AdamW with warm-up and cosine
        decay. With `snapshots`, the adapter as it stood a quarter of the way through is kept under "weak"."""
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten, tree_map

        optimizer = optim.AdamW(learning_rate=cfg["lr"], weight_decay=0.0)
        value_and_grad = nn.value_and_grad(lm, _completion_loss)
        total = math.ceil(len(rows) / cfg["batch"]) * cfg["epochs"]
        order, rows, history, step = random.Random(cfg["seed"]), list(rows), [], 0
        for _epoch in range(cfg["epochs"]):
            order.shuffle(rows)
            for offset in range(0, len(rows), cfg["batch"]):
                batch = rows[offset : offset + cfg["batch"]]
                grads, losses = None, []
                for row in batch:
                    loss, grad = value_and_grad(lm, *_encode(tokenizer, row))
                    grads = grad if grads is None else tree_map(lambda a, b: a + b, grads, grad)
                    mx.eval(grads)
                    losses.append(float(loss.item()))
                step += 1
                optimizer.learning_rate = learning_rate(step, total, cfg["lr"])
                clipped, _ = optim.clip_grad_norm(tree_map(lambda g: g / len(batch), grads), 1.0)
                optimizer.update(lm, clipped)
                mx.eval(lm.trainable_parameters(), optimizer.state)
                history.append({"adapter": name, "step": step, "loss": sum(losses) / len(losses)})
                if step == 1 or step % 10 == 0 or step == total:
                    log(f"[target {name}] step {step}/{total} loss {history[-1]['loss']:.3f}")
                if snapshots is not None and step == max(1, total // 4):  # the weak version of this adapter
                    snapshots["weak"] = {k: mx.array(v) for k, v in tree_flatten(lm.trainable_parameters())}
                    mx.eval(snapshots["weak"])
                    if stop_at_weak:
                        return history
        return history

    def _generate(lm, tokenizer, row: dict, max_new: int, good=None, bad=None, weight: float = 0.0, floor: float = 0.1, bad_row: dict | None = None, blank_weight: float = 0.0) -> str:
        """Greedy decoding of one tool call. With `good` and `bad` adapter weights it is guided: each step runs
        both adapters, each with its own KV cache, and scores a token by
        log p_good + weight · (log p_good − log p_bad), over the tokens P_good itself finds plausible
        (probability at least `floor` × its top token). That restriction stops the difference from promoting
        tokens that are merely unlikely under P_bad.

        With `bad_row` the contrast reads a different prompt (for example the same row with the request blanked);
        without `bad` it then uses the same adapter, so the difference is what the request itself contributes.
        `blank_weight` adds that request contrast on top of an adapter contrast, so both steer at once."""
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_unflatten

        def use(weights):
            lm.update(tree_unflatten(list(weights.items())))

        def step(tokens, cache):
            out = lm(tokens, cache=cache)
            return nn.log_softmax((out.logits if hasattr(out, "logits") else out)[:, -1, :].astype(mx.float32), axis=-1)

        stop = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
        tokens = mx.array([tokenizer.encode(target_prompt(tokenizer, row), add_special_tokens=False)])
        other_tokens = tokens if bad_row is None else mx.array([tokenizer.encode(target_prompt(tokenizer, bad_row), add_special_tokens=False)])
        blank_tokens = mx.array([tokenizer.encode(target_prompt(tokenizer, {**row, "speech": ""}), add_special_tokens=False)]) if blank_weight else None
        cache_good, cache_bad, cache_blank, written = lm.make_cache(), lm.make_cache(), lm.make_cache(), []
        for _ in range(max_new):
            if good is not None:
                use(good)
            scores = step(tokens, cache_good)
            blank = step(blank_tokens, cache_blank) if blank_weight else None
            if bad is not None or bad_row is not None or blank_weight:
                contrast = scores
                if bad is not None or bad_row is not None:
                    if bad is not None:
                        use(bad)
                    contrast = step(other_tokens, cache_bad)
                keep = scores >= mx.max(scores, axis=-1, keepdims=True) + math.log(floor)
                guided = scores + weight * (scores - contrast) + (blank_weight * (scores - blank) if blank_weight else 0.0)
                scores = mx.where(keep, guided, mx.array(-1e9, dtype=mx.float32))
            token = int(mx.argmax(scores, axis=-1).item())
            if token in stop:
                break
            written.append(token)
            if tokenizer.decode(written).rstrip().endswith("</tool_call>"):
                break
            tokens = other_tokens = blank_tokens = mx.array([[token]])
        return tokenizer.decode(written)

    def score_call(text: str, row: dict) -> dict:
        """Execute a generated tool call: is it a well-formed call, does it run, does it return the reference value?"""
        code = parse_tool_call(text.strip())
        if code is None:
            return {"well_formed": False, "runs": False, "correct": False, "code": None}
        result = run_nu(code, row["env"])
        return {"well_formed": True, "runs": result["ok"], "correct": result["ok"] and same_output(result["stdout"], row["expected"]), "code": code}

    def _rate(scores: list[dict], key: str = "correct"):
        return sum(s[key] for s in scores) / len(scores) if scores else None

    TARGET_GUIDANCE_GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]

    def quick_target_run(cfg: dict, dataset: dict, log=print) -> dict:
        """Train one P_good configuration and measure it: loss on held-out reference calls, and the share of
        held-out requests whose generated call returns the reference value. Used for sweeps and ablations;
        nothing is saved."""
        import mlx.core as mx

        started = time.monotonic()
        train, dev = dataset["target"]["train"], dataset["target"]["dev"]
        spread = lambda rows: sorted(rows, key=lambda r: (r["id"].rsplit("/", 1)[1], r["id"]))
        fit_rows, eval_rows = train[: cfg["n_good"]], spread(dev)[: cfg["n_eval"]]
        model, tokenizer, lm, layers, n_params = _load(cfg, cfg["seed"])
        try:
            history = _fit(lm, tokenizer, fit_rows, cfg, cfg["name"], log)
            loss = _mean_loss(lm, tokenizer, eval_rows[: cfg["n_loss"]])
            scores = [score_call(_generate(lm, tokenizer, r, cfg["max_new"]), r) for r in eval_rows]
        finally:
            del model, lm, layers
            gc.collect()
            mx.clear_cache()
        result = {
            "config": {k: v for k, v in cfg.items() if k not in ("guidance",)}, "trainable_parameters": n_params, "fit_rows": len(fit_rows),
            "dev_loss": loss, "well_formed": _rate(scores, "well_formed"), "runs": _rate(scores, "runs"), "correct": _rate(scores),
            "final_train_loss": history[-1]["loss"], "seconds": round(time.monotonic() - started, 1),
        }
        log(f"[target {cfg['name']}] dev loss {loss:.3f} · executed-correct {result['correct']:.0%} of {len(eval_rows)}")
        return result

    def run_target_experiments(base: dict, dataset: dict, sizes: dict, log=print) -> dict:
        """The target's full protocol, each stage choosing by executed-correct on held-out environments (dev loss
        breaks ties):
        1. a sweep over learning rate and rank with the adapter on every linear layer;
        2. an ablation of where the adapter goes (attention, MLP or convolution layers only) at the best setting;
        3. the full pipeline at the chosen setting: P_good on the full budget, the probe, P_bad, and the
           guidance comparison (against P_bad, against a weaker P_good, against the base model).
        Everything is written to `runs/target/experiments.json`."""
        TARGET_DIR.mkdir(parents=True, exist_ok=True)
        quick = {**base, "n_good": sizes["n_sweep"], "n_eval": sizes["n_sweep_eval"], "n_loss": sizes["n_sweep_eval"], "epochs": 1, "placement": "all_linear"}
        better = lambda r: (r["correct"], -r["dev_loss"])
        sweep = [
            quick_target_run({**quick, "name": f"all_linear-r{rank}-lr{lr:g}", "rank": rank, "alpha": 2.0 * rank, "lr": lr}, dataset, log)
            for rank in sizes["ranks"] for lr in sizes["learning_rates"]
        ]
        best = max(sweep, key=better)["config"]
        ablation = [
            quick_target_run({**best, "name": f"{placement}-r{best['rank']}-lr{best['lr']:g}", "placement": placement}, dataset, log)
            for placement in ("attention_only", "mlp_only", "conv_only")
        ]
        chosen = max([r for r in sweep if r["config"]["name"] == best["name"]] + ablation, key=better)["config"]
        final = run_target_pipeline(
            {**base, **sizes["full"], "rank": chosen["rank"], "alpha": chosen["alpha"], "lr": chosen["lr"], "placement": chosen["placement"]}, dataset, log
        )
        experiments = {
            "created": time.time(), "dataset": dataset["fingerprint"], "sizes": sizes, "sweep": sweep, "sweep_best": best["name"],
            "placement_ablation": ablation, "chosen": chosen["name"],
            "final": {k: final[k] for k in ("config", "rows", "loss", "execution", "reference_logp", "probe", "seconds", "trainable_parameters")},
        }
        (TARGET_DIR / "experiments.json").write_text(json.dumps(experiments, indent=1), encoding="utf-8")
        return experiments

    def run_target_pipeline(cfg: dict, dataset: dict, log=print) -> dict:
        """The whole of step 7. Training chains are divided into fit chains and probe chains; the dev chains are
        used only for the final scores.

        1. Generate with the base model on dev requests, train P_good on the fit chains, then generate with
           P_good on dev requests and on probe requests.
        2. Execute everything. Execution decides what is wrong.
        3. Ask the grader what kind of wrong each failed probe call is. It is never asked whether a call is
           wrong: execution has already answered that.
        4. Train P_bad on the probe outputs that failed plus synthetic breaks of the failure types the grader
           found most often in them.
        5. Decode dev requests guided by P_good against P_bad and execute the results."""
        import mlx.core as mx
        out_dir = type(TARGET_DIR)(cfg.get("out_dir") or TARGET_DIR)  # the smoke test writes somewhere else
        import mlx.nn as nn
        from mlx.utils import tree_flatten, tree_unflatten

        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        rng = random.Random(cfg["seed"])
        train, dev = dataset["target"]["train"], dataset["target"]["dev"]
        chains = sorted({r["chain_id"] for r in train})
        rng.shuffle(chains)
        probe_chains = set(chains[: max(1, round(len(chains) * cfg["probe_share"]))])
        spread = lambda rows: sorted(rows, key=lambda r: (r["id"].rsplit("/", 1)[1], r["id"]))  # one request per chain first
        fit_rows = [r for r in train if r["chain_id"] not in probe_chains][: cfg["n_good"]]
        probe_rows = spread([r for r in train if r["chain_id"] in probe_chains])[: cfg["n_probe"]]
        eval_rows = spread(dev)[: cfg["n_eval"]]
        loss_rows = eval_rows[: cfg["n_loss"]]

        # 1. base outputs, P_good, P_good outputs
        model, tokenizer, lm, layers, n_params = _load(cfg, cfg["seed"])
        try:
            loss_before = _mean_loss(lm, tokenizer, loss_rows)  # LoRA starts as the identity, so this is the base model
            log(f"[target] base model: generating {len(eval_rows)} dev calls")
            base_out = [_generate(lm, tokenizer, r, cfg["max_new"]) for r in eval_rows]
            snapshots = {}
            history = _fit(lm, tokenizer, fit_rows, cfg, "P_good", log, snapshots)
            loss_good = _mean_loss(lm, tokenizer, loss_rows)
            log(f"[target] P_good: generating {len(eval_rows)} dev and {len(probe_rows)} probe calls")
            good_out = [_generate(lm, tokenizer, r, cfg["max_new"]) for r in eval_rows]
            probe_out = [_generate(lm, tokenizer, r, cfg["max_new"]) for r in probe_rows]
            mx.save_safetensors(str(out_dir / "adapter_good.safetensors"), dict(tree_flatten(lm.trainable_parameters())))
            if snapshots.get("weak"):  # kept so the 3D view can show autoguidance token by token
                mx.save_safetensors(str(out_dir / "adapter_weak.safetensors"), snapshots["weak"])
        finally:
            del model, lm, layers
            gc.collect()
            mx.clear_cache()

        # 2. execution
        base_scores = [score_call(t, r) for t, r in zip(base_out, eval_rows)]
        good_scores = [score_call(t, r) for t, r in zip(good_out, eval_rows)]
        probe_scores = [score_call(t, r) for t, r in zip(probe_out, probe_rows)]

        # 3. execution has said which probe calls are wrong; the grader says what kind of wrong each one is
        wrong = [(r, t) for r, t, sc in zip(probe_rows, probe_out, probe_scores) if not sc["correct"]]
        failure_types, weak_types = None, []
        if wrong and (GRADER_DIR / "adapter.safetensors").exists():
            log(f"[target] {len(wrong)} of {len(probe_rows)} probe calls are wrong by execution; asking the grader what kind")
            grades = grade_calls([{"prompt": r["speech"], "completion": t.strip()} for r, t in wrong], cfg["grader_model_path"])
            failure_types = Counter(g["type_if_broken"] for g in grades)
            weak_types = [name for name, _ in failure_types.most_common(3)]
        elif wrong:
            log("[target] no trained grader found: P_bad will use every failure type")

        # 4. P_bad data: the target's own failures, topped up with synthetic breaks of its weak types
        own = [
            {**r, "completion": t.strip(), "source": "target output"}
            for r, t, s in zip(probe_rows, probe_out, probe_scores)
            if not s["correct"] and t.strip().endswith("</tool_call>")
        ]
        pool = [g for g in dataset["grader"]["train"] if g["is_broken"] and (not weak_types or g["failure_type"] in weak_types)]
        rng.shuffle(pool)
        setting = {r["chain_id"]: r for r in train + dev}  # a chain's files, context and references
        synthetic = [
            {**setting[g["chain_id"]], "id": g["id"], "speech": g["prompt"], "completion": g["completion"], "source": g["failure_type"]}
            for g in pool[: max(0, cfg["n_bad"] - len(own))]
        ]
        bad_rows = own + synthetic
        held_out_bad = [
            {**setting[g["chain_id"]], "id": g["id"], "speech": g["prompt"], "completion": g["completion"]}
            for g in dataset["grader"]["dev"]
            if g["is_broken"] and (not weak_types or g["failure_type"] in weak_types)
        ][: cfg["n_loss"]]

        # 5. P_bad, then guided decoding
        model, tokenizer, lm, layers, _ = _load(cfg, cfg["seed"] + 64)
        try:
            bad_before = _mean_loss(lm, tokenizer, held_out_bad)
            history += _fit(lm, tokenizer, bad_rows, cfg, "P_bad", log)
            bad_after = _mean_loss(lm, tokenizer, held_out_bad)
            bad_on_good = _mean_loss(lm, tokenizer, loss_rows)
            bad_weights = dict(tree_flatten(lm.trainable_parameters()))
            mx.save_safetensors(str(out_dir / "adapter_bad.safetensors"), bad_weights)
            good_weights = mx.load(str(out_dir / "adapter_good.safetensors"))
            # three things P_good can be steered away from: the adapter trained on mistakes, its own weaker
            # self (autoguidance), and the base model
            against = {
                "P_bad": bad_weights,
                "weak P_good": snapshots.get("weak"),
                "base": {k: (mx.zeros_like(v) if k.endswith("lora_b") else v) for k, v in good_weights.items()},
            }
            guided = {}
            for other, weight, floor in cfg["guidance"]:  # decoding only, no retraining
                if against.get(other) is None:
                    continue
                name = f"vs {other} w={weight} floor={floor}"
                log(f"[target] guided decoding {name}: {len(eval_rows)} dev calls")
                outs = [_generate(lm, tokenizer, r, cfg["max_new"], good_weights, against[other], weight, floor) for r in eval_rows]
                guided[name] = {"outputs": outs, "scores": [score_call(t, r) for t, r in zip(outs, eval_rows)]}
            # the same comparison without sampling: how probable the reference call is under each guided
            # distribution, log p = log p_good + w · (log p_good − log p_other), renormalised over the vocabulary
            logp_grid = {other: {} for other, weights in against.items() if weights is not None}

            def reference_logits(weights, row):
                lm.update(tree_unflatten(list(weights.items())))
                inputs, expected, start = _encode(tokenizer, row)
                hidden = lm.model(mx.array([inputs]))[:, start:]
                logits = lm.model.embed_tokens.as_linear(hidden) if lm.config.tie_word_embeddings else lm.lm_head(hidden)
                return nn.log_softmax(logits.astype(mx.float32), axis=-1)[0], mx.array(expected)

            totals = {other: {w: [] for w in TARGET_GUIDANCE_GRID} for other in logp_grid}
            for row in loss_rows:
                good_lp, expected = reference_logits(good_weights, row)
                for other in logp_grid:
                    other_lp, _ = reference_logits(against[other], row)
                    for w in TARGET_GUIDANCE_GRID:
                        mixed = nn.log_softmax(good_lp + w * (good_lp - other_lp), axis=-1)
                        totals[other][w].append(float(mx.take_along_axis(mixed, expected[:, None], axis=-1).mean().item()))
            for other in logp_grid:
                logp_grid[other] = {str(w): sum(v) / max(1, len(v)) for w, v in totals[other].items()}
        finally:
            del model, lm, layers
            gc.collect()
            mx.clear_cache()

        arms = {"base": base_scores, "P_good": good_scores, **{f"guided {name}": g["scores"] for name, g in guided.items()}}
        report = {
            "created": time.time(),
            "dataset": dataset["fingerprint"],
            "config": cfg,
            "trainable_parameters": n_params,
            "rows": {"fit": len(fit_rows), "probe": len(probe_rows), "eval": len(eval_rows), "bad_own": len(own), "bad_synthetic": len(synthetic)},
            "history": history,
            "loss": {
                "good_dev_base": loss_before, "good_dev_P_good": loss_good, "good_dev_P_bad": bad_on_good,
                "bad_dev_base": bad_before, "bad_dev_P_bad": bad_after,
            },
            "execution": {
                name: {"well_formed": _rate(scores, "well_formed"), "runs": _rate(scores, "runs"), "correct": _rate(scores)}
                for name, scores in arms.items()
            },
            "reference_logp": logp_grid,
            "probe": {
                "correct": _rate(probe_scores),
                "wrong": len(wrong),
                "weak_types": weak_types,
                "failure_types": dict(failure_types) if failure_types else None,
            },
            "samples": [
                {
                    "request": r["speech"],
                    "reference": parse_tool_call(r["completion"]),
                    "base": base_out[i].strip(), "base_correct": base_scores[i]["correct"],
                    "P_good": good_out[i].strip(), "P_good_correct": good_scores[i]["correct"],
                    **{f"guided {name}": g["outputs"][i].strip() for name, g in guided.items()},
                    **{f"guided {name} correct": g["scores"][i]["correct"] for name, g in guided.items()},
                }
                for i, r in enumerate(eval_rows)
            ],
            "seconds": round(time.monotonic() - started, 1),
            "peak_memory_gb": round(mx.get_peak_memory() / 1024**3, 2),
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out_dir / "prompt_format.json").write_text(
            json.dumps({"system": TARGET_SYSTEM, "tools": TARGET_TOOLS, "assistant_prefix": "<tool_call>\n", "stop": ["</tool_call>", "<|im_end|>"]}, indent=2),
            encoding="utf-8",
        )
        log(f"[target] executed-correct on dev: " + ", ".join(f"{k} {v['correct']:.0%}" for k, v in report["execution"].items()))
        return report

    def crossfit_wrong_calls(dataset: dict, folds: int = 5, log=print) -> dict:
        """P_good's mistakes on training requests it was not fitted on. The training chains are split into
        `folds` parts; for each part an adapter with the final settings is trained on the other parts and
        writes the calls for this one, which are executed. Every wrong call is kept with its request, in
        `runs/target/crossfit.json`. A part that is already there is not repeated."""
        import mlx.core as mx

        cfg = json.loads((TARGET_DIR / "report.json").read_text(encoding="utf-8"))["config"]
        path = TARGET_DIR / "crossfit.json"
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"dataset": dataset["fingerprint"], "folds": folds, "parts": {}}
        if saved.get("dataset") != dataset["fingerprint"] or saved.get("folds") != folds:
            saved = {"dataset": dataset["fingerprint"], "folds": folds, "parts": {}}
        train = dataset["target"]["train"]
        chains = sorted({r["chain_id"] for r in train})
        random.Random(cfg["seed"] + 5).shuffle(chains)
        for k in range(folds):
            if str(k) in saved["parts"]:
                continue
            held = set(chains[k::folds])
            fit_rows, out_rows = [r for r in train if r["chain_id"] not in held], [r for r in train if r["chain_id"] in held]
            model, tokenizer, lm, layers, _ = _load(cfg, cfg["seed"] + k)
            try:
                _fit(lm, tokenizer, fit_rows, cfg, f"cross-fit {k + 1}/{folds}", log)
                outputs = []
                for done, row in enumerate(out_rows, 1):
                    outputs.append(_generate(lm, tokenizer, row, cfg["max_new"]))
                    if done % 40 == 0 or done == len(out_rows):
                        log(f"[target cross-fit {k + 1}/{folds}] wrote {done}/{len(out_rows)} calls")
            finally:
                del model, lm, layers
                gc.collect()
                mx.clear_cache()
            scores = [score_call(t, r) for t, r in zip(outputs, out_rows)]
            saved["parts"][str(k)] = {
                "rows": len(out_rows), "correct": sum(s["correct"] for s in scores),
                "wrong": [{"id": r["id"], "completion": t.strip(), "runs": s["runs"]} for r, t, s in zip(out_rows, outputs, scores) if not s["correct"]],
            }
            path.write_text(json.dumps(saved, indent=1), encoding="utf-8")
            log(f"[target cross-fit {k + 1}/{folds}] {saved['parts'][str(k)]['correct']} of {len(out_rows)} correct")
        return saved

    def train_bad_adapter(rows: list[dict], name: str, log=print) -> None:
        """Fit an adapter with the final settings on calls that are known to be wrong, as `runs/target/adapter_<name>.safetensors`."""
        import mlx.core as mx
        from mlx.utils import tree_flatten

        cfg = json.loads((TARGET_DIR / "report.json").read_text(encoding="utf-8"))["config"]
        model, tokenizer, lm, layers, _ = _load(cfg, cfg["seed"] + 64)
        try:
            _fit(lm, tokenizer, rows, cfg, name, log)
            mx.save_safetensors(str(TARGET_DIR / f"adapter_{name}.safetensors"), dict(tree_flatten(lm.trainable_parameters())))
        finally:
            del model, lm, layers
            gc.collect()
            mx.clear_cache()

    def score_decoders(dataset: dict, arms: list[dict], log=print) -> dict:
        """Write and execute the call for every held-out request under each decoder in `arms`, and keep which
        requests each one gets right in `runs/target/decoders.json`. An arm is {name, weight, floor} plus either
        `adapter` (steer P_good away from `runs/target/adapter_<adapter>.safetensors`) or `blank_request: true`
        (steer P_good away from itself reading the prompt with the request removed). An arm with neither is
        plain P_good. Arms that are already there are not repeated."""
        import mlx.core as mx

        cfg = json.loads((TARGET_DIR / "report.json").read_text(encoding="utf-8"))["config"]
        path = TARGET_DIR / "decoders.json"
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if saved.get("dataset") != dataset["fingerprint"]:
            saved = {"dataset": dataset["fingerprint"], "arms": {}}
        rows = sorted(dataset["target"]["dev"], key=lambda r: (r["id"].rsplit("/", 1)[1], r["id"]))
        todo = [a for a in arms if a["name"] not in saved["arms"]]
        if not todo:
            return saved
        model, tokenizer, lm, layers, _ = _load(cfg, cfg["seed"])
        try:
            good = mx.load(str(TARGET_DIR / "adapter_good.safetensors"))
            for arm in todo:
                other = mx.load(str(TARGET_DIR / f"adapter_{arm['adapter']}.safetensors")) if arm.get("adapter") else None
                outputs = []
                for done, row in enumerate(rows, 1):
                    outputs.append(
                        _generate(
                            lm, tokenizer, row, cfg["max_new"], good=good, bad=other, weight=arm.get("weight", 0.0), floor=arm.get("floor", 0.1),
                            bad_row={**row, "speech": ""} if arm.get("blank_request") else None, blank_weight=arm.get("blank_weight", 0.0),
                        )
                    )
                    if done % 40 == 0 or done == len(rows):
                        log(f"[decoders] {arm['name']}: {done}/{len(rows)} calls")
                scores = [score_call(t, r) for t, r in zip(outputs, rows)]
                saved["arms"][arm["name"]] = {
                    **{k: v for k, v in arm.items() if k != "name"},
                    "well_formed": _rate(scores, "well_formed"), "runs": _rate(scores, "runs"), "correct": _rate(scores),
                    "rows": [{"id": r["id"], "correct": s["correct"], "code": s["code"]} for r, s in zip(rows, scores)],
                }
                path.write_text(json.dumps(saved, indent=1), encoding="utf-8")
                log(f"[decoders] {arm['name']}: {_rate(scores):.1%} of {len(rows)} return the reference value")
        finally:
            del model, lm, layers
            gc.collect()
            mx.clear_cache()
        return saved

    def score_low_bit(dataset: dict, bits, retrain: bool = False, log=print, rank: int | None = None) -> dict:
        """The target with its base weights stored at `bits` bits per weight (MLX affine quantisation, groups of
        64), scored by execution on every held-out request. With `retrain=False` the existing P_good adapter is
        used as it is; with `retrain=True` an adapter with the same settings and training rows is fitted on top
        of the low-bit weights first. Results are kept in `runs/target/low_bit.json`."""
        import mlx.core as mx
        from mlx.utils import tree_flatten

        cfg = {**json.loads((TARGET_DIR / "report.json").read_text(encoding="utf-8"))["config"], "bits": bits}
        if retrain and rank:  # a wider adapter has more room to make up for what the low-bit weights lost
            cfg.update(rank=rank, alpha=2.0 * rank)
        label = "ternary" if bits == "ternary" else f"{bits}-bit"
        path = TARGET_DIR / "low_bit.json"
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if saved.get("dataset") != dataset["fingerprint"]:
            saved = {"dataset": dataset["fingerprint"], "runs": {}}
        name = f"{label}, " + (f"adapter retrained on it (rank {cfg['rank']})" if retrain else "existing adapter")
        if name in saved["runs"]:
            return saved
        rows = sorted(dataset["target"]["dev"], key=lambda r: (r["id"].rsplit("/", 1)[1], r["id"]))
        model, tokenizer, lm, layers, _ = _load(cfg, cfg["seed"])
        try:
            if retrain:
                rng = random.Random(cfg["seed"])
                train = dataset["target"]["train"]
                chains = sorted({r["chain_id"] for r in train})
                rng.shuffle(chains)
                probe_chains = set(chains[: max(1, round(len(chains) * cfg["probe_share"]))])
                fit_rows = [r for r in train if r["chain_id"] not in probe_chains][: cfg["n_good"]]
                _fit(lm, tokenizer, fit_rows, cfg, f"P_good on {label}", log)
                mx.save_safetensors(str(TARGET_DIR / f"adapter_good_{label}_r{cfg['rank']}.safetensors"), dict(tree_flatten(lm.trainable_parameters())))
            else:
                lm.load_weights(str(TARGET_DIR / "adapter_good.safetensors"), strict=False)
            outputs = []
            for done, row in enumerate(rows, 1):
                outputs.append(_generate(lm, tokenizer, row, cfg["max_new"]))
                if done % 40 == 0 or done == len(rows):
                    log(f"[low-bit] {name}: {done}/{len(rows)} calls")
        finally:
            del model, lm, layers
            gc.collect()
            mx.clear_cache()
        scores = [score_call(t, r) for t, r in zip(outputs, rows)]
        saved["runs"][name] = {
            "bits": bits, "retrained": retrain, "rank": cfg["rank"], "well_formed": _rate(scores, "well_formed"), "runs": _rate(scores, "runs"), "correct": _rate(scores),
            "rows": [{"id": r["id"], "correct": s["correct"], "code": s["code"]} for r, s in zip(rows, scores)],
        }
        path.write_text(json.dumps(saved, indent=1), encoding="utf-8")
        log(f"[low-bit] {name}: {_rate(scores):.1%} of {len(rows)} return the reference value")
        return saved

    def rebuild_weak_target(dataset: dict, log=print) -> None:
        """Write `runs/target/adapter_weak.safetensors` for a P_good that was trained before the weak snapshot was
        saved: the same rows, seed and schedule, stopped a quarter of the way through."""
        import mlx.core as mx

        cfg = json.loads((TARGET_DIR / "report.json").read_text(encoding="utf-8"))["config"]
        rng = random.Random(cfg["seed"])
        train = dataset["target"]["train"]
        chains = sorted({r["chain_id"] for r in train})
        rng.shuffle(chains)
        probe_chains = set(chains[: max(1, round(len(chains) * cfg["probe_share"]))])
        fit_rows = [r for r in train if r["chain_id"] not in probe_chains][: cfg["n_good"]]
        model, tokenizer, lm, layers, _ = _load(cfg, cfg["seed"])
        try:
            snapshots = {}
            _fit(lm, tokenizer, fit_rows, cfg, "weak P_good", log, snapshots, stop_at_weak=True)
            mx.save_safetensors(str(TARGET_DIR / "adapter_weak.safetensors"), snapshots["weak"])
        finally:
            del model, lm, layers
            gc.collect()

    def score_saved_target(dataset: dict, log=print) -> dict:
        """Score the saved P_good on every held-out request that took no part in choosing its settings or in the
        report's own scores, and add the result to `runs/target/report.json` as `unused_held_out`."""
        report = json.loads((TARGET_DIR / "report.json").read_text(encoding="utf-8"))
        cfg = report["config"]
        spread = lambda rows: sorted(rows, key=lambda r: (r["id"].rsplit("/", 1)[1], r["id"]))
        rows = spread(dataset["target"]["dev"])[cfg["n_eval"]:]
        model, tokenizer, lm, layers, _ = _load(cfg, cfg["seed"])
        try:
            lm.load_weights(str(TARGET_DIR / "adapter_good.safetensors"), strict=False)
            outputs = []
            for done, row in enumerate(rows, 1):
                outputs.append(_generate(lm, tokenizer, row, cfg["max_new"]))
                if done % 20 == 0 or done == len(rows):
                    log(f"[target] unused held-out requests: {done}/{len(rows)}")
        finally:
            del model, lm, layers
            gc.collect()
        scores = [score_call(t, r) for t, r in zip(outputs, rows)]
        by_env = {}
        for row, sc in zip(rows, scores):
            by_env.setdefault(row["env"], []).append(sc["correct"])
        report["unused_held_out"] = {
            "rows": len(rows), "well_formed": _rate(scores, "well_formed"), "runs": _rate(scores, "runs"), "correct": _rate(scores),
            "by_environment": {env: {"rows": len(v), "correct": sum(v) / len(v)} for env, v in sorted(by_env.items())},
            "wrong": [{"request": r["speech"], "reference": parse_tool_call(r["completion"]), "got": sc["code"]} for r, sc in zip(rows, scores) if not sc["correct"]],
        }
        (TARGET_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report["unused_held_out"]

    return TARGET_DIR, TARGET_GUIDANCE_GRID, TARGET_SYSTEM, TARGET_TOOLS, crossfit_wrong_calls, rebuild_weak_target, score_decoders, score_low_bit, train_bad_adapter, score_saved_target, quick_target_run, run_target_experiments, run_target_pipeline, score_call, target_prompt


@app.cell
def _(MODEL_FOLDERS, PLACEMENTS, TARGET_GUIDANCE_GRID, TangleLatex, fader, knob, mo):
    # Not shown: a folder under $MODELS_DIR, or a path. Change it here to train another checkpoint.
    target_model_path = mo.ui.text(value=MODEL_FOLDERS["target"])
    # smoke: 8 rows, 4 held-out requests (proves the code path) · standard: all rows, 2 epochs, 40 held-out requests,
    # guidance comparison · protocol: sweep, placement ablation, then the full pipeline
    target_preset = mo.ui.dropdown(options=["smoke", "standard", "protocol"], value="smoke", label="Budget")
    target_placement = mo.ui.dropdown(options=list(PLACEMENTS), value="all_linear", label="Layers")
    target_lr = fader("Learning rate ×1e-4", 0.5, 8.0, 0.5, 2.0)
    target_floor = fader("Plausibility floor", 0.01, 0.5, 0.01, 0.1)
    target_rank = knob("Rank", 4, 32, 4, 8)
    target_alpha_mult = knob("Alpha ÷ rank", 1.0, 4.0, 0.5, 2.0)
    target_batch = knob("Batch", 1, 8, 1, 4)
    target_train_btn = mo.ui.run_button(label="▶")
    target_tangle = mo.ui.anywidget(
        TangleLatex(
            latex=r"\log p \;=\; \log p_{\text{good}} + \tangle{w}\,\bigl(\log p_{\text{good}} - \log p_{\text{other}}\bigr)",
            parameters={"w": {"value": 0.5, "min_value": min(TARGET_GUIDANCE_GRID), "max_value": max(TARGET_GUIDANCE_GRID), "step": 0.25, "digits": 2}},
        )
    )
    return (
        target_alpha_mult,
        target_batch,
        target_floor,
        target_lr,
        target_model_path,
        target_placement,
        target_preset,
        target_rank,
        target_tangle,
        target_train_btn,
    )


@app.cell
async def _(
    TARGET_DIR,
    dataset,
    fval,
    grader_model_path,
    ival,
    json,
    mlx_thread,
    mo,
    model_path,
    run_target_experiments,
    run_target_pipeline,
    target_alpha_mult,
    target_batch,
    target_floor,
    target_lr,
    target_model_path,
    target_placement,
    target_preset,
    target_rank,
    target_tangle,
    target_train_btn,
    shutil,
    time,
):
    target_error, _log = None, []
    if target_train_btn.value:
        _w, _f = float(target_tangle.widget.values["w"]) or 0.5, fval(target_floor)
        _base = {
            "model_path": model_path(target_model_path.value), "grader_model_path": model_path(grader_model_path.value),
            "batch": ival(target_batch), "seed": 73, "probe_share": 0.15, "max_new": 220,
        }
        _own = {"rank": ival(target_rank), "alpha": ival(target_rank) * fval(target_alpha_mult), "lr": fval(target_lr) * 1e-4, "placement": target_placement.value}
        _arms = [["P_bad", 0.25, _f], ["P_bad", _w, _f], ["P_bad", 2 * _w, _f], ["P_bad", _w, 0.03], ["P_bad", _w, 0.3], ["weak P_good", _w, _f], ["weak P_good", 2 * _w, _f], ["base", _w, _f]]
        _smoke = None
        with mo.status.spinner(title="Training the target…"):
            try:
                if target_preset.value != "smoke" and TARGET_DIR.exists():  # a real run replaces the saved adapters: keep the old ones
                    shutil.copytree(TARGET_DIR, TARGET_DIR.parent / "previous" / f"target-{int(time.time())}")
                if target_preset.value == "protocol":
                    _sizes = {
                        "ranks": [8, 16], "learning_rates": [1e-4, 2e-4, 4e-4], "n_sweep": 200, "n_sweep_eval": 24,
                        "full": {"n_good": 10_000, "n_bad": 160, "n_probe": 48, "n_eval": 40, "n_loss": 24, "epochs": 2, "guidance": _arms},
                    }
                    await mlx_thread(run_target_experiments, _base, dataset, _sizes, _log.append)
                elif target_preset.value == "standard":
                    _cfg = {**_base, **_own, "n_good": 10_000, "n_bad": 160, "n_probe": 48, "n_eval": 40, "n_loss": 24, "epochs": 2, "guidance": _arms}
                    await mlx_thread(run_target_pipeline, _cfg, dataset, _log.append)
                else:
                    _cfg = {**_base, **_own, "n_good": 8, "n_bad": 8, "n_probe": 4, "n_eval": 4, "n_loss": 4, "epochs": 1, "guidance": [["P_bad", _w, _f], ["weak P_good", _w, _f]]}
                    # the smoke budget only proves the code path; it is written to runs/smoke, never over a real adapter
                    _smoke = await mlx_thread(run_target_pipeline, {**_cfg, "out_dir": str(TARGET_DIR.parent / "smoke")}, dataset, _log.append)
            except Exception as _exc:
                target_error = f"{type(_exc).__name__}: {_exc}"
    _file = TARGET_DIR / "report.json"
    target_report = json.loads(_file.read_text(encoding="utf-8")) if _file.exists() else None
    if target_train_btn.value and not target_error and target_preset.value == "smoke" and _smoke:
        target_report = _smoke
    _file = TARGET_DIR / "experiments.json"
    target_experiments = json.loads(_file.read_text(encoding="utf-8")) if _file.exists() else None
    target_ran = target_preset.value if target_train_btn.value and not target_error else None
    return target_error, target_experiments, target_ran, target_report


@app.cell
def _(
    alt,
    dataset,
    math,
    mo,
    pd,
    row,
    space,
    target_alpha_mult,
    target_batch,
    target_error,
    target_experiments,
    target_floor,
    target_lr,
    target_placement,
    target_preset,
    target_rank,
    target_report,
    target_tangle,
    target_train_btn,
    guidance_tokens,
    fval,
):
    _pct = lambda v: "–" if v is None else f"{v:.0%}"
    _notes, _charts, _guidance = [], [], []
    if target_error:
        _notes.append(mo.callout(mo.md(f"The pipeline failed: `{target_error}`"), kind="danger"))
    if target_report:
        _r = target_report
        if _r["dataset"] != dataset["fingerprint"]:
            _notes.append(mo.callout(mo.md("Trained on a different dataset than the one built above."), kind="warn"))
        _loss = mo.ui.altair_chart(
            alt.Chart(pd.DataFrame(_r["history"]))
            .mark_line()
            .encode(x=alt.X("step:Q", title="Update"), y=alt.Y("loss:Q", title=None), color=alt.Color("adapter:N", title=None, legend=alt.Legend(orient="bottom")))
            .properties(title="Loss", height=200, width="container")
        )
        if target_experiments:
            _e = target_experiments
            _row = lambda r: {
                "configuration": r["config"]["name"], "LoRA on": r["config"]["placement"], "rank": r["config"]["rank"], "learning rate": r["config"]["lr"],
                "rows": r["fit_rows"], "dev loss": round(r["dev_loss"], 3), "runs": _pct(r["runs"]), "returns the reference value": _pct(r["correct"]),
                "parameters": r["trainable_parameters"], "seconds": r["seconds"],
            }
            _loss = mo.ui.tabs(
                {
                    "Loss": _loss,
                    "Sweep": mo.ui.table(pd.DataFrame([_row(r) for r in _e["sweep"]]), selection=None, pagination=False),
                    "LoRA layers": mo.ui.table(pd.DataFrame([_row(r) for r in _e["placement_ablation"]] + [_row(r) for r in _e["sweep"] if r["config"]["name"] == _e["sweep_best"]]), selection=None, pagination=False),
                }
            )
        # where the grader's confusion matrix sits in the D1 panel: what execution says about every decoder
        _scores = pd.DataFrame([{"decoder": name, "share": v["correct"]} for name, v in _r["execution"].items()])
        _bars = mo.ui.altair_chart(
            alt.Chart(_scores)
            .mark_bar()
            .encode(
                y=alt.Y("decoder:N", title=None, sort=None, axis=alt.Axis(labelLimit=190)),
                x=alt.X("share:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=5, labelOverlap=True)),
                tooltip=["decoder", alt.Tooltip("share:Q", format=".0%")],
            )
            .properties(
                title=alt.TitleParams(
                    "Executed: returns the reference value",
                    subtitle=[
                        f"{_r['rows']['eval']} held-out requests"
                        + (f" · P_good on the other {_r['unused_held_out']['rows']}: {_r['unused_held_out']['correct']:.0%}" if _r.get("unused_held_out") else ""),
                        f"P_bad: {_r['rows']['bad_own']} wrong P_good calls + {_r['rows']['bad_synthetic']} generated breaks",
                    ],
                    anchor="start",
                ),
                height=max(140, 22 * len(_scores)), width="container",
            )
        )
        _charts = [_loss, mo.ui.tabs({"Execution": _bars, "Calls": mo.ui.table(pd.DataFrame(_r["samples"]), selection=None, page_size=4)})]
        # Guidance, as pictures that follow the dragged weight and the floor fader.
        _w, _f = float(target_tangle.widget.values["w"]), fval(target_floor)
        _named = {"P_bad": "P_bad", "bad": "P_bad", "weak P_good": "weak P_good", "weak": "weak P_good", "base": "base"}
        _visuals = []
        _grid = _r.get("reference_logp") or {}
        if _grid:
            _lines = pd.DataFrame([{"away from": _named.get(other, other), "weight": float(k), "log p": v} for other, table in _grid.items() for k, v in table.items()])
            _visuals.append(
                (
                    alt.Chart(_lines).mark_line(point=True).encode(
                        x=alt.X("weight:Q", title="weight w"), y=alt.Y("log p:Q", title="log p of the reference call, per token"),
                        color=alt.Color("away from:N", legend=alt.Legend(orient="bottom", title=None)),
                    )
                    + alt.Chart(pd.DataFrame({"weight": [_w]})).mark_rule(strokeDash=[4, 3]).encode(x="weight:Q")
                ).properties(height=220, width="container")
            )
        if guidance_tokens and guidance_tokens.get("margin"):
            _limit = math.log(1 / _f)
            _leads = pd.DataFrame({"lead": [min(19.75, m) for m in guidance_tokens["margin"]]})
            _leads["within reach"] = _leads["lead"] <= _limit
            _visuals.append(
                (
                    alt.Chart(_leads).mark_bar().encode(
                        x=alt.X("lead:Q", bin=alt.Bin(step=0.5), title="P_good's lead over its runner-up (nats)"), y=alt.Y("count():Q", title="tokens"),
                        color=alt.Color("within reach:N", legend=None, scale=alt.Scale(domain=[False, True], range=["#2a78d6", "#eb6834"])),
                    )
                    + alt.Chart(pd.DataFrame({"lead": [_limit]})).mark_rule(strokeDash=[4, 3]).encode(x="lead:Q")
                ).properties(height=220, width="container")
            )
            _key = f"{min((0.25, 0.5, 1.0), key=lambda k: abs(k - _w))}|{min((0.03, 0.1, 0.3), key=lambda k: abs(k - _f))}"
            _moves = pd.DataFrame(
                [
                    {"away from": _named.get(other, other), "token moved": kind, "tokens": rec["flips"][_key][field]}
                    for other, rec in guidance_tokens["against"].items()
                    for kind, field in (("onto the reference", "onto_reference"), ("away from it", "away_from_reference"))
                ]
            )
            _visuals.append(
                alt.Chart(_moves).mark_bar().encode(
                    y=alt.Y("away from:N", title=None), x=alt.X("tokens:Q", title=f"tokens guidance changes, of {guidance_tokens['tokens']:,}", axis=alt.Axis(tickMinStep=1)),
                    color=alt.Color("token moved:N", legend=alt.Legend(orient="bottom", title=None), scale=alt.Scale(domain=["onto the reference", "away from it"], range=["#1baf7a", "#e34948"])),
                ).properties(height=220, width="container")
            )
        _guidance = [space(24), mo.md("## Guidance"), target_tangle] + ([row(*[(chart, 1, 300) for chart in _visuals])] if _visuals else [])
    mo.vstack(
        [
            space(56, "step-target"),
            mo.md("## Train LFM"),
            *_notes,
            mo.hstack(
                [target_lr, target_floor, target_rank, target_alpha_mult, target_batch, mo.vstack([target_train_btn, target_placement, target_preset], gap=0.75)],
                justify="start", align="end", gap=1.5, wrap=True,
            ),
            *([row(*[(chart, 1, 380) for chart in _charts])] if _charts else []),
            *_guidance,
        ],
        gap=1.5,
    )
    return


@app.cell
def _(parse_tool_call, Path, ROOT, TARGET_DIR, gc, json, shutil, time):
    # -------------------------------------------------------------------------
    # Step 8: export for the browser. P_good is added to the 4-bit ONNX build of the target that
    # Transformers.js runs on WebGPU, then the result is checked by generating with ONNX Runtime.
    # -------------------------------------------------------------------------
    BROWSER_DIR = ROOT / "webgpu-demo"
    BASE_ONNX_DIR = BROWSER_DIR / "models" / "onnx-community" / "LFM3-1.7B-ONNX"
    EXPORT_ONNX_DIR = BROWSER_DIR / "models" / "nushell-lfm3-1.7b"
    # (name in the ONNX graph, name in the MLX model); the graph keeps the attention gate as the last rows of q_proj
    _LAYERS = [
        ("conv_in_proj", "conv.in_proj"), ("conv_out_proj", "conv.out_proj"),
        ("mlp_gate_proj", "feed_forward.w1"), ("mlp_up_proj", "feed_forward.w3"), ("mlp_down_proj", "feed_forward.w2"),
        ("attn_q_proj", "self_attn.q_proj"), ("attn_k_proj", "self_attn.k_proj"),
        ("attn_v_proj", "self_attn.v_proj"), ("attn_o_proj", "self_attn.o_proj"),
    ]

    def export_for_browser(model_path: str, adapter_path, rank: int, alpha: float, log=print) -> dict:
        """Put a LoRA adapter into the 4-bit ONNX graph without touching the quantised weights: beside each
        4-bit matrix product y = x·W, add the low-rank branch (x·A)·B and sum the two. The adapter stays in
        half precision and exact; re-quantising W + A·B to 4 bits instead loses much of the small update.
        Runs on the MLX thread."""
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        import onnx
        from mlx_vlm import load as vlm_load
        from onnx import helper, numpy_helper

        started = time.monotonic()
        graph = onnx.load(str(BASE_ONNX_DIR / "onnx" / "model_q4f16.onnx"))
        tensors = {t.name: t for t in graph.graph.initializer}
        producers = {n.input[1]: n for n in graph.graph.node if n.op_type == "MatMulNBits"}
        adapter = {k: np.array(v.astype(mx.float32)) for k, v in mx.load(str(adapter_path)).items()}

        def nibbles(packed):  # two 4-bit values per byte, low half first
            return np.stack([packed & 15, packed >> 4], -1).reshape(*packed.shape[:-1], -1)

        def read(prefix):
            """The quantised weight as floats, [outputs, inputs]. Only used to line rows up with the MLX model."""
            packed = numpy_helper.to_array(tensors[prefix + "_quant"])
            scales = numpy_helper.to_array(tensors[prefix + "_scales"]).astype(np.float32)
            rows, blocks, _ = packed.shape
            zero = nibbles(numpy_helper.to_array(tensors[prefix + "_zp"]))[:, :blocks] if prefix + "_zp" in tensors else np.full((rows, blocks), 8)
            return ((nibbles(packed).astype(np.float32) - zero[..., None]) * scales[..., None]).reshape(rows, -1)

        def factors(layer: int, name: str):
            """A [inputs, rank] and B [rank, outputs], with the LoRA scale folded into B."""
            return adapter[f"model.layers.{layer}.{name}.lora_a"], (alpha / rank) * adapter[f"model.layers.{layer}.{name}.lora_b"]

        def row_order(onnx_rows, mlx_rows, head: int):
            """The ONNX build stores the outputs of q_proj and k_proj in a different order inside each attention
            head. Recover it from the first head by matching rows, and check it on the whole weight."""
            unit = lambda m: (m - m.mean(1, keepdims=True)) / (np.linalg.norm(m - m.mean(1, keepdims=True), axis=1, keepdims=True) + 1e-9)
            inside = (unit(onnx_rows[:head]) @ unit(mlx_rows[:head]).T).argmax(1)
            order = np.concatenate([inside + start for start in range(0, mlx_rows.shape[0], head)])
            match = float(np.mean(np.sum(unit(onnx_rows) * unit(mlx_rows[order]), axis=1)))
            if sorted(inside.tolist()) != list(range(head)) or match < 0.95:
                raise ValueError(f"could not align attention rows (match {match:.2f})")
            return order

        model, _processor = vlm_load(model_path)
        try:
            head = getattr(model.language_model.config, "head_dim", 128)
            added, extra = 0, []
            for index, block in enumerate(model.language_model.layers):
                modules = dict(block.named_modules())
                for onnx_name, mlx_name in _LAYERS:
                    prefix = f"model_layers_{index}_{onnx_name}_MatMul_weight"
                    node = producers.get(prefix + "_quant")
                    if node is None or f"model.layers.{index}.{mlx_name}.lora_a" not in adapter:
                        continue
                    a, b = factors(index, mlx_name)
                    if onnx_name in ("attn_q_proj", "attn_k_proj"):
                        layer = modules[mlx_name]
                        base = np.array(mx.dequantize(layer.weight, layer.scales, layer.biases, layer.group_size, layer.bits).astype(mx.float32)) if isinstance(layer, nn.QuantizedLinear) else np.array(layer.weight.astype(mx.float32))
                        weight = read(prefix)
                        b = b[:, row_order(weight[: base.shape[0]], base, head)]
                        if onnx_name == "attn_q_proj" and weight.shape[0] > b.shape[1] and f"model.layers.{index}.self_attn.gate_proj.lora_a" in adapter:
                            # the graph computes the attention gate as extra outputs of q_proj: stack the two adapters
                            gate_a, gate_b = factors(index, "self_attn.gate_proj")
                            a = np.concatenate([a, gate_a], axis=1)
                            b = np.block([[b, np.zeros((b.shape[0], gate_b.shape[1]), np.float32)], [np.zeros((gate_b.shape[0], b.shape[1]), np.float32), gate_b]])
                    name = f"lora_{index}_{onnx_name}"
                    graph.graph.initializer.extend(
                        [numpy_helper.from_array(a.astype(np.float16), name + "_A"), numpy_helper.from_array(b.astype(np.float16), name + "_B")]
                    )
                    output = node.output[0]
                    node.output[0] = output + "_base"
                    extra.append(
                        (
                            node.name,
                            [
                                helper.make_node("MatMul", [node.input[0], name + "_A"], [name + "_down"], name=name + "/down"),
                                helper.make_node("MatMul", [name + "_down", name + "_B"], [name + "_up"], name=name + "/up"),
                                helper.make_node("Add", [output + "_base", name + "_up"], [output], name=name + "/add"),
                            ],
                        )
                    )
                    added += 1
        finally:
            del model, _processor
            gc.collect()
            mx.clear_cache()

        after = dict(extra)  # keep the graph in execution order: each branch right after the product it joins
        nodes = []
        for node in graph.graph.node:
            nodes.append(node)
            nodes.extend(after.get(node.name, []))
        del graph.graph.node[:]
        graph.graph.node.extend(nodes)

        if EXPORT_ONNX_DIR.exists():
            shutil.rmtree(EXPORT_ONNX_DIR)
        (EXPORT_ONNX_DIR / "onnx").mkdir(parents=True)
        onnx.save(
            graph, str(EXPORT_ONNX_DIR / "onnx" / "model_q4f16.onnx"), save_as_external_data=True,
            all_tensors_to_one_file=True, location="model_q4f16.onnx_data", size_threshold=1024,
        )
        for item in BASE_ONNX_DIR.iterdir():
            if item.is_file():
                shutil.copyfile(item, EXPORT_ONNX_DIR / item.name)
        shutil.copyfile(TARGET_DIR / "prompt_format.json", BROWSER_DIR / "prompt_format.json")
        summary = {
            "adapted_weights": added,
            "seconds": round(time.monotonic() - started, 1),
            "size_mb": round(sum(f.stat().st_size for f in (EXPORT_ONNX_DIR / "onnx").iterdir()) / 1e6),
        }
        log(f"[export] added the adapter beside {added} weights, {summary['size_mb']} MB")
        return summary

    def check_browser_export(rows: list[dict], score_call, target_prompt, max_new: int = 220, model_dir=None, log=print) -> dict:
        """Generate with the exported ONNX model on ONNX Runtime, using the tokenizer and chat template that
        ship in the export, and execute each call. This is the same graph and prompt the browser runs."""
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_dir = Path(model_dir or EXPORT_ONNX_DIR)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        session = ort.InferenceSession(str(model_dir / "onnx" / "model_q4f16.onnx"), providers=["CPUExecutionProvider"])
        past_names = [i.name for i in session.get_inputs() if i.name.startswith("past")]
        shapes = {i.name: i.shape for i in session.get_inputs()}
        outputs = [o.name for o in session.get_outputs()]
        stop = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
        results = []
        for row in rows:
            ids = tokenizer.encode(target_prompt(tokenizer, row), add_special_tokens=False)
            past = {name: np.zeros([1 if isinstance(d, str) and "batch" in d else 0 if isinstance(d, str) else d for d in shapes[name]], np.float16) for name in past_names}
            feed, written, seen = np.array([ids], np.int64), [], len(ids)
            for _ in range(max_new):
                out = session.run(outputs, {"input_ids": feed, "attention_mask": np.ones((1, seen), np.int64), "num_logits_to_keep": np.array(1, np.int64), **past})
                token = int(out[0][0, -1].argmax())
                past = {name.replace("present_conv", "past_conv").replace("present.", "past_key_values."): value for name, value in zip(outputs[1:], out[1:])}
                if token in stop:
                    break
                written.append(token)
                if tokenizer.decode(written).rstrip().endswith("</tool_call>"):
                    break
                feed, seen = np.array([[token]], np.int64), seen + 1
            text = tokenizer.decode(written)
            results.append({"request": row["speech"], "output": text.strip(), **score_call(text, row)})
        rate = lambda key: sum(r[key] for r in results) / max(1, len(results))
        summary = {"requests": len(results), "well_formed": rate("well_formed"), "runs": rate("runs"), "correct": rate("correct"), "samples": results}
        log(f"[export] ONNX Runtime on {len(results)} held-out requests: {summary['correct']:.0%} return the reference value")
        return summary

    def write_browser_requests(rows: list[dict], environments: list[dict]) -> None:
        """What the demo page needs besides the model: each held-out environment's files and requests."""
        projects = []
        for environment in environments:
            own = [r for r in rows if r["env"] == environment["id"]]
            if own:
                projects.append(
                    {
                        "id": environment["id"],
                        "title": environment["title"],
                        "context": own[0]["context"],
                        "files": environment["files"],
                        # `code` is the reference call: the page runs it in its own Nushell to get the value to compare with,
                        # because directory order and directory sizes differ between a real disk and the browser.
                        "requests": [
                            {"request": r["speech"], "references": r["references"], "expected": r["expected"], "code": parse_tool_call(r["completion"])}
                            for r in own
                        ],
                    }
                )
        (BROWSER_DIR / "environments.json").write_text(json.dumps(projects, indent=1), encoding="utf-8")

    return BASE_ONNX_DIR, BROWSER_DIR, EXPORT_ONNX_DIR, check_browser_export, export_for_browser, write_browser_requests


@app.cell
async def _(
    BASE_ONNX_DIR,
    BROWSER_DIR,
    check_browser_export,
    dataset,
    environments,
    export_for_browser,
    json,
    mlx_thread,
    mo,
    model_path,
    score_call,
    target_model_path,
    target_prompt,
    target_ran,
    target_report,
    write_browser_requests,
):
    # After a real training run, P_good is added to the 4-bit ONNX build of the target as low-rank branches beside
    # each quantised weight, the exported graph generates on ONNX Runtime, and its calls are executed. The result is
    # written to webgpu-demo/export.json. `mise run experiments -- --only export` does the same from the command line.
    export_error = None
    if target_ran in ("standard", "protocol") and target_report and (BASE_ONNX_DIR / "onnx" / "model_q4f16.onnx").exists():
        _rows = sorted(dataset["target"]["dev"], key=lambda r: (r["id"].rsplit("/", 1)[1], r["id"]))[:24]
        _cfg = target_report["config"]

        def _export():
            merged = export_for_browser(model_path(target_model_path.value), BROWSER_DIR.parent / "runs" / "target" / "adapter_good.safetensors", _cfg["rank"], _cfg["alpha"], lambda line: None)
            checked = check_browser_export(_rows, score_call, target_prompt, _cfg["max_new"], log=lambda line: None)
            write_browser_requests(sorted(dataset["target"]["dev"], key=lambda r: r["id"]), environments)
            (BROWSER_DIR / "export.json").write_text(json.dumps({"merge": merged, "check": checked, "mlx_P_good": target_report["execution"]["P_good"]}, indent=1), encoding="utf-8")

        with mo.status.spinner(title="Exporting the target for the browser and checking it…"):
            try:
                await mlx_thread(_export)
            except Exception as _exc:
                export_error = f"{type(_exc).__name__}: {_exc}"
    return (export_error,)


@app.cell
def _(BROWSER_DIR, ROOT):
    # The demo page and the WebAssembly build of Nushell are static files, served from inside the notebook so
    # that neither the REPL above nor "Load Nu" needs another process.
    import http.server
    import socket
    import threading

    DEMO_PORT = 8377

    class _DemoFiles(http.server.SimpleHTTPRequestHandler):
        extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".mjs": "text/javascript", ".jinja": "text/plain"}

        # The folders are kept on the class: the server outlives the browser session that started it, and by
        # then this cell's own names are gone.
        nu_dir, demo_dir = str(ROOT / "public" / "nu-wasm"), str(BROWSER_DIR)

        def translate_path(self, path):
            path = path.split("?", 1)[0].split("#", 1)[0]
            if path.startswith("/nu-wasm/"):
                return self.nu_dir + "/" + path[len("/nu-wasm/"):]
            return self.demo_dir + "/" + (path.lstrip("/") or "index.html")

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")  # the notebook page loads the WebAssembly build from here
            self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def log_message(self, *args):
            pass

    def _port_open(port: int) -> bool:
        with socket.socket() as probe:
            probe.settimeout(0.3)
            return probe.connect_ex(("127.0.0.1", port)) == 0

    demo_serving = _port_open(DEMO_PORT)
    if not demo_serving:
        try:
            _server = http.server.ThreadingHTTPServer(("127.0.0.1", DEMO_PORT), _DemoFiles)
            threading.Thread(target=_server.serve_forever, daemon=True).start()
            demo_serving = True
        except OSError:
            demo_serving = False
    return DEMO_PORT, demo_serving


@app.cell
def _(BASE_ONNX_DIR, BROWSER_DIR, DEMO_PORT, EXPORT_ONNX_DIR, demo_serving, export_error, mo, space):
    # Opens the finished model on WebGPU and a Nushell session over a held-out environment, full window, in a new tab.
    _notes = []
    if export_error:
        _notes.append(f"Export failed: `{export_error}`")
    if not (BASE_ONNX_DIR / "onnx" / "model_q4f16.onnx").exists():
        _notes.append("The ONNX build of the base model is missing: run `mise run fetch-onnx`.")
    elif not ((EXPORT_ONNX_DIR / "onnx" / "model_q4f16.onnx").exists() and (BROWSER_DIR / "environments.json").exists()):
        _notes.append("No exported model yet: train the target at the standard or protocol budget.")
    if not (BROWSER_DIR / "node_modules" / "@huggingface" / "transformers").exists():
        _notes.append("Transformers.js is not installed: run `mise run setup`.")
    if not demo_serving:
        _notes.append(f"Port {DEMO_PORT} is taken by something else, so the demo cannot be served from here.")
    mo.vstack(
        [
            space(56, "step-nu"),
            mo.Html(
                f"""
                <a href="http://localhost:{DEMO_PORT}/?load=1" target="_blank" rel="noopener"
                   style="display:block; text-align:center; padding: clamp(18px, 4vw, 34px) 16px; margin: 0 0 56px; font: 700 clamp(1.5rem, 5vw, 2.4rem)/1.1 system-ui, sans-serif; text-decoration:none; color:inherit;
                          border: 3px solid currentColor; border-radius: 14px; background: color-mix(in srgb, currentColor 6%, transparent);">
                  Load Nu!
                </a>
                """
            ),
        ]
        + ([mo.callout(mo.md(" ".join(_notes)), kind="warn")] if _notes else [])
    )
    return


@app.cell
def _(fader, mo):
    from wigglystuff import EnvConfig

    # Up to three OpenAI-compatible endpoints. Values start from the environment when they are set there
    # (mise.local.toml is a good place); what is typed here stays in this session and is never written to disk.
    PROVIDER_SLOTS = 3
    provider_config = mo.ui.anywidget(EnvConfig([f"PROVIDER_{i}_{part}" for i in range(1, PROVIDER_SLOTS + 1) for part in ("URL", "KEY", "MODELS")]))
    provider_requests = fader("Requests", 20, 240, 20, 40)
    provider_run_btn = mo.ui.run_button(label="▶")
    return PROVIDER_SLOTS, provider_config, provider_requests, provider_run_btn


@app.cell
async def _(
    PROVIDER_SLOTS,
    ROOT,
    TARGET_SYSTEM,
    TARGET_TOOLS,
    alt,
    asyncio,
    dataset,
    ival,
    mo,
    parse_tool_call,
    pd,
    provider_config,
    provider_requests,
    provider_run_btn,
    row,
    run_nu,
    same_output,
    space,
):
    # Other models on the same held-out requests: same system prompt, tool and user text, and the same rule
    # (the call is executed and must return the reference value). scripts/eval_endpoint.py does the work, so
    # `mise run eval-endpoint` gives the same numbers. PROVIDER_n_MODELS is a comma-separated list.
    import importlib.util as _util

    _spec = _util.spec_from_file_location("eval_endpoint", ROOT / "scripts" / "eval_endpoint.py")
    _eval = _util.module_from_spec(_spec)
    _spec.loader.exec_module(_eval)
    _held_out = len(dataset["target"]["dev"])
    _limit = min(ival(provider_requests), _held_out)
    _notes = []
    if provider_run_btn.value:
        _defs = {
            "dataset": dataset, "TARGET_SYSTEM": TARGET_SYSTEM, "TARGET_TOOLS": TARGET_TOOLS,
            "parse_tool_call": parse_tool_call, "run_nu": run_nu, "same_output": same_output,
        }
        _jobs = []
        for _i in range(1, PROVIDER_SLOTS + 1):
            _url, _models = provider_config.widget.get(f"PROVIDER_{_i}_URL"), provider_config.widget.get(f"PROVIDER_{_i}_MODELS")
            if _url and _models:
                _jobs += [(_url, _m.strip(), provider_config.widget.get(f"PROVIDER_{_i}_KEY") or "") for _m in _models.split(",") if _m.strip()]
        if not _jobs:
            _notes.append("Nothing to run: set a URL and at least one model name for a provider.")
        for _url, _model, _key in _jobs:
            with mo.status.spinner(title=f"Asking {_model} for {_limit} calls…"):
                try:
                    _saved = await asyncio.to_thread(_eval.run_endpoint, _defs, _url, _model, _key, limit=None if _limit >= _held_out else _limit, log=lambda *_: None)
                    _failed = [r for r in _saved["rows"] if "error" in r]
                    if _failed:
                        _notes.append(f"`{_model}`: {len(_failed)} of {len(_saved['rows'])} requests got no answer. First error: `{_failed[0]['error'][:240]}`")
                except Exception as _exc:
                    _notes.append(f"`{_model}`: {type(_exc).__name__}: {_exc}")
    _results = _eval.saved_results()
    _table = [{"model": r["name"], **r["summary"], "ours": False} for r in _results if r.get("dataset") == dataset["fingerprint"]]
    _stale = len(_results) - len(_table)
    for _n in sorted({r["requests"] for r in _table} | {_held_out}):
        _mine = _eval.ours(None if _n >= _held_out else _n)
        if _mine:
            _table.append({"model": "LFM3 1.7B + P_good (this notebook)", **_mine, "ours": True})
    _views = [
        space(56, "step-providers"),
        mo.md("## Other models"),
        row((provider_config, 0, 520), (mo.vstack([provider_requests, provider_run_btn], align="center", gap=1), 0, 150)),
    ]
    if _notes:
        _views.append(mo.callout(mo.md("\n\n".join(_notes)), kind="warn"))
    if _stale:
        _views.append(mo.md(f"{_stale} saved results are for a different dataset and are not shown."))
    if _table:
        _frame = pd.DataFrame(_table)
        _frame["label"] = _frame["model"] + " · " + _frame["requests"].astype(str)
        _views.append(
            alt.Chart(_frame)
            .mark_bar()
            .encode(
                y=alt.Y("label:N", title=None, sort="-x", axis=alt.Axis(labelLimit=260)),
                x=alt.X("correct:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=5, labelOverlap=True)),
                color=alt.Color("ours:N", legend=None, scale=alt.Scale(domain=[False, True], range=["#64748b", "#f59e0b"])),
                tooltip=["model", "requests", alt.Tooltip("correct:Q", format=".1%"), alt.Tooltip("runs:Q", format=".0%"), alt.Tooltip("well_formed:Q", format=".0%")],
            )
            .properties(
                title=alt.TitleParams("Executed: returns the reference value", subtitle=["same held-out requests, prompt and tool; label ends with the number of requests"], anchor="start"),
                height=max(80, 26 * len(_frame)), width="container",
            )
        )
        _pct = lambda v: f"{v:.0%}"
        _views.append(
            mo.md(
                "| Model | Requests | Well-formed | Runs | Correct | s / request | No answer |\n|:---|:---|:---|:---|:---|:---|:---|\n"
                + "\n".join(
                    f"| {r['model']} | {r['requests']} | {_pct(r['well_formed'])} | {_pct(r['runs'])} | {r['correct']:.1%} | {r.get('seconds_per_request') or ''} | {r.get('errors') or ''} |"
                    for r in sorted(_table, key=lambda r: -r["correct"])
                )
            )
        )
    mo.vstack(_views, gap=1.25)
    return


if __name__ == "__main__":
    app.run()
