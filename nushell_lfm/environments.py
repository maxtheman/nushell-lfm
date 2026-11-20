"""Generated projects: files, logs, tables and configs from a seed, and the sandboxed Nushell runner."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile

from .core import RUNS_DIR, nu_path
from .core import IN_BROWSER

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
_files = {}  # environment id -> (its files, the files of its second copy)

def build_environments(count: int) -> list[dict]:
    """Generate `count` environments and write each to disk twice: as it is, and as its second copy."""
    environments = []
    for seed in range(1, count + 1):
        layout = environment_layout(seed)
        files, variant = environment_files(layout), environment_files(layout, 1, extra=True)
        _files[layout["id"]] = (files, variant)
        if not IN_BROWSER:  # in the browser the files are handed to the WebAssembly Nushell from memory
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
    if _browser_nu is not None:
        result = _run_in_browser(code, env, variant)
        if cache:
            _cache[key] = result
        return result
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

_browser_nu = None

def use_browser_nu(module) -> None:
    """Run Nushell through its WebAssembly build (the `nu_wasm.js` module, already initialised) whenever a
    result is not in the cache. The build reports the same release as the native binary the cache was made with."""
    global _browser_nu, _nu_version
    _browser_nu, _nu_version = module, str(module.nu_version())

def _run_in_browser(code: str, env: str, variant: bool) -> dict:
    _browser_nu.set_files(json.dumps(_files[env][1 if variant else 0]))
    out = json.loads(str(_browser_nu.run(code)))
    return {
        "ok": bool(out["ok"]), "returncode": 0 if out["ok"] else 1, "stdout": out["stdout"].strip(), "stderr": out["stderr"].strip(),
        "parse_error": bool(out["parse_error"]), "changed": sorted(out.get("changed", [])),
    }

def run_nu_many(jobs: list[tuple]) -> list[dict]:
    """`(code, env)` or `(code, env, variant)` jobs, run eight at a time; results in order."""
    if IN_BROWSER:  # no threads there
        return [run_nu(*job) for job in jobs]
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
