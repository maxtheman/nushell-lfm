"""Requests written by the base model, the splits by environment, and the dataset on disk."""

from collections import Counter
import gc
import hashlib
import json
import random
import re

from .core import RUNS_DIR
from .catalog import COMMAND_CATALOG

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
