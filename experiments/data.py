"""The dataset the grader and guidance experiments were run on: the target rows plus, for the grader, every
chain's broken and equivalent variants. Its fingerprint (edd14fe576df with the default settings) is the one in
the saved experiment results."""

from collections import Counter
import hashlib
import json
import random

from nushell_lfm.catalog import COMMAND_CATALOG
from nushell_lfm.dataset import composed_requests, load_model_requests
from nushell_lfm.pipeline import DEFAULTS, build

from breaks import build_samples

_CATALOG = {c["name"]: c for c in COMMAND_CATALOG}

def assemble_full_dataset(environments: list[dict], chains: list[dict], samples: list[dict], k: int, model_requests: dict, seed: int = 73) -> dict:
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

def build_full(breaks_per_chain: int = 4, **settings) -> dict:
    """The shipped pipeline's environments and chains, with the full dataset in place of the target-only one."""
    built = build(**settings)
    k = {**DEFAULTS, **settings}["requests_per_chain"]
    samples = build_samples(built["chains"], breaks_per_chain)
    return {**built, "samples": samples, "dataset": assemble_full_dataset(built["environments"], built["chains"], samples, k, load_model_requests(k))}
