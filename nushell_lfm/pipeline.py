"""The data pipeline in one call, for scripts: environments → chains → requests → splits.
The notebook runs the same functions cell by cell, with these values as its starting controls."""

from .catalog import ALL_CATEGORIES
from .chains import PRIOR_INFO, build_chains
from .dataset import assemble_dataset, load_model_requests
from .environments import build_environments

DEFAULTS = {"environments": 20, "chains": 400, "ported_share": 0.25, "length_bias": 0.5, "requests_per_chain": 3, "categories": list(ALL_CATEGORIES)}


def build(**settings) -> dict:
    """Build the dataset with `DEFAULTS`, overridden by `settings`. Returns the environments, the verified
    chains, the dataset and the settings used (which `save_dataset` writes into the manifest)."""
    s = {**DEFAULTS, **settings}
    environments = build_environments(s["environments"])
    chains, _dropped, summary = build_chains(environments, s["categories"], s["chains"], s["ported_share"], s["length_bias"])
    dataset = assemble_dataset(environments, chains, s["requests_per_chain"], load_model_requests(s["requests_per_chain"]))
    return {"environments": environments, "chains": chains, "dataset": dataset, "settings": {**s, "priors": PRIOR_INFO, "chain_summary": summary}}
