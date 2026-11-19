#!/usr/bin/env python3
"""Score any OpenAI-compatible chat endpoint on the held-out requests.

Each held-out request is sent with the same system prompt, the same `run_nushell` tool and the same user text
the fine-tuned model sees. The call that comes back is executed in a throwaway copy of the request's
environment, and counts as correct only if it returns the reference value: the same rule as for our own model.

    export PROVIDER_API_KEY=...            # the key is read from the environment, never from a flag or a file
    mise run eval-endpoint -- --base-url https://api.example.com/v1 --model some-model --api-key-env PROVIDER_API_KEY

    mise run eval-endpoint -- --export     # only write the evaluation set to runs/eval/eval_set.jsonl
    mise run eval-endpoint -- --compare    # table of every endpoint scored so far, next to our model

Useful flags: --limit 40 (the 40 requests our model is compared with its base model on), --no-tools (for models without
tool calling: ask for the code in a fenced block), --extra '{"reasoning_effort": "low"}' (extra body fields),
--name (label for the result file), --concurrency, --max-tokens.

Requests are sent to the endpoint you name: the generated project file names and the request text leave this
machine. Nothing else does.
"""

import argparse
import concurrent.futures
import functools
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nushell_lfm.core import RUNS_DIR  # noqa: E402

OUT = RUNS_DIR / "providers"
RUN_LOCK = __import__("threading").Lock()
NO_TOOLS_NOTE = " There is no tool here: reply with only the Nushell source, in one ```nu fenced block."


def user_text(row: dict) -> str:
    """The user turn, exactly as `nushell_lfm.target.target_prompt` builds it."""
    refs = "\n\n".join(f"{doc['title']} (nushell)\n{doc['text']}" for doc in row.get("references", []))
    text = f"\nSyntax references:\n{refs}\n\n" if refs else "\n"
    return text + f"Request: {row['speech']}\nWorkspace: {row['context']}\nFiles: " + ", ".join(row["workspace_files"])


def load_defs() -> dict:
    """What scoring needs from the pipeline: the dataset, the prompt and tool, and how a call is run and compared."""
    from nushell_lfm.calls import parse_tool_call
    from nushell_lfm.environments import run_nu, same_output
    from nushell_lfm.pipeline import build
    from nushell_lfm.target import TARGET_SYSTEM, TARGET_TOOLS

    return {
        "dataset": build()["dataset"], "TARGET_SYSTEM": TARGET_SYSTEM, "TARGET_TOOLS": TARGET_TOOLS,
        "parse_tool_call": parse_tool_call, "run_nu": run_nu, "same_output": same_output,
    }


def build_cases(defs: dict, limit: int | None) -> list[dict]:
    """The held-out requests in the order the report uses, so --limit 40 is the set our model was compared with its base on."""
    dev = sorted(defs["dataset"]["target"]["dev"], key=lambda r: (r["id"].rsplit("/", 1)[1], r["id"]))
    cases = [
        {
            "id": row["id"], "env": row["env"], "category": row.get("category"), "style": row.get("style"),
            "messages": [{"role": "system", "content": defs["TARGET_SYSTEM"]}, {"role": "user", "content": user_text(row)}],
            "reference": defs["parse_tool_call"](row["completion"]), "expected": row["expected"],
        }
        for row in dev
    ]
    return cases[:limit] if limit else cases


def extract_code(message: dict, parse_tool_call) -> tuple[str | None, str]:
    """The Nushell source in a reply, and how it was given: a tool call, our text format, or a fenced block."""
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name") == "run_nushell":
            arguments = function.get("arguments")
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            except ValueError:
                return None, "tool call with arguments that are not JSON"
            if isinstance(arguments, dict) and isinstance(arguments.get("code"), str):
                return arguments["code"], "tool_call"
            return None, "tool call without a code argument"
    content = message.get("content") or ""
    if isinstance(content, list):  # some endpoints return content parts
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    found = parse_tool_call(content) or parse_tool_call(content.removeprefix("<tool_call>").strip())
    if found is not None:
        return found, "text_tool_call"
    block = re.search(r"```(?:nu|nushell|sh|shell|bash)?\s*\n(.*?)```", content, flags=re.S)
    if block:
        return block.group(1).strip(), "fenced_block"
    return None, "no call in the reply"


class Endpoint:
    def __init__(self, base_url: str, model: str, key: str, tools: list[dict], *, no_tools: bool = False, tool_choice: str = "function",
                 temperature: float = 0.0, max_tokens: int = 600, timeout: float = 120.0, extra: dict | None = None):
        base = base_url.strip().rstrip("/")
        self.url = base if base.endswith("/chat/completions") else base + "/chat/completions"
        self.key, self.model, self.timeout, self.tools, self.no_tools = key, model, timeout, tools, no_tools
        self.body = {"temperature": temperature, "max_tokens": max_tokens, **(extra or {})}
        if not no_tools:
            self.body["tool_choice"] = {"type": "function", "function": {"name": "run_nushell"}} if tool_choice == "function" else tool_choice

    def ask(self, messages: list[dict]) -> dict:
        """One chat completion. Retries rate limits and server errors; adapts once to endpoints that refuse
        `max_tokens`, `temperature` or a forced tool choice."""
        if self.no_tools:
            messages = [{**messages[0], "content": messages[0]["content"] + NO_TOOLS_NOTE}, *messages[1:]]
        body = {"model": self.model, "messages": messages, **self.body, **({} if self.no_tools else {"tools": self.tools})}
        headers = {"Content-Type": "application/json", "User-Agent": "nushell-eval/1.0"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        last = "no attempt made"
        for attempt in range(6):
            started = time.monotonic()
            try:
                request = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read())
                return {"data": data, "seconds": time.monotonic() - started}
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", "replace")[:600]
                last = f"HTTP {error.code}: {detail}"
                if error.code == 400:
                    low = detail.lower()
                    # Some reasoning models refuse function tools on this route unless reasoning is switched off.
                    if "reasoning_effort" in low and body.get("reasoning_effort") != "none":
                        body["reasoning_effort"] = self.body["reasoning_effort"] = "none"
                        continue
                    if "max_tokens" in low and "max_tokens" in body:
                        body["max_completion_tokens"] = body.pop("max_tokens")
                        self.body["max_completion_tokens"] = self.body.pop("max_tokens", body["max_completion_tokens"])
                        continue
                    if "temperature" in low and "temperature" in body:
                        body.pop("temperature"); self.body.pop("temperature", None)
                        continue
                    if "tool_choice" in low and isinstance(body.get("tool_choice"), dict):
                        body["tool_choice"] = self.body["tool_choice"] = "required"
                        continue
                    if "tool_choice" in low and body.get("tool_choice") == "required":
                        body["tool_choice"] = self.body["tool_choice"] = "auto"
                        continue
                if error.code in (408, 409, 425, 429) or error.code >= 500:
                    time.sleep(min(60, 2 ** attempt + float(error.headers.get("Retry-After") or 0)))
                    continue
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as error:
                last = f"{type(error).__name__}: {error}"
                time.sleep(min(30, 2 ** attempt))
        return {"error": last}


def score(case: dict, endpoint: Endpoint, defs: dict) -> dict:
    out = {"id": case["id"], "category": case["category"], "style": case["style"], "well_formed": False, "runs": False, "correct": False}
    answer = endpoint.ask(case["messages"])
    if "error" in answer:
        return {**out, "error": answer["error"]}
    data = answer["data"]
    choice = (data.get("choices") or [{}])[0]
    code, how = extract_code(choice.get("message") or {}, defs["parse_tool_call"])
    usage = data.get("usage") or {}
    out.update({
        "seconds": round(answer["seconds"], 2), "how": how, "code": code, "finish": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
    })
    if code is None:
        return out
    with RUN_LOCK:  # one Nushell process at a time: the runner keeps a cache file
        result = defs["run_nu"](code, case["env"])
    out.update({"well_formed": True, "runs": result["ok"], "correct": result["ok"] and defs["same_output"](result["stdout"], case["expected"])})
    return out


def run_endpoint(defs: dict, base_url: str, model: str, key: str, *, name: str | None = None, limit: int | None = None, concurrency: int = 4,
                 log=print, **options) -> dict:
    """Score one model of one endpoint on the held-out requests and save the result under runs/providers/.
    The key is used for the requests and is not written anywhere. `defs` needs dataset, TARGET_SYSTEM,
    TARGET_TOOLS, parse_tool_call, run_nu and same_output (see `load_defs`)."""
    cases = build_cases(defs, limit)
    endpoint = Endpoint(base_url, model, key, defs["TARGET_TOOLS"], **options)
    name = name or model
    log(f"{name}: {len(cases)} requests to {endpoint.url}, {concurrency} at a time")
    rows, done = [None] * len(cases), 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(score, case, endpoint, defs): i for i, case in enumerate(cases)}
        for future in concurrent.futures.as_completed(futures):
            rows[futures[future]] = future.result()
            done += 1
            if done % 20 == 0 or done == len(cases):
                log(f"  {name}: {done}/{len(cases)} · correct so far {sum(r['correct'] for r in rows if r) / done:.0%}")
    saved = {
        "name": name, "model": model, "base_url": base_url, "dataset": defs["dataset"]["fingerprint"], "created": time.time(),
        # `body` is what was finally sent with every request, after any adjustment the endpoint asked for
        "settings": {"tools": not endpoint.no_tools, "tool_choice": endpoint.body.get("tool_choice"), "temperature": endpoint.body.get("temperature"), "extra": options.get("extra") or {}, "body": dict(endpoint.body)},
        "summary": summarise(rows), "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (re.sub(r"[^\w.-]+", "_", name) + ".json")
    path.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    log(f"saved {path}")
    return saved


def saved_results() -> list[dict]:
    """Every endpoint scored so far (summaries only), oldest first."""
    found = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(OUT.glob("*.json"))]
    return [{k: v for k, v in r.items() if k != "rows"} for r in sorted(found, key=lambda r: r.get("created", 0))]


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    answered = [r for r in rows if "error" not in r]
    rate = lambda key: sum(r[key] for r in rows) / n if n else 0.0
    total = lambda key: sum(r.get(key) or 0 for r in answered)
    return {
        "requests": n, "errors": n - len(answered), "well_formed": rate("well_formed"), "runs": rate("runs"), "correct": rate("correct"),
        "seconds_per_request": round(total("seconds") / len(answered), 2) if answered else None,
        "prompt_tokens": total("prompt_tokens"), "completion_tokens": total("completion_tokens"),
    }


def ours(limit: int | None) -> dict | None:
    """Our fine-tuned model on the same requests, from the saved report (40 compared + the 191 others)."""
    path = RUNS_DIR / "target" / "report.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    first = report["execution"]["P_good"]
    rest = report.get("unused_held_out")
    n = len(report["samples"])
    if limit and limit <= n and limit != n:
        return None
    if limit == n or not rest:
        return {"requests": n, **{k: first[k] for k in ("well_formed", "runs", "correct")}}
    m = rest["rows"]
    timing = report.get("timing") or {}  # measured on this machine with MLX, over all held-out requests
    return {"requests": n + m, **{k: (first[k] * n + rest[k] * m) / (n + m) for k in ("well_formed", "runs", "correct")}, "seconds_per_request": timing.get("seconds_per_request")}


def table(rows: list[tuple[str, dict]]) -> None:
    print(f"\n{'model':<44} {'requests':>8} {'well-formed':>12} {'runs':>7} {'correct':>8} {'s/request':>10} {'errors':>7}")
    for name, s in rows:
        per = f"{s['seconds_per_request']:.2f}" if s.get("seconds_per_request") else ""
        print(f"{name[:44]:<44} {s['requests']:>8} {s['well_formed']:>12.0%} {s['runs']:>7.0%} {s['correct']:>8.1%} {per:>10} {s.get('errors', 0) or '':>7}")


def compare() -> None:
    rows = []
    for path in sorted(OUT.glob("*.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        rows.append((saved["name"], saved["summary"]))
    sizes = {s["requests"] for _, s in rows}
    for size in sorted(sizes | {None}, key=lambda v: v or 0):
        mine = ours(size)
        if mine:
            rows.append((f"ours: LFM3 1.7B + LoRA ({mine['requests']})", mine))
    table(rows)
    if len(sizes) > 1:
        print("\nNot every endpoint was scored on the same number of requests; compare rows with the same count.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"), help="ends in /v1 for most providers (default: $OPENAI_BASE_URL)")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="name of the environment variable that holds the key; '' for none")
    parser.add_argument("--name", help="label for the result file (default: the model name)")
    parser.add_argument("--limit", type=int, help="score only the first N held-out requests")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--tool-choice", default="function", choices=["function", "required", "auto"], help="function: force run_nushell (default)")
    parser.add_argument("--no-tools", action="store_true", help="send no tools; ask for the code in a fenced block")
    parser.add_argument("--extra", help="JSON object merged into the request body")
    parser.add_argument("--export", action="store_true", help="write the evaluation set and stop")
    parser.add_argument("--compare", action="store_true", help="print the table of saved results and stop")
    args = parser.parse_args()
    if args.compare:
        compare()
        return 0
    if not args.export:
        if not args.base_url or not args.model:
            parser.error("--base-url and --model are needed (or --export / --compare)")
        if args.api_key_env and not os.environ.get(args.api_key_env):
            parser.error(f"${args.api_key_env} is not set; export the key, or pass --api-key-env '' for an endpoint without one")

    print("building the dataset…")
    defs = load_defs()
    cases = build_cases(defs, args.limit)
    tools = defs["TARGET_TOOLS"]
    if args.export:
        path = RUNS_DIR / "eval" / "eval_set.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps({**case, "tools": tools}) + "\n" for case in cases), encoding="utf-8")
        print(f"wrote {len(cases)} requests to {path} (dataset {defs['dataset']['fingerprint']})")
        return 0

    name = args.name or args.model
    key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
    saved = run_endpoint(
        defs, args.base_url, args.model, key, name=name, limit=args.limit, concurrency=args.concurrency, no_tools=args.no_tools,
        tool_choice=args.tool_choice, temperature=args.temperature, max_tokens=args.max_tokens, timeout=args.timeout, extra=json.loads(args.extra or "{}"),
    )
    summary, rows = saved["summary"], saved["rows"]
    reference = ours(args.limit)
    table([(name, summary)] + ([(f"ours: LFM3 1.7B + LoRA ({reference['requests']})", reference)] if reference else []))
    failed = [r for r in rows if "error" in r]
    if failed:
        print(f"\n{len(failed)} requests got no answer; the first error: {failed[0]['error'][:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
