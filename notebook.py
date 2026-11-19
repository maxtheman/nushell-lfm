import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full", app_title="Teaching a small model to drive Nushell")


@app.cell
def _():
    import asyncio
    import json
    import shutil
    import time
    from collections import Counter

    import altair as alt
    import marimo as mo
    import pandas as pd

    from nushell_lfm.core import MODELS_DIR, MODEL_FOLDERS, ROOT, RUNS_DIR, SANDBOX, function_source, mlx_thread, model_path, nu_path

    return (
        asyncio,
        Counter,
        MODELS_DIR,
        MODEL_FOLDERS,
        ROOT,
        RUNS_DIR,
        SANDBOX,
        alt,
        function_source,
        json,
        mlx_thread,
        mo,
        model_path,
        nu_path,
        pd,
        shutil,
        time,
    )


@app.cell
def _(DEMO_PORT, MODELS_DIR, MODEL_FOLDERS, ROOT, RUNS_DIR, demo_notes, json, mo, nu_path, row):
    _ready = [("Nushell", bool(nu_path)), ("LFM3 checkpoint", (MODELS_DIR / MODEL_FOLDERS["target"] / "config.json").exists())]
    _missing = [name for name, ok in _ready if not ok]
    _gif = ROOT / "assets" / "usage.gif"
    # The header states the results, read from the saved reports (nothing is shown for a part that is missing).
    _points = []
    _file = RUNS_DIR / "target" / "report.json"
    if _file.exists():
        _r = json.loads(_file.read_text(encoding="utf-8"))
        _n, _base, _good, _rest = _r["rows"]["eval"], _r["execution"]["base"]["correct"], _r["execution"]["P_good"]["correct"], _r.get("unused_held_out")
        _points.append(f"**Before training:** {_base:.0%} of {_n} held-out requests.")
        _all = f", and {(_good * _n + _rest['correct'] * _rest['rows']) / (_n + _rest['rows']):.1%} of all {_n + _rest['rows']}" if _rest else ""
        _points.append(f"**After training:** {_good:.0%} of the same {_n}{_all}.")
        _others = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((RUNS_DIR / "providers").glob("*.json"))]
        _others = sorted((o for o in _others if o.get("dataset") == _r["dataset"]), key=lambda o: -o["summary"]["correct"])
        if _others:
            _points.append("**Other models:** " + ", ".join(f"{o['name']} {o['summary']['correct']:.1%} of {o['summary']['requests']}" for o in _others) + ".")
        _c = _r["config"]
        _points.append(
            f"**Training:** one LoRA of rank {_c['rank']} on {_c['placement'].replace('_', ' ')} layers ({_r['trainable_parameters'] / 1e6:.1f} M parameters), "
            f"{_r['rows']['fit']} rows × {_c['epochs']} epochs, {max(h['step'] for h in _r['history'] if h['adapter'] == 'P_good')} updates, on a Mac with MLX."
        )
    _points.append("**In this notebook:** the generated environments, the command chains and requests, the training run, the model in 3D with and without its LoRA, and the comparison with other models.")
    _intro = mo.md(
        "# Teaching a small model to drive Nushell\n\n"
        "LFM3 1.7B turns a plain request into one `run_nushell` tool call. A call counts only if running it returns the reference value.\n\n"
        + "\n".join(f"- {point}" for point in _points)
    )
    _side = (
        mo.image(src=str(_gif), rounded=True, width="100%")
        if _gif.exists()
        else mo.Html('<div style="aspect-ratio: 16/10; border: 1px dashed currentColor; border-radius: 10px; display: grid; place-items: center; opacity: 0.5;">GIF of usage</div>')
    )
    # opens the finished model on WebGPU and a Nushell session over a held-out environment, in a new tab
    _demo = mo.Html(
        f'''<div><a href="http://localhost:{DEMO_PORT}/?load=1" target="_blank" rel="noopener"
              style="display: inline-block; padding: 6px 14px; font: 600 0.9rem/1.2 system-ui, sans-serif; text-decoration: none; color: inherit;
                     border: 1.5px solid currentColor; border-radius: 8px;">Open demo ↗</a></div>'''
    )
    mo.vstack(
        [row((mo.vstack([_intro, _demo], gap=0.75), 3, 360), (_side, 2, 300))]
        + ([mo.callout(mo.md(" ".join(demo_notes)), kind="warn")] if demo_notes else [])
        + ([mo.callout(mo.md("Missing: " + ", ".join(_missing) + ". Run `mise run setup` and see the README for the checkpoints."), kind="warn")] if _missing else [])
    )
    return


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
                ["step-target", "3. Train LFM"],
                ["step-providers", "4. Other models"],
            ]
        )
    )
    mo.vstack([_wrap, _bar], gap=0)
    return


@app.cell
async def _(RUNS_DIR, dataset, mlx_thread, mo, target_ran):
    from nushell_lfm import model_view

    # What the 3D view draws is measured: the model over held-out requests and their reference calls, with and
    # without its adapter. After a real training run it is captured again from the new adapter.
    activation_error = None
    _file = RUNS_DIR / "activation" / "examples.json"
    if target_ran in ("standard", "protocol"):

        def _capture():
            model_view.save_payload(_file, model_view.capture_activations(model_view.pick_rows(dataset, 4)), model_view.model_layout())

        with mo.status.spinner(title="Capturing activations from the new adapter…"):
            try:
                await mlx_thread(_capture)
            except Exception as _exc:
                activation_error = f"{type(_exc).__name__}: {_exc}"
    activation_payload = model_view.load_payload(_file)
    return activation_error, activation_payload, model_view


@app.cell
def _(activation_error, activation_payload, mo, model_view, space):
    _views = [space(32, "step-model")]
    if activation_error:
        _views.append(mo.callout(mo.md(f"Capture failed: `{activation_error}`"), kind="danger"))
    if activation_payload:
        _views.append(
            mo.ui.anywidget(
                model_view.ModelView(
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

    from wigglystuff import Fader, Knob
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
        .liveedit-return { display: none !important; }
        @container (max-width: 560px) {
          .liveedit-card { grid-template-columns: minmax(0, 1fr); grid-template-rows: auto auto; }
          .liveedit-code { border-right: 0; border-bottom: 1px solid var(--liveedit-border); }
        }
        """

    def fader(label, low, high, step, value, length=150, orientation="vertical"):
        return mo.ui.anywidget(Fader(min_value=low, max_value=high, step=step, value=value, label=label, length=length, orientation=orientation))

    def knob(label, low, high, step, value, size=46):
        return mo.ui.anywidget(Knob(min_value=low, max_value=high, step=step, value=value, label=label, size=size))

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

    return LiveEdit, fader, fval, ival, knob, row, space


@app.cell
def _(fader):
    env_count_slider = fader("Environments", 2, 24, 1, 20, 200, "horizontal")
    return (env_count_slider,)


@app.cell
def _():
    from nushell_lfm.environments import GENERATOR_SCOPE, build_environments, render_tree, run_nu, same_output

    return GENERATOR_SCOPE, build_environments, render_tree, run_nu, same_output


@app.cell
def _(build_environments, env_count_slider, ival, mo):
    environments = build_environments(ival(env_count_slider))
    _short = lambda e: " · ".join(e["title"].split(" · ")[:2])  # id and language; the folders are in the tree
    env_picker = mo.ui.dropdown(options={_short(e): e["id"] for e in environments}, value=_short(environments[0]), label="Environment")
    return env_picker, environments


@app.cell
def _(env_picker, environments):
    shown_env = next(e for e in environments if e["id"] == env_picker.value)
    return (shown_env,)


@app.cell
def _(GENERATOR_SCOPE, LiveEdit, env_count_slider, env_picker, function_source, mo, render_tree, row, shown_env, space):
    # The code shown is the generator itself, running for the selected seed: each line carries the value it produced.
    _code = LiveEdit(function_source("environment_layout"), args=(shown_env["layout"]["seed"],), function_name="environment_layout", globalns=GENERATOR_SCOPE)
    mo.vstack(
        [
            space(56, "step-env"),
            mo.md("## Env"),
            mo.hstack([env_count_slider, env_picker], justify="start", align="end", gap=2, wrap=False),
            row((mo.ui.anywidget(_code), 1, 560), (mo.json(shown_env["layout"]), 0, 250), (mo.plain_text(render_tree(list(shown_env["files"]))), 0, 180), gap="20px 28px"),
        ],
        gap=1.5,
    )
    return


@app.cell
def _():
    from nushell_lfm.catalog import ALL_CATEGORIES, COMMAND_CATALOG

    return ALL_CATEGORIES, COMMAND_CATALOG


@app.cell
def _(ALL_CATEGORIES, fader, mo):
    category_boxes = mo.ui.array([mo.ui.checkbox(value=True, label=name.replace("_", " ")) for name in ALL_CATEGORIES])
    chains_total_slider = fader("Chains", 20, 400, 20, 400, 84)
    ported_share_slider = fader("Unix", 0.0, 0.3, 0.05, 0.25, 84)
    length_bias_slider = fader("Longer", 0.0, 1.0, 0.1, 0.5, 84)
    k_variants_slider = fader("Requests", 1, 6, 1, 3, 84)
    return category_boxes, chains_total_slider, k_variants_slider, length_bias_slider, ported_share_slider


@app.cell
def _():
    from nushell_lfm.chains import NATIVE, PORTED, PRIOR_INFO, SOURCES, STAGES, build_chains

    return NATIVE, PORTED, PRIOR_INFO, SOURCES, STAGES, build_chains


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
            mo.md("## Data gen & rephrase"),
            row(
                (
                    mo.vstack(
                        [
                            mo.hstack([chains_total_slider, ported_share_slider, length_bias_slider, k_variants_slider], justify="start", gap=0.25, wrap=False),
                            mo.hstack(list(category_boxes), justify="start", gap=0.75, wrap=True),
                            request_btn,
                        ],
                        gap=1.5,
                    ),
                    0, 360,
                ),
                (mo.vstack([chain_tabs, chain_table], gap=0.75), 1, 440),
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
def _(mo):
    request_btn = mo.ui.run_button(label="Generate new data")
    return (request_btn,)


@app.cell
def _():
    from nushell_lfm.dataset import assemble_dataset, load_model_requests, save_dataset, write_requests_with_model

    return assemble_dataset, load_model_requests, save_dataset, write_requests_with_model


@app.cell
async def _(
    MODEL_FOLDERS,
    PRIOR_INFO,
    assemble_dataset,
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
    dataset = assemble_dataset(environments, verified_chains, ival(k_variants_slider), load_model_requests(ival(k_variants_slider)))
    save_dataset(
        dataset,
        {
            "chains": ival(chains_total_slider), "ported_share": fval(ported_share_slider), "length_bias": fval(length_bias_slider),
            "environments": len(environments), "requests_per_chain": ival(k_variants_slider),
            "categories": selected_categories, "priors": PRIOR_INFO, "chain_summary": chain_summary,
        },
    )
    _train, _held_out = len(dataset["target"]["train"]), len(dataset["target"]["dev"])
    mo.vstack(
        ([mo.callout(mo.md(f"Request writing failed: `{_error}`"), kind="danger")] if _error else [])
        + [
            space(40),
            mo.md(f"Final data size = {_train:,} + {_held_out:,} = **{_train + _held_out:,}**"),
        ]
    )
    return (dataset,)


@app.cell
def _():
    from nushell_lfm.calls import parse_tool_call
    from nushell_lfm.lora import PLACEMENTS
    from nushell_lfm.target import TARGET_DIR, TARGET_SYSTEM, TARGET_TOOLS, run_target_experiments, run_target_pipeline, score_call, target_prompt

    return PLACEMENTS, TARGET_DIR, TARGET_SYSTEM, TARGET_TOOLS, parse_tool_call, run_target_experiments, run_target_pipeline, score_call, target_prompt


@app.cell
def _(MODEL_FOLDERS, PLACEMENTS, fader, knob, mo):
    # Not shown: a folder under $MODELS_DIR, or a path. Change it here to train another checkpoint.
    target_model_path = mo.ui.text(value=MODEL_FOLDERS["target"])
    # smoke: 8 rows, 4 held-out requests (proves the code path) · standard: all rows, 2 epochs, every held-out
    # request · protocol: sweep over rank and learning rate, placement ablation, then the final adapter
    target_preset = mo.ui.dropdown(options=["smoke", "standard", "protocol"], value="smoke", label="Budget")
    target_placement = mo.ui.dropdown(options=list(PLACEMENTS), value="all_linear", label="Layers")
    target_lr = fader("LR ×1e-4", 0.5, 8.0, 0.5, 2.0, 84)
    target_rank = knob("Rank", 4, 32, 4, 8)
    target_alpha_mult = knob("α ÷ rank", 1.0, 4.0, 0.5, 2.0)
    target_batch = knob("Batch", 1, 8, 1, 4)
    target_train_btn = mo.ui.run_button(label="▶")
    return target_alpha_mult, target_batch, target_lr, target_model_path, target_placement, target_preset, target_rank, target_train_btn


@app.cell
async def _(
    TARGET_DIR,
    dataset,
    fval,
    ival,
    json,
    mlx_thread,
    mo,
    model_path,
    run_target_experiments,
    run_target_pipeline,
    shutil,
    target_alpha_mult,
    target_batch,
    target_lr,
    target_model_path,
    target_placement,
    target_preset,
    target_rank,
    target_train_btn,
    time,
):
    target_error, _log = None, []
    if target_train_btn.value:
        _base = {"model_path": model_path(target_model_path.value), "batch": ival(target_batch), "seed": 73, "probe_share": 0.15, "max_new": 220}
        _own = {"rank": ival(target_rank), "alpha": ival(target_rank) * fval(target_alpha_mult), "lr": fval(target_lr) * 1e-4, "placement": target_placement.value}
        _full = {"n_good": 10_000, "n_eval": 40, "n_loss": 24, "epochs": 2}
        _smoke = None
        with mo.status.spinner(title="Training LFM3…"):
            try:
                if target_preset.value != "smoke" and TARGET_DIR.exists():  # a real run replaces the saved adapter: keep the old one
                    shutil.copytree(TARGET_DIR, TARGET_DIR.parent / "previous" / f"target-{int(time.time())}")
                if target_preset.value == "protocol":
                    _sizes = {"ranks": [8, 16], "learning_rates": [1e-4, 2e-4, 4e-4], "n_sweep": 200, "n_sweep_eval": 24, "full": _full}
                    await mlx_thread(run_target_experiments, _base, dataset, _sizes, _log.append)
                elif target_preset.value == "standard":
                    await mlx_thread(run_target_pipeline, {**_base, **_own, **_full}, dataset, _log.append)
                else:
                    # the smoke budget only proves the code path; it is written to runs/smoke, never over a real adapter
                    _cfg = {**_base, **_own, "n_good": 8, "n_eval": 4, "n_loss": 4, "epochs": 1, "score_all": False, "out_dir": str(TARGET_DIR.parent / "smoke")}
                    _smoke = await mlx_thread(run_target_pipeline, _cfg, dataset, _log.append)
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
    mo,
    pd,
    row,
    space,
    target_alpha_mult,
    target_batch,
    target_error,
    target_experiments,
    target_lr,
    target_placement,
    target_preset,
    target_rank,
    target_report,
    target_train_btn,
):
    _pct = lambda v: "–" if v is None else f"{v:.0%}"
    _notes, _charts = [], []
    if target_error:
        _notes.append(mo.callout(mo.md(f"The pipeline failed: `{target_error}`"), kind="danger"))
    if target_report:
        _r = target_report
        if _r["dataset"] != dataset["fingerprint"]:
            _notes.append(mo.callout(mo.md("Trained on a different dataset than the one built above."), kind="warn"))
        _loss = mo.ui.altair_chart(
            alt.Chart(pd.DataFrame([h for h in _r["history"] if h["adapter"] == "P_good"]))
            .mark_line()
            .encode(x=alt.X("step:Q", title="Update"), y=alt.Y("loss:Q", title=None))
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
        _scores = pd.DataFrame([{"decoder": {"base": "base model", "P_good": "with LoRA"}[name], "share": v["correct"]} for name, v in _r["execution"].items() if name in ("base", "P_good")])
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
                        + (f" · with LoRA on the other {_r['unused_held_out']['rows']}: {_r['unused_held_out']['correct']:.0%}" if _r.get("unused_held_out") else ""),
                    ],
                    anchor="start",
                ),
                height=max(140, 22 * len(_scores)), width="container",
            )
        )
        _charts = [_loss, mo.ui.tabs({"Execution": _bars, "Calls": mo.ui.table(pd.DataFrame([{"request": x["request"], "reference": x["reference"], "base model": x["base"], "base correct": x["base_correct"], "with LoRA": x["P_good"], "LoRA correct": x["P_good_correct"]} for x in _r["samples"]]), selection=None, page_size=4)})]
    mo.vstack(
        [
            space(56, "step-target"),
            mo.md("## Train LFM"),
            *_notes,
            mo.hstack(
                [target_lr, target_rank, target_alpha_mult, target_batch, mo.vstack([target_train_btn, target_placement, target_preset], gap=0.4)],
                justify="start", align="end", gap=0.4, wrap=False,
            ),
            *([row(*[(chart, 1, 380) for chart in _charts])] if _charts else []),
        ],
        gap=1.5,
    )
    return


@app.cell
def _():
    from nushell_lfm.export import BASE_ONNX_DIR, BROWSER_DIR, EXPORT_ONNX_DIR, check_browser_export, export_for_browser, write_browser_requests

    return BASE_ONNX_DIR, BROWSER_DIR, EXPORT_ONNX_DIR, check_browser_export, export_for_browser, write_browser_requests


@app.cell
async def _(
    BASE_ONNX_DIR,
    BROWSER_DIR,
    TARGET_DIR,
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
    SANDBOX,
):
    # After a real training run, the adapter is added to the 4-bit ONNX build of the target as low-rank branches beside
    # each quantised weight, the exported graph generates on ONNX Runtime, and its calls are executed. The result is
    # written to webgpu-demo/export.json. `mise run train -- --only export` does the same from the command line.
    export_error = None
    if target_ran in ("standard", "protocol") and target_report and not SANDBOX and (BASE_ONNX_DIR / "onnx" / "model_q4f16.onnx").exists():
        _rows = sorted(dataset["target"]["dev"], key=lambda r: (r["id"].rsplit("/", 1)[1], r["id"]))[:24]
        _cfg = target_report["config"]

        def _export():
            merged = export_for_browser(model_path(target_model_path.value), TARGET_DIR / "adapter_good.safetensors", _cfg["rank"], _cfg["alpha"], lambda line: None)
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
    # that "Open demo" needs no other process.
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
def _(BASE_ONNX_DIR, BROWSER_DIR, DEMO_PORT, EXPORT_ONNX_DIR, demo_serving, export_error):
    # What would stop the demo from working; shown under the Open demo button at the top.
    demo_notes = []
    if export_error:
        demo_notes.append(f"Export failed: `{export_error}`")
    if not (BASE_ONNX_DIR / "onnx" / "model_q4f16.onnx").exists():
        demo_notes.append("The ONNX build of the base model is missing: run `mise run fetch-onnx`.")
    elif not ((EXPORT_ONNX_DIR / "onnx" / "model_q4f16.onnx").exists() and (BROWSER_DIR / "environments.json").exists()):
        demo_notes.append("No exported model yet: train at the standard or protocol budget.")
    if not (BROWSER_DIR / "node_modules" / "@huggingface" / "transformers").exists():
        demo_notes.append("Transformers.js is not installed: run `mise run setup`.")
    if not demo_serving:
        demo_notes.append(f"Port {DEMO_PORT} is taken by something else, so the demo cannot be served from here.")
    return (demo_notes,)


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
            _table.append({"model": "LFM3 1.7B + LoRA (this notebook)", **_mine, "ours": True})
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
