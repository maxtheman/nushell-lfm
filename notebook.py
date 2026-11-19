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

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    from nushell_lfm.core import MODELS_DIR, MODEL_FOLDERS, ROOT, RUNS_DIR, SANDBOX, function_source, mlx_thread, model_path, nu_path, show_path

    return (
        asyncio,
        Counter,
        MODELS_DIR,
        MODEL_FOLDERS,
        Path,
        ROOT,
        RUNS_DIR,
        SANDBOX,
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
        nu_path,
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
def _(MODELS_DIR, MODEL_FOLDERS, ROOT, mo, nu_path, row):
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
    RUNS_DIR,
):
    import activation_view

    # What the 3D view draws is measured: both models over held-out requests and their reference calls, each with
    # and without its adapter, and guided decoding token by token. After a real training run it is captured again
    # from the new adapters.
    activation_error = None
    _file = RUNS_DIR / "activation" / "examples.json"
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
    _views = [space(32, "step-model")]
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
        .liveedit-return { display: none !important; }
        @container (max-width: 560px) {
          .liveedit-card { grid-template-columns: minmax(0, 1fr); grid-template-rows: auto auto; }
          .liveedit-code { border-right: 0; border-bottom: 1px solid var(--liveedit-border); }
        }
        """

    def heading(text, section=None):
        return mo.md(f"## {text}")

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

    return LiveEdit, TangleLatex, fader, fval, heading, ival, knob, row, space


@app.cell
def _(fader):
    env_count_slider = fader("Environments", 2, 24, 1, 20, 200, "horizontal")
    return (env_count_slider,)


@app.cell
def _():
    from nushell_lfm.environments import (
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
    _short = lambda e: " · ".join(e["title"].split(" · ")[:2])  # id and language; the folders are in the tree
    env_picker = mo.ui.dropdown(options={_short(e): e["id"] for e in environments}, value=_short(environments[0]), label="Environment")
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
            mo.md("## Env"),
            mo.hstack([env_count_slider, env_picker], justify="start", align="end", gap=2, wrap=False),
            row((mo.ui.anywidget(_code), 1, 560), (mo.json(shown_env["layout"]), 0, 250), (mo.plain_text(render_tree(list(shown_env["files"]))), 0, 180), gap="20px 28px"),
        ],
        gap=1.5,
    )
    return


@app.cell
def _():
    from nushell_lfm.catalog import (
        ALL_CATEGORIES,
        CATEGORY_NOTES,
        COMMAND_CATALOG,
        NU_COMMANDS,
    )

    return (
        ALL_CATEGORIES,
        CATEGORY_NOTES,
        COMMAND_CATALOG,
        NU_COMMANDS,
    )


@app.cell
def _(ALL_CATEGORIES, fader, mo):
    category_boxes = mo.ui.array([mo.ui.checkbox(value=True, label=name.replace("_", " ")) for name in ALL_CATEGORIES])
    chains_total_slider = fader("Chains", 20, 400, 20, 400, 84)
    ported_share_slider = fader("Unix", 0.0, 0.3, 0.05, 0.25, 84)
    length_bias_slider = fader("Longer", 0.0, 1.0, 0.1, 0.5, 84)
    breaks_per_chain_slider = fader("Breaks", 1, 10, 1, 4, 84)
    k_variants_slider = fader("Requests", 1, 6, 1, 3, 84)
    return breaks_per_chain_slider, category_boxes, chains_total_slider, k_variants_slider, length_bias_slider, ported_share_slider


@app.cell
def _():
    from nushell_lfm.chains import (
        AUTO_STAGES,
        KIND_GRAPH,
        NATIVE,
        PORTED,
        PORTED_LIMIT,
        PRIOR_INFO,
        SOURCES,
        STAGES,
        build_chains,
        structural_breaks,
    )

    return (
        AUTO_STAGES,
        KIND_GRAPH,
        NATIVE,
        PORTED,
        PORTED_LIMIT,
        PRIOR_INFO,
        SOURCES,
        STAGES,
        build_chains,
        structural_breaks,
    )


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
            mo.md("## Data gen & rephrase"),
            row(
                (
                    mo.vstack(
                        [
                            mo.hstack([chains_total_slider, ported_share_slider, length_bias_slider, breaks_per_chain_slider, k_variants_slider], justify="start", gap=0.25, wrap=False),
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
def _():
    from nushell_lfm.breaks import (
        BREAK_LABELS,
        BREAK_SCOPE,
        FAILURE_LABELS,
        FAILURE_TYPES,
        VERIFIED_BY,
        build_samples,
        parse_tool_call,
        wrap_tool_call,
    )

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
            space(24),
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
def _():
    from nushell_lfm.dataset import (
        REQUEST_STYLES,
        assemble_dataset,
        load_model_requests,
        save_dataset,
        write_requests_with_model,
    )

    return (
        REQUEST_STYLES,
        assemble_dataset,
        load_model_requests,
        save_dataset,
        write_requests_with_model,
    )


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
            mo.md(f"Final data size = {_kind['valid']:,} + {_kind['broken']:,} + {_kind['equivalent']:,} = **{sum(_kind.values()):,}**"),
        ]
    )
    return (dataset,)


@app.cell
def _():
    from nushell_lfm.grader import (
        GRADER_DIR,
        GUIDANCE_GRID,
        PLACEMENTS,
        attach_lora,
        balanced,
        encode_grader_row,
        grade_calls,
        grader_readouts,
        guidance_table,
        learning_rate,
        run_grader_experiments,
        run_grader_sweep,
        set_lora,
    )

    return (
        GRADER_DIR,
        GUIDANCE_GRID,
        PLACEMENTS,
        attach_lora,
        balanced,
        encode_grader_row,
        grade_calls,
        grader_readouts,
        guidance_table,
        learning_rate,
        run_grader_experiments,
        run_grader_sweep,
        set_lora,
    )


@app.cell
def _(MODEL_FOLDERS, PLACEMENTS, TangleLatex, fader, knob, mo):
    # Not shown: a folder under $MODELS_DIR, or a path. Change it here to train another checkpoint.
    grader_model_path = mo.ui.text(value=MODEL_FOLDERS["grader"])
    # smoke: 20 rows, 1 epoch (proves the code path) · standard: 400 rows, 2 epochs, the settings shown ·
    # protocol: sweep over rank and learning rate, placement ablation, final adapter
    grader_preset = mo.ui.dropdown(options=["smoke", "standard", "protocol"], value="smoke", label="Budget")
    grader_placement = mo.ui.dropdown(options=list(PLACEMENTS), value="mlp_only", label="Layers")
    grader_lr = fader("LR ×1e-4", 0.5, 8.0, 0.5, 1.0, 84)
    grader_dropout = fader("Dropout", 0.0, 0.2, 0.05, 0.05, 84)
    grader_clip = fader("Clip", 0.5, 5.0, 0.5, 1.0, 84)
    grader_rank = knob("Rank", 4, 32, 4, 8)
    grader_alpha_mult = knob("α ÷ rank", 1.0, 4.0, 0.5, 2.0)
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
            mo.md("## Train grader"),
            *_notes,
            mo.hstack(
                [grader_lr, grader_dropout, grader_clip, grader_rank, grader_alpha_mult, grader_batch, mo.vstack([grader_train_btn, grader_placement, grader_preset], gap=0.4)],
                justify="start", align="end", gap=0.4, wrap=False,
            ),
            *([row(*[(chart, 1, 380) for chart in _charts])] if _charts else []),
        ],
        gap=1.5,
    )
    return


@app.cell
def _():
    from nushell_lfm.target import (
        TARGET_DIR,
        TARGET_GUIDANCE_GRID,
        TARGET_SYSTEM,
        TARGET_TOOLS,
        crossfit_wrong_calls,
        rebuild_weak_target,
        score_decoders,
        score_low_bit,
        train_bad_adapter,
        score_saved_target,
        quick_target_run,
        run_target_experiments,
        run_target_pipeline,
        score_call,
        target_prompt,
    )

    return (
        TARGET_DIR,
        TARGET_GUIDANCE_GRID,
        TARGET_SYSTEM,
        TARGET_TOOLS,
        crossfit_wrong_calls,
        rebuild_weak_target,
        score_decoders,
        score_low_bit,
        train_bad_adapter,
        score_saved_target,
        quick_target_run,
        run_target_experiments,
        run_target_pipeline,
        score_call,
        target_prompt,
    )


@app.cell
def _(MODEL_FOLDERS, PLACEMENTS, TARGET_GUIDANCE_GRID, TangleLatex, fader, knob, mo):
    # Not shown: a folder under $MODELS_DIR, or a path. Change it here to train another checkpoint.
    target_model_path = mo.ui.text(value=MODEL_FOLDERS["target"])
    # smoke: 8 rows, 4 held-out requests (proves the code path) · standard: all rows, 2 epochs, 40 held-out requests,
    # guidance comparison · protocol: sweep, placement ablation, then the full pipeline
    target_preset = mo.ui.dropdown(options=["smoke", "standard", "protocol"], value="smoke", label="Budget")
    target_placement = mo.ui.dropdown(options=list(PLACEMENTS), value="all_linear", label="Layers")
    target_lr = fader("LR ×1e-4", 0.5, 8.0, 0.5, 2.0, 84)
    target_floor = fader("Floor", 0.01, 0.5, 0.01, 0.1, 84)  # used by guided decoding during a protocol run; not on the page
    target_rank = knob("Rank", 4, 32, 4, 8)
    target_alpha_mult = knob("α ÷ rank", 1.0, 4.0, 0.5, 2.0)
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
                        + (f" · P_good on the other {_r['unused_held_out']['rows']}: {_r['unused_held_out']['correct']:.0%}" if _r.get("unused_held_out") else ""),
                    ],
                    anchor="start",
                ),
                height=max(140, 22 * len(_scores)), width="container",
            )
        )
        _charts = [_loss, mo.ui.tabs({"Execution": _bars, "Calls": mo.ui.table(pd.DataFrame([{k: v for k, v in sample.items() if not k.startswith("guided")} for sample in _r["samples"]]), selection=None, page_size=4)})]
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
    from nushell_lfm.export import (
        BASE_ONNX_DIR,
        BROWSER_DIR,
        EXPORT_ONNX_DIR,
        check_browser_export,
        export_for_browser,
        write_browser_requests,
    )

    return (
        BASE_ONNX_DIR,
        BROWSER_DIR,
        EXPORT_ONNX_DIR,
        check_browser_export,
        export_for_browser,
        write_browser_requests,
    )


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
    SANDBOX,
):
    # After a real training run, P_good is added to the 4-bit ONNX build of the target as low-rank branches beside
    # each quantised weight, the exported graph generates on ONNX Runtime, and its calls are executed. The result is
    # written to webgpu-demo/export.json. `mise run experiments -- --only export` does the same from the command line.
    export_error = None
    if target_ran in ("standard", "protocol") and target_report and not SANDBOX and (BASE_ONNX_DIR / "onnx" / "model_q4f16.onnx").exists():
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
