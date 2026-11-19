import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="LFM3 activations (real attention)")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    ROOT = Path(__file__).resolve().parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import activation_view as av

    CACHE = ROOT / "runs" / "activation" / "examples.json"
    return CACHE, av, mo


@app.cell
def _(CACHE, av, mo):
    payload = av.load_payload(CACHE)
    if payload is None:
        # No cache yet: run the notebook pipeline headless, capture 4 held-out rows, and cache them.
        import notebook

        _, _defs = notebook.app.run()
        _examples = av.capture_activations(_defs, av.pick_rows(_defs, 4), max_examples=4)
        av.save_payload(CACHE, _examples, av.model_layout(_defs))
        payload = av.load_payload(CACHE)
    mo.md(f"{len(payload['examples'])} examples from `{CACHE.relative_to(CACHE.parents[2])}`")
    return (payload,)


@app.cell
def _(av, mo, payload):
    widget = mo.ui.anywidget(
        av.ActivationWidget(
            examples=payload["examples"],
            layer_types=payload["layer_types"],
            conv_window=payload["conv_window"],
        )
    )
    widget
    return (widget,)


if __name__ == "__main__":
    app.run()
