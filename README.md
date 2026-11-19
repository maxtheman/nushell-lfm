# Teaching a small model to drive Nushell

One [marimo](https://marimo.io) notebook that generates a Nushell tool-call dataset from scratch and fine-tunes LFM3 1.7B on it, on Apple Silicon with MLX, offline. The model turns a plain request into one `run_nushell` tool call. A call counts as correct only if executing it returns the reference value.

On 231 requests from four generated projects the model never saw, the fine-tuned model is correct on **214 (92.6%)**; the base model is correct on 8% of the 40 it was run on. The finished model also runs in the browser on WebGPU, next to Nushell compiled to WebAssembly.

The page is `notebook.py`; the pipeline it drives is the `nushell_lfm` package, and the scripts under `scripts/` call the same functions from the command line.

## Setup

You need an Apple Silicon Mac, [mise](https://mise.jdx.dev), and the MLX checkpoint of LFM3 1.7B.

1. Put the checkpoint in a folder, keeping this folder name:

   ```
   models/
     lfm3-1.7b-mlx-8bit/
   ```

   To keep it elsewhere, create `mise.local.toml` (it is not committed):

   ```toml
   [env]
   MODELS_DIR = "/path/to/your/models"
   ```

2. Install the toolchain (uv, Nushell, Node) and the Python and JavaScript packages:

   ```sh
   mise run setup
   ```

3. For the browser export only, fetch the 4-bit ONNX build of the model (about 1.1 GB):

   ```sh
   mise run fetch-onnx
   ```

Model loading, data generation and training run offline. Besides setup (`mise run setup`, and `build-wasm` if you rebuild the module), the three `fetch-*` tasks are the only steps that use the network, and their outputs (`data/*.json`) are committed.

## Run

```sh
mise run notebook      # open the notebook for editing
mise run app           # serve it as an app
mise run sandbox       # the same, on a copy of runs/, so nothing saved is replaced
mise run test          # tests that need no model
mise run smoke         # train on 8 rows to prove the code path, about a minute (writes to runs/smoke)
mise run requests      # have the base model write the requests (cached)
mise run train         # train the adapter, score it on every held-out request, export it for the browser
mise run train -- --preset protocol   # sweep over rank and learning rate, placement ablation, final adapter
mise run check-wasm    # run every chain's reference call in WebAssembly Nushell and compare with native
mise run eval-endpoint -- --base-url URL --model NAME   # score any OpenAI-compatible endpoint on the same requests
mise run demo          # serve the browser demo
```

The smoke size proves the code path and is too small to learn anything.

### Use the pieces

```python
from nushell_lfm.pipeline import build
from nushell_lfm.environments import run_nu

built = build(environments=6, chains=60)           # environments, verified chains, dataset, settings
row = built["dataset"]["target"]["dev"][0]         # request, files, reference call, reference value
run_nu("ls | length", row["env"])                  # run Nushell in a throwaway copy of that project
```

`nushell_lfm.widgets` has the notebook's widgets for any anywidget host (marimo, Jupyter): `StepBar` (a sticky bar of section links), `NuRepl` (Nushell in WebAssembly over a set of files) and `ModelView` (the 3D view of the model's layers, drawn from a captured payload).

### Scoring other models

`scripts/eval_endpoint.py` sends the held-out requests to any OpenAI-compatible chat endpoint with the same system prompt, tool and user text our model gets, executes the call that comes back, and applies the same rule: correct means it returns the reference value. The key is read from an environment variable (`--api-key-env NAME`), results go to `runs/providers/`, and `--compare` prints every endpoint scored so far next to our model. `--export` writes the evaluation set to `runs/eval/eval_set.jsonl`. The notebook's last section, **Other models**, runs the same thing from the page: enter up to three endpoints in the form (`PROVIDER_n_URL`, `PROVIDER_n_KEY`, `PROVIDER_n_MODELS` as a comma-separated list, or set those variables in `mise.local.toml`), pick how many requests, press ▶. This is the one step that sends data off the machine: the request text and the generated file names go to the endpoint you name.

## How the data is made

Nothing in the training data is written by hand. Each stage is seeded and can be regenerated; `runs/generated/manifest.json` records the environment seeds and generator settings of a build.

| Stage | What happens |
|---|---|
| Environments | Each environment is a small project generated from a seed: source files, logs, a JSON table, a TOML config, docs, a script, with its own folder names, language, file names, column names, people and data. Faker supplies names and prose; polyfactory builds config and table records from dataclasses. |
| Commands | `assets/nu_commands.json` is every built-in command of the installed `nu` with its input and output types and parameters (`mise run cache-nu-docs`). |
| Chains | A chain is a walk on a graph of data kinds: a source, then stages whose input matches what the last step produced. Commands that need arguments have recipes that draw them from the environment; commands that take none are added automatically from their declared types. A sampled chain is kept only if it runs, gives the same output twice, returns something, and every automatically added stage changes the result. |
| Weights | The main prior is Nushell-native: how often each command appears in the example code of the Nushell book and cookbook (`mise run fetch-nu-examples`). Public Unix usage data (`mise run fetch-usage`) sets how much of the dataset each category gets and steers at most 30% of the chains. |
| Requests | Written by the base model, K per chain, in six styles: inconsistent formatting, spelling mistakes or abbreviations, less technical language, naming a part for the whole, an analogy from the same domain, an analogy from another domain. A request that loses a term, path or number the command needs falls back to a description composed from the chain's steps. |
| Splits | One environment in five is held out whole. Every score is for projects the model never saw. |

## How the model is trained and scored

- **Adapter.** LoRA on LFM3 1.7B, trained on the reference calls. The protocol is a sweep over rank and learning rate, an ablation of where the adapter goes (all linear layers, attention only, MLP only, convolution only), and a final adapter at the chosen setting. Each stage is chosen by the share of held-out requests whose generated call returns the reference value when executed.
- **Execution.** Every command runs in a throwaway copy of its environment. On macOS the Nu process is confined with `sandbox-exec`: no network, and no writes outside the copy.

## Results

Default settings: 20 generated environments, 401 verified chains, 3 requests each (972 training rows, 231 held-out rows); 4 environments held out whole. Dataset `a23bbb10902d`. Every number is for the held-out environments. `notes/results-tables.md` has these tables with their counts.

| LFM3 1.7B | Requests | Well-formed | Runs | Returns the reference value |
|---|---|---|---|---|
| Base model | 40 | 78% | 48% | 8% |
| With the adapter (all linear layers, rank 8, lr 2e-4, 825 rows × 2 epochs) | 40 | 100% | 100% | 95% |
| With the adapter, held-out requests used for nothing else | 191 | 100% | 97% | 92% |
| With the adapter, all held-out requests | 231 | 100% | 98% | **92.6%** |

- The 40 include the 24 the sweep chose settings on, so the 191 are the cleaner figure: 176 of 191, by environment 88%, 98%, 94%, 90%.
- Of the 17 wrong calls in the 231, 6 are requests that do not contain what the reference call needs (the request-writing model dropped a key name or echoed its instruction). Leaving those out: 214 of 225 (95%).
- The 4-bit ONNX export returns the reference value on 22 of 24 requests under ONNX Runtime. The 8-bit MLX adapter gets the same 22 of the same 24.

Sweep and placement ablation (200 rows, one epoch, 24 held-out requests; one request is about 4 points, so neighbours are within noise):

| Rank | lr 1e-4 | lr 2e-4 | lr 4e-4 |
|---|---|---|---|
| 8 | 71% | **75%** | 42% |
| 16 | 71% | 62% | 54% |

| Adapter on | Returns the reference value |
|---|---|
| all linear layers | **75%** |
| MLP only | 71% |
| convolution only | 71% |
| attention only | 62% |

Other models on the first 40 held-out requests, with the same system prompt, tool and user text, no examples and no fine-tuning (`mise run eval-endpoint`):

| Model | Requests | Well-formed | Runs | Returns the reference value |
|---|---|---|---|---|
| LFM3 1.7B with the adapter | 40 | 100% | 100% | 95% |
| gpt-6-luna (reasoning off) | 40 | 100% | 78% | 32.5% |
| claude-haiku-5-5 | 40 | 100% | 85% | 17.5% |
| LFM3 1.7B base | 40 | 78% | 48% | 8% |

The comparison is on this dataset's own convention: one reference value per request, compared exactly. A call that answers the request in another shape counts as wrong, which the fine-tuned model has learned to avoid and the others have not been shown.

**What the held-out score measures.** No environment is shared between training and held-out data. Within the 77 held-out chains, 15 have a command identical to a training command (the project differs, the command happens not to), 50 follow a chain pattern that also occurs in training, and 27 are new combinations of familiar steps.

## In the browser

- `nu-wasm/` is a small Rust crate that links the Nushell crates of the same release as the native `nu` and adds `ls`, `glob`, `open`, `save`, `cd`, `path type` and friends over an in-memory workspace. The built module is in `public/nu-wasm/` (`mise run build-wasm` rebuilds it). `mise run check-wasm` runs every chain's reference call both ways: of 401, 382 print the same answer and 19 differ only in the order `glob` returns entries (directory order on disk, sorted in the browser). The demo page therefore gets its reference value by running the reference call in the same in-browser Nushell.
- The export step adds the adapter to the 4-bit ONNX build of the model as low-rank branches beside each quantised weight, then generates with the exported graph on ONNX Runtime and executes the calls to confirm it still works.
- **Open demo** at the top of the notebook opens `webgpu-demo/` in a new tab: the exported model on WebGPU (Transformers.js) and a Nushell session, on one page. `mise run demo` serves the same page without the notebook.
- `MODEL=owner/name mise run build-demo` collects the page into `dist/demo` as a static site that downloads the model from that repository on the Hugging Face Hub. `webgpu-demo/config.json` (or `?model=`) says where the model is.

## Limits to keep in mind

- **Correct means "returns the reference value".** A generated call counts as correct only if executing it gives the reference output. A call can run and still not be what the user meant in ways this does not catch, and a different but reasonable answer counts as wrong. That simplification is deliberate for this first version.
- **Held-out environments share chain patterns with training.** Scores measure a known kind of request in an unseen project, not new kinds of request.
- **Requests are written by a small model.** About a fifth fall back to composed descriptions, which read stiffly. About 4% of the requests in this build repeat part of the rewriting instruction ("one line", "don't write code"); new builds reject those.
- **The adapter was fitted on 825 of its 972 training rows.** 15% of the training chains are kept out of fitting (`probe_share`); the guidance experiments read the adapter's mistakes on them.
- **Training from the notebook replaces the saved adapter.** The ▶ button at the standard and protocol budgets first copies the current one to `runs/previous/`; the smoke budget never touches it. `mise run sandbox` opens the notebook on a copy of `runs/`.
- **The budgets are small.** Sweeps use a few hundred rows so the whole protocol finishes in hours on a 24 GB machine. 
- **The browser model is 4-bit.** The adapter was trained on 8-bit weights; the export step reports the difference it measures.
- **Confinement is macOS-only.** Without `sandbox-exec`, commands still run in a throwaway copy but are not otherwise restricted. Generated commands are executed during scoring.
- **Some answers depend on the machine.** `glob` returns entries in directory order, `ls` reports a size for directories, and modified times print in the local time zone. All three come from the Mac the data was generated on, so a few reference values would differ on another file system or in another time zone. (The browser Nushell reports directory sizes the way APFS does.)
- **Usage data has known skews.** One source is a task corpus dominated by `find`; the shell-history source has weak provenance. They decide category shares and a minority of chains, nothing else.

## Layout

```
notebook.py            the page: controls, layout and charts; it calls the package below
nushell_lfm/           the pipeline as a library
  environments.py        generated projects and the sandboxed Nushell runner
  catalog.py             Nushell's built-in commands with their types
  chains.py              command chains, sampled and verified by running them
  dataset.py             requests, splits, the dataset on disk
  pipeline.py            all of the above in one call, for scripts
  calls.py               the text of a tool call
  lora.py                where the adapter goes, the learning-rate schedule
  target.py              training, decoding, scoring by execution, the sweep
  export.py              ONNX export for the browser
  model_view.py          what the model does on held-out examples, and the 3D view of it
  widgets/               notebook widgets: StepBar, NuRepl, ModelView
scripts/               command-line entry points and the data fetchers
tests/                 tests that need no model
nu-wasm/               Nushell for WebAssembly (Rust)
public/nu-wasm/        the built module
webgpu-demo/           the browser page (model on WebGPU + Nushell session)
assets/, data/         Nu's command list, usage data, Unix-to-Nushell map, Nushell-native prior
experiments/           not part of the pipeline: a grader model, guided decoding, low-bit weights (see its README)
notes/                 lab notes, results tables and charts
runs/                  generated datasets, adapters and reports (not committed)
```

`training-sketch.png` and `training-sketch-transcription.md` are the original design notes. Sources and papers are in `CITATIONS.md`. Code and data are under the Apache License 2.0.
