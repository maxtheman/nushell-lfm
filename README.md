# Teaching a small model to drive Nushell

One [marimo](https://marimo.io) notebook that generates a Nushell tool-call dataset from scratch and fine-tunes two small models on Apple Silicon with MLX, offline:

- a **target** (LFM3 1.7B) that turns a plain request into a `run_nushell` tool call;
- a **grader** (D1 2B) that reads a request and a call and says whether it is broken, how, and where.

The finished target also runs in the browser on WebGPU, next to Nushell compiled to WebAssembly.

The page is `notebook.py`; the pipeline it drives is the `nushell_lfm` package, and the scripts under `scripts/` call the same functions from the command line.

## Setup

You need an Apple Silicon Mac, [mise](https://mise.jdx.dev), and the two MLX checkpoints.

1. Put the checkpoints in a folder, keeping these folder names:

   ```
   models/
     d1-2b-mlx-8bit/         grader
     lfm3-1.7b-mlx-8bit/     target
   ```

   To keep them elsewhere, create `mise.local.toml` (it is not committed):

   ```toml
   [env]
   MODELS_DIR = "/path/to/your/models"
   ```

2. Install the toolchain (uv, Nushell, Node) and the Python and JavaScript packages:

   ```sh
   mise run setup
   ```

3. For the browser export only, fetch the 4-bit ONNX build of the target (about 1.1 GB):

   ```sh
   mise run fetch-onnx
   ```

Model loading, data generation and training run offline. Besides setup (`mise run setup`, and `build-wasm` if you rebuild the module), the three `fetch-*` tasks are the only steps that use the network, and their outputs (`data/*.json`) are committed.

## Run

```sh
mise run notebook      # open the notebook for editing
mise run app           # serve it as an app
mise run smoke         # the whole pipeline at its smallest size, a few minutes (writes to runs/smoke)
mise run requests      # have the base model write the requests (cached)
mise run experiments   # sweeps, ablations, final adapters, guidance comparison, export, attention capture (about three hours)
mise run check-wasm    # run every dataset sample in WebAssembly Nushell and compare with native
mise run eval-endpoint -- --base-url URL --model NAME   # score any OpenAI-compatible endpoint on the same held-out requests
mise run experiments -- --only grader   # one stage: grader, target, export or activation
```

The smoke size proves the code path and is too small to learn anything.

### Scoring other models

`scripts/eval_endpoint.py` sends the 231 held-out requests to any OpenAI-compatible chat endpoint with the same system prompt, tool and user text our model gets, executes the call that comes back, and applies the same rule: correct means it returns the reference value. The key is read from an environment variable (`--api-key-env NAME`), results go to `runs/providers/`, and `--compare` prints every endpoint scored so far next to our model. `--export` writes the evaluation set to `runs/eval/eval_set.jsonl`. The notebook's last section, **Other models**, runs the same thing from the page: enter up to three endpoints in the form (`PROVIDER_n_URL`, `PROVIDER_n_KEY`, `PROVIDER_n_MODELS` as a comma-separated list, or set those variables in `mise.local.toml`), pick how many requests, press ▶. This is the one step that sends data off the machine: the request text and the generated file names go to the endpoint you name.

## How the data is made

Nothing in the training data is written by hand. Each stage is seeded and can be regenerated; `runs/generated/manifest.json` records the environment seeds and generator settings of a build.

| Stage | What happens |
|---|---|
| Environments | Each environment is a small project generated from a seed: source files, logs, a JSON table, a TOML config, docs, a script, with its own folder names, language, file names, column names, people and data. Faker supplies names and prose; polyfactory builds config and table records from dataclasses. |
| Commands | `assets/nu_commands.json` is every built-in command of the installed `nu` with its input and output types and parameters (`mise run cache-nu-docs`). |
| Chains | A chain is a walk on a graph of data kinds: a source, then stages whose input matches what the last step produced. Commands that need arguments have recipes that draw them from the environment; commands that take none are added automatically from their declared types. A sampled chain is kept only if it runs, gives the same output twice, returns something, and every automatically added stage changes the result. |
| Weights | The main prior is Nushell-native: how often each command appears in the example code of the Nushell book and cookbook (`mise run fetch-nu-examples`). Public Unix usage data (`mise run fetch-usage`) sets how much of the dataset each category gets and steers at most 30% of the chains. |
| Breaks | The ten failure types of the original design, each generated from its listed examples and kept only if running it agrees with the label (table below). |
| Requests | Written by the base model, K per chain, in six styles: inconsistent formatting, spelling mistakes or abbreviations, less technical language, naming a part for the whole, an analogy from the same domain, an analogy from another domain. A request that loses a term, path or number the command needs falls back to a description composed from the chain's steps. |
| Splits | One environment in five is held out whole. Every score is for projects the models never saw. |

### How each failure type is confirmed

| Failure type | Kept only if |
|---|---|
| `request_misunderstood`, `wrong_data_source`, `wrong_operation_or_argument`, `broken_chain_or_function` | it runs and its output differs from the reference |
| `wrong_output_or_side_effect` | it runs and either the output differs or a file was written |
| `runtime_or_environment_failure` | it parses, then fails or times out |
| `invalid_nushell` | Nu reports a parse error |
| `wrong_tool_or_malformed_call` | it does not parse as a `run_nushell` call |
| `grader_false_pass` | a hard-coded answer that matches the reference here, where the reference changes in a second copy of the environment |
| `grader_false_fail` | a correct answer that is not identical to the reference but is the same once formatting, order and type are set aside; labelled not broken |

## How the models are trained and scored

- **Grader.** LoRA on the D1 model for three questions read in one pass: broken (yes/no), failure type, and the faulty span (start and end pointer). The protocol is a sweep over rank and learning rate, an ablation of where the adapter goes (all linear layers, attention only, MLP only, convolution only), and a final adapter. Guidance is compared two ways: extrapolating from the base model through the adapter, and autoguidance against the same adapter a quarter of the way through training.
- **Target.** The same sweep and placement ablation, chosen by the share of held-out requests whose generated call returns the reference value when executed. Then the full pipeline: `P_good` is trained on the reference calls; its outputs on requests it was not fitted on are executed; execution alone decides which are wrong, and the grader is asked only what kind of mistake each wrong one is; `P_bad` is trained on those wrong calls plus generated breaks of the same kinds. Guided decoding steers `P_good` away from `P_bad`, from a weaker version of itself, or from the base model, and each is scored by executing its calls.
- **Execution.** Every command runs in a throwaway copy of its environment. On macOS the Nu process is confined with `sandbox-exec`: no network, and no writes outside the copy.

## Results

From `mise run experiments` on the default settings: 20 generated environments, 401 verified chains, 3 requests each; 4 environments held out whole. Dataset `edd14fe576df`. Every number is for the held-out environments.

**Target (LFM3 1.7B).** A call counts only if executing it returns the reference value.

The number to quote is **92%**: the final adapter on the 191 held-out requests that took no part in choosing its settings (176 of 191; by environment 88%, 98%, 94%, 90%). The base model is at 8% on the comparison set below.

| On the 40 held-out requests every decoder was run on | Well-formed | Runs | Returns the reference value |
|---|---|---|---|
| Base model | 78% | 48% | 8% |
| `P_good` (LoRA, all linear layers, rank 8, lr 2e-4, 825 rows × 2 epochs) | 100% | 100% | 95% |
| `P_good` steered away from `P_bad` (weight 0.25, floor 0.1) | 100% | 100% | 95% |
| `P_good` steered away from `P_bad` (weight 0.5 or 1.0) | 100% | 98–100% | 92% |
| `P_good` steered away from a weaker `P_good` (weight 0.5 / 1.0) | 100% | 95–100% | 95% / 92% |
| `P_good` steered away from the base model (weight 0.5) | 100% | 100% | 92% |

- These 40 include the 24 the sweep chose settings on, and come from two of the four held-out environments, so read them as a comparison between decoders, not as the score. One request is 2.5 points.
- Guidance did not improve on the plain adapter at any weight tried, and the token-level record says why. Read along the reference calls of those 40 requests (1,290 tokens), `P_good`'s first choice is a median of 11.1 nats ahead of its runner-up. The plausibility floor of 0.1 only lets guidance pick a token within 2.3 nats, which leaves it a choice at 19 tokens. Steering away from `P_bad` at weight 0.5 changes 3 tokens, 2 of them away from the reference. `P_bad` itself saw only 3 of `P_good`'s own mistakes (3 wrong calls in 48 probes); the rest of its data is generated breaks. The notebook's Guidance section has the inequality, and the 3D view shows it token by token.
- The 4-bit ONNX export returns the reference value on 22 of 24 requests under ONNX Runtime. The 8-bit MLX adapter gets the same 22 of the same 24, so the export lost nothing measurable.

Sweep and placement ablation (200 rows, one epoch, 24 held-out requests; one request is about 4 points, so neighbours are within noise and the choice among the top settings is not reliable):

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

**Grader (D1 2B).** Final adapter: MLP layers only, rank 8, lr 1e-4, 2,400 rows × 2 epochs, scored on 400 held-out rows.

| | Failure type | Broken or not | Span found |
|---|---|---|---|
| Base model | 27% | 61% | |
| Final adapter | **96%** | **97%** | 94% |

- Half of those rows are not broken, which is the easy half. On the 200 broken rows alone the failure type is right for 92.5%; the weakest type is `runtime_or_environment_failure` (18 of 22).
- The sweep (160 rows, one epoch, 100 held-out rows) put rank 8 and rank 16 at lr 1e-4 level (54% and 53% on failure type); lr 4e-4 diverged at both ranks. The four placements were within five points of each other (54–58%); MLP-only was chosen by dev loss.
- Rows mattered far more than settings: the same configuration scored 58% on 160 rows, 85% on 800 and 96% on 2,400.
- Guidance changed failure-type accuracy by at most half a point in either scheme.

**More data helped the target too,** though the comparison is loose: an earlier run of the same protocol on 162 chains in 10 environments gave 72% on its own 40 held-out requests, with different held-out environments and a different chosen setting. Its misses were mostly patterns with too few training chains, such as opening a JSON file.

**What the held-out score measures.** No environment is shared between training and held-out data. Within the 77 held-out chains, 15 have a command identical to a training command (the project differs, the command happens not to), 50 follow a chain pattern that also occurs in training, and 27 are new combinations of familiar steps.

## In the browser

- `nu-wasm/` is a small Rust crate that links the Nushell crates of the same release as the native `nu` and adds `ls`, `glob`, `open`, `save`, `cd`, `path type` and friends over an in-memory workspace. The built module is in `public/nu-wasm/` (`mise run build-wasm` rebuilds it). The notebook's REPL uses it. `mise run check-wasm` runs every sample both ways: of 580 correct calls, 561 print the same answer and 19 differ only in the order `glob` returns entries (directory order on disk, sorted in the browser). The demo page therefore gets its reference value by running the reference call in the same in-browser Nushell.
- The export step adds `P_good` to the 4-bit ONNX build of the target as low-rank branches beside each quantised weight, then generates with the exported graph on ONNX Runtime and executes the calls to confirm it still works.
- **Load Nu** in the notebook opens `webgpu-demo/` in a new tab: the exported target on WebGPU (Transformers.js) and a Nushell session, on one page. `mise run demo` serves the same page without the notebook.

## Limits to keep in mind

- **Correct means "returns the reference value".** A generated call counts as correct only if executing it gives the reference output. A call can run and still not be what the user meant in ways this does not catch, and a different but reasonable answer counts as wrong. That simplification is deliberate for this first version.
- **Held-out environments share chain patterns with training.** Scores measure a known kind of request in an unseen project, not new kinds of request.
- **Requests are written by a small model.** About a fifth fall back to composed descriptions, which read stiffly. About 4% of the requests in this build repeat part of the rewriting instruction ("one line", "don't write code"); new builds reject those.
- **The target was fitted on 825 of its 972 training rows.** The other 15% of chains are kept aside to probe it for mistakes.
- **Training from the notebook replaces the saved adapters.** The ▶ buttons at the standard and protocol budgets first copy the current ones to `runs/previous/`; the smoke budget never touches them.
- **The budgets are small.** Sweeps use a few hundred rows so the whole protocol finishes in hours on a 24 GB machine. Models are loaded one at a time.
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
  breaks.py              the ten failure types
  dataset.py             requests, splits, the dataset on disk
  grader.py              D1 LoRA, readouts, the grader's experiments
  target.py              LFM3 LoRA, guided decoding, the target's experiments
  export.py              ONNX export for the browser
  model_view.py          what both models do on held-out examples, and the 3D view of it
  widgets/               notebook widgets: StepBar, NuRepl, ModelView
lowbit.py              2-bit and 1-bit versions of the target, retrained and distilled
scripts/               command-line entry points and the data fetchers
nu-wasm/               Nushell for WebAssembly (Rust)
public/nu-wasm/        the built module
webgpu-demo/           the browser page (model on WebGPU + Nushell session)
assets/, data/         Nu's command list, usage data, Unix-to-Nushell map, Nushell-native prior
notes/                 lab notes and their charts
runs/                  generated datasets, adapters and reports (not committed)
```

`training-sketch.png` and `training-sketch-transcription.md` are the original design notes; `activation_3d_sketch.py` and `activation_view_demo.py` are earlier studies of the 3D view. `train.py` and `data/train.jsonl` are the first version of this project, a single LoRA on hand-written requests. The notebook does not use them.
