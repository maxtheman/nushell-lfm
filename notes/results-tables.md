# Results tables

Dataset `a23bbb10902d`: 20 generated environments, 401 verified chains, 3 requests each. 972 training rows, 231 held-out rows from 4 environments held out whole. A call is correct only if executing it returns the reference value. Numbers are read from `runs/target/report.json`, `runs/target/experiments.json`, `runs/providers/*.json` and `webgpu-demo/export.json`.

## 1. The model

| LFM3 1.7B | Requests | Well-formed | Runs | Correct | Count |
|---|---|---|---|---|---|
| Base model | 40 | 78% | 48% | 8% | 3 of 40 |
| With the adapter | 40 | 100% | 100% | 95% | 38 of 40 |
| With the adapter, requests used for nothing else | 191 | 100% | 97% | 92% | 176 of 191 |
| With the adapter, all held-out requests | 231 | 100% | 98% | 92.6% | 214 of 231 |
| With the adapter, 6 faulty requests left out | 225 | | | 95.1% | 214 of 225 |

Adapter: LoRA on all linear layers, rank 8, α 16, learning rate 2e-4, batch 4, 825 rows × 2 epochs, 8.2 M trainable parameters. Training and the first scoring pass took 50 minutes; peak memory 3.5 GB.

| Held-out environment | Requests | Correct |
|---|---|---|
| env-03 | 48 | 88% |
| env-12 | 44 | 98% |
| env-13 | 51 | 94% |
| env-14 | 48 | 90% |

(The 191 requests used for nothing else.)

## 2. Other models, same 40 requests

Same system prompt, tool definition and user text; no examples; no fine-tuning.

| Model | Well-formed | Runs | Correct | Count | Seconds per request |
|---|---|---|---|---|---|
| LFM3 1.7B with the adapter | 100% | 100% | 95% | 38 of 40 | |
| gpt-6-luna, reasoning off | 100% | 78% | 32.5% | 13 of 40 | 2.2 |
| claude-haiku-5-5 | 100% | 85% | 17.5% | 7 of 40 | 1.2 |
| LFM3 1.7B base | 78% | 48% | 8% | 3 of 40 | |

- One reference value per request, compared exactly. A call that answers in another shape counts as wrong.
- gpt-6-luna does not accept function tools with reasoning on at this endpoint; it ran with `reasoning_effort: "none"`.
- 40 requests: the 95% interval on one score is about ±13 points.

## 3. Sweep and placement

200 training rows, 1 epoch, 24 held-out requests. One request is 4.2 points.

| Rank | lr 1e-4 | lr 2e-4 | lr 4e-4 |
|---|---|---|---|
| 8 | 71% (17) | **75% (18)** | 42% (10) |
| 16 | 71% (17) | 62% (15) | 54% (13) |

| Adapter on (rank 8, lr 2e-4) | Correct | Held-out loss |
|---|---|---|
| All linear layers | 75% (18) | 0.127 |
| MLP only | 71% (17) | 0.134 |
| Convolution only | 71% (17) | 0.163 |
| Attention only | 62% (15) | 0.148 |

## 4. Export and browser

| Build | Requests | Correct |
|---|---|---|
| MLX, 8-bit base + adapter | 24 | 22 |
| ONNX, 4-bit base + adapter, ONNX Runtime | 24 | 22 (the same 22) |
| ONNX in the browser, WebGPU | 60 | 54 |

- Exported model: 1.10 GB (142 adapted weights).
- WebAssembly Nushell against native on the 401 reference calls: 382 identical, 19 differ only in the order `glob` returns entries.

## 5. Data

| | Count |
|---|---|
| Environments (held out) | 20 (4) |
| Verified chains (held out) | 401 (77) |
| Requests per chain | 3 |
| Requests written by the model | 935 |
| Requests that fell back to the composed description | 268 (22%) |
| Training rows / held-out rows | 972 / 231 |
| Held-out chains with a command identical to a training command | 15 of 77 |
| Held-out chains whose pattern also occurs in training | 50 of 77 |
| Held-out chains that are new combinations | 27 of 77 |

## 6. The 17 wrong calls

| Cause | Count |
|---|---|
| Leaves out a step the request names | 6 |
| Writes a different step than asked | 5 |
| Request does not contain what the reference needs | 6 |

The 11 model-side errors come from 5 chain patterns.

## 7. Experiments not in the shipped pipeline

| Experiment | Result |
|---|---|
| Grader (D1 2B + LoRA), 400 held-out rows | failure type 96%, broken or not 97%, span 94% (base 27%, 61%) |
| Guided decoding, 8 decoders, 40 requests | 92–95% against 95% unguided |
| Steering away from the prompt without its request, 231 requests | 89.6–90.9% against 92.6% unguided |
| 4-bit base weights, existing adapter, 231 requests | 91.3% |
| 3-bit | 73.2% |
| 2-bit, plain requantisation | 0% |
