# Publishing: what goes where

Nothing is uploaded yet.

## Hugging Face account

- Signed in on this machine as `maxhugger` (Pro, no organisations); the stored token can write.
- There now, all private: model `d1-2B-NeuralEngine`, six `autotrain-data-*` datasets, Space `TRELLIS`.

## What can go there

| Kind | Contents | Size |
|---|---|---|
| Model repo | The LoRA adapter (`adapter_good.safetensors`), prompt format, report, model card | tens of MB |
| Model repo | The 4-bit ONNX build with the adapter merged in (what the browser loads) | 1.10 GB |
| Dataset repo | Training and held-out rows, manifest with seeds, the evaluation set, dataset card | under 10 MB |
| Space (static) | The demo page: Nushell in WebAssembly plus the model on WebGPU | about 96 MB; the model is fetched from the ONNX repo |
| Space (static) | Nushell in WebAssembly alone, no model | 13 MB plus a page |

## The Nushell WebAssembly runner as a Space

- A static Space serves plain files; the runner is `nu_wasm.js` and `nu_wasm_bg.wasm` plus a page.
- The demo already has a `nu` mode for typing Nushell directly.
- A runner-only Space loads at once and needs no WebGPU. Its page does not exist yet.
- The model in the demo needs WebGPU; without it the demo falls back to WebAssembly and is slow.

## Needed before upload

Names and visibility

1. GitHub: account or organisation, repository name, public from the start or private first.
2. Hugging Face names for: the LoRA adapter, the ONNX build the browser loads, the dataset, the demo Space, and the runner-only Space (if wanted).
3. Public or private for each of those at first.

What goes up

4. The LoRA adapter and the ONNX build: both, or only one.
5. The dataset: only the 1,203 request rows, or also the generated environments (seeds are enough to regenerate them).
6. The experiments' artefacts on Hugging Face (grader adapter, grader rows with breaks), or only their code on GitHub.
7. Saved results in the repository: `runs/` is not committed. The reports behind the README numbers and the two provider result files could be committed under `results/`, or stay out.
8. A runner-only Space (Nushell in WebAssembly, no model): yes or no.

Where things are hosted

9. The demo: Hugging Face static Space, or your own domain, or both.
10. "The training process notebook" link in the demo: a read-only export of the notebook hosted somewhere (where), or the notebook file on GitHub.
11. The lab notes: where they are published, and whether the repository links to them.

Text and credit

12. The lab-notes draft, to work the results into.
13. Author name and contact as they should appear on the model card, the dataset card and the README.
14. How the base model is named and linked on the cards.

Timing

15. Wait for the 2-bit and cross-fit results, or publish now and add them later.
16. Whether the hosted models are run on all 231 requests first (40 so far).

## In the repository, not code

| Files | What they are |
|---|---|
| `data/command_usage.json`, `data/unix_to_nushell.json`, `data/nushell_examples_usage.json` | Usage counts and the Unix-to-Nushell map, with sources and checksums |
| `assets/nu_commands.json` | Every built-in Nushell command with its types |
| `public/nu-wasm/nu_wasm.js`, `nu_wasm_bg.wasm` | The built WebAssembly module |
| `webgpu-demo/environments.json`, `export.json`, `prompt_format.json` | Held-out environments and requests, the export check, the prompt format |
| `webgpu-demo/package.json`, `package-lock.json` | The demo's JavaScript packages |
| `training-sketch.png`, `training-sketch-transcription.md` | The original design sketch |
