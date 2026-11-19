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

- Repository names for the adapter, the ONNX build, the dataset and the Spaces.
- Public or private for each.
- The GitHub repository name.

## In the repository, not code

| Files | What they are |
|---|---|
| `data/command_usage.json`, `data/unix_to_nushell.json`, `data/nushell_examples_usage.json` | Usage counts and the Unix-to-Nushell map, with sources and checksums |
| `assets/nu_commands.json` | Every built-in Nushell command with its types |
| `public/nu-wasm/nu_wasm.js`, `nu_wasm_bg.wasm` | The built WebAssembly module |
| `webgpu-demo/environments.json`, `export.json`, `prompt_format.json` | Held-out environments and requests, the export check, the prompt format |
| `webgpu-demo/package.json`, `package-lock.json` | The demo's JavaScript packages |
| `training-sketch.png`, `training-sketch-transcription.md` | The original design sketch |
