#!/usr/bin/env bash
# Collect the browser demo into dist/demo as a static site: the page, the held-out environments, Nushell in
# WebAssembly, Transformers.js and the ONNX Runtime files it loads. The model is not copied: set MODEL to its
# repository on the Hugging Face Hub (owner/name), which the page then downloads from.
#
#   MODEL=owner/nushell-lfm3-1.7b-onnx mise run build-demo
#
# NOTEBOOK_URL, MODEL_URL and GITHUB_URL fill the three links in the page's opening text.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUT=dist/demo
rm -rf "$OUT"
mkdir -p "$OUT/nu-wasm" "$OUT/node_modules/@huggingface/transformers/dist" "$OUT/node_modules/onnxruntime-web/dist"
cp webgpu-demo/index.html webgpu-demo/environments.json webgpu-demo/prompt_format.json "$OUT/"
cp public/nu-wasm/nu_wasm.js public/nu-wasm/nu_wasm_bg.wasm "$OUT/nu-wasm/"
cp webgpu-demo/node_modules/@huggingface/transformers/dist/transformers.min.js "$OUT/node_modules/@huggingface/transformers/dist/"
cp webgpu-demo/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded*.mjs webgpu-demo/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded*.wasm "$OUT/node_modules/onnxruntime-web/dist/"
printf '{\n  "model": "%s",\n  "links": { "notebook": "%s", "model": "%s", "github": "%s" }\n}\n' "${MODEL:-nushell-lfm3-1.7b}" "${NOTEBOOK_URL:-}" "${MODEL_URL:-}" "${GITHUB_URL:-}" > "$OUT/config.json"
du -sh "$OUT"
echo "model: ${MODEL:-nushell-lfm3-1.7b (local: copy webgpu-demo/models/nushell-lfm3-1.7b to $OUT/models/ to serve it yourself)}"
