#!/usr/bin/env bash
# Collect the browser demo into dist/demo as a static site: the page, the held-out environments, Nushell in
# WebAssembly, Transformers.js and the ONNX Runtime files it loads. The model is not copied: set MODEL to its
# repository on the Hugging Face Hub (owner/name), which the page then downloads from.
#
#   MODEL=owner/nushell-lfm3-1.7b-onnx mise run build-demo
#
# NOTEBOOK_URL, MODEL_URL and GITHUB_URL fill the three links in the page's opening text. dist/runner is the
# Nushell runner alone (no model); DEMO_URL and GITHUB_URL fill its links.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUT=dist/demo
rm -rf "$OUT"
mkdir -p "$OUT/nu-wasm" "$OUT/vendor/@huggingface/transformers/dist" "$OUT/vendor/onnxruntime-web/dist"
# the libraries sit in vendor/ in the built site (some static hosts skip a folder called node_modules)
sed "s|./node_modules/|./vendor/|g" webgpu-demo/index.html > "$OUT/index.html"
cp webgpu-demo/environments.json webgpu-demo/prompt_format.json "$OUT/"
cp public/nu-wasm/nu_wasm.js public/nu-wasm/nu_wasm_bg.wasm "$OUT/nu-wasm/"
cp webgpu-demo/node_modules/@huggingface/transformers/dist/transformers.min.js "$OUT/vendor/@huggingface/transformers/dist/"
cp webgpu-demo/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded*.mjs webgpu-demo/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded*.wasm "$OUT/vendor/onnxruntime-web/dist/"
printf '{\n  "model": "%s",\n  "links": { "notebook": "%s", "model": "%s", "github": "%s" }\n}\n' "${MODEL:-nushell-lfm3-1.7b}" "${NOTEBOOK_URL:-}" "${MODEL_URL:-}" "${GITHUB_URL:-}" > "$OUT/config.json"

# The runner alone: Nushell in WebAssembly and the generated projects, no model.
RUN=dist/runner
rm -rf "$RUN"
mkdir -p "$RUN/nu-wasm"
cp webgpu-demo/runner.html "$RUN/index.html"
cp webgpu-demo/environments.json "$RUN/"
cp public/nu-wasm/nu_wasm.js public/nu-wasm/nu_wasm_bg.wasm "$RUN/nu-wasm/"
printf '{\n  "links": { "demo": "%s", "github": "%s" }\n}\n' "${DEMO_URL:-}" "${GITHUB_URL:-}" > "$RUN/config.json"
du -sh "$OUT" "$RUN"
echo "model: ${MODEL:-nushell-lfm3-1.7b (local: copy webgpu-demo/models/nushell-lfm3-1.7b to $OUT/models/ to serve it yourself)}"
