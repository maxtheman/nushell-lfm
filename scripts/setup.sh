#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "==> [1/4] Installing the toolchain with mise (uv, nushell, node)"
mise trust --yes >/dev/null 2>&1 || true
mise install

echo "==> [2/4] Installing Python dependencies into .venv"
if [ ! -d ".venv" ]; then
  mise exec -- uv venv --python 3.12 .venv
fi
mise exec -- uv pip install --python .venv/bin/python -r requirements.txt

echo "==> [3/4] Installing the browser demo's JavaScript packages"
(cd webgpu-demo && mise exec -- npm install --no-audit --no-fund)

echo "==> [4/4] Checking the environment"
mise run verify-env

echo ""
echo "Ready. Open the notebook with:  mise run notebook"
