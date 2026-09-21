#!/usr/bin/env bash
# Starts a persistent vLLM OpenAI-compatible server for the robot pipeline's stage-1
# (primary) VLM -- run this once, before robot_implementation.py / robot_web_app.py.
#
# Why a separate script and a separate conda env: vLLM pins its own torch/CUDA build
# (currently torch 2.13.0+cu130 as of vllm 0.29.0), which does not match the
# torch build the `sam_oed` env's detectron2 was compiled against (torch 2.11.0+cu128).
# Installing vllm into sam_oed would force a torch reinstall there and likely break
# detectron2's compiled CUDA extensions. Keeping vLLM in its own env avoids that
# entirely -- the robot pipeline only needs an HTTP client (httpx) to talk to it, not
# vllm itself.
#
# One-time setup (already done if you're re-reading this):
#   conda create -n vllm_serve python=3.12 -y
#   conda activate vllm_serve
#   pip install vllm
#
# Usage:
#   robot_code/start_vllm_server.sh                       # defaults below
#   robot_code/start_vllm_server.sh --gpu 0 --port 8001    # override
#
# Then point the robot pipeline at it (both scripts default --vllm-url to this
# server's address already, so this is only needed if you changed --port/--gpu):
#   python robot_code/robot_implementation.py --vllm-url http://127.0.0.1:8000/v1
#   python robot_code/robot_web_app.py --vllm-url http://127.0.0.1:8000/v1
#
# Stage-2 (the bigger --verify-model-id checkpoint, Qwen3-VL-8B by default) is NOT
# served here -- it's hit rarely (only when stage 1 says "unknown"), so it isn't worth
# a second always-on server; it stays on the local transformers path, lazy-loaded onto
# whichever GPU has the most free memory once this server has claimed its own (see
# QWEN_DEVICE_MAP=single in robot_implementation.py).

set -euo pipefail

GPU=1
PORT=8000
MODEL="Qwen/Qwen3-VL-4B-Instruct"
UTIL=0.85
MAX_MODEL_LEN=8192

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --gpu-memory-utilization) UTIL="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

VLLM_ENV_BIN="$(conda info --base 2>/dev/null)/envs/vllm_serve/bin"
if [[ ! -x "$VLLM_ENV_BIN/vllm" ]]; then
  echo "vllm CLI not found at $VLLM_ENV_BIN -- run the one-time setup in this script's header first." >&2
  exit 1
fi

# One of vllm's dependencies (sqlite3's ICU extension, pulled in transitively) links
# against the *system* libstdc++ at runtime instead of the conda env's own newer one,
# and the system copy is too old (missing CXXABI_1.3.15+) -- fails with an ImportError
# on `import sqlite3` deep inside vllm's own startup otherwise. The env ships a libstdc++
# new enough; just make sure the dynamic linker finds it first.
export LD_LIBRARY_PATH="$(conda info --base 2>/dev/null)/envs/vllm_serve/lib:${LD_LIBRARY_PATH:-}"

echo "[start_vllm_server] serving $MODEL on GPU $GPU, port $PORT" \
     "(gpu-memory-utilization=$UTIL, max-model-len=$MAX_MODEL_LEN) ..."
echo "[start_vllm_server] first start downloads/compiles and can take a few minutes;" \
     "subsequent starts are faster (cached)."

exec env CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_ENV_BIN/vllm" serve "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --served-model-name "$MODEL" \
  --trust-remote-code
