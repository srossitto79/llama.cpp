#!/usr/bin/env bash
# Perplexity evaluation — runs on CPU only (-ngl 0) so GPU training is unaffected.
# Usage:
#   ./tools/moe-pruning/eval-ppl.sh <model.gguf> [adapter.gguf] [max_chunks]
#
# Examples:
#   # Base model only, all chunks (~9 h)
#   ./tools/moe-pruning/eval-ppl.sh ~/nemotron-3-trader-15b-Q4_K_M.gguf
#
#   # With LoRA adapter, quick 20-chunk run (~50 min)
#   ./tools/moe-pruning/eval-ppl.sh ~/nemotron-3-trader-15b-Q4_K_M.gguf ~/nemotron-3-trader-15b-lora.gguf.ckpt3.gguf 20

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PPL_BIN="$REPO_ROOT/build/bin/llama-perplexity"
EVAL_FILE="$SCRIPT_DIR/ppl-eval-val.txt"

MODEL="${1:-}"
LORA="${2:-}"
MAX_CHUNKS="${3:-0}"   # 0 = all chunks

if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 <model.gguf> [adapter.gguf] [max_chunks]" >&2
    exit 1
fi

if [[ ! -f "$PPL_BIN" ]]; then
    echo "Error: llama-perplexity not found at $PPL_BIN — did you run cmake --build?" >&2
    exit 1
fi

if [[ ! -f "$EVAL_FILE" ]]; then
    echo "Error: eval file not found: $EVAL_FILE — run extract_ppl.py first" >&2
    exit 1
fi

# If max_chunks requested, truncate input to approx that many 20k-token chunks.
# ~4 chars/token × 20480 tokens/chunk = ~81920 chars/chunk.
ACTIVE_FILE="$EVAL_FILE"
TMPFILE=""
if [[ "$MAX_CHUNKS" -gt 0 ]]; then
    MAX_CHARS=$(( MAX_CHUNKS * 81920 ))
    TMPFILE="$(mktemp /tmp/ppl-eval-XXXXXX.txt)"
    head -c "$MAX_CHARS" "$EVAL_FILE" > "$TMPFILE"
    ACTIVE_FILE="$TMPFILE"
fi

LORA_ARG=""
if [[ -n "$LORA" ]]; then
    LORA_ARG="--lora $LORA"
    echo "=== PPL eval: $(basename "$MODEL") + $(basename "$LORA") (CPU only) ==="
else
    echo "=== PPL eval: $(basename "$MODEL") base (CPU only) ==="
fi

echo "    eval file : $ACTIVE_FILE"
echo "    context   : 16384 tokens, batch 16384, stride 8192"
if [[ "$MAX_CHUNKS" -gt 0 ]]; then
    echo "    max chunks: $MAX_CHUNKS  (~$MAX_CHUNKS × 2.5 min)"
fi
echo ""

"$PPL_BIN" \
    --model "$MODEL" \
    $LORA_ARG \
    -f "$ACTIVE_FILE" \
    -c 16384 \
    -b 16384 \
    -ub 512 \
    --ppl-stride 8192 \
    -ngl 0 \
    -t "$(nproc)"

if [[ -n "$TMPFILE" ]]; then
    rm -f "$TMPFILE"
fi
