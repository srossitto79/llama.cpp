#!/usr/bin/env bash
# Convert pruned Nemotron-3-Nano-30B-A3B (42 experts) to GGUF and quantize
# Run from WSL terminal: bash /mnt/f/llm-arena-storage/agent_workspace/run_convert_quantize.sh

set -e

LLAMA_CPP=../..
PRUNED_MODEL=~/nemotron-pruned-0.33
GGUF_F16=$PRUNED_MODEL/model-f16.gguf
Q4_OUT=$PRUNED_MODEL/model-q4km.gguf
Q5_OUT=$PRUNED_MODEL/model-q5km.gguf

PYTHON=python3
QUANTIZE=$LLAMA_CPP/build/bin/llama-quantize

export LD_LIBRARY_PATH="$LLAMA_CPP/build/bin:$LD_LIBRARY_PATH"
export PYTHONPATH="$LLAMA_CPP/gguf-py:$PYTHONPATH"

echo "=== Step 1: Convert HF safetensors -> GGUF (F16) ==="
echo "  Input : $PRUNED_MODEL"
echo "  Output: $GGUF_F16"
echo ""

"$PYTHON" "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$PRUNED_MODEL" \
    --outfile "$GGUF_F16" \
    --outtype f16

echo ""
echo "=== GGUF conversion done. File size: ==="
ls -lh "$GGUF_F16"

echo ""
echo "=== Step 2: Quantize GGUF to Q4_K_M ==="
"$QUANTIZE" "$GGUF_F16" "$Q4_OUT" Q4_K_M

echo ""
echo "=== Step 3: Quantize GGUF to Q5_K_M ==="
"$QUANTIZE" "$GGUF_F16" "$Q5_OUT" Q5_K_M

echo ""
echo "=== All done ==="
ls -lh "$PRUNED_MODEL"/*.gguf
