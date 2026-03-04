#!/usr/bin/env bash
# Phase 2: Prune Nemotron-3-Nano-30B-A3B using REAP expert stats
# Loads model in bf16 on CPU (~60GB RAM), prunes MoE experts, saves pruned model
# Run from WSL terminal: bash /mnt/f/llm-arena-storage/agent_workspace/run_prune.sh

set -e

WORKSPACE=~
PYTHON=python

echo "=== Nemotron REAP Expert Pruning ==="
echo "Keep ratio: 0.5 (64/128 experts per MoE layer)"
echo "Stats file: $WORKSPACE/expert_stats_reap.json"
echo "Output file: $WORKSPACE/nemotron-3-trader-15b-Q4_K_M.gguf"
echo ""

"$PYTHON" gguf_prune.py \
    --input ~/nemotron-3-nano-30b-Q4_K_M.gguf \
    --output ~/nemotron-3-trader-15b-Q4_K_M.gguf \
    --stats expert_stats_reap.json \
    --keep_ratio 0.5

echo ""
echo "=== Prune complete ==="
echo "Pruned model saved to: $WORKSPACE/nemotron-3-trader-15b-Q4_K_M.gguf"
