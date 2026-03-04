#!/usr/bin/env bash
# Phase 2: Prune Nemotron-3-Nano-30B-A3B using REAP expert stats
# Loads model in bf16 on CPU (~60GB RAM), prunes MoE experts, saves pruned model
# Run from WSL terminal: bash /mnt/f/llm-arena-storage/agent_workspace/run_prune.sh

set -e

WORKSPACE=~
PYTHON=python3

echo "=== Nemotron REAP Expert Pruning ==="
echo "Keep ratio: 0.33 (42/128 experts per MoE layer)"
echo "Stats file: $WORKSPACE/expert_stats_reap.json"
echo "Output dir: $WORKSPACE/nemotron-pruned-0.33"
echo ""

cd "$WORKSPACE"

"$PYTHON" nemotron_reap.py prune \
    --model unsloth/Nemotron-3-Nano-30B-A3B \
    --stats expert_stats_reap.json \
    --keep_ratio 0.33 \
    --output "$WORKSPACE/nemotron-pruned-0.33"

echo ""
echo "=== Prune complete ==="
echo "Pruned model saved to: $WORKSPACE/nemotron-pruned-0.33"
