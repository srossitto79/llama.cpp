#!/usr/bin/env bash

/mnt/w/openclaw_agent/model_pruning_dependencies/llama.cpp/build_expert/bin/llama-expert-profile \
    -m ~/nemotron-3-nano-30b.gguf \
    --jsonl /mnt/f/llm-arena-storage/agent_workspace/training-data.jsonl \
    --output /mnt/f/llm-arena-storage/agent_workspace/expert_stats_reap.json \
    --n-experts 128 \
    --ctx-size 16384 \
    --type-k f16 \
    --type-v f16 \
    -ngl 32 \
    --save-every 1 \
    -t 24