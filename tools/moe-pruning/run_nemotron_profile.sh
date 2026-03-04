#!/usr/bin/env bash

../../build_expert/bin/llama-expert-profile \
    -m ~/nemotron-3-nano-30b.gguf \
    --jsonl ./training-data.jsonl \
    --output ./expert_stats_reap.json \
    --n-experts 128 \
    --ctx-size 16384 \
    --type-k f16 \
    --type-v f16 \
    -ngl 32 \
    --save-every 1 \
    -t 24