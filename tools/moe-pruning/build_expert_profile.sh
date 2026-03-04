#!/usr/bin/env bash
# build_expert_profile.sh
# Builds llama.cpp with the expert-profile tool in WSL2 with CUDA.
# Run this from WSL2: bash /mnt/f/llm-arena-storage/agent_workspace/build_expert_profile.sh

set -e

LLAMA_SRC="/mnt/w/openclaw_agent/model_pruning_dependencies/llama.cpp"
BUILD_DIR="$LLAMA_SRC/build_expert"

echo "=== Building llama.cpp + expert-profile tool ==="
echo "  Source : $LLAMA_SRC"
echo "  Build  : $BUILD_DIR"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CUDA
cmake "$LLAMA_SRC" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    2>&1 | tail -20

# Build only the expert-profile target (fast)
cmake --build . --target llama-expert-profile --config Release -j$(nproc)

echo ""
echo "=== Build complete ==="
echo "  Binary: $BUILD_DIR/tools/expert-profile/llama-expert-profile"
echo ""
echo "=== Usage ==="
echo "  $BUILD_DIR/tools/expert-profile/llama-expert-profile \\"
echo "    -m /mnt/f/llm-arena-storage/nemotron-q4.gguf \\"
echo "    --jsonl /mnt/f/llm-arena-storage/agent_workspace/training-data.jsonl \\"
echo "    --output /mnt/f/llm-arena-storage/agent_workspace/expert_stats.json \\"
echo "    --n-experts 128 \\"
echo "    --ctx-size 2048 \\"
echo "    -ngl 99"
