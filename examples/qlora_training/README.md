# llama.cpp/examples/qlora_training

Native QLoRA + Reward-Weighted SFT + GRPO training pipeline for quantized GGUF models.

The base model weights remain **frozen** (quantized tensors are skipped by `llama_set_param` because they are not `GGML_TYPE_F32`). Only freshly-allocated F32 LoRA A/B tensors are trained. The saved adapter GGUF is directly compatible with the existing `llama_adapter_lora_init` loader and `llama-export-lora` merge tool.

---

## Tools

| Binary | Description |
|---|---|
| `llama-finetune-qlora` | Phase 1 — QLoRA SFT on a quantized GGUF |
| `llama-finetune-rwsft` | Phase 2 — Reward-Weighted SFT (reward in dataset) |
| `llama-finetune-grpo` | Phase 3 — GRPO with pluggable reward function |

---

## Build

```bash
cmake -B build -DLLAMA_CURL=OFF
cmake --build build --target llama-finetune-qlora -j$(nproc)
```

---

## Phase 1 — QLoRA SFT (`llama-finetune-qlora`)

Trains LoRA adapters on a quantized GGUF model. The adapter is saved as a GGUF file that can be hot-loaded for inference without merging.

### Usage

```bash
llama-finetune-qlora \
  --model models/llama-3.2-1b-q4_k_m.gguf \
  --train-file data/train.jsonl \
  --lora-rank 16 \
  --lora-alpha 16 \
  --lora-targets "attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down" \
  --lora-out adapter.gguf \
  -e 3 -c 2048 --seed 42
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Path to quantized GGUF model |
| `--train-file` | *(required)* | JSONL training dataset |
| `--lora-rank` | `16` | LoRA rank (r) |
| `--lora-alpha` | `0` (= rank) | LoRA alpha; scale = alpha/rank |
| `--lora-targets` | see below | Comma-separated tensor name substrings to target |
| `--lora-out` | `adapter.gguf` | Output adapter GGUF path |
| `-e` / `--epochs` | `3` | Number of training epochs |
| `-c` / `--ctx-size` | `512` | Training context length |
| `--seed` | `42` | Random seed for LoRA init |
| `-lr` / `--learning-rate` | `1e-4` | Learning rate |

### Default LoRA targets

llama.cpp uses **internal GGUF tensor names**, not HuggingFace names:

| llama.cpp internal | HuggingFace equivalent | Layer type |
|---|---|---|
| `attn_q` | `q_proj` | Attention |
| `attn_k` | `k_proj` | Attention (**not in default** — see note) |
| `attn_v` | `v_proj` | Attention (**not in default** — see note) |
| `attn_output` | `o_proj` | Attention |
| `ffn_gate` | `gate_proj` | MLP |
| `ffn_up` | `up_proj` | MLP |
| `ffn_down` | `down_proj` | MLP |
| `ssm_in` | `in_proj` | Mamba / NemotronH SSM |
| `ssm_out` | `out_proj` | Mamba / NemotronH SSM |

Default: `attn_q,attn_output,ffn_gate,ffn_up,ffn_down,ssm_in,ssm_out`

> **Note on K/V projections**: `attn_k` and `attn_v` are excluded from the default targets because the KV cache write path uses `ggml_set_rows` (a scatter op). In ggml's current backward graph, the read and write branches of the KV cache are disconnected — gradients cannot flow from the attention output back through `set_rows` to the LoRA K/V tensors. Including them costs VRAM but contributes no learning signal. A custom fork of ggml that routes gradients through the KV cache write would be required to train K/V adapters correctly.

### Dataset format (JSONL)

Three formats are accepted:

**Chat format** (loss on assistant turns only):
```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
```

**Prompt/response format** (loss on response only):
```json
{"prompt": "What is the capital of France?", "response": "Paris."}
```

**Plain text** (loss on all tokens):
```json
{"text": "The quick brown fox jumps over the lazy dog."}
```

### Verify the adapter

```bash
# Hot-load for inference (no merge needed)
llama-cli --model models/llama-3.2-1b-q4_k_m.gguf \
  --lora adapter.gguf -p "Hello"

# Merge adapter into base model
llama-export-lora \
  --model models/llama-3.2-1b-q4_k_m.gguf \
  --lora adapter.gguf \
  --output merged.gguf
```

---

## Phase 2 — Reward-Weighted SFT (`llama-finetune-rwsft`)

Same as Phase 1 but each sample carries a `reward` field. The cross-entropy loss is scaled by the reward before backprop, so high-quality samples drive larger gradient updates.

### Additional dataset field

```json
{"prompt": "...", "response": "...", "reward": 0.85}
```

Reward is expected in `[0, 1]`. Samples with `reward=0` produce no gradient.

---

## Phase 3 — GRPO (`llama-finetune-grpo`)

Group Relative Policy Optimization. For each prompt, G responses are sampled from the current policy, scored by a reward function, and the group-normalized advantage drives a REINFORCE-style policy gradient update.

### Reward function

Two modes:

**Embedded stub** (default — edit `reward_fn.cpp` and recompile):
```cpp
float compute_reward(const char * prompt, const char * response) {
    // your scoring logic here
    return 1.0f;
}
```

**Runtime plugin** (`--reward-lib`):
```bash
# Build your reward shared library
g++ -shared -fPIC -o myreward.so myreward.cpp

# Pass it at runtime
llama-finetune-grpo \
  --model models/base.gguf \
  --train-file prompts.jsonl \
  --reward-lib myreward.so \
  --grpo-group-size 8 \
  --lora-out grpo_adapter.gguf
```

The plugin must export:
```cpp
extern "C" float compute_reward(const char * prompt, const char * response);
```

### Additional flags (Phase 3)

| Flag | Default | Description |
|---|---|---|
| `--grpo-group-size` | `8` | Number of responses sampled per prompt (G) |
| `--grpo-temperature` | `0.8` | Sampling temperature for response generation |
| `--reward-lib` | *(embedded stub)* | Path to shared library implementing `compute_reward` |

---

## Implementation notes

- **Natural freeze**: quantized tensors (`q4_k`, `q8_0`, etc.) are skipped by `llama_set_param` because they are not `GGML_TYPE_F32`. No explicit freeze logic is needed.
- **Graph sizing**: the adapter GGUF is pre-saved and registered in `params.lora_adapters` before context creation so `sched_reserve` accounts for the extra LoRA nodes (~4–6 nodes per A/B pair).
- **No warmup**: `params.warmup = false` is forced — warmup runs inference with PARAM-flagged tensors which causes a segfault.
- **Adapter compatibility**: saved adapter GGUF uses the metadata format expected by `llama_adapter_lora_init` (`general.type=adapter`, `adapter.type=lora`, `adapter.lora.alpha`).
- **MoE expert exclusion**: tensors whose names contain `_exps` (e.g. `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) are automatically excluded from LoRA targets. They use `ggml_mul_mat_id` (sparse expert dispatch) which has no backward implementation. Dense FFN layers on non-MoE layers are still trainable via `ffn_gate`, `ffn_up`, `ffn_down` targets.
- **Inplace ops in backward graph**: ggml's backward expander cannot compute gradients through inplace ops (CLAMP, SCALE, SET_ROWS, etc. with `view_src` set). These are silently skipped with a warning — they correspond to side-effect writes (KV cache, MoE weight normalization) that are not part of the loss computation path anyway.
