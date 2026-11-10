# MoE QLoRA follow-up plan

As of 2026-11-10. QLoRA trains real dense and MoE models on CPU, Vulkan and CUDA, and MoE pruning works on any
architecture with stacked experts. Metal, Qwen3.5 (hybrid) training and a large MoE run are untested or unsupported.

## Where things stand

| Commit | What it does | Verified by |
| --- | --- | --- |
| (this branch) | CI fixes; architecture-generic MoE pruning; four pruning bugs fixed; Q8_0 reference back to upstream; real-model tests | `ctest -L main` 23/23 on Vulkan and CUDA; `ctest -L real-model` 5/5; `test-backend-ops -o SET_ROWS` on Vulkan1 |
| 7fbc7444f | Gradient flows through quantized experts (dequantized to F32 for the transpose); CUDA dequantizing CPY for every type; Vulkan Q2_K/Q5_K/Q6_K to F32 | `test-moe-backward` quantized cases (fail without the fix); `test-backend-ops -o CPY` 276/276 on CUDA0 and Vulkan1 |
| 83e45e771 | MUL_MAT_ID backward with broadcast src1; batched GET_ROWS backward flattened to the 2D kernel; new QLoRA tests | `test-moe-backward`; `test-qlora-finetune-*` on dense, MoE, MoE Q4_0, MoE Q4_K, CPU and GPU losses equal |
| fa42c0626 | Upstream merge (cf34d3113) | Vulkan and CUDA builds |

Run the checks with:

```sh
ctest -C Release -L main --output-on-failure
ctest -C Release -L real-model --output-on-failure   # downloads about 850 MB once into models/hf-cache/
```

### Real-model results (2026-11-10, RTX 3090)

QLoRA, rank 16, lr 1e-4, ctx 128, 8 identical samples, train loss per epoch (the CPU run agrees within 0.004):

| Model | Epoch 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B Q4_0 (dense) | 0.835 | 0.520 | 0.153 | 0.027 |
| Qwen3-0.6B-MoE-A0.5B Q4_K_M (32 experts, top 16) | 0.920 | 0.598 | 0.454 | 0.314 |

Pruning the same MoE model (157-conversation calibration set, Vulkan): perplexity 93.89 baseline, 93.92 at 12.5%,
102.09 at 25%, 133.54 at 50% of the experts removed. Hard pruning at 25%: 416.5 MB to 371.6 MB, within 0.2% of the
soft-pruned perplexity.

### Fixed in this branch

- CI: QLoRA tests referenced `llama-export-lora` in builds that don't have it (GGML_BACKEND_DL, wasm, android);
  the RPC header asserted upstream's op count; arm64 repacked weights broke the QLoRA backward pass
  (`llama-finetune-qlora` now disables repacking); OpenVINO cannot run training graphs (tests skipped there).
- `test-quantize-fns` q8_0: ff2fd5217 had changed the Q8_0 reference quantizer (C and gguf-py) to `max/-128`;
  restored upstream's `amax/127`, so Q8_0 files match upstream again.
- Vulkan SET_ROWS claimed Q3_K/Q4_K/MXFP4 support without pipelines.
- Pruning: the collector read the strided `ffn_moe_topk` view as contiguous and attributed routing to the wrong
  experts (importance cache bumped to version 2); `std::rename` failed on Windows when the target existed; the hard
  step kept its report open while replacing it; calibration needed `ffn_moe_weights_norm`, which only some
  architectures compute.
- `test-qlora-finetune.cmake` parsed `0.6024` as 624 (CMake's `REGEX REPLACE` re-applies `^`); it only showed on
  a real model, whose loss has inner zeros.

## Open items

Effort: S = hours, M = a day or two, L = several days.

| # | Item | Problem | Proposed fix | Proof it works | Effort |
| --- | --- | --- | --- | --- | --- |
| 1 | Re-run Gemma 4 pruning | Existing Gemma 4 26B A4B profiles and importance caches came from the collector bug above | Re-run `llama-prune analyze` (version 1 caches are ignored automatically) and regenerate the hard-pruned GGUFs | New perplexity table; compare with the old one | S, needs the model |
| 2 | Qwen3.5 / hybrid models cannot be trained | `GATED_DELTA_NET` has no backward pass, so QLoRA aborts on Qwen3.5 (tried Qwen3.5-0.8B-Artemis) | Backward for GATED_DELTA_NET (and SSM_CONV if used) on CPU, CUDA, Vulkan | `test-backend-ops grad`-style reference check; Artemis loss falls | L |
| 3 | Unsupported backward op aborts | `llama-finetune-qlora` dies in `ggml_compute_backward` instead of reporting the op | Check the graph for ops without backward before training and exit with the op name | Artemis run prints a clear error | S |
| 4 | Large MoE fine-tune unmeasured | VRAM and step time of the expert-gradient path measured only on a 0.6B MoE | Train Qwen3-30B-A3B Q4_K_M on the RTX 3090, Vulkan and CUDA | Peak VRAM, s/step and loss curve | S, 18 GB download |
| 5 | Memory cost of the expert gradient | The backward pass dequantizes each expert stack to F32 and transposes it: about 1.6 GB peak for Qwen3-30B-A3B, 2 GB for Gemma 4 26B | Only if item 4 shows it matters: a transposed MUL_MAT_ID kernel that dequantizes on the fly, or dequantize in groups of experts | `test-moe-backward`; `test-backend-ops` against CPU; peak VRAM back near the pre-fix level | L |
| 6 | Slow Vulkan K-quant to F32 copy | [copy_from_quant_k.comp](../ggml/src/ggml-vulkan/vulkan-shaders/copy_from_quant_k.comp) runs one thread per 256-value block; 255 of 256 threads idle | Spread each block over the workgroup, as `copy_from_quant.comp` does | `test-backend-ops -o CPY`; `test-backend-ops perf -o CPY` before and after | M |
| 7 | Metal unverified, Q6_K falls back | Metal copies only Q3_K/Q4_K among K-quants to F32, so Q4_K_M experts (Q6_K) dequantize on the CPU; none of these paths ran on Metal | Add Q2_K/Q5_K/Q6_K to F32 copy kernels; run the QLoRA tests on a Mac | `test-qlora-finetune-*` and `test-backend-ops -o CPY` on Metal | M, needs a Mac |
| 8 | Router gradient barely tested | In the dummy MoE the router gradient is about 2e-7, so the end-to-end test hardly checks it | Add a [test-moe-backward.cpp](../tests/test-moe-backward.cpp) case for softmax, top-k, GET_ROWS and the weighted sum, against a reference | The case fails on the GET_ROWS backward before 83e45e771 | S |
| 9 | `test-backend-ops grad` trusted wrongly | Its grad mode passed the broken batched GET_ROWS backward | Find why (likely only 2D cases), add batched GET_ROWS and broadcast MUL_MAT_ID grad cases | The new cases fail on the old code | M |
| 10 | Pruning on more architectures | Tested on Gemma 4 and qwen3moe only; grouped routing (DeepSeek V3) can only be soft-pruned | Run `test-moe-prune-model.cmake` on a small DeepSeek-style and a gpt-oss-style model | Analyze passes the disabled-expert check; hard prune validates | M |

## Pre-existing failures

| Failure | Cause | Proposed fix | Effort |
| --- | --- | --- | --- |
| `test-backend-ops -o OUT_PROD` segfaults | Base-type quants (Q4_0, Q8_0, ...) as src0 with a non-transposed b | Reproduce on CPU, fix the stride handling in the fork's quantized OUT_PROD | M |
| CUDA build with the Visual Studio generator | "No CUDA toolset found"; only the Ninja build inside `vcvars64.bat` works | Install the CUDA Visual Studio integration, or document Ninja as the only Windows CUDA build | S |

## Order of work

1. Check CI on the pushed master; fix what is still red.
2. Re-run Gemma 4 pruning (item 1).
3. Clear the OUT_PROD segfault, so `test-backend-ops` is green and new breakage shows.
4. Clean error for unsupported backward ops (item 3), then decide on GATED_DELTA_NET backward (item 2).
5. Strengthen the tests (items 8, 9, 10).
6. Large MoE run (item 4); performance and memory: item 6, then item 5 if needed.
7. Metal (item 7), when a Mac is available.
