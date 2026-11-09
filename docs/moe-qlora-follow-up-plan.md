# MoE QLoRA follow-up plan

As of 2026-11-09. MoE QLoRA trains correctly on CPU, Vulkan and CUDA. Metal and a real MoE model are untested.

## Where things stand

| Commit | What it does | Verified by |
| --- | --- | --- |
| 7fbc7444f | Gradient flows through quantized experts (dequantized to F32 for the transpose); CUDA dequantizing CPY for every type; Vulkan Q2_K/Q5_K/Q6_K to F32 | `test-moe-backward` quantized cases (fail without the fix); `test-backend-ops -o CPY` 276/276 on CUDA0 and Vulkan1 |
| 83e45e771 | MUL_MAT_ID backward with broadcast src1; batched GET_ROWS backward flattened to the 2D kernel; new QLoRA tests | `test-moe-backward`; `test-qlora-finetune-*` on dense, MoE, MoE Q4_0, MoE Q4_K, CPU and GPU losses equal |
| fa42c0626 | Upstream merge (cf34d3113) | Vulkan and CUDA builds; `ctest -L main` 22/23, one known failure |

A real dense run (Qwen3-1.7B, Vulkan) reached loss 0.0000 with the expected answer.

Run the QLoRA checks with:

```sh
ctest -C Release -R "test-qlora|test-moe-backward|generate-models" --output-on-failure
```

## Open items

Items 1 and 2 gate the rest: measuring a real MoE run decides whether item 3 is needed at all.
Effort: S = hours, M = a day or two, L = several days.

| # | Item | Problem | Proposed fix | Proof it works | Effort |
| --- | --- | --- | --- | --- | --- |
| 1 | Real MoE fine-tune | Only dummy models trained; VRAM and step time of the expert-gradient path are unmeasured | Train Qwen3-30B-A3B Q4_K_M on an RTX 3090, Vulkan and CUDA, before and after 7fbc7444f | Peak VRAM, s/step and loss curve for both; loss falls faster with the fix | S |
| 2 | Publish master | No CI run on the merged tree | Push `master`, let the fork CI build it (the macOS runner gives a Metal build) | Green CI on the pushed commit | S |
| 3 | Memory cost of the expert gradient | The backward pass dequantizes each expert stack to F32 and transposes it: about 1.6 GB peak for Qwen3-30B-A3B, 2 GB for Gemma 4 26B | Only if item 1 shows it matters: a transposed MUL_MAT_ID kernel that dequantizes on the fly (CPU, CUDA, Vulkan, Metal), or dequantize in groups of experts | `test-moe-backward`; `test-backend-ops` against CPU; peak VRAM back near the pre-fix level | L |
| 4 | Slow Vulkan K-quant to F32 copy | [copy_from_quant_k.comp](../ggml/src/ggml-vulkan/vulkan-shaders/copy_from_quant_k.comp) runs one thread per 256-value block; 255 of 256 threads idle | Spread each block over the workgroup, as `copy_from_quant.comp` does | `test-backend-ops -o CPY`; `test-backend-ops perf -o CPY` before and after | M |
| 5 | Metal unverified, Q6_K falls back | Metal copies only Q3_K/Q4_K among K-quants to F32, so Q4_K_M experts (Q6_K) dequantize on the CPU; none of these paths ran on Metal | Add Q2_K/Q5_K/Q6_K to F32 copy kernels; run the QLoRA tests on a Mac | `test-qlora-finetune-*` and `test-backend-ops -o CPY` on Metal | M, needs a Mac |
| 6 | Router gradient barely tested | In the dummy MoE the router gradient is about 2e-7, so the end-to-end test hardly checks it | Add a [test-moe-backward.cpp](../tests/test-moe-backward.cpp) case for softmax, top-k, GET_ROWS and the weighted sum, against a reference | The case fails on the GET_ROWS backward before 83e45e771 | S |
| 7 | `test-backend-ops grad` trusted wrongly | Its grad mode passed the broken batched GET_ROWS backward | Find why (likely only 2D cases), add batched GET_ROWS and broadcast MUL_MAT_ID grad cases | The new cases fail on the old code | M |

## Pre-existing failures

These fail on the fork's pre-merge master too and pass on plain upstream. Each keeps a suite red, which hides new failures.

| Failure | Cause | Proposed fix | Effort |
| --- | --- | --- | --- |
| `test-quantize-fns`: "q8_0 reference implementation error" | The fork's `quantize_row_q8_0_ref` uses a `max/-128` scale; upstream uses `amax/127` | Open question: why the fork changed it. If no reason holds, restore upstream; otherwise update the test's reference | S |
| `test-backend-ops -o OUT_PROD` segfaults | Base-type quants (Q4_0, Q8_0, ...) as src0 with a non-transposed b | Reproduce on CPU, fix the stride handling in the fork's quantized OUT_PROD | M |
| `test-backend-ops -o SET_ROWS -b Vulkan*` aborts | Vulkan reports SET_ROWS support for Q3_K/Q4_K/MXFP4 without set_rows pipelines | Remove those types from `supports_op`, or add the pipelines | S |
| CUDA build with the Visual Studio generator | "No CUDA toolset found"; only the Ninja build inside `vcvars64.bat` works | Install the CUDA Visual Studio integration, or document Ninja as the only Windows CUDA build | S |

## Out of scope

- MoE pruning tests: `llama-prune` supports only Gemma 4 26B A4B, and the pruning tool in use is a Python script outside this repo.

## Order of work

1. Push master and check CI (item 2).
2. Measure the real MoE fine-tune (item 1); decide on item 3 from its VRAM numbers.
3. Clear the pre-existing failures, so `ctest -L main` and `test-backend-ops` are green and new breakage shows.
4. Strengthen the tests (items 6, 7).
5. Performance and memory: item 4, then item 3 if needed.
6. Metal (item 5), when a Mac is available.
