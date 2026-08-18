#pragma once

#include "llama.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// Static MoE pruning (common/moe-prune.cpp and tools/llama-prune/) currently supports exactly one
// checkpoint family: the Gemma 4 26B A4B Q4_0 QAT GGUF layout loaded by src/models/gemma4.cpp.
// common_moe_prune_is_supported_architecture() is the single source of truth for that check on the
// GGUF-metadata side (common/moe-prune.cpp, tools/llama-prune/hard-prune.cpp). The runtime soft-prune
// path in src/llama-model.cpp cannot depend on common/, so it re-expresses the same constraint via
// LLM_ARCH_GEMMA4 / LLM_TYPE_26B_A4B (llama_model_set_moe_prune) — keep that check's layer-count
// assumption in sync with COMMON_MOE_PRUNE_GEMMA4_LAYER_COUNT when either changes.
constexpr const char * COMMON_MOE_PRUNE_GEMMA4_ARCHITECTURE = "gemma4";
constexpr int32_t COMMON_MOE_PRUNE_GEMMA4_LAYER_COUNT = 30;

bool common_moe_prune_is_supported_architecture(const std::string & architecture, int32_t layer_count);

struct common_moe_prune_layer {
    std::vector<int32_t> disabled_experts;
};

struct common_moe_prune_profile {
    int32_t version = 1;
    std::string mode = "soft";
    std::string architecture;
    std::string model_hash;
    std::string expert_tensor_hash;
    int32_t expert_count = 0;
    int32_t experts_used = 0;
    std::string dataset_hash;
    std::string ppl_mask;
    std::string metric;
    int64_t evaluated_tokens = 0;
    double requested_ratio = 0.0;
    double actual_ratio = 0.0;
    std::map<int32_t, common_moe_prune_layer> layers;
};

struct common_moe_prune_model_info {
    std::string architecture;
    std::string model_hash;
    std::string expert_tensor_hash;
    int32_t layer_count = 0;
    int32_t expert_count = 0;
    int32_t experts_used = 0;
    std::vector<int32_t> moe_layers;
    uint64_t expert_bytes = 0;
};

// Per-expert calibration accumulators for the REAP saliency criterion
// (arXiv:2510.13999, Equation 9). See common/moe-prune.cpp for provenance.
struct common_moe_prune_expert_stats {
    uint64_t selection_count = 0;     // tokens routed to this expert
    double probability_sum = 0.0;     // sum of router probabilities g_j(t)
    double output_norm_sum = 0.0;     // sum of ||f_j(t)||_2                 (EAN, ungated)
    double weighted_output_sum = 0.0; // sum of g_j(t) * ||f_j(t)||_2        (REAP)

    double mean_probability() const;
    double mean_output_norm() const;
    double importance() const; // REAP score: the `router-output` metric
};

using common_moe_prune_stats = std::map<int32_t, std::vector<common_moe_prune_expert_stats>>;

std::string common_moe_prune_sha256_file(const std::string & path);
common_moe_prune_model_info common_moe_prune_inspect_model(const std::string & path);
common_moe_prune_profile common_moe_prune_profile_load(const std::string & path);
void common_moe_prune_profile_write(const common_moe_prune_profile & profile, const std::string & path);
void common_moe_prune_profile_validate(const common_moe_prune_profile & profile, const common_moe_prune_model_info & model);
void common_moe_prune_profile_apply(llama_model * model, const common_moe_prune_profile & profile);

std::vector<common_moe_prune_profile> common_moe_prune_make_profiles(
        const common_moe_prune_model_info & model,
        const common_moe_prune_stats & stats,
        const std::vector<double> & ratios,
        double max_layer_ratio,
        const std::string & dataset_hash,
        const std::string & ppl_mask,
        const std::string & metric,
        int64_t evaluated_tokens);
