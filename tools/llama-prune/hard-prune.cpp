#include "hard-prune.h"

#include "ggml.h"
#include "gguf.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

struct gguf_deleter {
    void operator()(gguf_context * ctx) const { gguf_free(ctx); }
};

struct ggml_deleter {
    void operator()(ggml_context * ctx) const { ggml_free(ctx); }
};

using gguf_ptr = std::unique_ptr<gguf_context, gguf_deleter>;
using ggml_ptr = std::unique_ptr<ggml_context, ggml_deleter>;

int32_t tensor_layer(const std::string & name) {
    int32_t layer = -1;
    return sscanf(name.c_str(), "blk.%d.", &layer) == 1 ? layer : -1;
}

// the highest dimension that is not 1 holds the experts; -1 if it is not n_expert
int expert_axis(const int64_t * ne, int32_t n_expert) {
    int axis = GGML_MAX_DIMS - 1;
    while (axis > 0 && ne[axis] == 1) --axis;
    return ne[axis] == n_expert ? axis : -1;
}

std::vector<int32_t> surviving_experts(int32_t n_expert, const std::vector<int32_t> & disabled) {
    std::vector<bool> removed(n_expert, false);
    for (int32_t expert : disabled) removed[expert] = true;
    std::vector<int32_t> result;
    for (int32_t expert = 0; expert < n_expert; ++expert) {
        if (!removed[expert]) result.push_back(expert);
    }
    return result;
}

void copy_range(std::ifstream & input, std::ofstream & output, uint64_t offset, uint64_t size) {
    input.clear();
    input.seekg((std::streamoff) offset);
    if (!input) throw std::runtime_error("failed to seek source GGUF");
    std::vector<char> buffer(4 * 1024 * 1024);
    while (size > 0) {
        const size_t chunk = (size_t) std::min<uint64_t>(size, buffer.size());
        input.read(buffer.data(), chunk);
        if ((size_t) input.gcount() != chunk) throw std::runtime_error("failed to read source GGUF tensor data");
        output.write(buffer.data(), chunk);
        if (!output) throw std::runtime_error("failed to write pruned GGUF tensor data");
        size -= chunk;
    }
}

uint64_t file_size(const std::string & path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("failed to stat file: " + path);
    return (uint64_t) in.tellg();
}

void write_report(const llama_prune_hard_prune_report & report, const std::string & output_path) {
    json mappings = json::array();
    for (const auto & layer : report.original_to_new) {
        json mapping = json::object();
        for (const auto & item : layer.second) mapping[std::to_string(item.first)] = item.second;
        mappings.push_back({ { "layer", layer.first }, { "original_to_new", mapping } });
    }
    json root = {
        { "format", "llama-moe-hard-prune-report" },
        { "version", 1 },
        { "source_bytes", report.source_bytes },
        { "output_bytes", report.output_bytes },
        { "expert_bytes_removed", report.expert_bytes_removed },
        { "layers", mappings },
    };
    const std::string path = output_path + ".report.json";
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to write hard-pruning report");
        out << root.dump(2) << '\n';
    }
    if (!common_moe_prune_replace_file(tmp, path)) {
        std::remove(tmp.c_str());
        throw std::runtime_error("failed to replace hard-pruning report");
    }
}

}

llama_prune_hard_prune_report llama_prune_hard_prune(
        const std::string & model_path,
        const common_moe_prune_profile & profile,
        const common_moe_prune_model_info & model_info,
        const std::string & output_path) {
    if (model_path == output_path) throw std::runtime_error("hard pruning never modifies the source GGUF in place");
    common_moe_prune_profile_validate(profile, model_info);
    if (model_info.expert_group_count > 1) {
        throw std::runtime_error("hard pruning does not support grouped expert routing (" + model_info.architecture +
            ".expert_group_count > 1), use the soft profile");
    }

    ggml_context * source_tensor_ctx_raw = nullptr;
    gguf_ptr source(gguf_init_from_file(model_path.c_str(), { true, &source_tensor_ctx_raw }));
    if (!source) throw std::runtime_error("failed to read source GGUF");
    ggml_ptr source_tensor_ctx(source_tensor_ctx_raw);
    gguf_ptr output(gguf_init_empty());
    gguf_set_kv(output.get(), source.get());
    gguf_set_val_u32(output.get(), GGUF_KEY_GENERAL_ALIGNMENT, (uint32_t) gguf_get_alignment(output.get()));
    gguf_set_val_u32(output.get(), (model_info.architecture + ".expert_count").c_str(),
        model_info.expert_count - (uint32_t) profile.layers.begin()->second.disabled_experts.size());

    const size_t tensor_count = (size_t) gguf_get_n_tensors(source.get());
    const size_t metadata_memory = std::max<size_t>(16 * 1024 * 1024, tensor_count * sizeof(ggml_tensor) * 4);
    ggml_ptr tensor_ctx(ggml_init({ metadata_memory, nullptr, true }));
    if (!tensor_ctx) throw std::runtime_error("failed to allocate GGUF tensor metadata context");

    struct tensor_plan {
        int64_t source_id = -1;
        std::vector<int32_t> survivors; // empty: copy the tensor unchanged
        size_t slice_size = 0;          // bytes of one expert's slice
        size_t old_size = 0;
        size_t new_size = 0;
    };
    std::vector<tensor_plan> plans;
    plans.reserve(tensor_count);
    std::set<int32_t> layers_with_experts;
    llama_prune_hard_prune_report report;

    for (int64_t i = 0; i < (int64_t) tensor_count; ++i) {
        const std::string name = gguf_get_tensor_name(source.get(), i);
        const ggml_tensor * old_tensor = ggml_get_tensor(source_tensor_ctx.get(), name.c_str());
        if (old_tensor == nullptr) throw std::runtime_error(name + ": missing source tensor metadata");
        const int64_t * old_ne = gguf_get_tensor_ne(source.get(), i);
        int64_t new_ne[GGML_MAX_DIMS] = { old_ne[0], old_ne[1], old_ne[2], old_ne[3] };
        const ggml_type type = gguf_get_tensor_type(source.get(), i);
        tensor_plan plan;
        plan.source_id = i;
        plan.old_size = gguf_get_tensor_size(source.get(), i);
        const int32_t layer = tensor_layer(name);
        auto profile_layer = profile.layers.find(layer);

        if (profile_layer != profile.layers.end() && common_moe_prune_is_expert_tensor(name)) {
            const int axis = expert_axis(old_ne, model_info.expert_count);
            if (axis < 0) throw std::runtime_error(name + ": no dimension holds the " + std::to_string(model_info.expert_count) + " experts");
            if (axis == 0) {
                if (ggml_blck_size(type) != 1) throw std::runtime_error(name + ": quantized per-expert vector");
                plan.slice_size = ggml_type_size(type);
            } else {
                plan.slice_size = ggml_row_size(type, old_ne[0]);
                for (int d = 1; d < axis; ++d) plan.slice_size *= old_ne[d];
            }
            if (plan.slice_size * model_info.expert_count != plan.old_size) {
                throw std::runtime_error(name + ": experts are interleaved or packed and cannot be copied safely");
            }
            plan.survivors = surviving_experts(model_info.expert_count, profile_layer->second.disabled_experts);
            new_ne[axis] = plan.survivors.size();
            if (name.find("exps.") != std::string::npos) layers_with_experts.insert(layer);
        }

        const int n_dims = ggml_n_dims(old_tensor);
        ggml_tensor * tensor = ggml_new_tensor(tensor_ctx.get(), type, n_dims, new_ne);
        ggml_set_name(tensor, name.c_str());
        gguf_add_tensor(output.get(), tensor);
        plan.new_size = ggml_nbytes(tensor);
        plans.push_back(std::move(plan));
    }
    for (const auto & layer : profile.layers) {
        if (layers_with_experts.count(layer.first) == 0) {
            throw std::runtime_error("layer " + std::to_string(layer.first) + " has no stacked expert tensors (blk.N.ffn_*_exps)");
        }
    }

    for (const auto & layer : profile.layers) {
        int32_t next = 0;
        const std::vector<int32_t> survivors = surviving_experts(model_info.expert_count, layer.second.disabled_experts);
        for (int32_t original : survivors) {
            report.original_to_new[layer.first][original] = next++;
        }
    }

    const std::string tmp_path = output_path + ".tmp";
    if (!gguf_write_to_file(output.get(), tmp_path.c_str(), true)) {
        throw std::runtime_error("failed to write pruned GGUF metadata");
    }
    std::ifstream input(model_path, std::ios::binary);
    std::ofstream out(tmp_path, std::ios::binary | std::ios::app);
    if (!input || !out) {
        std::remove(tmp_path.c_str());
        throw std::runtime_error("failed to open GGUF tensor streams");
    }
    const uint64_t source_data = gguf_get_data_offset(source.get());
    const size_t alignment = gguf_get_alignment(output.get());
    uint64_t output_cursor = 0;
    for (size_t i = 0; i < plans.size(); ++i) {
        const tensor_plan & plan = plans[i];
        const uint64_t target_offset = gguf_get_tensor_offset(output.get(), i);
        if (target_offset < output_cursor) throw std::runtime_error("invalid output GGUF tensor offsets");
        std::vector<char> padding((size_t) (target_offset - output_cursor), 0);
        out.write(padding.data(), padding.size());
        const uint64_t source_offset = source_data + gguf_get_tensor_offset(source.get(), plan.source_id);
        if (plan.survivors.empty()) {
            copy_range(input, out, source_offset, plan.old_size);
        } else {
            for (int32_t expert : plan.survivors) copy_range(input, out, source_offset + expert * plan.slice_size, plan.slice_size);
        }
        output_cursor = target_offset + plan.new_size;
        if (plan.old_size > plan.new_size) report.expert_bytes_removed += plan.old_size - plan.new_size;
    }
    const size_t end_padding = (alignment - output_cursor % alignment) % alignment;
    std::vector<char> padding(end_padding, 0);
    out.write(padding.data(), padding.size());
    out.close();
    if (!out) {
        std::remove(tmp_path.c_str());
        throw std::runtime_error("failed to finish pruned GGUF");
    }
    input.close();

    report.source_bytes = file_size(model_path);
    report.output_bytes = file_size(tmp_path);
    if (!common_moe_prune_replace_file(tmp_path, output_path)) {
        std::remove(tmp_path.c_str());
        throw std::runtime_error("failed to atomically replace hard-pruned GGUF output");
    }
    write_report(report, output_path);
    return report;
}
