#include "moe-prune.h"
#include "dataset.h"
#include "hard-prune.h"

#include "chat.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

// Portable, collision-resistant scratch directory: std::filesystem::temp_directory_path() resolves
// to the platform temp dir (unlike a hardcoded "/tmp", which doesn't exist on Windows), and the
// timestamp suffix keeps concurrent test runs from colliding on the same files.
static std::filesystem::path test_temp_dir;

struct fixture_layout {
    const char * arch;
    uint32_t block_count;
    ggml_type expert_type;
    int64_t n_embd;
    bool biases; // router bias [n_expert] and expert selection bias, as in DeepSeek V3 / gpt-oss layouts
};

// one MoE layer (blk.0) with 8 experts; every byte of expert e's slices is e
static void make_fixture(const std::string & path, const fixture_layout & layout) {
    const int64_t n_expert = 8;
    gguf_context * gguf = gguf_init_empty();
    const std::string arch = layout.arch;
    gguf_set_val_str(gguf, "general.architecture", layout.arch);
    gguf_set_val_u32(gguf, (arch + ".block_count").c_str(), layout.block_count);
    gguf_set_val_u32(gguf, (arch + ".expert_count").c_str(), n_expert);
    gguf_set_val_u32(gguf, (arch + ".expert_used_count").c_str(), 2);

    ggml_context * tensors = ggml_init({ 4 * 1024 * 1024, nullptr, false });
    std::vector<ggml_tensor *> per_expert;
    auto add = [&](ggml_tensor * t, const char * name) {
        ggml_set_name(t, name);
        gguf_add_tensor(gguf, t);
        per_expert.push_back(t);
    };
    add(ggml_new_tensor_2d(tensors, GGML_TYPE_F32, layout.n_embd, n_expert), "blk.0.ffn_gate_inp.weight");
    add(ggml_new_tensor_3d(tensors, layout.expert_type, layout.n_embd, 4, n_expert), "blk.0.ffn_gate_up_exps.weight");
    add(ggml_new_tensor_3d(tensors, layout.expert_type, layout.n_embd, layout.n_embd, n_expert), "blk.0.ffn_down_exps.weight");
    if (layout.biases) {
        add(ggml_new_tensor_1d(tensors, GGML_TYPE_F32, n_expert), "blk.0.ffn_gate_inp.bias");
        add(ggml_new_tensor_1d(tensors, GGML_TYPE_F32, n_expert), "blk.0.exp_probs_b.bias");
        add(ggml_new_tensor_2d(tensors, GGML_TYPE_F32, 4, n_expert), "blk.0.ffn_down_exps.bias");
    }
    // not per expert: copied unchanged
    ggml_tensor * norm = ggml_new_tensor_1d(tensors, GGML_TYPE_F32, layout.n_embd);
    ggml_set_name(norm, "blk.0.ffn_norm.weight");
    gguf_add_tensor(gguf, norm);
    memset(norm->data, 0x7f, ggml_nbytes(norm));

    for (ggml_tensor * t : per_expert) {
        const size_t slice = ggml_nbytes(t) / n_expert;
        for (int expert = 0; expert < n_expert; ++expert) memset((char *) t->data + expert * slice, expert, slice);
    }
    if (!gguf_write_to_file(gguf, path.c_str(), false)) throw std::runtime_error("failed to write synthetic GGUF fixture");
    ggml_free(tensors);
    gguf_free(gguf);
}

static void expect_failure(const std::function<void()> & fn) {
    bool failed = false;
    try {
        fn();
    } catch (const std::exception &) {
        failed = true;
    }
    if (!failed) throw std::runtime_error("expected failure");
}

static void require(bool value) {
    if (!value) throw std::runtime_error("test assertion failed");
}

static void test_dataset() {
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    llama_model * model = llama_model_load_from_file(LLAMA_PRUNE_TEST_GEMMA_VOCAB, params);
    require(model != nullptr);
    common_chat_templates_ptr templates = common_chat_templates_init(model, "");

    const std::string path = (test_temp_dir / "llama-moe-prune-test-data.jsonl").string();
    {
        FILE * file = fopen(path.c_str(), "wb");
        require(file != nullptr);
        fputs("{\"messages\":[{\"role\":\"user\",\"content\":\"Question\"},{\"role\":\"assistant\",\"content\":\"Answer\"}]}\n", file);
        fputs("{\"messages\":[{\"role\":\"user\",\"content\":\"Solve\"},{\"role\":\"assistant\",\"reasoning\":\"Work\",\"content\":\"Final\"}]}\n", file);
        fputs("{\"messages\":[{\"role\":\"user\",\"content\":\"Q1\"},{\"role\":\"assistant\",\"content\":\"A1\"},{\"role\":\"user\",\"content\":\"Q2\"},{\"role\":\"assistant\",\"content\":\"A2\"},{\"role\":\"user\",\"content\":\"Q3\"},{\"role\":\"assistant\",\"content\":\"A3\"},{\"role\":\"user\",\"content\":\"Q4\"},{\"role\":\"assistant\",\"content\":\"A4\"},{\"role\":\"user\",\"content\":\"Q5\"},{\"role\":\"assistant\",\"content\":\"A5\"}]}\n", file);
        fclose(file);
    }
    const llama_prune_dataset dataset = llama_prune_dataset_load(path, model, templates.get(), 1);
    const llama_prune_dataset parallel_dataset = llama_prune_dataset_load(path, model, templates.get(), 4);
    require(dataset.records.size() == 3);
    require(parallel_dataset.records.size() == dataset.records.size());
    require(parallel_dataset.total_tokens == dataset.total_tokens);
    for (size_t i = 0; i < dataset.records.size(); ++i) {
        require(parallel_dataset.records[i].line == dataset.records[i].line);
        require(parallel_dataset.records[i].tokens == dataset.records[i].tokens);
        require(parallel_dataset.records[i].token_fields == dataset.records[i].token_fields);
    }
    size_t assistant = 0;
    size_t reasoning = 0;
    size_t content = 0;
    for (const llama_prune_dataset_record & record : dataset.records) {
        for (size_t i = 0; i < record.tokens.size(); ++i) {
            assistant += llama_prune_token_is_evaluated(record, i, llama_prune_ppl_mask::ASSISTANT);
            reasoning += llama_prune_token_is_evaluated(record, i, llama_prune_ppl_mask::REASONING);
            content += llama_prune_token_is_evaluated(record, i, llama_prune_ppl_mask::CONTENT);
        }
    }
    require(assistant > 0 && reasoning > 0 && content > 0);

    {
        FILE * file = fopen(path.c_str(), "wb");
        require(file != nullptr);
        fputs("{\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}]}\n", file);
        fputs("{\"messages\":42}\n", file);
        fclose(file);
    }
    bool line_error = false;
    try {
        (void) llama_prune_dataset_load(path, model, templates.get());
    } catch (const std::exception & e) {
        line_error = std::string(e.what()).find("line 2") != std::string::npos;
    }
    require(line_error);
    std::remove(path.c_str());
    llama_model_free(model);
}

static void test_hard_prune(const fixture_layout & layout) {
    const std::string arch = layout.arch;
    const std::string source_path = (test_temp_dir / ("llama-moe-prune-test-source-" + arch + ".gguf")).string();
    const std::string output_path = (test_temp_dir / ("llama-moe-prune-test-output-" + arch + ".gguf")).string();
    make_fixture(source_path, layout);
    const common_moe_prune_model_info fixture_info = common_moe_prune_inspect_model(source_path);
    require(fixture_info.architecture == arch);
    require(fixture_info.layer_count == (int32_t) layout.block_count);
    require(fixture_info.expert_count == 8 && fixture_info.experts_used == 2);
    require(fixture_info.moe_layers == std::vector<int32_t>{ 0 });
    require(fixture_info.model_hash == common_moe_prune_sha256_file(source_path));
    common_moe_prune_profile fixture_profile;
    fixture_profile.architecture = fixture_info.architecture;
    fixture_profile.model_hash = fixture_info.model_hash;
    fixture_profile.expert_tensor_hash = fixture_info.expert_tensor_hash;
    fixture_profile.expert_count = fixture_info.expert_count;
    fixture_profile.experts_used = fixture_info.experts_used;
    fixture_profile.layers[0].disabled_experts = { 1, 3 };
    const llama_prune_hard_prune_report report = llama_prune_hard_prune(source_path, fixture_profile, fixture_info, output_path);
    require(report.original_to_new.at(0).at(0) == 0);
    require(report.original_to_new.at(0).at(2) == 1);
    require(report.expert_bytes_removed > 0);

    ggml_context * pruned_tensors = nullptr;
    gguf_context * pruned = gguf_init_from_file(output_path.c_str(), { false, &pruned_tensors });
    require(pruned != nullptr);
    const int64_t expert_key = gguf_find_key(pruned, (arch + ".expert_count").c_str());
    require(expert_key >= 0 && gguf_get_val_u32(pruned, expert_key) == 6);
    const int expected[] = { 0, 2, 4, 5, 6, 7 };
    std::vector<std::string> names = { "blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_up_exps.weight", "blk.0.ffn_down_exps.weight" };
    if (layout.biases) {
        names.insert(names.end(), { "blk.0.ffn_gate_inp.bias", "blk.0.exp_probs_b.bias", "blk.0.ffn_down_exps.bias" });
    }
    for (const std::string & name : names) {
        const ggml_tensor * t = ggml_get_tensor(pruned_tensors, name.c_str());
        require(t != nullptr);
        require(t->ne[ggml_n_dims(t) - 1] == 6);
        const size_t slice = ggml_nbytes(t) / 6;
        for (int expert = 0; expert < 6; ++expert) {
            const unsigned char * data = (const unsigned char *) t->data + expert * slice;
            require(data[0] == expected[expert] && data[slice - 1] == expected[expert]);
        }
    }
    const ggml_tensor * norm = ggml_get_tensor(pruned_tensors, "blk.0.ffn_norm.weight");
    require(norm != nullptr && norm->ne[0] == layout.n_embd && *(const unsigned char *) norm->data == 0x7f);
    ggml_free(pruned_tensors);
    gguf_free(pruned);

    // grouped routing cannot be hard-pruned
    common_moe_prune_model_info grouped = fixture_info;
    grouped.expert_group_count = 2;
    expect_failure([&]() { (void) llama_prune_hard_prune(source_path, fixture_profile, grouped, output_path + ".grouped"); });

    std::remove(source_path.c_str());
    std::remove(output_path.c_str());
    std::remove((output_path + ".report.json").c_str());
}

int main() {
    test_temp_dir = std::filesystem::temp_directory_path() /
        ("llama-moe-prune-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(test_temp_dir);

    test_dataset();
    common_moe_prune_model_info model;
    model.architecture = "gemma4";
    model.model_hash = "sha256:model";
    model.expert_tensor_hash = "sha256:experts";
    model.layer_count = 30;
    model.expert_count = 8;
    model.experts_used = 2;
    model.moe_layers = { 1, 3 };

    common_moe_prune_stats stats;
    for (int32_t layer : model.moe_layers) {
        stats[layer].resize(model.expert_count);
        for (int32_t expert = 0; expert < model.expert_count; ++expert) {
            auto & value = stats[layer][expert];
            value.selection_count = 10;
            value.probability_sum = expert + 1;
            value.output_norm_sum = 2 * (expert + 1);
            value.weighted_output_sum = expert + layer;
        }
    }

    const auto profiles = common_moe_prune_make_profiles(
        model, stats, { 0.25, 0.50 }, 0.50, "sha256:dataset", "assistant", "router-output", 100);
    require(profiles.size() == 2);
    for (int32_t layer : model.moe_layers) {
        const auto & small = profiles[0].layers.at(layer).disabled_experts;
        const auto & large = profiles[1].layers.at(layer).disabled_experts;
        require(small.size() == 2);
        require(large.size() == 4);
        for (int32_t expert : small) require(std::find(large.begin(), large.end(), expert) != large.end());
    }

    const std::string path = (test_temp_dir / "llama-moe-prune-test-profile.json").string();
    common_moe_prune_profile_write(profiles[0], path);
    const common_moe_prune_profile loaded = common_moe_prune_profile_load(path);
    common_moe_prune_profile_validate(loaded, model);
    require(loaded.layers.at(1).disabled_experts == profiles[0].layers.at(1).disabled_experts);
    std::remove(path.c_str());

    common_moe_prune_profile invalid = loaded;
    invalid.model_hash = "sha256:wrong";
    expect_failure([&]() { common_moe_prune_profile_validate(invalid, model); });
    invalid = loaded;
    invalid.layers.at(1).disabled_experts = { 0, 0 };
    expect_failure([&]() { common_moe_prune_profile_validate(invalid, model); });
    invalid = loaded;
    invalid.layers.at(1).disabled_experts = { 0, 8 };
    expect_failure([&]() { common_moe_prune_profile_validate(invalid, model); });
    invalid = loaded;
    invalid.layers.at(1).disabled_experts = { 0, 1, 2, 3, 4, 5, 6 };
    invalid.layers.at(3).disabled_experts = invalid.layers.at(1).disabled_experts;
    expect_failure([&]() { common_moe_prune_profile_validate(invalid, model); });

    require(std::abs(stats[1][2].mean_probability() - 0.3) < 1e-12);
    require(std::abs(stats[1][2].mean_output_norm() - 0.6) < 1e-12);
    require(std::abs(stats[1][2].importance() - 0.3) < 1e-12);

    // Gemma 4 26B A4B (Q4_0) and a generic qwen3moe layout with K-quant experts and per-expert biases
    test_hard_prune({ "gemma4",   30, GGML_TYPE_Q4_0, 32,  false });
    test_hard_prune({ "qwen3moe", 4,  GGML_TYPE_Q4_K, 256, true  });

    // a dense model is rejected
    {
        const std::string dense_path = (test_temp_dir / "llama-moe-prune-test-dense.gguf").string();
        gguf_context * gguf = gguf_init_empty();
        gguf_set_val_str(gguf, "general.architecture", "llama");
        gguf_set_val_u32(gguf, "llama.block_count", 2);
        require(gguf_write_to_file(gguf, dense_path.c_str(), false));
        gguf_free(gguf);
        expect_failure([&]() { (void) common_moe_prune_inspect_model(dense_path); });
    }
    std::filesystem::remove_all(test_temp_dir);
    return 0;
}
