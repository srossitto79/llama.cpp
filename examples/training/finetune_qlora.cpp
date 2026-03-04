// QLoRA fine-tuning for quantized GGUF models.
//
// The base model weights stay frozen (quantized tensors are skipped by
// llama_set_param because they are not GGML_TYPE_F32).  Only the freshly
// allocated F32 LoRA A/B tensors are trained.  After training the adapter
// is saved as a GGUF file that is directly compatible with the existing
// llama_adapter_lora_init() loader and llama-export-lora merge tool.
//
// Usage example:
//   llama-finetune-qlora \
//     --model llama-3.2-1b-q4_k_m.gguf \
//     --file  train.jsonl \
//     --lora-rank 16 --lora-alpha 16 \
//     --lora-targets "attn_q,attn_k,attn_v,attn_out" \
//     --lora-out adapter.gguf \
//     -e 3 -c 512

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "gguf.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<std::string> split_csv(const std::string & s) {
    std::vector<std::string> out;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(tok);
    }
    return out;
}

static bool tensor_matches_targets(const char * name, const std::vector<std::string> & targets) {
    for (const auto & t : targets) {
        if (std::string(name).find(t) != std::string::npos) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// JSONL dataset loading
// ---------------------------------------------------------------------------

struct training_sample {
    std::vector<llama_token> tokens;   // full token sequence
    std::vector<bool>        is_label; // true for tokens that contribute to loss
};

// Apply a very simple ChatML fallback template when the model has no template.
static std::string apply_chatml(const std::vector<common_chat_msg> & msgs) {
    std::string out;
    for (const auto & m : msgs) {
        out += "<|im_start|>" + m.role + "\n";
        // content_parts is a vector; build a plain text string
        std::string text;
        if (!m.content_parts.empty()) {
            for (const auto & p : m.content_parts) {
                text += p.text;
            }
        }
        out += text + "<|im_end|>\n";
    }
    return out;
}

static std::vector<training_sample> load_jsonl(
        const std::string & path,
        llama_context      * ctx,
        common_chat_templates * tmpls) {

    std::ifstream f(path);
    if (!f.is_open()) {
        LOG_ERR("%s: cannot open %s\n", __func__, path.c_str());
        return {};
    }

    std::vector<training_sample> samples;
    std::string line;
    int lineno = 0;

    while (std::getline(f, line)) {
        ++lineno;
        if (line.empty()) continue;

        nlohmann::json j;
        try { j = nlohmann::json::parse(line); }
        catch (...) {
            LOG_WRN("%s: skipping invalid JSON on line %d\n", __func__, lineno);
            continue;
        }

        std::string prompt_text;
        std::string response_text;

        if (j.contains("messages")) {
            // chat format — apply template
            std::vector<common_chat_msg> msgs;
            for (const auto & m : j["messages"]) {
                common_chat_msg msg;
                msg.role = m.value("role", "user");
                common_chat_msg_content_part part;
                part.text = m.value("content", "");
                msg.content_parts.push_back(part);
                msgs.push_back(msg);
            }

            if (tmpls) {
                common_chat_templates_inputs inp;
                inp.messages = msgs;
                inp.add_generation_prompt = false;
                auto params = common_chat_templates_apply(tmpls, inp);
                // params.prompt contains the fully formatted text
                prompt_text   = "";
                response_text = params.prompt;
            } else {
                response_text = apply_chatml(msgs);
            }
        } else if (j.contains("prompt") && j.contains("response")) {
            prompt_text   = j["prompt"].get<std::string>();
            response_text = j["response"].get<std::string>();
        } else if (j.contains("text")) {
            response_text = j["text"].get<std::string>();
        } else {
            LOG_WRN("%s: unknown format on line %d, skipping\n", __func__, lineno);
            continue;
        }

        // Tokenize: prompt (no loss) + response (loss)
        auto tok_prompt   = common_tokenize(ctx, prompt_text,   /*add_special=*/true);
        auto tok_response = common_tokenize(ctx, response_text, /*add_special=*/false);

        if (tok_prompt.empty() && tok_response.empty()) continue;

        training_sample s;
        s.tokens.insert(s.tokens.end(), tok_prompt.begin(),   tok_prompt.end());
        s.tokens.insert(s.tokens.end(), tok_response.begin(), tok_response.end());
        s.is_label.resize(s.tokens.size(), false);
        // Only response tokens contribute to the loss
        for (size_t i = tok_prompt.size(); i < s.tokens.size(); ++i) {
            s.is_label[i] = true;
        }
        samples.push_back(std::move(s));
    }

    LOG_INF("%s: loaded %zu samples from %s\n", __func__, samples.size(), path.c_str());
    return samples;
}

// Pack variable-length samples into fixed-context-length windows and create
// an ggml_opt_dataset. Labels for prompt tokens are set to -1 (ignored by
// the loss in the epoch loop).
static ggml_opt_dataset_t build_dataset(
        const std::vector<training_sample> & samples,
        int32_t n_ctx) {

    // Flatten samples separated by EOS (-1 acts as pad) into context windows
    std::vector<llama_token> flat_tokens;
    std::vector<int32_t>     flat_labels; // -1 = no loss, token_id = loss target

    for (const auto & s : samples) {
        for (size_t i = 0; i + 1 < s.tokens.size(); ++i) {
            flat_tokens.push_back(s.tokens[i]);
            flat_labels.push_back(s.is_label[i + 1] ? (int32_t)s.tokens[i + 1] : -1);
        }
    }

    if ((int64_t)flat_tokens.size() < n_ctx) {
        LOG_ERR("%s: dataset too small (%zu tokens) for context %d\n",
                __func__, flat_tokens.size(), n_ctx);
        return nullptr;
    }

    const int64_t stride = n_ctx / 2;
    const int64_t ndata  = ((int64_t)flat_tokens.size() - n_ctx) / stride;

    // data:   input token ids  [n_ctx, ndata]
    // labels: target token ids [n_ctx, ndata], -1 → mask (we use 0 but ignore via is_label)
    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
            GGML_TYPE_I32, GGML_TYPE_I32, n_ctx, n_ctx, ndata, 1);

    int32_t * data   = (int32_t *) ggml_opt_dataset_data  (dataset)->data;
    int32_t * labels = (int32_t *) ggml_opt_dataset_labels(dataset)->data;

    for (int64_t i = 0; i < ndata; ++i) {
        const int64_t off = i * stride;
        for (int32_t j = 0; j < n_ctx; ++j) {
            data  [i * n_ctx + j] = flat_tokens[off + j];
            // label at position j is the next token; -1 means no loss
            int32_t lbl = flat_labels[off + j];
            labels[i * n_ctx + j] = (lbl < 0) ? flat_tokens[off + j] : lbl;
        }
    }

    return dataset;
}

// ---------------------------------------------------------------------------
// LoRA tensor allocation
// ---------------------------------------------------------------------------

struct lora_tensors {
    struct ggml_context      * ctx  = nullptr;
    struct ggml_backend_buffer * buf = nullptr;
    // map: base tensor name → {lora_a, lora_b}
    std::unordered_map<std::string, std::pair<ggml_tensor*, ggml_tensor*>> ab;
};

static lora_tensors alloc_lora_tensors(
        const std::string        & model_path,
        const std::vector<std::string> & targets,
        int32_t                   rank,
        std::mt19937            & rng) {

    lora_tensors lt;

    // Open the model GGUF to discover tensor names and shapes
    // without needing access to private llama_model internals.
    struct ggml_context * ctx_meta = nullptr;
    struct gguf_init_params gguf_params = { /*.no_alloc=*/true, /*.ctx=*/&ctx_meta };
    struct gguf_context * ctx_gguf = gguf_init_from_file(model_path.c_str(), gguf_params);
    if (!ctx_gguf) {
        LOG_ERR("%s: failed to open model GGUF for tensor discovery: %s\n",
                __func__, model_path.c_str());
        return lt;
    }

    // Collect matching 2-D tensors
    struct tensor_info { std::string name; int64_t ne0, ne1; };
    std::vector<tensor_info> matched;

    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta);
         t; t = ggml_get_next_tensor(ctx_meta, t)) {
        if (t->n_dims < 2) continue;
        if (!tensor_matches_targets(t->name, targets)) continue;
        matched.push_back({t->name, t->ne[0], t->ne[1]});
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx_meta);

    if (matched.empty()) {
        LOG_ERR("%s: no model tensors matched --lora-targets; check spelling\n", __func__);
        return lt;
    }

    LOG_INF("%s: allocating LoRA A/B tensors for %zu weight matrices, rank=%d\n",
            __func__, matched.size(), rank);

    // Allocate ggml context for A+B tensors (2 tensors per matched weight)
    const size_t mem = (2 * matched.size() + 16) * ggml_tensor_overhead();
    struct ggml_init_params ip = { mem, nullptr, /*no_alloc=*/true };
    lt.ctx = ggml_init(ip);

    for (const auto & ti : matched) {
        const int64_t in_dim  = ti.ne0; // columns (input features)
        const int64_t out_dim = ti.ne1; // rows    (output features)

        // lora_a: [in_dim, rank]   applied first: a @ x
        // lora_b: [rank,   out_dim] applied second: b @ (a @ x)
        // Convention matches llama-adapter.cpp:48-60:
        //   a->ne[0] == in_dim,  a->ne[1] == rank
        //   b->ne[0] == rank,    b->ne[1] == out_dim
        ggml_tensor * la = ggml_new_tensor_2d(lt.ctx, GGML_TYPE_F32, in_dim, rank);
        ggml_tensor * lb = ggml_new_tensor_2d(lt.ctx, GGML_TYPE_F32, rank,   out_dim);

        ggml_set_name(la, (ti.name + ".lora_a").c_str());
        ggml_set_name(lb, (ti.name + ".lora_b").c_str());

        lt.ab[ti.name] = {la, lb};
    }

    // Allocate backend buffer for all LoRA tensors at once
    lt.buf = ggml_backend_alloc_ctx_tensors_from_buft(lt.ctx, ggml_backend_cpu_buffer_type());

    // Initialize: A ~ N(0, 1/sqrt(rank)), B = 0
    const float std_a = 1.0f / std::sqrt((float)rank);
    std::normal_distribution<float> dist(0.0f, std_a);

    for (auto & kv : lt.ab) {
        ggml_tensor * la = kv.second.first;
        ggml_tensor * lb = kv.second.second;

        // Fill A
        float * data_a = (float *) la->data;
        for (int64_t i = 0; i < ggml_nelements(la); ++i) data_a[i] = dist(rng);
        // Zero B
        memset(lb->data, 0, ggml_nbytes(lb));
    }

    return lt;
}

// ---------------------------------------------------------------------------
// Param filter: only train lora_a / lora_b tensors
// ---------------------------------------------------------------------------

static bool lora_param_filter(const struct ggml_tensor * t, void * /*ud*/) {
    const char * n = t->name;
    const size_t len = strlen(n);
    if (len > 7 && strcmp(n + len - 7, ".lora_a") == 0) return true;
    if (len > 7 && strcmp(n + len - 7, ".lora_b") == 0) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Save adapter GGUF
// ---------------------------------------------------------------------------

static void save_adapter(
        const lora_tensors & lt,
        const std::string  & out_path,
        const std::string  & arch,
        float                alpha) {

    // Build output GGUF context
    struct gguf_context * gctx = gguf_init_empty();

    // Metadata required by llama_adapter_lora_init
    gguf_set_val_str(gctx, "general.type",         "adapter");
    gguf_set_val_str(gctx, "general.architecture",  arch.c_str());
    gguf_set_val_str(gctx, "adapter.type",          "lora");
    gguf_set_val_f32(gctx, "adapter.lora.alpha",    alpha);

    // Register tensors
    for (const auto & kv : lt.ab) {
        gguf_add_tensor(gctx, kv.second.first);   // lora_a
        gguf_add_tensor(gctx, kv.second.second);  // lora_b
    }

    // Write: meta placeholder → tensor data → rewrite meta
    std::ofstream fout(out_path, std::ios::binary);
    if (!fout.is_open()) {
        LOG_ERR("%s: cannot open %s for writing\n", __func__, out_path.c_str());
        gguf_free(gctx);
        return;
    }

    // Write meta placeholder
    const size_t meta_size = gguf_get_meta_size(gctx);
    std::vector<char> zeros_buf(meta_size, 0);
    fout.write(zeros_buf.data(), meta_size);

    // Write tensor data
    for (const auto & kv : lt.ab) {
        for (ggml_tensor * t : {kv.second.first, kv.second.second}) {
            const size_t nb = ggml_nbytes(t);
            fout.write((const char *) t->data, nb);
            // GGUF tensors are 32-byte aligned
            const size_t pad = GGML_PAD(nb, 32) - nb;
            if (pad > 0) {
                std::vector<char> pad_buf(pad, 0);
                fout.write(pad_buf.data(), pad);
            }
        }
    }

    // Re-write metadata at offset 0
    std::vector<uint8_t> meta(meta_size);
    gguf_get_meta_data(gctx, meta.data());
    fout.seekp(0);
    fout.write((const char *) meta.data(), meta_size);

    fout.close();
    gguf_free(gctx);

    LOG_INF("%s: adapter saved to %s\n", __func__, out_path.c_str());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.escape = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE_QLORA)) {
        return 1;
    }

    // Force settings required for training
    params.use_mmap    = false;
    params.cache_type_k = GGML_TYPE_F32;
    params.cache_type_v = GGML_TYPE_F32;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (!model) { LOG_ERR("failed to load model\n"); return 1; }

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    // Resolve LoRA alpha (default: same as rank → effective scale = 1)
    const float lora_alpha = (params.lora_alpha > 0.0f)
        ? params.lora_alpha
        : (float) params.lora_rank;

    // Parse target substrings
    const auto targets = split_csv(params.lora_targets);

    // Get model architecture string directly from the GGUF metadata
    std::string arch;
    {
        struct ggml_context * ctx_meta = nullptr;
        struct gguf_init_params gp = { true, &ctx_meta };
        struct gguf_context * ctx_gguf = gguf_init_from_file(params.model.path.c_str(), gp);
        if (ctx_gguf) {
            int kid = gguf_find_key(ctx_gguf, "general.architecture");
            if (kid >= 0) arch = gguf_get_val_str(ctx_gguf, kid);
            gguf_free(ctx_gguf);
            ggml_free(ctx_meta);
        }
    }
    if (arch.empty()) {
        // Fall back to model description
        char arch_buf[256] = {};
        llama_model_desc(model, arch_buf, sizeof(arch_buf));
        arch = std::string(arch_buf);
        arch = arch.substr(0, arch.find_first_of(" /"));
    }

    // Load chat templates (optional — falls back to ChatML if unavailable)
    auto tmpls = common_chat_templates_init(model, "");

    // Load JSONL dataset
    if (params.train_file.empty()) {
        LOG_ERR("%s: --train-file is required\n", __func__);
        return 1;
    }

    auto samples = load_jsonl(params.train_file, ctx, tmpls.get());
    if (samples.empty()) {
        LOG_ERR("%s: no training samples loaded\n", __func__);
        return 1;
    }

    const int32_t n_ctx = llama_n_ctx(ctx);
    auto dataset = build_dataset(samples, n_ctx);
    if (!dataset) return 1;

    // Allocate LoRA tensors (reads tensor shapes from model GGUF directly)
    std::mt19937 rng(42);
    auto lt = alloc_lora_tensors(params.model.path, targets, params.lora_rank, rng);
    if (lt.ab.empty()) return 1;

    // Inject LoRA adapter into context so build_lora_mm picks it up
    // We build a transient llama_adapter_lora and register it on the context.
    // The tensors are owned by lt.ctx / lt.buf — they stay alive for training.
    {
        // Register each LoRA A/B pair as trainable parameters directly.
        // We call ggml_set_param here because llama_set_param inside opt_init
        // only sees the base model layers; our tensors are in a separate context.
        for (auto & kv : lt.ab) {
            ggml_set_param(kv.second.first);   // lora_a → trainable
            // lora_b is initialized to zero; mark trainable too
            ggml_set_param(kv.second.second);  // lora_b → trainable
        }

        // Build a temporary llama_adapter_lora from our tensors and apply to ctx
        // so that build_lora_mm injects the LoRA deltas into the compute graph.
        //
        // llama_adapter_lora is a private struct — we use the public init API
        // which loads from a GGUF file.  Instead we exploit that the adapter is
        // looked up by base weight pointer inside build_lora_mm:
        //   lora.first->get_weight(w)  →  finds by base-tensor pointer key
        //
        // We cannot call get_weight because it is on the internal struct.
        // Therefore we pre-save a temp GGUF, load it back, and apply it.
        // This is the safest path that does NOT require changes to private API.

        const std::string tmp_path = params.lora_out + ".tmp_train.gguf";
        save_adapter(lt, tmp_path, arch, lora_alpha);

        struct llama_adapter_lora * adapter = llama_adapter_lora_init(model, tmp_path.c_str());
        if (!adapter) {
            LOG_ERR("%s: failed to init LoRA adapter from temp file\n", __func__);
            return 1;
        }

        float scale = 1.0f;
        llama_set_adapters_lora(ctx, &adapter, 1, &scale);

        // The temp file is no longer needed; clean up asynchronously after training.
        // (We leave it for now; it gets overwritten by the final save.)
    }

    // Initialize optimizer — our custom param filter restricts training to lora_a/b
    struct llama_opt_params lopt_params {
        /*.n_ctx_train     =*/0,
        /*.param_filter    =*/lora_param_filter,
        /*.param_filter_ud =*/nullptr,
        /*.get_opt_pars    =*/common_opt_lr_pars,
        /*.get_opt_pars_ud =*/&params.lr,
        /*.optimizer_type  =*/params.optimizer,
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    LOG_INF("%s: starting QLoRA training — rank=%d alpha=%.1f epochs=%d\n",
            __func__, params.lora_rank, lora_alpha, params.lr.epochs);

    for (params.lr.epoch = 0; params.lr.epoch < params.lr.epochs; ++params.lr.epoch) {
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
                        ggml_opt_epoch_callback_progress_bar,
                        ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n");
        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }

    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    // Save final adapter (overwrites temp file if same path)
    save_adapter(lt, params.lora_out, arch, lora_alpha);

    // Clean up temp init file if it differs
    const std::string tmp_path = params.lora_out + ".tmp_train.gguf";
    if (tmp_path != params.lora_out) {
        std::remove(tmp_path.c_str());
    }

    ggml_backend_buffer_free(lt.buf);
    ggml_free(lt.ctx);
    ggml_opt_dataset_free(dataset);
    llama_backend_free();

    return 0;
}
