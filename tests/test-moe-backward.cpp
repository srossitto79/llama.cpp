// Backward pass of the ops the MoE FFN uses in training, checked against reference gradients:
//   - GET_ROWS with batched rows, as the router selects the probabilities of the top-k experts
//     (the indices are a view of the argsort result, as ggml_top_k returns them)
//   - MUL_MAT_ID with src1 broadcast over the expert slots, as the gate/up projections use it,
//     with F32 experts and with quantized (frozen) experts, whose src1 gradient must still flow
//
// The gradient graph is built by ggml.c and computed on the CPU backend.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <initializer_list>
#include <cstdio>
#include <vector>

static int n_failed = 0;

static float * data_f32(ggml_tensor * t) { return (float   *) t->data; }
static int   * data_i32(ggml_tensor * t) { return (int32_t *) t->data; }

static void fill(ggml_tensor * t, float scale) {
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        data_f32(t)[i] = scale * (float) ((i * 7919) % 23 - 11);
    }
}

static void check(const char * name, const ggml_tensor * grad, const std::vector<float> & ref) {
    double max_err = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_err = std::fmax(max_err, std::fabs(((const float *) grad->data)[i] - ref[i]));
    }
    const bool ok = max_err < 1e-3;
    printf("%-50s max_abs_err=%-12g %s\n", name, max_err, ok ? "OK" : "FAIL");
    n_failed += ok ? 0 : 1;
}

static ggml_cgraph * compute_backward(ggml_context * ctx, ggml_tensor * loss) {
    ggml_set_loss(loss);
    ggml_cgraph * gb = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(gb, loss);
    ggml_build_backward_expand(ctx, gb, nullptr);
    ggml_graph_reset(gb);
    ggml_graph_compute_with_ctx(ctx, gb, 2);
    return gb;
}

// in [n, m, B], rows [r, B]: out[:, k, t] = in[:, rows[k, t], t]
static void test_get_rows_batched(bool rows_view) {
    const int n = 3, m = 8, r = 2, B = 5;

    ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, m, B);
    ggml_set_param(in);
    fill(in, 0.1f);

    // a view keeps the first r of m indices per batch, like ggml_top_k over an argsort
    ggml_tensor * rows_full = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, rows_view ? m : r, B);
    for (int t = 0; t < B; ++t) {
        for (int k = 0; k < rows_full->ne[0]; ++k) {
            data_i32(rows_full)[t*rows_full->ne[0] + k] = (t + 3*k + 1) % m;
        }
    }
    ggml_tensor * rows = rows_view ? ggml_view_2d(ctx, rows_full, r, B, rows_full->nb[1], 0) : rows_full;

    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, r, B);
    fill(w, 1.0f);

    ggml_tensor * out = ggml_get_rows(ctx, in, rows);
    ggml_cgraph * gb  = compute_backward(ctx, ggml_sum(ctx, ggml_mul(ctx, out, w)));

    std::vector<float> ref(n*m*B, 0.0f);
    for (int t = 0; t < B; ++t) {
        for (int k = 0; k < r; ++k) {
            const int row = data_i32(rows_full)[t*rows_full->ne[0] + k];
            for (int c = 0; c < n; ++c) {
                ref[(t*m + row)*n + c] += data_f32(w)[(t*r + k)*n + c];
            }
        }
    }
    check(rows_view ? "GET_ROWS batched, rows view: grad(in)" : "GET_ROWS batched: grad(in)", ggml_graph_get_grad(gb, in), ref);

    ggml_free(ctx);
}

// as [K, M, E], b [K, nb1, T] with nb1 = 1 (broadcast) or n_used, ids [n_used, T]
//   out[m, i, t] = sum_k as[k, m, ids[i, t]] * b[k, i % nb1, t]
// A quantized as is frozen: only grad(b) is checked, against the dequantized weights.
static void test_mul_mat_id(bool broadcast, ggml_type type) {
    const bool quantized = ggml_is_quantized(type);
    const int K = quantized ? (int) ggml_blck_size(type) : 4, M = 3, E = 4, n_used = 2, T = 3;
    const int nb1 = broadcast ? 1 : n_used;

    ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * as_f32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, M, E);
    fill(as_f32, 0.05f);
    ggml_tensor * as = as_f32;
    if (quantized) {
        as = ggml_new_tensor_3d(ctx, type, K, M, E);
        ggml_quantize_chunk(type, data_f32(as_f32), as->data, 0, M*E, K, nullptr);
        ggml_get_type_traits(type)->to_float(as->data, data_f32(as_f32), K*M*E); // reference weights
    } else {
        ggml_set_param(as);
    }

    ggml_tensor * b  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, nb1, T);
    ggml_set_param(b);
    fill(b, 0.2f);

    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, T);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < n_used; ++i) {
            data_i32(ids)[t*n_used + i] = (t + 2*i) % E;
        }
    }

    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, M, n_used, T);
    fill(w, 1.0f);

    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_cgraph * gb  = compute_backward(ctx, ggml_sum(ctx, ggml_mul(ctx, out, w)));

    std::vector<float> ref_as(K*M*E, 0.0f);
    std::vector<float> ref_b(K*nb1*T, 0.0f);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < n_used; ++i) {
            const int e = data_i32(ids)[t*n_used + i];
            const int j = i % nb1;
            for (int mm = 0; mm < M; ++mm) {
                const float g = data_f32(w)[(t*n_used + i)*M + mm];
                for (int k = 0; k < K; ++k) {
                    ref_as[(e*M + mm)*K + k] += data_f32(b)[(t*nb1 + j)*K + k] * g;
                    ref_b[(t*nb1 + j)*K + k] += data_f32(as_f32)[(e*M + mm)*K + k] * g;
                }
            }
        }
    }
    char name[96];
    if (!quantized) {
        snprintf(name, sizeof(name), "MUL_MAT_ID%s: grad(as)", broadcast ? " broadcast src1" : "");
        check(name, ggml_graph_get_grad(gb, as), ref_as);
    }
    snprintf(name, sizeof(name), "MUL_MAT_ID %s%s: grad(b)", ggml_type_name(type), broadcast ? " broadcast src1" : "");
    ggml_tensor * grad_b = ggml_graph_get_grad(gb, b);
    if (grad_b == nullptr) {
        printf("%-50s no gradient FAIL\n", name);
        n_failed++;
    } else {
        check(name, grad_b, ref_b);
    }

    ggml_free(ctx);
}

int main() {
    test_get_rows_batched(false);
    test_get_rows_batched(true);
    for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K }) {
        test_mul_mat_id(false, type);
        test_mul_mat_id(true,  type);
    }

    if (n_failed > 0) {
        printf("%d check(s) failed\n", n_failed);
        return 1;
    }
    return 0;
}
