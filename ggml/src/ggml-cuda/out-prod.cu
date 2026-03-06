#include "out-prod.cuh"
#include "convert.cuh"

#include <cstdint>
#include <vector>

void ggml_cuda_out_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    GGML_ASSERT(ne01 == ne11);
    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);

    GGML_ASSERT(ne2 % src0->ne[2] == 0);
    GGML_ASSERT(ne3 % src0->ne[3] == 0);

    GGML_ASSERT(ne2 == src1->ne[2]);
    GGML_ASSERT(ne3 == src1->ne[3]);

    cudaStream_t   stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const bool src1_T = ggml_is_transposed(src1);
    const cublasOperation_t src1_cublas_op = src1_T ? CUBLAS_OP_N : CUBLAS_OP_T;
    const int64_t ldb = (src1_T ? nb10 : nb11) / sizeof(float);
    GGML_ASSERT(    (src1_T ? nb11 : nb10) == sizeof(float));

    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // dps == dst per src0, used for group query attention broadcasting
    const int64_t dps2 = ne2 / ne02;
    const int64_t dps3 = ne3 / ne03;

    if (src0->type == GGML_TYPE_F32) {
        // Fast path: all F32, use cuBLAS directly (original implementation)
        const float * src0_d = (const float *) src0->data;

        const int64_t lda = nb01 / sizeof(float);
        const int64_t ldc = nb1  / sizeof(float);

        const size_t s02 = nb02 / sizeof(float);
        const size_t s03 = nb03 / sizeof(float);
        const size_t s12 = nb12 / sizeof(float);
        const size_t s13 = nb13 / sizeof(float);
        const size_t s2  = nb2  / sizeof(float);
        const size_t s3  = nb3  / sizeof(float);

        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                CUBLAS_CHECK(
                    cublasSgemm(handle, CUBLAS_OP_N, src1_cublas_op,
                            ne0, ne1, ne01,
                            &alpha, src0_d + (i3/dps3)*s03 + (i2/dps2)*s02, lda,
                                    src1_d +  i3      *s13 +  i2      *s12, ldb,
                            &beta,  dst_d  +  i3      *s3  +  i2      *s2,  ldc));
            }
        }
    } else {
        // Quantized src0 path: dequantize each 2D slice to F32 on GPU, then cuBLAS.
        // This enables the backward pass of MUL_MAT(quantized_weight, activation) to
        // run entirely on GPU instead of falling back to the CPU out_prod_q_f32 path.
        //
        // OUT_PROD(W[ne00,ne01], B_transposed[ne10,ne11]):
        //   dst[i0, i1] = sum_{k} W[i0,k] * B_T[i1,k]
        //   = W @ B_T^T  (standard matmul)
        //
        // We dequantize W to F32 and call cublasSgemm with:
        //   C[ne00, ne10] = W_f32[ne00, ne01] @ B[ne11, ne10]^T   (ne01==ne11)
        //   lda = ne00 (column-major leading dim for W)
        //   ldb = ne10 or ne11 depending on src1 transposition (same as F32 path)
        //   ldc = ne0 = ne00

        const to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(src0->type);
        GGML_ASSERT(to_fp32 != nullptr);

        // Allocate temp F32 buffer for one 2D slice of src0 (ne00 × ne01 floats)
        const int64_t slice_elems = ne00 * ne01;
        ggml_cuda_pool_alloc<float> src0_f32(ctx.pool(), slice_elems);

        // Byte strides for src0 slices in dims 2/3
        const size_t s02_bytes = nb02;
        const size_t s03_bytes = nb03;
        const size_t s12 = nb12 / sizeof(float);
        const size_t s13 = nb13 / sizeof(float);
        const size_t s2  = nb2  / sizeof(float);
        const size_t s3  = nb3  / sizeof(float);

        const int64_t lda = ne00;  // dequantized slice is always contiguous: col-major [ne00, ne01]
        const int64_t ldc = nb1 / sizeof(float);  // dst leading dim — matches F32 path

        const char * src0_base = (const char *) src0->data;

        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                const char * src0_slice = src0_base
                    + (i3/dps3) * s03_bytes
                    + (i2/dps2) * s02_bytes;

                // Dequantize this [ne00, ne01] slice to F32 on GPU
                to_fp32(src0_slice, src0_f32.get(), slice_elems, stream);

                CUBLAS_CHECK(
                    cublasSgemm(handle, CUBLAS_OP_N, src1_cublas_op,
                            ne0, ne1, ne01,
                            &alpha, src0_f32.get(),                          lda,
                                    src1_d + i3*s13 + i2*s12,               ldb,
                            &beta,  dst_d  + i3*s3  + i2*s2,                ldc));
            }
        }
    }
}

// ggml_cuda_out_prod_id — scattered outer-product for MUL_MAT_ID backward (grad w.r.t. src0).
//
//   a   = src0: [cols, n_expert_used, n_tokens]  F32
//   b   = src1: [rows, n_expert_used, n_tokens]  F32
//   ids = src2: [n_expert_used, n_tokens]        i32  (may be CPU-resident in backward graph)
//   dst:        [cols, rows, n_expert, 1]         F32
//
//   dst[:, :, e] += a[:, i, t] ⊗ b[:, i, t]   ∀ (i,t) : ids[i,t] == e
//
// Algorithm:
//   1. Read ids from CPU (memcpy) or GPU (cudaMemcpy).
//   2. Bucket tokens per expert.
//   3. For each expert with >0 tokens: gather a_tokens and b_tokens into contiguous buffers,
//      then cublasSgemm with beta=1 to accumulate into dst[:,:,e].

void ggml_cuda_out_prod_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // a [cols, n_exp_used, n_tokens]
    const ggml_tensor * src1 = dst->src[1]; // b [rows, n_exp_used, n_tokens]
    const ggml_tensor * ids  = dst->src[2]; // [n_exp_used, n_tokens] i32

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t cols       = src0->ne[0];
    const int64_t n_exp_used = src0->ne[1];
    const int64_t n_tokens   = src0->ne[2];
    const int64_t rows       = src1->ne[0];
    const int64_t n_expert   = dst->ne[2];

    cudaStream_t   stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    // Zero dst
    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));

    // Read ids to host
    const size_t ids_nbytes = ggml_nbytes(ids);
    std::vector<char> ids_host(ids_nbytes);
    if (ids->buffer && !ggml_backend_buffer_is_host(ids->buffer)) {
        CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ids_nbytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        memcpy(ids_host.data(), ids->data, ids_nbytes);
    }

    // Build per-expert token lists: for each (iexp, itok) pair, find which expert it maps to
    std::vector<std::vector<int64_t>> expert_tokens(n_expert); // expert_tokens[e] = list of flat (iexp*n_tokens+itok)
    for (int64_t itok = 0; itok < n_tokens; ++itok) {
        for (int64_t iexp = 0; iexp < n_exp_used; ++iexp) {
            const int32_t expert_id = *(const int32_t *)(ids_host.data()
                + itok * ids->nb[1] + iexp * ids->nb[0]);
            GGML_ASSERT(expert_id >= 0 && expert_id < n_expert);
            expert_tokens[expert_id].push_back(iexp * n_tokens + itok);
        }
    }

    // For each expert, gather tokens and do cublasSgemm
    const float alpha_one = 1.0f;
    const float beta_acc  = 1.0f; // accumulate (dst already zeroed above)

    const float * a_base = (const float *) src0->data; // [cols, n_exp_used, n_tokens]
    const float * b_base = (const float *) src1->data; // [rows, n_exp_used, n_tokens]
    float       * d_base = (float       *) dst->data;  // [cols, rows, n_expert]

    // Stride layout: src0 nb[0]=sizeof(float), nb[1]=cols*sizeof(float), nb[2]=cols*n_exp_used*sizeof(float)
    const int64_t a_stride_exp  = src0->nb[1] / sizeof(float); // cols
    const int64_t a_stride_tok  = src0->nb[2] / sizeof(float); // cols * n_exp_used
    const int64_t b_stride_exp  = src1->nb[1] / sizeof(float); // rows
    const int64_t b_stride_tok  = src1->nb[2] / sizeof(float); // rows * n_exp_used
    const int64_t dst_stride_e  = dst->nb[2]  / sizeof(float); // cols * rows

    for (int64_t e = 0; e < n_expert; ++e) {
        const auto & toks = expert_tokens[e];
        if (toks.empty()) continue;

        const int64_t ntoks_e = (int64_t) toks.size();

        // Allocate gathered a_e [cols, ntoks_e] and b_e [rows, ntoks_e] on GPU
        ggml_cuda_pool_alloc<float> a_gathered(ctx.pool(), cols * ntoks_e);
        ggml_cuda_pool_alloc<float> b_gathered(ctx.pool(), rows * ntoks_e);

        // Gather token columns from GPU src0/src1 using device-to-device cudaMemcpyAsync.
        // src0->data and src1->data are GPU pointers — must NOT memcpy on the host.
        for (int64_t ti = 0; ti < ntoks_e; ++ti) {
            const int64_t flat = toks[ti];
            const int64_t iexp = flat / n_tokens;
            const int64_t itok = flat % n_tokens;
            const float * a_col = a_base + iexp * a_stride_exp + itok * a_stride_tok;
            const float * b_col = b_base + iexp * b_stride_exp + itok * b_stride_tok;
            CUDA_CHECK(cudaMemcpyAsync(a_gathered.ptr + ti * cols, a_col,
                        cols * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(b_gathered.ptr + ti * rows, b_col,
                        rows * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }

        // cublasSgemm: C[cols,rows] += A[cols,ntoks_e] @ B[rows,ntoks_e]^T
        //   (column-major: A is [cols, ntoks_e] with lda=cols, B is [rows, ntoks_e] with ldb=rows)
        float * dst_e = d_base + e * dst_stride_e;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                cols, rows, ntoks_e,
                &alpha_one, a_gathered.ptr, cols,
                            b_gathered.ptr, rows,
                &beta_acc,  dst_e,          cols));
    }
}
