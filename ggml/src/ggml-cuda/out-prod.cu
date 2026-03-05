#include "out-prod.cuh"
#include "convert.cuh"

#include <cstdint>

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
