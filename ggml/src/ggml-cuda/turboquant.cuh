// CUDA support for GGML_TYPE_TQ3_K256 (TurboQuant 3-bit, head_dim=256).
//
// Everything tq3_k256-specific lives in a single TU (turboquant.cu) to
// avoid the cross-TU __device__ symbol problem nvcc has without RDC.
// This header just exposes the host-callable entry points used by
// set-rows.cu's dispatch.

#pragma once

#include "common.cuh"

// Lazy host init: copies the Pi rotation matrix and the Lloyd-Max
// codebook into device memory on first call. Idempotent + thread-safe.
extern "C" void ggml_tq3_k256_ensure_cuda_init(void);

// Host launch wrappers for SET_ROWS on a TQ3_K256 destination tensor.
// Mirrors the parameter list of set_rows_cuda_quant in set-rows.cu but
// with the source/dest types fixed and the per-block quantize function
// resolved internally to a tq3_k256 specialization. Two specializations
// for the index type (int64_t or int32_t).
void ggml_cuda_set_rows_tq3_k256_i64(
        const float * src0_d, const int64_t * src1_d, void * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream);

void ggml_cuda_set_rows_tq3_k256_i32(
        const float * src0_d, const int32_t * src1_d, void * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream);
