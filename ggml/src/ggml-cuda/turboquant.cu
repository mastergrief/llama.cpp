// CUDA implementation for GGML_TYPE_TQ3_K256 (TurboQuant 3-bit, head_dim=256).
//
// Self-contained TU: device tables + per-block quantize function +
// SET_ROWS kernel + host launch wrappers + lazy init. Mirrors the
// quantize_row_tq3_k256_ref reference in ggml-quants.c on a per-block
// basis (one CUDA thread = one 256-element vector). The CPU reference
// is the algorithmic oracle; this kernel must produce bit-equivalent
// output on the same input.
//
// Why one TU: nvcc without RDC treats `extern __device__` declarations
// as definitions, which causes link conflicts when device symbols are
// shared across .cu files. Putting all tq3_k256 device state and code
// in one TU avoids the issue entirely. set-rows.cu only calls our host
// launch wrappers (declared in turboquant.cuh).

#include "turboquant.cuh"
#include "ggml-common.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mutex>
#include <cstdio>
#include <cstdint>

// Pull in the host-side Pi data + dimension constants. The header is
// `static const`, so this TU gets its own ~1.2 MB host-side copy. We
// only touch it once at init and then cudaMemcpyToSymbol it onto the
// device.
#include "../turboquant_tables.h"

// CPU-side codebook is computed in ggml-quants.c via a closed-form
// Lloyd-Max iteration. We forward-declare the getters to avoid pulling
// ggml-quants.h (which exposes block_tq3_k256 and would conflict with
// ggml-common.h's typedef).
extern "C" {
    void          ggml_tq3_k256_init_impl(void);
    const float * ggml_tq3_k256_get_centroids(void);
    const float * ggml_tq3_k256_get_boundaries(void);
}

// CUDA SET_ROWS uses 256-thread blocks per the existing convention.
#ifndef CUDA_SET_ROWS_BLOCK_SIZE
#define CUDA_SET_ROWS_BLOCK_SIZE 256
#endif

// ============================================================
// Device-side tables (one TU = one symbol set, no cross-TU sharing).
// ============================================================
__device__   static float g_tq3_k256_pi_d[256 * 256];     // 256 KB
__constant__ static float g_tq3_k256_centroids_d[8];      // 32 B
__constant__ static float g_tq3_k256_boundaries_d[7];     // 28 B

// ============================================================
// Per-block quantize function. One CUDA thread does the entire
// 256-element vector: norm → Pi rotation → quantize → 3-bit pack.
// Mirrors quantize_row_tq3_k256_ref's per-block loop body in
// ggml-quants.c. Bit-equivalent output on the same input.
// ============================================================
static __device__ void quantize_f32_tq3_k256_block(
        const float    * __restrict__ x,
        block_tq3_k256 * __restrict__ y) {

    // 1. L2 norm
    float norm_sq = 0.0f;
    for (int j = 0; j < 256; ++j) {
        norm_sq += x[j] * x[j];
    }
    const float norm     = sqrtf(norm_sq);
    const float inv_norm = (norm > 1e-8f) ? (1.0f / norm) : 0.0f;

    // 2 + 3. Stream Pi rotation + 3-bit quantization in groups of 8.
    //        rotated[ii] = inv_norm * <x, Pi[ii, :]>
    //        Pi is row-major: row ii starts at g_tq3_k256_pi_d + ii*256.
    for (int g = 0; g < 32; ++g) {  // 32 groups × 8 = 256
        uint32_t packed = 0;
        for (int s = 0; s < 8; ++s) {
            const int     ii  = g * 8 + s;
            const float * row = g_tq3_k256_pi_d + (size_t) ii * 256;
            float acc = 0.0f;
            for (int j = 0; j < 256; ++j) {
                acc += x[j] * row[j];
            }
            const float v = inv_norm * acc;

            // 7-boundary monotonic scan → centroid index 0..7.
            int idx;
            if      (v < g_tq3_k256_boundaries_d[0]) idx = 0;
            else if (v < g_tq3_k256_boundaries_d[1]) idx = 1;
            else if (v < g_tq3_k256_boundaries_d[2]) idx = 2;
            else if (v < g_tq3_k256_boundaries_d[3]) idx = 3;
            else if (v < g_tq3_k256_boundaries_d[4]) idx = 4;
            else if (v < g_tq3_k256_boundaries_d[5]) idx = 5;
            else if (v < g_tq3_k256_boundaries_d[6]) idx = 6;
            else                                     idx = 7;

            packed |= ((uint32_t) idx) << (s * 3);
        }
        y->qs[g * 3 + 0] = (uint8_t)( packed        & 0xFFu);
        y->qs[g * 3 + 1] = (uint8_t)((packed >>  8) & 0xFFu);
        y->qs[g * 3 + 2] = (uint8_t)((packed >> 16) & 0xFFu);
    }

    // 4. fp16 norm
    y->d = __float2half(norm);
}

// ============================================================
// SET_ROWS kernel. Replicates k_set_rows_quant from set-rows.cu's
// generic template, specialized for tq3_k256 (block_tq3_k256, qk=256).
// We don't reuse the template because it's `static` in set-rows.cu;
// inlining the logic here keeps everything tq3-related in one TU.
// ============================================================
template <typename idx_t>
static __global__ void k_set_rows_tq3_k256(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_tq3_k256 * __restrict__ dst,
        const int64_t ne_total,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne11, const int64_t ne12) {

    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i >= ne_total) {
        return;
    }

    // Decompose linear block index `i` into (i00, i01, i02, i03) where i00
    // is in units of qk (256). One thread = one 256-elem block = one row.
    const int64_t i_base = i * 256;
    int64_t       tmp    = i_base;

    const int64_t i00 = tmp % ne00;  tmp /= ne00;
    const int64_t i01 = tmp % ne01;  tmp /= ne01;
    const int64_t i02 = tmp % ne02;
    const int64_t i03 = tmp / ne02;

    const int64_t i12 = i03 % ne12;
    const int64_t i11 = i02 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = src1[i10*s10 + i11*s11 + i12*s12];

    const float    * src0_row    = src0 + i01*s01 + i02*s02 + i03*s03;
    block_tq3_k256 * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_tq3_k256);

    const float    * src_block = src0_row + i00;
    block_tq3_k256 * dst_block = dst_row_ptr + i00 / 256;

    quantize_f32_tq3_k256_block(src_block, dst_block);
}

// ============================================================
// Host launch wrappers (one per index type).
// ============================================================
template <typename idx_t>
static void launch_set_rows_tq3_k256(
        const float * src0_d, const idx_t * src1_d, void * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    (void) ne10; (void) ne13;  // unused, mirror set-rows.cu signature

    if (ne00 % 256 != 0 || ne00 <= 0 || ne01 <= 0 || ne02 <= 0 || ne11 <= 0 || ne12 <= 0) {
        return;
    }

    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / 256;
    if (ne_total <= 0) {
        return;
    }

    const int num_blocks = (int)((ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = (int64_t) (nb01 / sizeof(float));
    const int64_t s02 = (int64_t) (nb02 / sizeof(float));
    const int64_t s03 = (int64_t) (nb03 / sizeof(float));
    const int64_t s10 = (int64_t) (nb10 / sizeof(idx_t));
    const int64_t s11 = (int64_t) (nb11 / sizeof(idx_t));
    const int64_t s12 = (int64_t) (nb12 / sizeof(idx_t));
    const int64_t s1  = (int64_t) nb1;
    const int64_t s2  = (int64_t) nb2;
    const int64_t s3  = (int64_t) nb3;

    k_set_rows_tq3_k256<idx_t><<<grid_size, block_size, 0, stream>>>(
        src0_d, src1_d, (block_tq3_k256 *) dst_d, ne_total,
        s01, s02, s03, s10, s11, s12, s1, s2, s3,
        ne00, ne01, ne02, ne11, ne12);
}

void ggml_cuda_set_rows_tq3_k256_i64(
        const float * src0_d, const int64_t * src1_d, void * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    launch_set_rows_tq3_k256<int64_t>(
        src0_d, src1_d, dst_d,
        ne00, ne01, ne02, ne03,
        ne10, ne11, ne12, ne13,
        nb01, nb02, nb03,
        nb10, nb11, nb12,
        nb1, nb2, nb3,
        stream);
}

void ggml_cuda_set_rows_tq3_k256_i32(
        const float * src0_d, const int32_t * src1_d, void * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    launch_set_rows_tq3_k256<int32_t>(
        src0_d, src1_d, dst_d,
        ne00, ne01, ne02, ne03,
        ne10, ne11, ne12, ne13,
        nb01, nb02, nb03,
        nb10, nb11, nb12,
        nb1, nb2, nb3,
        stream);
}

// ============================================================
// Host init: populates the device tables. Idempotent + thread-safe.
// Called from set-rows.cu's TQ3_K256 dispatch branch on every op (the
// std::call_once ensures the actual work happens once per process).
// ============================================================
static std::once_flag g_tq3_k256_cuda_init_flag;

extern "C" void ggml_tq3_k256_ensure_cuda_init(void) {
    std::call_once(g_tq3_k256_cuda_init_flag, []() {
        // Make sure the CPU codebook is computed before we try to copy it.
        ggml_tq3_k256_init_impl();

        const float * centroids  = ggml_tq3_k256_get_centroids();
        const float * boundaries = ggml_tq3_k256_get_boundaries();

        cudaError_t err;

        err = cudaMemcpyToSymbol(g_tq3_k256_pi_d, TQ3_K256_PI,
                                 sizeof(float) * 256 * 256);
        if (err != cudaSuccess) {
            fprintf(stderr, "ggml_tq3_k256_ensure_cuda_init: copy Pi failed: %s\n",
                    cudaGetErrorString(err));
            return;
        }
        err = cudaMemcpyToSymbol(g_tq3_k256_centroids_d, centroids,
                                 sizeof(float) * 8);
        if (err != cudaSuccess) {
            fprintf(stderr, "ggml_tq3_k256_ensure_cuda_init: copy centroids failed: %s\n",
                    cudaGetErrorString(err));
            return;
        }
        err = cudaMemcpyToSymbol(g_tq3_k256_boundaries_d, boundaries,
                                 sizeof(float) * 7);
        if (err != cudaSuccess) {
            fprintf(stderr, "ggml_tq3_k256_ensure_cuda_init: copy boundaries failed: %s\n",
                    cudaGetErrorString(err));
            return;
        }

        // Block until symbol copies have actually landed on device.
        cudaDeviceSynchronize();
    });
}
