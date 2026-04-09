// Template instance for tq3_k256 K + tq3_k256 V flash attention.
//
// Self-contained TU: device tables (Pi rotation matrix + Lloyd-Max
// codebook) + vec_dot_KQ + dequantize_V template specializations +
// lazy init via std::call_once + DECL_FATTN_VEC_CASE expansion.
//
// Why one TU: nvcc without RDC (relocatable device code) cannot share
// __device__ symbols across translation units. The Pi matrix and other
// tables already live in turboquant.cu's TU for the SET_ROWS path; this
// TU has its own private copies for the FA read path. ~256 KB extra
// device memory total — negligible against the 8 GB budget, and the
// alternative (RDC) would slow down the entire CUDA build.
//
// Algorithmic correctness: bit-equivalent to the CPU dispositive haiku
// validation by construction (same Pi data from turboquant_tables.h,
// same C-side codebook via ggml_tq3_k256_get_centroids, same dequant
// arithmetic).

#include "../common.cuh"
#include "../fattn-common.cuh"
#include "../fattn-vec.cuh"

#include "ggml-common.h"
#include "../../turboquant_tables.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mutex>
#include <cstdio>
#include <cstdint>
#include <type_traits>

// Forward declarations from ggml-quants.c (CPU side). The CPU code
// computes the Lloyd-Max codebook + boundaries from a closed-form
// Gaussian E[x|a<x<b] (no RNG, no scipy). We reuse those values on the
// device side via cudaMemcpyToSymbol — keeps a single source of truth
// for the codebook regardless of which path (CPU, CUDA SET_ROWS, CUDA FA)
// reads it.
extern "C" {
    void          ggml_tq3_k256_init_impl(void);
    const float * ggml_tq3_k256_get_centroids(void);
    const float * ggml_tq3_k256_get_boundaries(void);
}

// ============================================================
// Device-side tables. One TU = one private symbol set.
// Populated lazily by ggml_fattn_tq3_ensure_init() on first FA dispatch
// for tq3_k256, called from ggml_cuda_flash_attn_ext() in fattn.cu.
// ============================================================
__device__   static float g_fattn_tq3_pi_d[256 * 256];   // 256 KB rotation matrix
__constant__ static float g_fattn_tq3_centroids_d[8];    // 32 B Lloyd-Max centroids
__constant__ static float g_fattn_tq3_boundaries_d[7];   // 28 B (kept for symmetry; FA only reads centroids)

// ============================================================
// Lazy host init. Idempotent + thread-safe via std::call_once.
// Mirrors turboquant.cu's ggml_tq3_k256_ensure_cuda_init pattern, but
// populates THIS TU's private symbols. Both inits can coexist; they
// touch different physical __device__ allocations.
// ============================================================
static std::once_flag g_fattn_tq3_init_flag;

extern "C" void ggml_fattn_tq3_ensure_init(void) {
    std::call_once(g_fattn_tq3_init_flag, []() {
        // Make sure the CPU codebook is computed before we copy it.
        ggml_tq3_k256_init_impl();

        const float * centroids  = ggml_tq3_k256_get_centroids();
        const float * boundaries = ggml_tq3_k256_get_boundaries();

        cudaError_t err;

        err = cudaMemcpyToSymbol(g_fattn_tq3_pi_d, TQ3_K256_PI,
                                 sizeof(float) * 256 * 256);
        if (err != cudaSuccess) {
            fprintf(stderr, "ggml_fattn_tq3_ensure_init: copy Pi failed: %s\n",
                    cudaGetErrorString(err));
            return;
        }
        err = cudaMemcpyToSymbol(g_fattn_tq3_centroids_d, centroids,
                                 sizeof(float) * 8);
        if (err != cudaSuccess) {
            fprintf(stderr, "ggml_fattn_tq3_ensure_init: copy centroids failed: %s\n",
                    cudaGetErrorString(err));
            return;
        }
        err = cudaMemcpyToSymbol(g_fattn_tq3_boundaries_d, boundaries,
                                 sizeof(float) * 7);
        if (err != cudaSuccess) {
            fprintf(stderr, "ggml_fattn_tq3_ensure_init: copy boundaries failed: %s\n",
                    cudaGetErrorString(err));
            return;
        }

        // Block until the symbol copies have actually landed on device.
        cudaDeviceSynchronize();
    });
}

// ============================================================
// vec_dot_fattn_vec_KQ_tq3_k256: per-thread Q · (Pi @ K_dequant) for
// the FA vec kernel's K @ Q^T inner loop.
//
// Each thread:
//   1. Reads the full K block (96 bytes qs + 2 bytes d). Coalesced
//      across the warp since all 32 threads read the same address.
//   2. Decodes all 256 codes from the 96-byte qs array (3-bit unpacking).
//   3. Computes its 8 output elements of (Pi @ K_dequant) — rows
//      [t*8, t*8+8) where t = threadIdx.x % nthreads.
//   4. Dot-products those 8 outputs with its 8 Q stripe elements (read
//      from Q_v which points to this thread's Q registers, laid out as
//      4 half2 or 4 float2 depending on V_DOT2_F32_F16_AVAILABLE).
//   5. Returns the partial sum. The warp_reduce is done by the caller.
//
// Performance: ~2k mul-adds per thread per K row (8 output elements ×
// 256-element Pi rotation each). Per warp per K row: ~65k mul-adds. At
// 4096 ctx + Ada Lovelace (4070), ~30-100 µs per token of FA dequant
// overhead. Acceptable for first-light; cooperative dequant via shared
// memory is a future optimization (would cut the redundant decode work
// at the cost of one __syncthreads per K row).
// ============================================================
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tq3_k256(
        const char * __restrict__ K_c,
        const void * __restrict__ Q_v,
        const int  * __restrict__ Q_q8,
        const void * __restrict__ Q_ds_v) {

    static_assert(D == 256, "tq3_k256 vec_dot only supports D=256");
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    const block_tq3_k256 * K_block = (const block_tq3_k256 *) K_c;
    const int t = (nthreads == WARP_SIZE) ? threadIdx.x : (threadIdx.x % nthreads);
    const float scale = __half2float(K_block->d);

    // Decode all 256 codes from the 96-byte packed qs array.
    // 32 groups × 8 codes × 3 bits = 24 bits per group = 3 bytes per group.
    // Stack-resident, ~256 bytes per thread (may spill to local memory but
    // L1-cached on Ada Lovelace). NOT unrolled at the outer level — the
    // compiler picks a sensible partial unroll. Fully unrolling all 256
    // iterations is what made cicc spend an hour on this file in earlier
    // attempts; let it choose.
    uint8_t codes[256];
    for (int g = 0; g < 32; ++g) {
        const uint32_t b0 = K_block->qs[g*3 + 0];
        const uint32_t b1 = K_block->qs[g*3 + 1];
        const uint32_t b2 = K_block->qs[g*3 + 2];
        const uint32_t packed = b0 | (b1 << 8) | (b2 << 16);
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            codes[g*8 + s] = (uint8_t)((packed >> (s*3)) & 0x07u);
        }
    }

    // Each thread computes its 8 rows of (Pi^T @ K_dequant_centroids) and
    // dot-products with its 8 Q stripe elements.
    //
    // CRITICAL: tq3_k256 quantize uses Pi (y = Pi @ x), so dequantize must
    // use Pi^T (x = Pi^T @ y). In Pi[a, b] = g_pi[a*256 + b] layout, that
    // means reading g_pi[j*256 + row] (column-strided), NOT g_pi[row*256 + j]
    // (row-strided). See ggml-quants.c:2626 dequantize_row_tq3_k256 — it uses
    // TQ3_K256_PI[j * HEAD_DIM + ii], same column-strided pattern. The
    // outer 8-iteration loop unrolls fine, but the inner 256-iteration
    // Pi^T rotation stays rolled — fully unrolling sends cicc into a
    // multi-hour spiral.
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = t * 8 + i;

        float pi_dot = 0.0f;
        for (int j = 0; j < 256; ++j) {
            pi_dot += g_fattn_tq3_pi_d[(size_t) j * 256 + row]
                    * g_fattn_tq3_centroids_d[codes[j]];
        }

        // Read Q[t*8 + i] from this thread's local register slice.
        // Q_v is laid out as half2[(D/2)/nthreads_KQ] = half2[4] (or
        // float2[4] when V_DOT2_F32_F16_AVAILABLE is undefined). Each
        // half2/float2 holds 2 fp values, so element i maps to slot i/2
        // and lo/hi based on parity.
        float q_val;
#ifdef V_DOT2_F32_F16_AVAILABLE
        const half2 q_h2 = ((const half2 *) Q_v)[i / 2];
        q_val = (i & 1) ? __high2float(q_h2) : __low2float(q_h2);
#else
        const float2 q_f2 = ((const float2 *) Q_v)[i / 2];
        q_val = (i & 1) ? q_f2.y : q_f2.x;
#endif

        sum += scale * pi_dot * q_val;
    }

    return sum;
}

// ============================================================
// dequantize_V_tq3_k256: per-thread dequantize for V tile reads.
//
// Same algorithmic shape as vec_dot_KQ but writes ne dequantized output
// elements at offset i0 instead of returning a dot product. Each thread
// call decodes the FULL 256-code V block to get the centroid lookups,
// then computes ne (=4) elements of (Pi @ V_dequant) at rows [i0, i0+ne).
//
// Wasteful per-call (full decode for ne=4 outputs) but the warp total
// adds up to one full V-block dequant when 32 threads each do their
// 4-element slice. Constant overhead is higher than cooperative dequant
// would be (32x redundant code decoding) but the dominant Pi rotation
// work is the same. Optimize later.
// ============================================================
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_tq3_k256(
        const void * __restrict__ vx,
        void       * __restrict__ dst,
        const int64_t i0) {

    static_assert(ne == 2 || ne == 4, "bad ne");

    // Block size 256 → ib = i0 / 256, off = i0 % 256.
    const int64_t ib  = i0 / 256;
    const int     off = (int)(i0 % 256);

    const block_tq3_k256 * V_block = ((const block_tq3_k256 *) vx) + ib;
    const float scale = __half2float(V_block->d);

    // Decode all 256 codes (same shape as vec_dot_KQ — outer 32-loop NOT
    // unrolled to keep cicc compile time tractable, only the tiny inner
    // 8-loop is unrolled).
    uint8_t codes[256];
    for (int g = 0; g < 32; ++g) {
        const uint32_t b0 = V_block->qs[g*3 + 0];
        const uint32_t b1 = V_block->qs[g*3 + 1];
        const uint32_t b2 = V_block->qs[g*3 + 2];
        const uint32_t packed = b0 | (b1 << 8) | (b2 << 16);
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            codes[g*8 + s] = (uint8_t)((packed >> (s*3)) & 0x07u);
        }
    }

    // Compute ne output elements of (Pi^T @ V_dequant_centroids) at rows
    // [off, off+ne). Same Pi^T column-strided indexing as vec_dot_KQ — see
    // the comment there for the math. Outer loop is tiny (ne=2 or 4) so
    // unroll it; inner 256-loop stays rolled.
    #pragma unroll
    for (int l = 0; l < ne; ++l) {
        const int row = off + l;

        float pi_dot = 0.0f;
        for (int j = 0; j < 256; ++j) {
            pi_dot += g_fattn_tq3_pi_d[(size_t) j * 256 + row]
                    * g_fattn_tq3_centroids_d[codes[j]];
        }

        const float v = scale * pi_dot;

        if constexpr (std::is_same_v<T, half>) {
            ((half *) dst)[l] = __float2half(v);
        } else if constexpr (std::is_same_v<T, float>) {
            ((float *) dst)[l] = v;
        } else {
            static_assert(std::is_same_v<T, void>, "bad type");
        }
    }
}

// ============================================================
// Template instantiation. Only D=256 is meaningful for tq3_k256
// (block size matches head_dim). The fattn dispatch only routes
// tq3_k256 K cache to D=256 layers anyway.
// ============================================================
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TQ3_K256, GGML_TYPE_TQ3_K256);
