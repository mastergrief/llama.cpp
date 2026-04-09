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
//
// Note on access patterns: g_fattn_tq3_pi_d is read with the per-thread
// pattern g_pi[j*256 + row]. CPU cache intuition says this is "column-
// strided and bad" but on a GPU warp, when 32 threads each have a
// different `row` (4 or 8 apart) and the same `j`, the resulting reads
// are at addresses [j*256+0, j*256+4, j*256+8, ...] which is SEQUENTIAL
// within row j of Pi — i.e., warp-coalesced. Switching to a Pi^T array
// (g_pi_t[row*256 + j]) breaks coalescing because consecutive threads
// then read addresses 1024 floats apart. We tried it; it was 38% slower.
// ============================================================
__device__   static float g_fattn_tq3_pi_d  [256 * 256];   // 256 KB Pi
__constant__ static float g_fattn_tq3_centroids_d [8];    // 32 B Lloyd-Max centroids
__constant__ static float g_fattn_tq3_boundaries_d[7];    // 28 B (kept for symmetry; FA only reads centroids)

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

        // Cross-TU pointer indirection: get the device address of THIS TU's
        // g_fattn_tq3_pi_d and stash it in fattn-common.cuh's per-TU
        // __constant__ slot g_fattn_tq3_pi_const_ptr. The kernel template
        // body in fattn-vec.cuh reads from g_fattn_tq3_pi_const_ptr to do
        // the Pi @ Q precompute. Since this TU is where the kernel
        // template gets instantiated (via DECL_FATTN_VEC_CASE below), the
        // __constant__ slot we set here is the same one the kernel reads.
        void * pi_d_addr = nullptr;
        err = cudaGetSymbolAddress(&pi_d_addr, g_fattn_tq3_pi_d);
        if (err != cudaSuccess) {
            fprintf(stderr, "ggml_fattn_tq3_ensure_init: cudaGetSymbolAddress(g_fattn_tq3_pi_d) failed: %s\n",
                    cudaGetErrorString(err));
            return;
        }
        const float * pi_d_ptr = (const float *) pi_d_addr;
        err = cudaMemcpyToSymbol(g_fattn_tq3_pi_const_ptr, &pi_d_ptr,
                                 sizeof(const float *));
        if (err != cudaSuccess) {
            fprintf(stderr, "ggml_fattn_tq3_ensure_init: set pi const ptr failed: %s\n",
                    cudaGetErrorString(err));
            return;
        }

        // Block until the symbol copies have actually landed on device.
        cudaDeviceSynchronize();
    });
}

// ============================================================
// vec_dot_fattn_vec_KQ_tq3_k256: per-thread Q · K_dequant for the FA
// vec kernel's K @ Q^T inner loop. ALGORITHMIC FAST PATH — relies on
// fattn-vec.cuh having already converted Q_reg from raw Q values into
// PiQ = Pi @ Q during the per-token precompute phase (see the constexpr
// branch in fattn-vec.cuh after Q load, gated on type_K == TQ3_K256).
//
// Math:
//   Q · K_dequant = Q · (Pi^T @ y) = (Pi @ Q) · y = PiQ · y
// where y[j] = scale * centroid[code[j]] is the dequantized centroid
// value at position j. The Pi rotation has already been folded into Q
// (one O(D²) precompute per token), so the per-K-row work is just
// per-thread bit decode + 8 mul-adds.
//
// Per-thread work per K row:
//   1. Read 3 bytes of K block (group t = bytes [t*3, t*3+3)). Each
//      thread reads its OWN unique 3 bytes — no redundant decoding.
//      The 32 threads' reads are coalesced (sequential 3-byte stride
//      across the warp, total 96 bytes per warp, fits in 1 cache line).
//   2. Decode 8 codes from those 3 bytes via 3-bit unpacking.
//   3. 8 mul-adds: PiQ_local[s] * scale * centroid[code[s]]
//   4. Return partial sum. Warp reduce is done by the caller.
//
// Total per warp per K row: 32 × 24 = 768 ops (vs ~65k ops in the prior
// implementation). ~85x reduction in per-K-row compute.
// ============================================================
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tq3_k256(
        const char * __restrict__ K_c,
        const void * __restrict__ Q_v,
        const int  * __restrict__ Q_q8,
        const void * __restrict__ Q_ds_v) {

    static_assert(D == 256, "tq3_k256 vec_dot only supports D=256");
    static_assert(nthreads == 32, "tq3_k256 requires nthreads=32 (1 thread per 3-byte group)");
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    const block_tq3_k256 * K_block = (const block_tq3_k256 *) K_c;
    const int t = (nthreads == WARP_SIZE) ? threadIdx.x : (threadIdx.x % nthreads);
    const float scale = __half2float(K_block->d);

    // Read this thread's 3-byte group (group t = bytes [t*3, t*3+3)) and
    // unpack 8 codes covering positions [t*8, t*8+8) in the original K
    // vector. Bit layout matches turboquant.cu's quantize:
    //   packed = qs[g*3+0] | qs[g*3+1]<<8 | qs[g*3+2]<<16
    //   code[g*8+s] = (packed >> (s*3)) & 0x07
    const uint32_t b0 = K_block->qs[t*3 + 0];
    const uint32_t b1 = K_block->qs[t*3 + 1];
    const uint32_t b2 = K_block->qs[t*3 + 2];
    const uint32_t packed = b0 | (b1 << 8) | (b2 << 16);

    // Compute partial dot product: sum_{s in 0..8} PiQ[t*8+s] * y[t*8+s]
    // where y[t*8+s] = scale * centroid[code_s]. PiQ values come from
    // Q_v which now points to this thread's PiQ slice (4 half2 / float2
    // for D=256, nthreads=32 → 8 fp values per thread).
    float sum = 0.0f;
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        const int   code  = (packed >> (s * 3)) & 0x07u;
        const float y_val = scale * g_fattn_tq3_centroids_d[code];

        // Read PiQ[t*8 + s] from the thread's per-warp register slice.
        float piq_val;
#ifdef V_DOT2_F32_F16_AVAILABLE
        const half2 piq_h2 = ((const half2 *) Q_v)[s / 2];
        piq_val = (s & 1) ? __high2float(piq_h2) : __low2float(piq_h2);
#else
        const float2 piq_f2 = ((const float2 *) Q_v)[s / 2];
        piq_val = (s & 1) ? piq_f2.y : piq_f2.x;
#endif

        sum += piq_val * y_val;
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
    // [off, off+ne). Uses the per-thread g_pi[j*256 + row] pattern: this
    // looks "column-strided" but on a GPU warp it's actually warp-COALESCED
    // because consecutive threads have row offsets 4 apart (not 1024 like
    // a row-strided Pi^T would be). See the device-tables comment block
    // above for the analysis. Outer loop is tiny (ne=2 or 4) so unroll it;
    // inner 256-loop stays rolled to keep cicc compile time tractable.
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
