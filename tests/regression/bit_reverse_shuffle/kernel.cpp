/*
 * kernel.cpp  –  Vortex GPGPU port of CULIBRAShuffle.hpp
 *
 * CUDA concept          →  Vortex equivalent
 * ─────────────────────────────────────────────
 * __global__ kernel     →  plain C++ function launched with vx_spawn_threads
 * blockIdx.x            →  blockIdx.x   (from vx_spawn.h)
 * threadIdx.x           →  threadIdx.x  (from vx_spawn.h)
 * blockDim.x            →  blockDim.x   (from vx_spawn.h)
 * __shared__ T buf[]    →  __local_mem(size)  (per-workgroup scratchpad)
 * __syncthreads()       →  __syncthreads()    (barrier macro in vx_spawn.h)
 * Multiple kernel args  →  packed into kernel_arg_t, passed through MSCRATCH
 */

#include "common.h"
#include <vx_spawn.h>

/* ------------------------------------------------------------------ */
/*  Bit-reversal helper – identical logic to the CUDA template         */
/*  W is passed as a runtime value because Vortex kernels are plain C++*/
/* ------------------------------------------------------------------ */
static inline int bit_reverse(int x, int W) {
    int reversed = 0;
    for (int j = 0; j < W; j++)
        reversed = (reversed << 1) | ((x >> j) & 1);
    return reversed;
}

/* ------------------------------------------------------------------ */
/*  libra – in-place bit-reversal permutation on a 2^LOG_SIZE buffer   */
/*  Operates on the per-block scratch tile; threads cooperate.         */
/*  Direct port of the CUDA __device__ template libra<T, LOG_SIZE>.   */
/* ------------------------------------------------------------------ */
static void libra_local(TYPE *signal, int LOG_SIZE) {
    /* Number of (index, bit_reverse(index)) pairs with index < bit_reverse */
    int lim = (((1 << LOG_SIZE) - (1 << (LOG_SIZE / 2))) / 2);

    for (int id = (int)threadIdx.x; id < lim; id += (int)blockDim.x) {
        /* Reconstruct (col, row) from the flat index id */
        int offset = 0;
        int col = 1;
        int row = 0;
        for (int i = 1; i < (1 << (LOG_SIZE / 2)); i++) {
            int ri = bit_reverse(i, LOG_SIZE / 2);
            if (id - offset >= 0) {
                col = i;
                row = id - offset;
            }
            offset += ri;
        }

        int up = row;
        int pu = bit_reverse(row, LOG_SIZE / 2);
        int esab = bit_reverse(col, LOG_SIZE / 2);

        if (LOG_SIZE % 2) {
            /* Odd LOG_SIZE: two interleaved swaps */
            {
                int index = (up << ((LOG_SIZE / 2) + 1)) | col;
                int xendi = (esab << ((LOG_SIZE / 2) + 1)) | pu;
                TYPE tmp = signal[index];
                signal[index] = signal[xendi];
                signal[xendi] = tmp;
            }
            {
                int index = (up << ((LOG_SIZE / 2) + 1)) | (1 << (LOG_SIZE / 2)) | col;
                int xendi = (esab << ((LOG_SIZE / 2) + 1)) | (1 << (LOG_SIZE / 2)) | pu;
                TYPE tmp = signal[index];
                signal[index] = signal[xendi];
                signal[xendi] = tmp;
            }
        } else {
            /* Even LOG_SIZE: single swap */
            int index = (up << (LOG_SIZE / 2)) | col;
            int xendi = (esab << (LOG_SIZE / 2)) | pu;
            TYPE tmp = signal[index];
            signal[index] = signal[xendi];
            signal[xendi] = tmp;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  mem2buffer – gather a tile from global memory into local scratch   */
/*  Mirrors the CUDA __device__ template mem2buffer<T,LOG_N,LBW>.     */
/* ------------------------------------------------------------------ */
static void mem2buffer_local(unsigned int page, const TYPE *__restrict__ src,
                             TYPE *__restrict__ dst) {
    for (unsigned a = threadIdx.x; a < BLOCK_SIZE; a += blockDim.x) {
        unsigned src_base = (a << (LOG_N - LOG_BLOCK_WIDTH)) | (page << LOG_BLOCK_WIDTH);
        unsigned dst_base = a << LOG_BLOCK_WIDTH;
        for (unsigned i = 0; i < BLOCK_SIZE; i++)
            dst[dst_base + i] = src[src_base + i];
    }
}

/* ------------------------------------------------------------------ */
/*  buffer2mem – scatter local scratch back to global memory           */
/*  Mirrors the CUDA __device__ template buffer2mem<T,LOG_N,LBW>.     */
/* ------------------------------------------------------------------ */
static void buffer2mem_local(unsigned int page, const TYPE *__restrict__ src,
                             TYPE *__restrict__ dst) {
    for (unsigned a = threadIdx.x; a < BLOCK_SIZE; a += blockDim.x) {
        unsigned dst_base = (a << (LOG_N - LOG_BLOCK_WIDTH)) | (page << LOG_BLOCK_WIDTH);
        unsigned src_base = a << LOG_BLOCK_WIDTH;
        for (unsigned i = 0; i < BLOCK_SIZE; i++)
            dst[dst_base + i] = src[src_base + i];
    }
}

/* ------------------------------------------------------------------ */
/*  Kernel body                                                         */
/*                                                                      */
/*  One Vortex "block" (workgroup) handles one page, exactly as one    */
/*  CUDA thread-block handled one page in the original.                */
/*                                                                      */
/*  Grid mapping:                                                       */
/*    gridDim.x  = num_pages  (set by vx_spawn_threads in main.cpp)   */
/*    blockDim.x = THREAD_LIMIT (32) – set in main.cpp                 */
/*                                                                      */
/*  The per-block scratch buffer comes from a pre-allocated pool in    */
/*  global memory (scratch_pool) exactly as in the CUDA version.       */
/*  __local_mem could be used if BUF_ELEMS*sizeof(TYPE) ≤ local store; */
/*  using the global-memory pool mirrors the CUDA design faithfully.   */
/* ------------------------------------------------------------------ */
void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
    TYPE *input = reinterpret_cast<TYPE *>(arg->input_addr);
    const uint32_t *pages = reinterpret_cast<const uint32_t *>(arg->pages_addr);
    TYPE *scratch_pool = reinterpret_cast<TYPE *>(arg->scratch_addr);

    /* Each block gets its own scratch slice – mirrors CUDA indexing */
    TYPE *buf = scratch_pool + (unsigned long)blockIdx.x * BUF_ELEMS;

    if (arg->is_diag) {
        /* ── diagonal pages (libra_batch_diag) ── */
        unsigned page = pages[blockIdx.x];

        mem2buffer_local(page, input, buf);
        __syncthreads();

        libra_local(buf, 2 * LOG_BLOCK_WIDTH);
        __syncthreads();

        buffer2mem_local(page, buf, input);

    } else {
        /* ── off-diagonal pairs (libra_batch) ── */
        unsigned src_page = pages[blockIdx.x];
        unsigned not_is_odd = (blockIdx.x % 2) ? 0u : 1u;
        unsigned dst_page = pages[((blockIdx.x / 2) * 2) + not_is_odd];

        mem2buffer_local(src_page, input, buf);
        __syncthreads();

        libra_local(buf, 2 * LOG_BLOCK_WIDTH);
        __syncthreads();

        buffer2mem_local(dst_page, buf, input);
    }
}

/* ------------------------------------------------------------------ */
/*  Kernel entry point – reads kernel_arg_t from MSCRATCH CSR and      */
/*  launches one workgroup per page.                                    */
/* ------------------------------------------------------------------ */
int main() {
    kernel_arg_t *arg = reinterpret_cast<kernel_arg_t *>(csr_read(VX_CSR_MSCRATCH));

    /*
     * Launch a 1-D grid of num_pages workgroups.
     * block_dim == nullptr → Vortex uses its default warp/thread config.
     * The host sets up two separate launches (off-diag then diagonal),
     * each with a freshly uploaded kernel_arg_t; this call handles one.
     */
    return vx_spawn_threads(1, &arg->num_pages, /* grid_dim  */
                            nullptr,            /* block_dim – use HW default */
                            reinterpret_cast<vx_kernel_func_cb>(kernel_body), arg);
}
