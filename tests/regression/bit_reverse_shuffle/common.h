#ifndef _COMMON_H_
#define _COMMON_H_

#include <cstdint>

/*
 * Compile-time parameters (can be overridden via -D flags):
 *
 *   LOG_N            – log2 of the total number of elements (e.g. 20 → 1M)
 *   LOG_BLOCK_WIDTH  – log2 of the tile edge length; must satisfy
 *                      2*LOG_BLOCK_WIDTH < 18 - log2(sizeof(TYPE))
 *                      so each scratch tile fits in L1 cache.
 *   TYPE             – element type (default: float)
 */
#ifndef TYPE
#define TYPE float
#endif

#ifndef LOG_N
#define LOG_N 20
#endif

#ifndef LOG_BLOCK_WIDTH
#define LOG_BLOCK_WIDTH 4 /* tile edge = 16, tile area = 256 */
#endif

/* Derived constants – mirroring the CUDA constexpr statics */
#define BLOCK_SIZE (1u << LOG_BLOCK_WIDTH)
#define BUF_ELEMS (BLOCK_SIZE * BLOCK_SIZE) /* elements per scratch tile */
#define NUM_B_BITS (LOG_N - 2 * LOG_BLOCK_WIDTH)
#define B_SIZE (1u << NUM_B_BITS)
#define N_ELEMS (1ul << LOG_N)

/*
 * Kernel argument block.
 *
 * input_addr   – device address of the data array  (N_ELEMS × sizeof(TYPE))
 * pages_addr   – device address of the page-index array (num_pages × uint32)
 * scratch_addr – device address of the scratch pool  (num_blocks × BUF_ELEMS ×
 * sizeof(TYPE)) num_pages    – length of the pages array (== gridDim for one launch)
 * is_diag      – 0 → off-diagonal pairs (libra_batch)
 *                1 → diagonal pages    (libra_batch_diag)
 */
typedef struct {
    uint64_t input_addr;
    uint64_t pages_addr;
    uint64_t scratch_addr;
    uint32_t num_pages;
    uint32_t is_diag;
} kernel_arg_t;

#endif /* _COMMON_H_ */
