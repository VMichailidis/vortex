#ifndef _SPARSE_BLOCK
#define _SPARSE_BLOCK
#include <bits/floatn-common.h>
#include <cstdint>
#ifndef TYPE
#define TYPE float
#endif
#ifndef SPARSE_LEN
#define SPARSE_LEN 64
#endif

typedef struct Sparse_block {
    unsigned C;
    unsigned char len;
    unsigned short size;
    unsigned samples[SPARSE_LEN];
    unsigned char mask[SPARSE_LEN];
    unsigned ptx[SPARSE_LEN];
    TYPE data[]; // Dynamically allocate val buffer based on channel count and samples
} Sparse_block;

#ifndef __IS_KERNEL
#include <cstdio>
#include <cstdlib>

void free_Sparse(Sparse_block *arr) { free(arr); }

void cnz(unsigned short &size, unsigned char &samples, TYPE *arr, unsigned N,
         unsigned C) {
    size = 0;
    samples = 0;
    for (unsigned n = 0; n < N; n++) {
        bool occupied = false;
        for (unsigned c = 0; c < C; c++) {
            if (arr[n * C + c] != 0) {
                size++;
                occupied = true;
            }
        }
        samples = occupied ? samples + 1 : samples;
    }
}

Sparse_block *compress(TYPE *arr, unsigned N, unsigned C) {
    unsigned char samples;
    unsigned short size;
    cnz(size, samples, arr, N, C);
    Sparse_block *block =
        (Sparse_block *)malloc(sizeof(Sparse_block) + size * sizeof(TYPE));
    block->C = C;
    block->len = samples;
    block->size = size;
    unsigned ptr = 0;
    unsigned j = 0;
    unsigned i = 0;
    for (unsigned n = 0; n < N; n++) {
        unsigned mask = 0;
        bool occupied = false;
        for (unsigned c = 0; c < C; c++) {
            if (arr[c * N + n] != 0) {
                occupied = true;
                block->data[j] = arr[c * N + n];
                j++;
                mask |= 1 << c;
            }
        }
        if (occupied) {
            block->samples[i] = n;
            block->mask[i] = mask;
            block->ptx[i] = ptr;
            ptr += __builtin_popcount(mask);
            i++;
        }
    }
    return block;
}

void accumulate_indexes(Sparse_block *sarr) {
    unsigned acc = 0;
    sarr->ptx[0] = 0;
    for (unsigned n = 0; n < sarr->len - 1; n++) {
        acc += __builtin_popcount(sarr->mask[n]);
        sarr->ptx[n + 1] = acc;
    }
}

void decompress(TYPE *arr, unsigned N, unsigned C, Sparse_block *sarr) {
    N = sarr->samples[sarr->len - 1];
    C = sarr->C;
    for (unsigned n = 0; n < N; n++)
        for (unsigned c = 0; c < C; c++)
            arr[c * N + n] = 0;

    for (unsigned s = 0; s < sarr->len; s++) {
        unsigned mask = sarr->mask[s];
        unsigned ptx = sarr->ptx[s];
        unsigned sample = sarr->samples[s];
        while (mask) {
            unsigned channel = __builtin_ffs(mask);
            unsigned c = channel - 1;
            unsigned offset = __builtin_popcount(~(~0 << c) & sarr->mask[s]);
            arr[c * N + sample] = sarr->data[ptx + offset];
            mask &= ~0 << channel;
        }
    }
    // for (unsigned n = 0; n < N; n++) {
    //     for (unsigned c = 0; c < C; c++) {
    //         bool occupied = sarr->mask[idx] & 1 << c;
    //         unsigned offset = __builtin_popcount(
    //             ~(~0 << c) & sarr->mask[idx]); // find the position of channel c in val
    //         if (occupied && n == sarr->samples[idx]) {
    //             arr[c * N + n] = sarr->data[ptx + offset];
    //         } else {
    //             arr[c * N + n] = 0.0f;
    //         }
    //     }
    //     if (n == sarr->samples[idx]) {
    //         ptx += __builtin_popcount(sarr->mask[idx]);
    //         idx += 1;
    //     }
}

#endif
#endif
