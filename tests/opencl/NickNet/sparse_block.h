#ifndef _SPARSE_ARRAY
#define _SPARSE_ARRAY
#include <bits/floatn-common.h>
#ifndef TYPE
#define TYPE float
#endif

typedef struct Sparse_block {
    // Number of samples in uncompressed array
    unsigned N;
    // Number of channels
    unsigned C;
    // Length of idx/mask/ptx arrays.
    unsigned char samples;
    // Length of val
    unsigned short size;

    struct Sparse_block *next;
    struct Sparse_block *prev;
    unsigned first;
    unsigned last;

    TYPE val[3 * 256];
    // array of the equivalent sparse index of the n-th element of val
    unsigned idx[256];
    // array of masks indicating the occupancy of each channel of element val[ptx[i]]
    // Example: [ch0,ch2] -> 101, [ch1,ch2] -> 110
    unsigned char mask[256];
    // array of pointers to val
    unsigned ptx[256];
} Sparse_block;

#ifndef __IS_KERNEL
#include <cstdio>
#include <cstdlib>
// TYPE get(unsigned n, unsigned c, Sparse_block *block) {
//     while (n > block->last) {
//         block = block->next;
//     }
//     int i = 0;
//     unsigned n_p;
//     for (n_p = block->idx[i]; n_p <= n; n_p = block->idx[i++])
//         ;
//     if (n_p == n && (block->mask[i] & 1 << c)) {
//         unsigned offset = __builtin_popcount(
//             ~(~0 << c) & block->mask[i]); // find the position of channel c in val

//         return block->val[n_p + offset];
//     } else {
//         return NAN;
//     }
// }

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
    Sparse_block *sarr = (Sparse_block *)malloc(sizeof(Sparse_block));
    sarr->N = N;
    sarr->C = C;
    cnz(sarr->size, sarr->samples, arr, N, C);
    unsigned j = 0;
    unsigned i = 0;
    for (unsigned n = 0; n < N; n++) {
        unsigned mask = 0;
        bool occupied = false;
        for (unsigned c = 0; c < C; c++) {
            if (arr[n * C + c] != 0) {
                occupied = true;
                sarr->val[j] = arr[n * C + c];
                j++;
                mask |= 1 << c;
            }
        }
        if (occupied) {
            sarr->idx[i] = n;
            sarr->mask[i] = mask;
            i++;
        }
    }
    return sarr;
}

void accumulate_indexes(Sparse_block *sarr) {
    unsigned acc = 0;
    sarr->ptx[0] = 0;
    for (unsigned n = 0; n < sarr->samples - 1; n++) {
        acc += __builtin_popcount(sarr->mask[n]);
        sarr->ptx[n + 1] = acc;
    }
}

void decompress(TYPE *arr, unsigned &N, unsigned &C, Sparse_block *sarr) {
    N = sarr->N;
    C = sarr->C;
    unsigned ptx = 0;
    unsigned idx = 0;
    for (unsigned n = 0; n < sarr->N; n++) {
        for (unsigned c = 0; c < C; c++) {
            bool occupied = sarr->mask[idx] & 1 << c;
            unsigned offset = __builtin_popcount(
                ~(~0 << c) & sarr->mask[idx]); // find the position of channel c in val
            if (occupied && n == sarr->idx[idx]) {
                arr[n * C + c] = sarr->val[ptx + offset];
            } else {
                arr[n * C + c] = 0.0f;
            }
        }
        if (n == sarr->idx[idx]) {
            ptx += __builtin_popcount(sarr->mask[idx]);
            idx += 1;
        }
    }
}

#endif
#endif
