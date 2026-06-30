#ifndef _SPARSE_CONV_CPU
#define _SPARSE_CONV_CPU
#include <cstdio>
#include <iostream>
#define __IS_KERNEL
#include "sparse_block.h"

#ifndef TYPE
#define TYPE float
#endif
template <typename T> void print_arr(T *arr, unsigned len) {
    printf("{");
    for (unsigned i = 0; i < len; i++) {
        std::cout << arr[i] << ", ";
    }
    printf("\b\b}\n");
}
template <> void print_arr<unsigned char>(unsigned char *arr, unsigned len) {
    printf("{");
    for (unsigned i = 0; i < len; i++) {
        printf("%u, ", arr[i]);
    }
    printf("\b\b}\n");
}
void sconv(TYPE *W, TYPE *B, unsigned K, unsigned F, Sparse_block *src,
           Sparse_block *dst) {
    dst->len = src->len - K + 1;
    dst->C = F;
    std::cout << "Mask: ";
    print_arr(src->mask, src->len);

    std::cout << "PTX";
    print_arr(src->ptx, src->len);
    std::cout << "SAMPLES";
    print_arr(src->samples, src->len);
    std::cout << "DATA: ";
    print_arr(src->data, src->size);

    for (unsigned f = 0; f < F; f++) {
        for (unsigned t = 0; t < dst->len; t++) {
            TYPE acc = B[f];
            unsigned sample = t;
            while ((sample < dst->len) && (src->samples[sample] < src->samples[t] + K)) {
                unsigned char mask = src->mask[sample];
                while (mask) {
                    unsigned char channel = __builtin_ffs(mask);
                    unsigned c = channel - 1;
                    unsigned offset = __builtin_popcount(~(~0 << c) & src->mask[sample]);
                    unsigned k_pos = src->samples[sample] - src->samples[t];
                    acc += W[f * src->C * K + c * K + k_pos] *
                           src->data[src->ptx[sample] + offset];
                    mask = mask & (~0 << channel);
                }
                sample++;
            }
            dst->data[t * F + f] = acc > 0 ? acc : 0;
            dst->ptx[t] = t * F;
            dst->samples[t] = src->samples[t];
        }
    }
}

#endif
