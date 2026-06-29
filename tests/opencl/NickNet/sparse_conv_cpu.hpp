#ifndef _SPARSE_CONV_CPU
#define _SPARSE_CONV_CPU
#define __IS_KERNEL
#include "sparse_block.h"

#ifndef TYPE
#define TYPE float
#endif
void sconv(TYPE *W, TYPE *B, unsigned K, unsigned F, Sparse_block *src,
           Sparse_block *dst) {
    dst->len = src->len - K + 1;
    dst->C = F;
    for (unsigned i = 0; i < dst->len; i++) {
        unsigned sample = i;
        for (unsigned f = 0; f < F; f++) {
            TYPE acc = B[f];
            while (sample < dst->len && src->samples[sample] < i + K) {
                unsigned char mask = src->mask[sample];
                unsigned char channel = __builtin_ffs(mask);
                while (mask) {
                    unsigned c = channel - 1;
                    unsigned offset = __builtin_popcount(~(~0 << c)) & src->mask[sample];
                    unsigned k_pos = src->samples[sample] - i;
                    acc += W[f * src->C * K + c * K + k_pos] *
                           src->data[src->ptx[sample] + offset];
                    mask >>= channel;
                    channel = __builtin_ffs(mask);
                }
                sample++;
            }
            dst->data[i * F + f] = acc > 0 ? acc : 0;
        }
        dst->ptx[i] = i * F;
        dst->samples[i] = src->samples[i];
    }
}

#endif
