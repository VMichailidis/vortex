#ifndef _SPARSE_UTILS_H
#define _SPARSE_UTILS_H
#include <cstdio>
#include <cstdlib>
#ifndef TYPE
#define TYPE float
#endif
// The sparse buffer represents an input array I[C][N] as a packed array in channel major
// order
//
// Terminology
//
// Sample: data of all channels some timestep
//
// Element: data at a specific channel at a specific timestep
//
typedef struct {
    unsigned C;            // number of channels
    unsigned N;            // number of elements in the original array
    unsigned num_samples;  // number of non-zero samples, length of sample, sample_offsets
                           // and bitmask
    unsigned num_elements; // number of non-zero elements

    unsigned *sample; // the index of the n-th non zero sample in the original array
    unsigned *sample_offset; // the offset of the n-th non-zero sample
    unsigned
        *bitmask; // mask determining the channel occupancy of the n-th non-zero sample
    TYPE *data;
} SparseBuffer;

static unsigned int round_up(unsigned int n, unsigned int multiple) {
    return ((n + multiple - 1) / multiple) * multiple;
}

static void print_sparse(SparseBuffer *b, bool sample = false, bool sample_offset = false,
                         bool bitmask = false, bool data = false) {
    printf("Num Channels: %u\n", b->C);
    printf("Original Num samples %u\n", b->N);
    printf("Number of non-zero samples: %u\n", b->num_samples);
    printf("Number of non-zero elements: %u\n", b->num_elements);

    if (sample) {

        printf("Index of non zero samples:\n");
        printf("{");
        for (unsigned i = 0; i < b->num_samples; i++) {
            printf("%u, ", b->sample[i]);
        }
        printf("\b\b}\n");
    }
    if (sample_offset) {

        printf("offsets of non zero samples in data array:\n");
        printf("{");
        for (unsigned i = 0; i < b->num_samples; i++) {
            printf("%u, ", b->sample_offset[i]);
        }
        printf("\b\b}\n");
    }

    if (bitmask) {

        printf("bitmask of each sample:\n");
        printf("{");
        for (unsigned i = 0; i < b->num_samples; i++) {
            printf("%x, ", b->bitmask[i]);
        }
        printf("\b\b}\n");
    }
    if (data) {

        printf("Raw packed data:\n");
        printf("{");
        for (unsigned i = 0; i < b->num_elements; i++) {
            printf("%f, ", b->data[i]);
        }
        printf("\b\b}\n");
    }
}

static void free_sparse(SparseBuffer *b) {
    free(b->bitmask);
    free(b->data);
    free(b->sample_offset);
    free(b->sample);
    free(b);
}
#endif
