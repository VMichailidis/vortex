#include "CL/cl.h"
#include "Convolution.hpp"
#include "Device.hpp"
#include <cstdio>
template <typename T> void print_arr(T *arr, int N) {
    printf("{ ");
    for (int n = 0; n < N; n++) {
        printf("%f, ", arr[n]);
    }
    printf("\b\b}\n");
}
template <typename T> T *generate_sparse_input(unsigned len, unsigned channels) {
    T *input = (T *)malloc(channels * len * sizeof(T));
    for (unsigned c = 0; c < channels; c++)
        for (unsigned s = 0; s < len; s++) {
            input[c * len + s] = rand() % 2 != 0 ? (c + 1) * 100 + s : 0;
        }
    return input;
}
template <uint32_t L> static TYPE **gen_weights(uint32_t *W) {
    TYPE **out = (float **)std::malloc(L * sizeof(TYPE *));
    auto rand_float = []() { return (static_cast<float>(rand()) / RAND_MAX) - 0.5f; };
    for (uint32_t l = 0; l < L; l++) {
        out[l] = (float *)std::malloc(W[l] * sizeof(TYPE));
        for (uint32_t w = 0; w < W[l]; w++) {
            out[l][w] = 1;
        }
    }
    return out;
}
int main() {
    TYPE *input = generate_sparse_input<TYPE>(IN_, C_);
    printf("\033[33mStarting Device\033[0m\n");
    Device dev;
    printf("\033[33mInitializing Sparse Layer\033[0m\n");
    Convolution<TYPE, IN_, K_, C_, F_> S(dev, "conv_relu");
    SparseBuffer *b = S.compress_cpu(input, IN_, C_);
    print_arr(input, IN_ * C_);

    printf("Compressed array\n");
    print_sparse(b, true);

    TYPE *decompressed = S.decompress_cpu(b);
    print_arr(decompressed, IN_ * C_);
    for (unsigned i = 0; i < IN_ * C_; i++)
        if (decompressed[i] != input[i])
            printf("\033[31mError!\033[0m %f should be %f\n", decompressed[i], input[i]);

    free(input);
    free(b);
    free(decompressed);
}
