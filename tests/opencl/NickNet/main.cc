#include "CL/cl.h"
#include "Convolution.hpp"
#include "Device.hpp"
#include "Network.hpp"
#include "SparseConv.hpp"
#include "common.h"
#include "sparse_block.h"
#include "sparse_conv_cpu.hpp"
#include <assert.h>
// #include <ios>
#include <cstddef>
#include <cstdint>
#include <unistd.h>
#include <vector>
#ifndef TYPE
#define TYPE float
#endif
#define FLOAT_ULP 20000

#define KERNEL_NAME "conv1"
#define _IN 32
#define _K 5
#define _C 2
#define _OUT (_IN - _K + 1)
#define _F 8
#define _K0 5
#define _K1 5
#define _K2 5
#define _C0 6
#define _C1 6
#define _C2 6
#define _OUT0 (_IN - _K0 + 1)
#define _OUT1 (_OUT0 - _K1 + 1)
#define _OUT2 (_OUT1 - _K2 + 1)

static bool compare_equal(float a, float b) {
    union fi_t {
        float f;
        int32_t i;
    };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    return d <= FLOAT_ULP;
}

template <uint32_t L> static TYPE **gen_weights(uint32_t *W) {
    TYPE **out = (float **)std::malloc(L * sizeof(TYPE *));
    auto rand_float = []() { return (static_cast<float>(rand()) / RAND_MAX) - 0.5f; };
    for (uint32_t l = 0; l < L; l++) {
        out[l] = (float *)std::malloc(W[l] * sizeof(TYPE));
        for (uint32_t w = 0; w < W[l]; w++) {
            out[l][w] = rand_float();
        }
    }
    return out;
}

template <typename T, uint32_t IN, uint32_t K, uint32_t C, uint32_t F>
static void convolution_cpu(TYPE *I,   // [NumChannels, SeqLength]
                            TYPE *W,   // [NumFilters, NumChannels, K]
                            TYPE *B,   // [NumFilters]
                            TYPE *O) { // [NumFilters, OutLen]
    const unsigned OUT = IN - K + 1;

    for (uint32_t f = 0; f < F; f++) {
        for (uint32_t t = 0; t < OUT; t++) {
            TYPE acc = B[f];
            for (uint32_t c = 0; c < C; c++) {
                for (uint32_t k = 0; k < K; k++) {
                    acc += I[c * IN + t + k] * W[f * C * K + c * K + k];
                }
            }
            O[f * OUT + t] = acc;
        }
    }
}
template <unsigned N> static void relu_cpu(TYPE *I) {
    for (unsigned i = 0; i < N; i++) {
        I[i] = I[i] > 0 ? I[i] : 0;
    }
}
template <typename T, unsigned IN, unsigned K0, unsigned C0, unsigned K1, unsigned C1,
          unsigned K2, unsigned C2, unsigned F>
static void nick_net_cpu(T *I, T **W, T **B, T *O) {
    static const unsigned OUT0 = IN - K0 + 1;
    static const unsigned OUT1 = OUT0 - K1 + 1;
    T *O0 = (T *)std::malloc(OUT0 * C1 * sizeof(T));
    T *O1 = (T *)std::malloc(OUT1 * C2 * sizeof(T));
    convolution_cpu<T, IN, K0, C0, C1>(I, W[0], B[0], O0);
    relu_cpu<OUT0 * C1>(O0);
    convolution_cpu<T, OUT0, K1, C1, C2>(O0, W[1], B[1], O1);
    relu_cpu<OUT1 * C2>(O1);
    convolution_cpu<T, OUT1, K2, C2, F>(O1, W[2], B[2], O);
    std::free(O0);
    std::free(O1);
}
void generate_sparse_input(std::vector<TYPE> *I) {
    for (size_t i = 0; i < I->size(); i++) {
    }
}
void print_arr(TYPE arr[], int N, int C) {
    printf("{ \n");
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            printf("%.1f, ", arr[n * C + c]);
        }
        printf("\n");
    }
    printf("}\n");
}

void print_arr(TYPE arr[], int N) {
    printf("{");
    for (int n = 0; n < N; n++) {
        printf("%.1f, ", arr[n]);
    }
    printf("\b\b}\n");
}
template <typename T> void print_arr(T arr[], int N) {
    printf("{ ");
    for (int n = 0; n < N; n++) {
        printf("%d, ", arr[n]);
    }
    printf("\b\b}\n");
}
int main() {
    bool testing_gpu = false;
    const unsigned N = 64;
    uint32_t weights[1] = {_K * _F * _C};
    uint32_t bias[1] = {_F};
    TYPE h_in[N * _C], h_out[(N + _K - 1) * _F], dev_out[(N + _K - 1) * _F];

    for (unsigned n = 0; n < N; n++) {
        for (int c = 0; c < _C; c++) {
            h_in[n * _C + c] = rand() % 5 != 0 ? n * 10 + c : 0;
        }
    }
    Sparse_block *input = compress(h_in, N, _C);
    Sparse_block *output =
        (Sparse_block *)malloc(sizeof(Sparse_block) + sizeof(TYPE) * input->len * _F);
    output->C = _F;
    TYPE *h_W = gen_weights<1>(weights)[0];
    TYPE *h_B = gen_weights<1>(bias)[0];
    convolution_cpu<TYPE, N, _K, _C, _F>(h_in, h_W, h_B, h_out);
    relu_cpu<N>(h_out);

    if (testing_gpu) {
        printf("\033[33mStarting Device\033[0m\n");
        Device dev;
        printf("\033[33mInitializing Network\033[0m\n");
        SparseConvolution<TYPE, N, _K, _C, _F> L0(dev, "conv_relu");
        L0.load_weights(h_W, h_B);
        printf("\033[33mRunning Network on device\033[0m\n");
        L0.load_input(input);
        L0.run();
        L0.get_output(output);
        printf("\033[33mTesting Network\033[0m\n");
    } else {
        sconv(h_W, h_B, _K, _F, input, output);

        printf("CPU values: {");
        for (unsigned i = 0; i < (N + _K - 1); i++) {
            if (h_out[i] != 0) {
                printf("%f, ", h_out[i]);
            }
        }
        printf("\b\b}\n");
        printf("SPARSE values: {");
        for (unsigned i = 0; i < input->size; i++) {
            if (output->data[i] != 0) {
                printf("%f, ", output->data[i]);
            }
        }
        printf("\b\b}\n");
    }
    if (testing_gpu) {
        unsigned len_tmp, f_tmp;
        decompress(dev_out, len_tmp, f_tmp, output);
        int errors = 0;
        // IMPORTANT: The two printed arrays should contain the same numbers!
        //  for (uint32_t i = 0; i < (N + _K - 1); ++i) {
        //      if (!compare_equal(dev_out[i], h_out[i])) {
        //          if (errors < 100)
        //              printf("*** \033[31merror\033[0m: [%d] expected=%f, actual=%f\n",
        //              i,
        //                     h_out[i], dev_out[i]);
        //          ++errors;
        //      }
        //  }
        //  if (errors != 0) {
        //      printf("\033[31mFAILED! - %d errors\033[0m\n", errors);
        //  } else {
        //      printf("\033[32mPASSED!\033[0m\n");
        //  }
    }

    free_Sparse(input);
}
