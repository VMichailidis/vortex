#include "CL/cl.h"
#include "Convolution.hpp"
#include "Device.hpp"
#include "Network.hpp"
#include "ReLU.hpp"
#include "common.h"
#include "weights_loader.hpp"
#include <assert.h>
// #include <ios>
#include <cstddef>
#include <cstdint>
#include <unistd.h>
#include <vector>
#ifndef TYPE
#define TYPE float
#endif
#define FLOAT_ULP 6

#define KERNEL_NAME "conv1"
#define _IN 8206
// #define _K 5
// #define _C 2
// #define _OUT (_IN - _K + 1)
#define _F 2
#define _K0 15
#define _K1 1
#define _K2 1
#define _C0 2
#define _C1 16
#define _C2 16
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
    auto d = std::abs(fa.f - fb.f);
    return d <= 1e-5;
}

template <uint32_t L> static TYPE **gen_weights(uint32_t *W) {
    printf("Generating Weights\n");
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

int main() {
    uint32_t weight_dims[3] = {_K0 * _C1 * _C0, _K1 * _C2 * _C1, _K2 * _F * _C2};
    // TYPE **h_W = gen_weights<3>(weight_dims);
    // uint32_t bias_dims[3] = {_C1, _C2, _F};
    // TYPE **h_B = gen_weights<3>(bias_dims);
    TYPE **h_W, **h_B;
    weights_io::load_sic_cnn_weights("model", _K0, _C0, _C1, _K1, _C2, _K2, _F, &h_W,
                                     &h_B);

    std::vector<TYPE> h_i(_IN * _C0);
    TYPE *sample = weights_io::load_input_channels_first("model/X_test.bin", 0, _IN, _C0);
    std::copy(sample, sample + _IN * _C0, h_i.begin());
    std::free(sample);
    std::vector<TYPE> h_o(_OUT2 * _F, 0.0f);
    std::vector<TYPE> ref_vec(_OUT2 * _F, 0.0f);
    // Generate input values
    auto rand_float = []() { return (static_cast<float>(rand()) / RAND_MAX) - 0.5f; };
    for (int32_t i = 0; i < _IN * _C0; i++) {
        h_i[i] = rand_float();
    }
    printf("Starting device\n");
    Device dev;

    printf("Initializing network\n");
    Network<TYPE, _IN, _K0, _C0, _K1, _C1, _K2, _C2, _F> Net(dev);

    printf("Running Network on device\n");
    Net.load_weights(h_W, h_B);
    Net.load_input(h_i.data());
    Net.run();
    Net.get_output(h_o.data());

    printf("Running network simulation\n");
    nick_net_cpu<TYPE, _IN, _K0, _C0, _K1, _C1, _K2, _C2, _F>(h_i.data(), h_W, h_B,
                                                              ref_vec.data());

    int errors = 0;
    int correct = 0;
    for (uint32_t i = 0; i < _OUT2 * _F; ++i) {
        if (!compare_equal(h_o[i], ref_vec[i])) {
            if (errors < 100)
                printf("*** error: [%d] expected=%f, actual=%f\n", i, ref_vec[i], h_o[i]);
            ++errors;
        } else {
            correct++;
        }
    }
    if (errors != 0) {
        printf("\033[31mFAILED! - %d errors, %d correct\033[0m\n", errors, correct);
    } else {
        printf("\033[32mPASSED!\033[0m\n");
    }
    dev.print_info();
}
