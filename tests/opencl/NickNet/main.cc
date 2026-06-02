#include "Convolution.hpp"
#include "Device.hpp"
#include "common.h"
#include <assert.h>
// #include <ios>
#include <unistd.h>
#include <vector>
#ifndef TYPE
#define TYPE float
#endif
#define FLOAT_ULP 6

#define KERNEL_NAME "conv1"
#define IN 128
#define K 5
#define F 8
#define C 2
#define OUT IN - K + 1

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

static void convolution_cpu(TYPE *I,   // [NumChannels, SeqLength]
                            TYPE *W,   // [NumFilters, NumChannels, K]
                            TYPE *B,   // [NumFilters]
                            TYPE *O) { // [NumFilters, OutLen]

    for (int f = 0; f < F; f++) {
        for (int t = 0; t < OUT; t++) {
            TYPE acc = B[f];
            for (int c = 0; c < C; c++) {
                for (int k = 0; k < K; k++) {
                    acc += I[c * IN + t + k] * W[f * C * K + c * K + k];
                }
            }
            O[f * OUT + t] = acc;
        }
    }
}

int main() {

    std::vector<TYPE> h_i(IN * C);
    std::vector<TYPE> h_w(IN * C * F);
    std::vector<TYPE> h_b(F);
    std::vector<TYPE> h_o(OUT * F, 0.0f);
    std::vector<TYPE> ref_vec(OUT * F, 0.0f);
    // Generate input values
    for (int32_t i = 0; i < IN + 1; i++) {
        h_i[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (uint32_t i = 0; i < IN * C * F; i++) {
        h_w[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (uint32_t i = 0; i < F; i++) {
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    Device dev;

    Convolution<TYPE, IN, K, C, F> conv(dev);
    conv.load_weights(h_w, h_b);
    conv.load_input(h_i);
    conv.run();
    conv.get_output(h_o);
    convolution_cpu(h_i.data(), h_w.data(), h_b.data(), ref_vec.data());
    int errors = 0;
    for (uint32_t i = 0; i < OUT * F; ++i) {
        if (!compare_equal(h_o[i], ref_vec[i])) {
            if (errors < 100)
                printf("*** error: [%d] expected=%f, actual=%f\n", i, ref_vec[i], h_o[i]);
            ++errors;
        }
    }
    if (errors != 0) {
        printf("FAILED! - %d errors\n", errors);
    } else {
        printf("PASSED!\n");
    }
}
