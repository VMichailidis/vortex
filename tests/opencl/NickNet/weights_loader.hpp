// weights_loader.hpp
//
// Loads the raw float32 weight files exported from the TensorFlow SavedModel
// (see ARCHITECTURE.md) and reshapes them into the layout expected by this
// project's convolution_cpu / Network / OpenCL kernel:
//
//   this codebase expects   -- weight: [F, C, K]  (k fastest)   -- from convolution_cpu:
//                                                                   W[f*C*K + c*K + k]
//                              input : [C, IN]     (t fastest)  -- from convolution_cpu:
//                                                                   I[c*IN + t]
//
//   the exported .bin files store TensorFlow's native layout instead:
//   weight: [K, C, F]  (f fastest)
//   input : [IN, C]    (c fastest)   (channels-last, TF's Conv1D convention)
//
// So both the kernel and the input need an axis transpose on the way in --
// this header does that.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#ifndef TYPE
#define TYPE float
#endif

namespace weights_io {

// Reads `count` TYPE elements starting at element offset `elem_offset` from a
// raw float32 file into a newly malloc'd buffer. Caller owns the buffer
// (free with std::free).
inline TYPE *read_raw_f32(const std::string &path, size_t count, size_t elem_offset = 0) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("could not open: " + path);
    }
    if (elem_offset != 0) {
        if (fseek(f, (long)(elem_offset * sizeof(TYPE)), SEEK_SET) != 0) {
            fclose(f);
            throw std::runtime_error("seek failed: " + path);
        }
    }
    TYPE *buf = (TYPE *)std::malloc(count * sizeof(TYPE));
    size_t n = fread(buf, sizeof(TYPE), count, f);
    fclose(f);
    if (n != count) {
        std::free(buf);
        throw std::runtime_error(path + ": expected " + std::to_string(count) +
                                 " floats, read " + std::to_string(n));
    }
    return buf;
}

// Converts a Conv1D kernel from TensorFlow's on-disk layout (K, C, F)
// row-major (F fastest) into this project's expected layout (F, C, K)
// row-major (K fastest).
inline TYPE *transpose_kcf_to_fck(const TYPE *src, uint32_t K, uint32_t C, uint32_t F) {
    TYPE *dst = (TYPE *)std::malloc((size_t)K * C * F * sizeof(TYPE));
    for (uint32_t k = 0; k < K; k++) {
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t f = 0; f < F; f++) {
                size_t src_idx = (size_t)k * C * F + (size_t)c * F + f;
                size_t dst_idx = (size_t)f * C * K + (size_t)c * K + k;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
    return dst;
}

// Loads one Conv1D layer's kernel + bias from
// "<weights_dir>/<base_name>_kernel.bin" / "..._bias.bin", already reshaped
// into the (F, C, K) layout convolution_cpu expects. Bias needs no reshape
// (it's just F values either way).
inline void load_layer(const std::string &weights_dir, const std::string &base_name,
                       uint32_t K, uint32_t C, uint32_t F, TYPE **out_weight,
                       TYPE **out_bias) {
    TYPE *raw_kernel =
        read_raw_f32(weights_dir + "/" + base_name + "_kernel.bin", (size_t)K * C * F);
    *out_weight = transpose_kcf_to_fck(raw_kernel, K, C, F);
    std::free(raw_kernel);

    *out_bias = read_raw_f32(weights_dir + "/" + base_name + "_bias.bin", F);
}

// Loads all 3 layers and returns them as TYPE*[3] arrays, matching the
// h_W / h_B shape your main() already builds with gen_weights(). Layer
// order is conv1d_1 -> conv1d_2 -> output, i.e. W[0]/B[0], W[1]/B[1],
// W[2]/B[2]. Free with free_sic_cnn_weights().
inline void load_sic_cnn_weights(const std::string &weights_dir, uint32_t K0, uint32_t C0,
                                 uint32_t C1, uint32_t K1, uint32_t C2, uint32_t K2,
                                 uint32_t F, TYPE ***out_W, TYPE ***out_B) {
    TYPE **W = (TYPE **)std::malloc(3 * sizeof(TYPE *));
    TYPE **B = (TYPE **)std::malloc(3 * sizeof(TYPE *));

    load_layer(weights_dir, "conv1d_1", K0, C0, C1, &W[0], &B[0]);
    load_layer(weights_dir, "conv1d_2", K1, C1, C2, &W[1], &B[1]);
    load_layer(weights_dir, "output", K2, C2, F, &W[2], &B[2]);

    *out_W = W;
    *out_B = B;
}

inline void free_sic_cnn_weights(TYPE **W, TYPE **B) {
    for (int i = 0; i < 3; i++) {
        std::free(W[i]);
        std::free(B[i]);
    }
    std::free(W);
    std::free(B);
}

// Loads ONE sample (non-batched) from a batched, channel-last .bin file
// (e.g. data/X_test.bin, shape [batch, seq_len, channels]) and transposes it
// into the channel-first [channels, seq_len] layout convolution_cpu expects.
// `sample_idx` selects which of the `batch` sequences to load.
inline TYPE *load_input(const std::string &path, uint32_t samples, uint32_t seq_len,
                        uint32_t channels) {
    TYPE *raw = read_raw_f32(path, (size_t)seq_len * channels *
                                       samples); // [BATCH, IN, C], c fastest

    return raw;
}

} // namespace weights_io
