#include "common.h"
#include <vx_spawn.h>

template <typename T, unsigned char LOG_N, unsigned char LOG_BLOCK_WIDTH>
void mem2buffer(unsigned int page, const T *__restrict__ src, T *__restrict__ dst) {
    constexpr unsigned BLOCK_SIZE = 1u << LOG_BLOCK_WIDTH;
    for (unsigned a = threadIdx.x; a < BLOCK_SIZE; a += blockDim.x) {
        unsigned src_base = (a << (LOG_N - LOG_BLOCK_WIDTH)) | (page << LOG_BLOCK_WIDTH);
        unsigned dst_base = a << LOG_BLOCK_WIDTH;
        for (unsigned i = 0; i < BLOCK_SIZE; i++) {
            dst[dst_base + i] = src[src_base + i];
        }
    }
}
template <typename T, unsigned char LOG_N, unsigned char LOG_BLOCK_WIDTH>
void buffer2mem(unsigned int page, const T *__restrict__ src, T *__restrict__ dst) {
    constexpr unsigned BLOCK_SIZE = 1u << LOG_BLOCK_WIDTH;
    for (unsigned a = threadIdx.x; a < BLOCK_SIZE; a += blockDim.x) {
        unsigned dst_base = (a << (LOG_N - LOG_BLOCK_WIDTH)) | (page << LOG_BLOCK_WIDTH);
        unsigned src_base = a << LOG_BLOCK_WIDTH;
        for (unsigned i = 0; i < BLOCK_SIZE; i++) {
            dst[dst_base + i] = src[src_base + i];
        }
    }
}

template <typename T, unsigned char LOG_N, unsigned char LOG_BLOCK_WIDTH>
void shuffle_batch(kernel_arg_t *__UNIFORM__ arg) {
    auto input = reinterpret_cast<int *>(arg->input);
    auto page = reinterpret_cast<const unsigned *>(arg->page);
    auto scratch_pool = reinterpret_cast<int *>(arg->scratch_pool);
    int *buf = scratch_pool + (unsigned long)blockIdx.x * BUF_ELEMS;
    unsigned src_page = pages[blockIdx.x];
    unsigned not_is_odd = blockIdx.x % 2 ? 0 : 1;
    unsigned dst_page = pages[((blockIdx.x / 2) * 2) + not_is_odd];
    mem2buffer<T, LOG_N, LOG_BLOCK_WIDTH>(src_page, input, buf);
    __syncthreads();

    libra<T, 2 * LOG_BLOCK_WIDTH>(buf);
    __syncthreads();
    buffer2mem<T, LOG_N, LOG_BLOCK_WIDTH>(dst_page, buf, input);

    dst_ptr[blockIdx.x] = src0_ptr[blockIdx.x] + src1_ptr[blockIdx.x];
}

int main() {
    kernel_arg_t *arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->num_points, nullptr, (vx_kernel_func_cb)kernel_body,
                            arg);
}
