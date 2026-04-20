#include "common.h"
#include "vx_intrinsics.h"
#include <cstdio>
#include <vx_spawn.h>
#define LOG_BLOCK_WIDTH_TEST 5

template <typename T, unsigned char LOG_BLOCK_WIDTH>
void mem2buffer(unsigned int page, unsigned int log_n, const T *__restrict__ src,
                T *__restrict__ dst) {
    constexpr unsigned BLOCK_SIZE = 1u << LOG_BLOCK_WIDTH;
    for (unsigned a = threadIdx.x; a < BLOCK_SIZE; a += blockDim.x) {
        unsigned src_base = (a << (log_n - LOG_BLOCK_WIDTH)) | (page << LOG_BLOCK_WIDTH);
        unsigned dst_base = a << LOG_BLOCK_WIDTH;
        for (unsigned i = 0; i < BLOCK_SIZE; i++) {
            dst[dst_base + i] = src[src_base + i];
        }
    }
}
template <typename T, unsigned char LOG_BLOCK_WIDTH>
void buffer2mem(unsigned int page, unsigned int log_n, const T *__restrict__ src,
                T *__restrict__ dst) {
    constexpr unsigned BLOCK_SIZE = 1u << LOG_BLOCK_WIDTH;
    for (unsigned a = threadIdx.x; a < BLOCK_SIZE; a += blockDim.x) {
        unsigned dst_base = (a << (log_n - LOG_BLOCK_WIDTH)) | (page << LOG_BLOCK_WIDTH);
        unsigned src_base = a << LOG_BLOCK_WIDTH;
        for (unsigned i = 0; i < BLOCK_SIZE; i++) {
            dst[dst_base + i] = src[src_base + i];
        }
    }
}

template <typename T, unsigned int LOG_SIZE> void shuffle(T *signal) {
    int lim = ((1ul << LOG_SIZE) - (1ul << (LOG_SIZE / 2))) / 2;
    for (int id = threadIdx.x; id < lim; id += blockDim.x) {
        // calculate index within the table
        int offset = 0;
        int col = 1;
        int row = 0;
        for (int i = 1; i < (1 << (LOG_SIZE / 2)); i++) {
            int ri = reverse<LOG_SIZE / 2>(i);
            if (id - offset >= 0) {
                col = i;
                row = id - offset;
            }
            offset += ri;
        }
        int up = row;
        int pu = reverse<LOG_SIZE / 2>(row);
        int esab = reverse<LOG_SIZE / 2>(col);
        if (LOG_SIZE % 2) {
            {
                int index = (up << ((LOG_SIZE / 2) + 1)) | col;
                int xendi = (esab << ((LOG_SIZE / 2) + 1)) | pu;
                // Swap
                T tmp = signal[index];
                signal[index] = signal[xendi];
                signal[xendi] = tmp;
            }
            {
                int index = (up << ((LOG_SIZE / 2) + 1)) | 1 << (LOG_SIZE / 2) | col;
                int xendi = (esab << ((LOG_SIZE / 2) + 1)) | 1 << (LOG_SIZE / 2) | pu;
                // Swap
                T tmp = signal[index];
                signal[index] = signal[xendi];
                signal[xendi] = tmp;
            }
        } else {
            int index = (up << (LOG_SIZE / 2)) | col;
            int xendi = (esab << (LOG_SIZE / 2)) | pu;
            // Swap
            T tmp = signal[index];
            signal[index] = signal[xendi];
            signal[xendi] = tmp;
        }
    }
}

template <typename T, unsigned char LOG_BLOCK_WIDTH>
void shuffle_batch(kernel_arg_t *__UNIFORM__ arg) {
    constexpr unsigned BUF_ELEMS = (1 << LOG_BLOCK_WIDTH) * (1 << LOG_BLOCK_WIDTH);

    auto input = reinterpret_cast<int *>(arg->input);
    auto pages = reinterpret_cast<const unsigned *>(arg->pages);
    auto scratch_pool = reinterpret_cast<int *>(arg->scratch_pool);
    auto log_n = reinterpret_cast<unsigned int *>(arg->log_n);

    int *buf = scratch_pool + (unsigned long)blockIdx.x * BUF_ELEMS;
    unsigned src_page = pages[blockIdx.x];
    unsigned not_is_odd = blockIdx.x % 2 ? 0 : 1;
    unsigned dst_page = pages[((blockIdx.x / 2) * 2) + not_is_odd];

    mem2buffer<int, LOG_BLOCK_WIDTH>(src_page, *log_n, input, buf);
    __syncthreads();
    shuffle<T, 2 * LOG_BLOCK_WIDTH>(buf);
    __syncthreads();
    buffer2mem<int, LOG_BLOCK_WIDTH>(dst_page, *log_n, buf, input);
}

int main() {
    kernel_arg_t *arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
    // return vx_spawn_threads(1, &arg->threads, nullptr,
    //                         (vx_kernel_func_cb)shuffle_batch<int,
    //                         LOG_BLOCK_WIDTH_TEST>, arg);
    auto input = reinterpret_cast<int *>(arg->input);
    input[0] = vx_num_threads();
    input[1] = vx_num_warps();
    input[2] = vx_num_cores();

    // printf("Num Threads: %d,\n Num Warps: %d,\n Num Cores: %d\n", vx_num_threads(),
    // vx_num_warps(), vx_num_cores());
}
