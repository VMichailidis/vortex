#include "./bra.hpp"
#include "common.h"
#include <vortex.h>
#define RT_CHECK(_expr)                                                                  \
    do {                                                                                 \
        int _ret = _expr;                                                                \
        if (0 == _ret)                                                                   \
            break;                                                                       \
        printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);                         \
        cleanup();                                                                       \
        exit(-1);                                                                        \
    } while (false)

template <unsigned char LOG_N> bool is_reversed(int a, int index) {
    return reverse<LOG_N>(index) == a;
}

template <unsigned char LOG_N> void generate(int *test_array) {
    for (int i = 0; i < 1 << LOG_N; i++) {
        test_array[i] = i;
    }
}
#define LOG_N_TEST 4
int buf_size = sizeof(int) * (1 << LOG_N_TEST);
vx_buffer_h input_buffer = nullptr;
vx_device_h device = nullptr;
kernel_arg_t kernel_arg = {};
const char *kernel_file = "kernel.vxbin";
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
void cleanup() {
    if (device) {
        vx_mem_free(krnl_buffer);
        vx_mem_free(args_buffer);
        vx_dev_close(device);
    }
}

void shuffle(int *v) {
    // open device connection
    std::cout << "open device connection" << std::endl;
    RT_CHECK(vx_dev_open(&device));

    // allocate device memory
    std::cout << "allocate device memory" << std::endl;
    RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ_WRITE, &input_buffer));
    RT_CHECK(vx_mem_address(input_buffer, &kernel_arg.input));

    // upload source input
    std::cout << "upload source buffer0" << std::endl;
    RT_CHECK(vx_copy_to_dev(input_buffer, v, 0, buf_size));

    // Upload kernel binary
    std::cout << "Upload kernel binary" << std::endl;
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

    // upload kernel argument
    std::cout << "upload kernel argument" << std::endl;
    RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

    std::cout << "start device" << std::endl;
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

    // wait for completion
    std::cout << "wait for completion" << std::endl;
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    // download destination buffer
    std::cout << "download destination buffer" << std::endl;
    RT_CHECK(vx_copy_from_dev(v, input_buffer, 0, buf_size));
}

int main() {
    int test_array[1 << LOG_N_TEST];
    generate<LOG_N_TEST>(test_array);
    int errors = 0;
    // Shuffle<int, LOG_N_TEST>::apply(test_array);
    shuffle(test_array);
    // for (int i = 0; i < 1 << LOG_N_TEST; i++) {
    //     if (!is_reversed<LOG_N_TEST>(test_array[i], i) && (errors < 100)) {
    //         printf("error at %d: expected %d, got %d\n", i, reverse<LOG_N_TEST>(i),
    //                test_array[i]);
    //     }
    // }
    // if (errors == 0) {
    //     printf("PASSED!\n");
    // }

    printf("Num Threads: %d,\n Num Warps: %d,\n Num Cores: %d\n", test_array[0],
           test_array[1], test_array[2]);
    return errors;
}
