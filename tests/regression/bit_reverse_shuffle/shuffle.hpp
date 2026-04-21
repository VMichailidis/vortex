#include "common.h"
#include <cstdio>
#include <iostream>
#include <vortex.h>
#define RT_CHECK(_expr)                                                                  \
    do {                                                                                 \
        int _ret = _expr;                                                                \
        if (0 == _ret)                                                                   \
            break;                                                                       \
        printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);                         \
        exit(-1);                                                                        \
    } while (false)

template <typename T> class vx_Shuffle {
  public:
    vx_Shuffle(const char *file) {
        // open device connection
        std::cout << "open device connection" << std::endl;
        RT_CHECK(vx_dev_open(&device));
        kernel_file = file;
        // Upload kernel binary
        std::cout << "Upload kernel binary" << std::endl;
        RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
    }
    ~vx_Shuffle() {
        if (device) {
            vx_mem_free(krnl_buffer);
            vx_mem_free(args_buffer);
            vx_dev_close(device);
            // TODO: free other bufflers
        }
    }
    template <unsigned LOG_N> void apply(T *v) {
        int buf_size = sizeof(T) * (1 << LOG_N);
        vx_buffer_h input_buffer = nullptr;
        kernel_arg_t kernel_arg;

        // allocate device memory
        std::cout << "allocate device memory" << std::endl;
        RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ_WRITE, &input_buffer));
        RT_CHECK(vx_mem_address(input_buffer, &kernel_arg.input));

        // upload source input
        std::cout << "upload source buffer" << std::endl;
        RT_CHECK(vx_copy_to_dev(input_buffer, v, 0, buf_size));
        // upload kernel argument
        std::cout << "upload kernel argument" << std::endl;
        RT_CHECK(
            vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));
        // Execute Kernel
        std::cout << "start device" << std::endl;
        RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
        // wait for completion
        std::cout << "wait for completion" << std::endl;
        RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
        // download destination buffer
        std::cout << "download destination buffer" << std::endl;
        RT_CHECK(vx_copy_from_dev(v, input_buffer, 0, buf_size));
    }

  private:
    vx_device_h device;
    const char *kernel_file;
    vx_buffer_h krnl_buffer = nullptr;
    vx_buffer_h args_buffer = nullptr;
};
