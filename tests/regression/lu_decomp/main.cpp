#include "common.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <vortex.h>

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                                                  \
    do {                                                                                 \
        int _ret = _expr;                                                                \
        if (0 == _ret)                                                                   \
            break;                                                                       \
        printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);                         \
        cleanup();                                                                       \
        exit(-1);                                                                        \
    } while (false)

///////////////////////////////////////////////////////////////////////////////

template <typename Type> class Comparator {};

template <> class Comparator<int> {
  public:
    static const char *type_str() { return "integer"; }
    static int generate() { return rand(); }
    static bool compare(int a, int b, int index, int errors) {
        if (a != b) {
            if (errors < 100) {
                printf("*** error: [%d] expected=%d, actual=%d\n", index, b, a);
            }
            return false;
        }
        return true;
    }
};

template <> class Comparator<float> {
  public:
    static const char *type_str() { return "float"; }
    static float generate() { return static_cast<float>(rand()) / RAND_MAX; }
    static bool compare(float a, float b, int index, int errors) {
        union fi_t {
            float f;
            int32_t i;
        };
        fi_t fa, fb;
        fa.f = a;
        fb.f = b;
        auto d = std::abs(fa.i - fb.i);
        if (d > FLOAT_ULP) {
            if (errors < 100) {
                printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
            }
            return false;
        }
        return true;
    }
};

static void lu_decomp_cpu(TYPE *L, TYPE *U, const TYPE *A, uint32_t n) {
    memset(L, 0, sizeof(TYPE) * n * n);
    memset(U, 0, sizeof(TYPE) * n * n);
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t k = 0; k < n; k++) {
            int sum = 0;
            for (uint32_t j = 0; j < k; j++) {
                sum += L[i * n + j] * A[j * n + k];
            }
            U[i * n + k] = L[i * n + k] - sum;
        }
        for (uint32_t k = 0; k < n; k++) {
            if (i == k) {
                L[i * n + i] = 1;

            } else {
                int sum = 0;
                for (uint32_t j = 0; j < n; j++) {
                    sum += L[k * n + j] * U[j * n + i];
                }
                L[k * n + i] = (A[k * n + i] - sum) / U[i * n + i];
            }
        }
    }
}

const char *kernel_file = "kernel.vxbin";
uint32_t size = 32;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h L_buffer = nullptr;
vx_buffer_h U_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
    std::cout << "Vortex Test." << std::endl;
    std::cout << "Usage: [-k: kernel] [-n size] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "n:k:h")) != -1) {
        switch (c) {
        case 'n':
            size = atoi(optarg);
            break;
        case 'k':
            kernel_file = optarg;
            break;
        case 'h':
            show_usage();
            exit(0);
            break;
        default:
            show_usage();
            exit(-1);
        }
    }
}

void cleanup() {
    if (device) {
        vx_mem_free(A_buffer);
        vx_mem_free(L_buffer);
        vx_mem_free(U_buffer);
        vx_mem_free(krnl_buffer);
        vx_mem_free(args_buffer);
        vx_dev_close(device);
    }
}

int main(int argc, char *argv[]) {
    // parse command arguments
    parse_args(argc, argv);

    std::srand(50);

    // open device connection
    std::cout << "open device connection" << std::endl;
    RT_CHECK(vx_dev_open(&device));

    uint32_t size_sq = size * size;
    uint32_t buf_size = size_sq * sizeof(TYPE);

    std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
    std::cout << "matrix size: " << size << "x" << size << std::endl;

    kernel_arg.len = size;
    kernel_arg.size = size;

    // allocate device memory
    std::cout << "allocate device memory" << std::endl;
    RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &A_buffer));
    RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
    RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &L_buffer));
    RT_CHECK(vx_mem_address(L_buffer, &kernel_arg.B_addr));
    RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_WRITE, &U_buffer));
    RT_CHECK(vx_mem_address(U_buffer, &kernel_arg.C_addr));

    std::cout << "A_addr=0x" << std::hex << kernel_arg.A_addr << std::endl;
    std::cout << "B_addr=0x" << std::hex << kernel_arg.B_addr << std::endl;
    std::cout << "C_addr=0x" << std::hex << kernel_arg.C_addr << std::endl;

    // generate source data
    std::vector<TYPE> h_A(size_sq);
    std::vector<TYPE> h_L(size_sq);
    std::vector<TYPE> h_U(size_sq);
    for (uint32_t i = 0; i < size_sq; ++i) {
        h_A[i] = Comparator<TYPE>::generate();
    }

    if (0) { // upload matrix A buffer
        {
            std::cout << "upload matrix A buffer" << std::endl;
            RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, buf_size));
        }

        // Upload kernel binary
        std::cout << "Upload kernel binary" << std::endl;
        RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

        // upload kernel argument
        std::cout << "upload kernel argument" << std::endl;
        RT_CHECK(
            vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

        auto time_start = std::chrono::high_resolution_clock::now();

        // start device
        std::cout << "start device" << std::endl;
        RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

        // wait for completion
        std::cout << "wait for completion" << std::endl;
        RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

        auto time_end = std::chrono::high_resolution_clock::now();
        double elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start)
                .count();
        printf("Elapsed time: %lg ms\n", elapsed);

        // download destination buffer
        std::cout << "download destination buffer" << std::endl;
        RT_CHECK(vx_copy_from_dev(h_L.data(), L_buffer, 0, buf_size));
        RT_CHECK(vx_copy_from_dev(h_U.data(), U_buffer, 0, buf_size));
    }
    // verify result
    std::cout << "verify result" << std::endl;
    int errors = 0;
    {
        std::vector<TYPE> h_ref_L(size_sq);
        std::vector<TYPE> h_ref_U(size_sq);
        lu_decomp_cpu(h_ref_L.data(), h_ref_U.data(), h_A.data(), size);

        for (uint32_t i = 0; i < h_ref_U.size(); ++i) {
            if (!Comparator<TYPE>::compare(h_U[i], h_ref_U[i], i, errors)) {
                ++errors;
            }
        }
        for (uint32_t i = 0; i < h_ref_L.size(); ++i) {
            if (!Comparator<TYPE>::compare(h_U[i], h_ref_L[i], i, errors)) {
                ++errors;
            }
        }
    }

    // cleanup
    std::cout << "cleanup" << std::endl;
    cleanup();

    if (errors != 0) {
        std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
        std::cout << "FAILED!" << std::endl;
        return errors;
    }

    std::cout << "PASSED!" << std::endl;

    return 0;
}
