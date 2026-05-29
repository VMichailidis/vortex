#include "common.h"
#include <CL/opencl.h>
#include <assert.h>
#include <chrono>
#include <ios>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#define FLOAT_ULP 6

#define KERNEL_NAME "conv1"

int SeqLength = 32;
int K = 5;
int Stride = 1;
int NumChannels = 2;
int NumFilters = 8;
int OutLen = (SeqLength - K) / Stride + 1;

#define CL_CHECK(_expr)                                                                  \
    do {                                                                                 \
        cl_int _err = _expr;                                                             \
        if (_err == CL_SUCCESS)                                                          \
            break;                                                                       \
        printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);                  \
        cleanup();                                                                       \
        exit(-1);                                                                        \
    } while (0)

#define CL_CHECK2(_expr)                                                                 \
    ({                                                                                   \
        cl_int _err = CL_INVALID_VALUE;                                                  \
        decltype(_expr) _ret = _expr;                                                    \
        if (_err != CL_SUCCESS) {                                                        \
            printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);              \
            cleanup();                                                                   \
            exit(-1);                                                                    \
        }                                                                                \
        _ret;                                                                            \
    })

static int read_kernel_file(const char *filename, uint8_t **data, size_t *size) {
    if (nullptr == filename || nullptr == data || 0 == size)
        return -1;

    FILE *fp = fopen(filename, "r");
    if (NULL == fp) {
        fprintf(stderr, "Failed to load kernel.");
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    rewind(fp);

    *data = (uint8_t *)malloc(fsize);
    *size = fread(*data, 1, fsize, fp);

    fclose(fp);

    return 0;
}

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

static void convolution_cpu(TYPE *I, // [NumChannels, SeqLength]
                            TYPE *W, // [NumFilters, NumChannels, K]
                            TYPE *B, // [NumFilters]
                            TYPE *O, // [NumFilters, OutLen]

                            const int SeqLength, const int K, const int Stride,
                            const int NumChannels, const int NumFilters) {

    for (int f = 0; f < NumFilters; f++) {
        for (int t = 0; t < OutLen; t++) {
            TYPE acc = B[f];
            for (int c = 0; c < NumChannels; c++) {
                for (int k = 0; k < K; k++) {
                    acc += I[c * SeqLength + t * Stride + k] *
                           W[f * NumChannels * K + c * K + k];
                }
            }
            O[f * OutLen + t] = acc;
        }
    }
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem i_memobj = NULL;
cl_mem w_memobj = NULL;
cl_mem b_memobj = NULL;
cl_mem o_memobj = NULL;
uint8_t *kernel_bin = NULL;

static void cleanup() {
    if (commandQueue) {
        printf("Releasing commandQueue\n");
        clReleaseCommandQueue(commandQueue);
    }
    if (kernel) {
        printf("Releasing kernel\n");
        clReleaseKernel(kernel);
    }
    if (program) {
        printf("Releasing program\n");
        clReleaseProgram(program);
    }
    if (i_memobj) {
        printf("Releasing i_memobj\n");
        clReleaseMemObject(i_memobj);
    }
    if (w_memobj) {
        printf("Releasing w_memobj\n");
        clReleaseMemObject(w_memobj);
    }
    if (b_memobj) {
        printf("Releasing b_memobj\n");
        clReleaseMemObject(b_memobj);
    }
    if (o_memobj) {
        printf("Releasing o_memobj\n");
        clReleaseMemObject(o_memobj);
    }
    if (context) {
        printf("Releasing context\n");
        clReleaseContext(context);
    }
    if (device_id) {
        printf("Releasing device_id\n");
        clReleaseDevice(device_id);
    }
    if (kernel_bin) {
        printf("Releasing kernel_bin\n");
        free(kernel_bin);
    }
}

static void show_usage() { printf("Usage: [-n size] [-h: help]\n"); }

static void parse_args(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "n:h")) != -1) {
        switch (c) {
        case 'n':
            SeqLength = atoi(optarg);
            OutLen = (SeqLength - K) / Stride + 1;
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

int main(int argc, char **argv) {
    // parse command arguments
    parse_args(argc, argv);

    printf("Input size=%d\n", SeqLength);

    uint32_t o_points = NumFilters * OutLen;
    uint32_t i_points = NumChannels * SeqLength;
    uint32_t w_points = NumFilters * NumChannels * K;
    uint32_t b_points = NumFilters;

    cl_platform_id platform_id;
    size_t kernel_size;

    // Getting platform and device information
    CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
    CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

    printf("Create context\n");
    context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

    char device_string[1024];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string,
                    NULL);
    printf("Using device: %s\n", device_string);

    printf("Allocate device buffers\n");
    size_t i_nbytes = i_points * sizeof(TYPE);
    size_t w_nbytes = w_points * sizeof(TYPE);
    size_t b_nbytes = b_points * sizeof(TYPE);
    size_t o_nbytes = o_points * sizeof(TYPE);
    i_memobj =
        CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, i_nbytes, NULL, &_err));
    w_memobj =
        CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, w_nbytes, NULL, &_err));
    b_memobj =
        CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, b_nbytes, NULL, &_err));
    o_memobj =
        CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, o_nbytes, NULL, &_err));

    printf("Create program from kernel source\n");
    if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
        return -1;
    program = CL_CHECK2(clCreateProgramWithSource(context, 1, (const char **)&kernel_bin,
                                                  &kernel_size, &_err));
    if (program == NULL) {
        cleanup();
        return -1;
    }

    // Build program
    CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

    // Create kernel
    kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

    size_t global_size[2] = {NumFilters, OutLen};
    size_t local_size[2] = {1, 1};

    // Set kernel arguments
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&i_memobj));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&w_memobj));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&b_memobj));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&o_memobj));

    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(uint32_t), &SeqLength));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(uint32_t), &K));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(uint32_t), &Stride));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(uint32_t), &NumChannels));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(uint32_t), &NumFilters));

    // Allocate memories for input arrays and output arrays.
    std::vector<float> h_i(i_points);
    std::vector<float> h_w(w_points);
    std::vector<float> h_b(b_points);
    std::vector<float> h_o(o_points, 0.0f);

    // Generate input values
    for (int32_t i = 0; i < SeqLength + 1; i++) {
        h_i[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (uint32_t i = 0; i < w_points; i++) {
        h_w[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (uint32_t i = 0; i < b_points; i++) {
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Creating command queue
    commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

    printf("Upload source buffers\n");
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, i_memobj, CL_TRUE, 0, i_nbytes,
                                  h_i.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, w_memobj, CL_TRUE, 0, w_nbytes,
                                  h_w.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, b_memobj, CL_TRUE, 0, b_nbytes,
                                  h_b.data(), 0, NULL, NULL));

    printf("Execute the kernel\n");
    printf("Executing kernel with OutLen: %d, Output buffer size: %d\n", OutLen,
           o_points);
    auto time_start = std::chrono::high_resolution_clock::now();
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, global_size,
                                    local_size, 0, NULL, NULL));
    CL_CHECK(clFinish(commandQueue));
    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start)
            .count();
    printf("Elapsed time: %lg ms\n", elapsed);

    printf("Download destination buffer\n");
    CL_CHECK(clEnqueueReadBuffer(commandQueue, o_memobj, CL_TRUE, 0, o_nbytes, h_o.data(),
                                 0, NULL, NULL));

    printf("Verify result\n");
    std::vector<float> ref_vec(o_points);
    convolution_cpu(h_i.data(), h_w.data(), h_b.data(), ref_vec.data(), SeqLength, K,
                    Stride, NumChannels, NumFilters);
    int errors = 0;
    for (uint32_t i = 0; i < o_points; ++i) {
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

    // Clean up
    cleanup();

    return errors;
}
