#include "CL/cl.h"
#include "CL/cl_platform.h"
#include "common.h"
#include <CL/opencl.h>
#include <assert.h>
#include <chrono>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>

#define PIVOT_KERNEL "compute_pivot"
#define LU_UPDATE "lu_update"

#define FLOAT_ULP 2048
#define FLOAT_TOLERANCE 1e-4

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

template <typename Type> class Comparator {};

template <> class Comparator<int> {
  public:
    static const char *type_str() { return "integer"; }
    static int generate() { return rand(); }
    static bool compare(int a, int b, int index, int errors) {
        if (a != b) {
            if (errors < 100) {
                printf("*** error: [%d] expected=%d, actual=%d\n", index, a, b);
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
        uint32_t d = std::abs(fa.f - fb.f);
        if (d > FLOAT_TOLERANCE) {
            if (errors < 100) {
                printf("*** error: [%d] expected=%f, actual=%f\n", d, a, b);
            }
            return false;
        }
        return true;
    }
};

static void lu_decomp_cpu(TYPE *L, TYPE *U, TYPE *A, int N) {
    memset(L, 0, sizeof(TYPE) * N * N);
    memset(U, 0, sizeof(TYPE) * N * N);

    // Decomposing matrix into Upper and Lower
    // triangular matrix
    for (int i = 0; i < N; i++) {
        // Upper Triangular
        for (int k = i; k < N; k++) {
            // Summation of L(i, j) * U(j, k)
            TYPE sum = 0;
            for (int j = 0; j < i; j++)
                sum += (L[i * N + j] * U[j * N + k]);

            // Evaluating U(i, k)
            U[i * N + k] = A[i * N + k] - sum;
        }

        // Lower Triangular
        for (int k = i; k < N; k++) {
            if (i == k)
                L[i * N + i] = 1; // Diagonal as 1
            else {
                // Summation of L(k, j) * U(j, i)
                TYPE sum = 0;
                for (int j = 0; j < i; j++)
                    sum += (L[k * N + j] * U[j * N + i]);

                // Evaluating L(k, i)
                L[k * N + i] = (A[k * N + i] - sum) / U[i * N + i];
            }
        }
    }
}
static void vecadd_alt(TYPE *L, TYPE *U, TYPE *A, int N) {
    memset(L, 0, sizeof(TYPE) * N * N);
    memset(U, 0, sizeof(TYPE) * N * N);

    // Decomposing matrix into Upper and Lower
    // triangular matrix
    for (int i = 0; i < N; i++) {
        // Upper Triangular
        for (int k = i; k < N; k++) {
            // Summation of L(i, j) * U(j, k)
            TYPE sum_l = 0;
            TYPE sum_u = 0;
            for (int j = 0; j < i; j++) {
                sum_l += (L[i * N + j] * U[j * N + k]);
                sum_u += (L[k * N + j] * U[j * N + i]);
            }
            // Evaluating U(i, k)
            U[i * N + k] = A[i * N + k] - sum_l;
            L[k * N + i] = (i == k) ? 1.0 : (A[k * N + i] - sum_u) / U[i * N + i];
        }
    }
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel pivot_kernel = NULL;
cl_kernel update_kernel = NULL;
cl_mem a_memobj = NULL;
cl_mem l_memobj = NULL;
cl_mem u_memobj = NULL;
uint8_t *kernel_bin = NULL;

static void cleanup() {
    if (commandQueue)
        clReleaseCommandQueue(commandQueue);
    if (pivot_kernel)
        clReleaseKernel(pivot_kernel);
    if (update_kernel)
        clReleaseKernel(update_kernel);
    if (program)
        clReleaseProgram(program);
    if (a_memobj)
        clReleaseMemObject(a_memobj);
    if (l_memobj)
        clReleaseMemObject(l_memobj);
    if (u_memobj)
        clReleaseMemObject(u_memobj);
    if (context)
        clReleaseContext(context);
    if (device_id)
        clReleaseDevice(device_id);

    if (kernel_bin)
        free(kernel_bin);
}

uint32_t size = 64;

static void show_usage() { printf("Usage: [-n size] [-h: help]\n"); }

static void parse_args(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "n:h")) != -1) {
        switch (c) {
        case 'n':
            size = atoi(optarg);
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

    printf("Workload size=%d\n", size);
}

int main(int argc, char **argv) {
    // parse command arguments
    parse_args(argc, argv);

    cl_platform_id platform_id;
    size_t kernel_size;

    // Getting platform and device information
    CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
    CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

    printf("Create context\n");
    context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

    printf("Allocate device buffers\n");
    auto size_sq = size * size;
    size_t nbytes = size_sq * sizeof(TYPE);
    a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, &_err));
    l_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &_err));
    u_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &_err));

    printf("Create program from kernel source\n");
    if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
        return -1;
    program = CL_CHECK2(clCreateProgramWithSource(context, 1, (const char **)&kernel_bin,
                                                  &kernel_size, &_err));

    // Build program
    CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

    // Create kernel
    pivot_kernel = CL_CHECK2(clCreateKernel(program, PIVOT_KERNEL, &_err));
    update_kernel = CL_CHECK2(clCreateKernel(program, LU_UPDATE, &_err));

    // Set kernel arguments
    CL_CHECK(clSetKernelArg(pivot_kernel, 0, sizeof(cl_mem), (void *)&l_memobj));
    CL_CHECK(clSetKernelArg(pivot_kernel, 1, sizeof(cl_mem), (void *)&u_memobj));
    CL_CHECK(clSetKernelArg(pivot_kernel, 2, sizeof(cl_mem), (void *)&a_memobj));
    CL_CHECK(clSetKernelArg(pivot_kernel, 3, sizeof(int), &size));
    CL_CHECK(clSetKernelArg(update_kernel, 0, sizeof(cl_mem), (void *)&l_memobj));
    CL_CHECK(clSetKernelArg(update_kernel, 1, sizeof(cl_mem), (void *)&u_memobj));
    CL_CHECK(clSetKernelArg(update_kernel, 2, sizeof(cl_mem), (void *)&a_memobj));
    CL_CHECK(clSetKernelArg(update_kernel, 3, sizeof(int), &size));

    // Allocate memories for input arrays and output arrays.
    std::vector<TYPE> h_l(size_sq);
    std::vector<TYPE> h_u(size_sq);
    std::vector<TYPE> h_a(size_sq);
    memset(h_l.data(), 0, sizeof(TYPE) * size_sq);
    memset(h_u.data(), 0, sizeof(TYPE) * size_sq);

    // Generate input values
    for (uint32_t i = 0; i < size_sq; ++i) {
        TYPE t = Comparator<TYPE>::generate();
        h_a[i] = t;
    }

    // Creating command queue
    commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

    printf("Upload source buffers\n");
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, nbytes, h_a.data(),
                                  0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, l_memobj, CL_TRUE, 0, nbytes, h_u.data(),
                                  0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, u_memobj, CL_TRUE, 0, nbytes, h_l.data(),
                                  0, NULL, NULL));

    printf("Execute the kernel\n");
    auto time_start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < size; i++) {
        // Step 1: compute pivot element U[i][i] safely (single work-item)
        CL_CHECK(clSetKernelArg(pivot_kernel, 4, sizeof(int), &i));
        size_t pivot_work_size[1] = {1};
        CL_CHECK(clEnqueueNDRangeKernel(commandQueue, pivot_kernel, 1, NULL,
                                        pivot_work_size, pivot_work_size, 0, NULL, NULL));
        CL_CHECK(clFinish(commandQueue)); // ensure pivot is written before update

        // Step 2: update remaining row/column in parallel (safe, pivot already written)
        if (i + 1 < size) {
            CL_CHECK(clSetKernelArg(update_kernel, 4, sizeof(int), &i));
            size_t update_work_size[1] = {size - i - 1};
            size_t local_work_size[1] = {1};
            CL_CHECK(clEnqueueNDRangeKernel(commandQueue, update_kernel, 1, NULL,
                                            update_work_size, local_work_size, 0, NULL,
                                            NULL));
        }
        CL_CHECK(clFinish(commandQueue));
    }
    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start)
            .count();
    printf("Elapsed time: %lg ms\n", elapsed);

    printf("Download destination buffer\n");
    CL_CHECK(clEnqueueReadBuffer(commandQueue, l_memobj, CL_TRUE, 0, nbytes, h_l.data(),
                                 0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(commandQueue, u_memobj, CL_TRUE, 0, nbytes, h_u.data(),
                                 0, NULL, NULL));

    printf("Verify result\n");
    std::vector<TYPE> h_ref_l(size_sq);
    std::vector<TYPE> h_ref_u(size_sq);
    lu_decomp_cpu(h_ref_l.data(), h_ref_u.data(), h_a.data(), size);
    // vecadd_alt(h_l.data(), h_u.data(), h_a.data(), size);
    int errors = 0;
    printf("Testing L\n");
    for (uint32_t i = 0; i < size_sq; ++i) {
        if (!Comparator<TYPE>::compare(h_ref_l[i], h_l[i], i, errors)) {
            ++errors;
        }
    }
    printf("Testing U\n");
    for (uint32_t i = 0; i < size_sq; ++i) {
        if (!Comparator<TYPE>::compare(h_ref_u[i], h_u[i], i, errors)) {
            ++errors;
        }
    }
    if (0 == errors) {
        printf("PASSED!\n");
    } else {
        printf("FAILED! - %d errors\n", errors);
    }

    // Clean up
    cleanup();

    return errors;
}
