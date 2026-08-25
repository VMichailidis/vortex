#ifndef _DEVICE_HPP
#define _DEVICE_HPP
#include "CL/cl.h"
#include "CL/cl_platform.h"
#include "common.h"
#include <CL/opencl.h>
#include <unordered_map>
#include <vector>

static const std::unordered_map<cl_device_info, const char *> cl_names = {
#define X(p, s) {p, s},
    CL_NAMES
#undef X
};
const char *get_cl_name(cl_device_info param) {
    auto it = cl_names.find(param);
    return it != cl_names.end() ? it->second : "UNKNOWN";
}
class Device {

  public:
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_platform_id platform_id;
    std::vector<cl_mem> buffers;
    // std::vector<cl_command_queue> queues;
    cl_command_queue q;
    cl_int cl_err;
    uint8_t *kernel_bin = NULL;
    size_t kernel_size;
    cl_program program;

    Device() {
        // Getting platform and device information
        CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
        CL_CHECK(
            clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

        printf("Create context\n");
        context =
            CL_CHECK2(cl_err, clCreateContext(NULL, 1, &device_id, NULL, NULL, &cl_err));
        char device_string[1024];
        clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string,
                        NULL);
        q = CL_CHECK2(cl_err, clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &cl_err));
        // Build program
        printf("Create program from kernel source\n");
        if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
            printf("Failed to read kernel file");
        program = CL_CHECK2(cl_err, clCreateProgramWithSource(context, 1,
                                                              (const char **)&kernel_bin,
                                                              &kernel_size, &cl_err));
        cl_int err = clBuildProgram(program, 1, &device_id, "-DTYPE=float", NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t log_size;

            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                                  &log_size);

            char *build_log = (char *)malloc(log_size + 1);

            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size,
                                  build_log, NULL);
            build_log[log_size] = '\0'; // Ensure null-termination

            fprintf(stderr, "OpenCL Program Build Failed. Build Log:\n%s\n", build_log);

            free(build_log);

            exit(1);
        }
        free(kernel_bin);
    }

    ~Device() {

        CL_CHECK(clFinish(q));
        if (q) {
            clReleaseCommandQueue(q);
        }
        if (program) {
            clReleaseProgram(program);
        }
        if (context) {

            printf("Releasing context\n");
            clReleaseContext(context);
        }
        if (device_id) {
            printf("Releasing device_id\n");
            clReleaseDevice(device_id);
        }
        if (!buffers.empty()) {
            for (cl_mem buffer : buffers) {
                if (buffer) {
                    clReleaseMemObject(buffer);
                }
            }
        }
    }
    template <typename T>
    T get_param(cl_device_info param, bool print = false, const char *prefix = "") {
        T ret;
        clGetDeviceInfo(device_id, param, sizeof(ret), &ret, NULL);
        if (print)
            printf("%s%s: %d\n", prefix, get_cl_name(param), ret);
        return ret;
    }
    cl_uint max_compute_units(bool print = false, const char *prefix = "") {
        return get_param<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS, print, prefix);
    }
    cl_uint max_work_item_dimensions(bool print = false, const char *prefix = "") {
        return get_param<cl_uint>(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, print, prefix);
    }
    size_t max_work_group_sizes(size_t *sizes = NULL, bool print = false,
                                const char *prefix = "") {
        cl_uint len;
        size_t max_size;
        clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(len), &len,
                        NULL);
        clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_size),
                        &max_size, NULL);
        size_t *ret = sizes = (size_t *)malloc(len * sizeof(size_t));
        clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * len,
                        ret, NULL);

        if (print) {
            printf("%sMax Work Group Size: %lu\n", prefix, max_size);
            printf("%sMax Work Group Sizes: {", prefix);
            for (cl_uint i = 0; i < len - 1; i++) {
                printf("%lu, ", ret[i]);
            }
            printf("%lu}\n", ret[len - 1]);
        }

        if (sizes == NULL)
            free(sizes);
        else {
            sizes = ret;
        }
        return max_size;
    }
    void max_mem(bool print = false, const char *prefix = "") {
        get_param<cl_ulong>(CL_DEVICE_MAX_MEM_ALLOC_SIZE, print, prefix);
        get_param<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE, print, prefix);
        get_param<cl_ulong>(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, print, prefix);
        get_param<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE, print, prefix);
        return;
    }
    void print_info() {
        printf("Device Info:\n");
        max_compute_units(true, "\t");
        max_work_item_dimensions(true, "\t");
        max_mem(true, "\t");
        max_work_group_sizes(NULL, true, "\t");
    }
};
#endif
