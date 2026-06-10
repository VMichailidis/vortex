#ifndef _DEVICE_HPP
#define _DEVICE_HPP
#include "CL/cl.h"
#include "CL/cl_platform.h"
#include "common.h"
#include <CL/opencl.h>
#include <unordered_map>
#include <vector>
#define CL_NAMES                                                                         \
    X(CL_DEVICE_TYPE, "Type")                                                            \
    X(CL_DEVICE_VENDOR_ID, "Vendor Id")                                                  \
    X(CL_DEVICE_MAX_COMPUTE_UNITS, "Max Compute Units")                                  \
    X(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "Max Work Item Dimensions")                    \
    X(CL_DEVICE_MAX_WORK_GROUP_SIZE, "Max Work Group Size")                              \
    X(CL_DEVICE_MAX_WORK_ITEM_SIZES, "Max Work Item Sizes")                              \
    X(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "Preferred Vector Width Char")              \
    X(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "Preferred Vector Width Short")            \
    X(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "Preferred Vector Width Int")                \
    X(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "Preferred Vector Width Long")              \
    X(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "Preferred Vector Width Float")            \
    X(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "Preferred Vector Width Double")          \
    X(CL_DEVICE_MAX_CLOCK_FREQUENCY, "Max Clock Frequency")                              \
    X(CL_DEVICE_ADDRESS_BITS, "Address Bits")                                            \
    X(CL_DEVICE_MAX_READ_IMAGE_ARGS, "Max Read Image Args")                              \
    X(CL_DEVICE_MAX_WRITE_IMAGE_ARGS, "Max Write Image Args")                            \
    X(CL_DEVICE_MAX_MEM_ALLOC_SIZE, "Max Mem Alloc Size")                                \
    X(CL_DEVICE_IMAGE2D_MAX_WIDTH, "Image2d Max Width")                                  \
    X(CL_DEVICE_IMAGE2D_MAX_HEIGHT, "Image2d Max Height")                                \
    X(CL_DEVICE_IMAGE3D_MAX_WIDTH, "Image3d Max Width")                                  \
    X(CL_DEVICE_IMAGE3D_MAX_HEIGHT, "Image3d Max Height")                                \
    X(CL_DEVICE_IMAGE3D_MAX_DEPTH, "Image3d Max Depth")                                  \
    X(CL_DEVICE_IMAGE_SUPPORT, "Image Support")                                          \
    X(CL_DEVICE_MAX_PARAMETER_SIZE, "Max Parameter Size")                                \
    X(CL_DEVICE_MAX_SAMPLERS, "Max Samplers")                                            \
    X(CL_DEVICE_MEM_BASE_ADDR_ALIGN, "Mem Base Addr Align")                              \
    X(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "Min Data Type Align Size")                    \
    X(CL_DEVICE_SINGLE_FP_CONFIG, "Single Fp Config")                                    \
    X(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, "Global Mem Cache Type")                          \
    X(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "Global Mem Cacheline Size")                  \
    X(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "Global Mem Cache Size")                          \
    X(CL_DEVICE_GLOBAL_MEM_SIZE, "Global Mem Size")                                      \
    X(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "Max Constant Buffer Size")                    \
    X(CL_DEVICE_MAX_CONSTANT_ARGS, "Max Constant Args")                                  \
    X(CL_DEVICE_LOCAL_MEM_TYPE, "Local Mem Type")                                        \
    X(CL_DEVICE_LOCAL_MEM_SIZE, "Local Mem Size")                                        \
    X(CL_DEVICE_ERROR_CORRECTION_SUPPORT, "Error Correction Support")                    \
    X(CL_DEVICE_PROFILING_TIMER_RESOLUTION, "Profiling Timer Resolution")                \
    X(CL_DEVICE_ENDIAN_LITTLE, "Endian Little")                                          \
    X(CL_DEVICE_AVAILABLE, "Available")                                                  \
    X(CL_DEVICE_COMPILER_AVAILABLE, "Compiler Available")                                \
    X(CL_DEVICE_EXECUTION_CAPABILITIES, "Execution Capabilities")                        \
    X(CL_DEVICE_QUEUE_PROPERTIES, "Queue Properties")                                    \
    X(CL_DEVICE_QUEUE_ON_HOST_PROPERTIES, "Queue On Host Properties")                    \
    X(CL_DEVICE_NAME, "Name")                                                            \
    X(CL_DEVICE_VENDOR, "Vendor")                                                        \
    X(CL_DRIVER_VERSION, "Version")                                                      \
    X(CL_DEVICE_PROFILE, "Profile")                                                      \
    X(CL_DEVICE_VERSION, "Version")                                                      \
    X(CL_DEVICE_EXTENSIONS, "Extensions")                                                \
    X(CL_DEVICE_PLATFORM, "Platform")                                                    \
    X(CL_DEVICE_DOUBLE_FP_CONFIG, "Double Fp Config")
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
        q = CL_CHECK2(cl_err, clCreateCommandQueue(context, device_id, 0, &cl_err));
        // Build program
        printf("Create program from kernel source\n");
        if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
            printf("Failed to read kernel file");
        program = CL_CHECK2(cl_err, clCreateProgramWithSource(context, 1,
                                                              (const char **)&kernel_bin,
                                                              &kernel_size, &cl_err));
        CL_CHECK(clBuildProgram(program, 1, &device_id, "-DTYPE=float", NULL, NULL));
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
        printf("Device Info:");
        max_compute_units(true, "\t");
        max_work_item_dimensions(true, "\t");
        max_mem(true, "\t");
        max_work_group_sizes(NULL, true, "\t");
    }
};
#endif
