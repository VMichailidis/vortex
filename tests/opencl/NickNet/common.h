
#ifndef _COMMON_H
#define _COMMON_H

#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define TYPE float
#include <CL/opencl.h>
#define CL_CHECK(_expr)                                                                  \
    do {                                                                                 \
        cl_int _err = _expr;                                                             \
        if (_err == CL_SUCCESS)                                                          \
            break;                                                                       \
        printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);                  \
        exit(-1);                                                                        \
    } while (0)

/* #define CL_CHECK2(_expr) \
    ({ \
        cl_int _err = CL_INVALID_VALUE; \
        decltype(_expr) _ret = _expr; \
        if (_err != CL_SUCCESS) { \
            printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
            exit(-1); \
        } \
        _ret; \
    }) */
#define CL_CHECK2(_err, _expr)                                                           \
    ({                                                                                   \
        decltype(_expr) _ret = (_expr);                                                  \
        if ((_err) != CL_SUCCESS) {                                                      \
            printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)(_err));            \
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

    *data = (uint8_t *)malloc(fsize + 1);
    *size = fread(*data, 1, fsize, fp);
    (*data)[*size] = '\0';

    fclose(fp);

    return 0;
}
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
#endif
