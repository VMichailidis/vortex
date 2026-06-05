#ifndef _DEVICE_HPP
#define _DEVICE_HPP
#include "CL/cl.h"
#include "common.h"
#include <CL/opencl.h>
#include <vector>

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
        if (q) {
            clReleaseCommandQueue(q);
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
    // cl_command_queue new_queue() {
    //     cl_command_queue q =
    //         CL_CHECK2(cl_err, clCreateCommandQueue(context, device_id, 0, &cl_err));
    //     queues.push_back(q);
    //     return q;
    // }
};
#endif
