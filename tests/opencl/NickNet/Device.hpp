#ifndef _DEVICE_HPP
#define _DEVICE_HPP
#include "common.h"
#include <CL/opencl.h>
#include <vector>

class Device {

  public:
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue commandQueue = NULL;
    cl_platform_id platform_id;
    std::vector<cl_mem> buffers;

    Device() {
        // Getting platform and device information
        CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
        CL_CHECK(
            clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

        printf("Create context\n");
        context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));
        char device_string[1024];
        clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string,
                        NULL);
        // Creating command queue
        commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));
        printf("Using device: %s\n", device_string);
    }
    ~Device() {
        if (commandQueue) {
            printf("Releasing commandQueue\n");
            clReleaseCommandQueue(commandQueue);
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
};
#endif
