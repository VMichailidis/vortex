#ifndef _CONVOLUTION_HPP
#define _CONVOLUTION_HPP
#ifndef TYPE
#define TYPE float
#endif

#include "Device.hpp"
#include <CL/opencl.h>
#include <vector>
// Convolution layer
// Input: C channels of IN samples
// Output: F channels of IN - K + 1 samples
// Parameters: K: kernel size

template <typename T, uint32_t IN, uint32_t K, uint32_t C, uint32_t F> class Convolution {
  private:
    // Required to pass template params to kernel
    uint32_t in = IN;
    uint32_t k = K;
    uint32_t c = C;
    uint32_t f = F;

  public:
    const unsigned OUT = IN - K + 1;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem i_memobj = NULL;
    cl_mem w_memobj = NULL;
    cl_mem b_memobj = NULL;
    cl_mem o_memobj = NULL;
    uint32_t o_points = F * OUT;
    uint32_t i_points = C * IN;
    uint32_t w_points = F * C * K;
    uint32_t b_points = F;
    size_t i_nbytes;
    size_t w_nbytes;
    size_t b_nbytes;
    size_t o_nbytes;

    uint8_t *kernel_bin = NULL;
    Device *dev;
    const size_t global_size[2] = {F, OUT};
    const size_t local_size[2] = {1, 1};
    Convolution(Device &dev_h) {
        i_nbytes = i_points * sizeof(TYPE);
        w_nbytes = w_points * sizeof(TYPE);
        b_nbytes = b_points * sizeof(TYPE);
        o_nbytes = o_points * sizeof(TYPE);
        size_t kernel_size;
        dev = &dev_h;

        i_memobj = CL_CHECK2(
            clCreateBuffer(dev_h.context, CL_MEM_READ_WRITE, i_nbytes, NULL, &_err));
        dev_h.buffers.push_back(i_memobj);
        w_memobj = CL_CHECK2(
            clCreateBuffer(dev_h.context, CL_MEM_READ_ONLY, w_nbytes, NULL, &_err));
        dev_h.buffers.push_back(w_memobj);
        b_memobj = CL_CHECK2(
            clCreateBuffer(dev_h.context, CL_MEM_READ_ONLY, b_nbytes, NULL, &_err));
        dev_h.buffers.push_back(b_memobj);
        o_memobj = CL_CHECK2(
            clCreateBuffer(dev_h.context, CL_MEM_READ_WRITE, o_nbytes, NULL, &_err));
        dev_h.buffers.push_back(o_memobj);
        printf("Create program from kernel source\n");
        if (0 != read_kernel_file("conv1.cl", &kernel_bin, &kernel_size))
            printf("Failed to read kernel file");
        program = CL_CHECK2(clCreateProgramWithSource(
            dev_h.context, 1, (const char **)&kernel_bin, &kernel_size, &_err));
        if (program == NULL) {
            printf("Failed to build program");
        }

        // Build program
        CL_CHECK(clBuildProgram(program, 1, &dev_h.device_id, NULL, NULL, NULL));

        // Create kernel
        kernel = CL_CHECK2(clCreateKernel(program, "conv1", &_err));

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&i_memobj));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&w_memobj));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&b_memobj));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&o_memobj));

        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(uint32_t), &in));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(uint32_t), &k));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(uint32_t), &c));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(uint32_t), &f));
        // Creating command queue
    }

    ~Convolution() {
        if (kernel) {
            printf("Releasing kernel\n");
            clReleaseKernel(kernel);
        }
        if (program) {
            printf("Releasing program\n");
            clReleaseProgram(program);
        }
        if (kernel_bin) {
            printf("Releasing kernel_bin\n");
            free(kernel_bin);
        }
    }
    void load_weights(TYPE *w, TYPE *b) {
        CL_CHECK(clFinish(dev->commandQueue));
        CL_CHECK(clEnqueueWriteBuffer(dev->commandQueue, w_memobj, CL_TRUE, 0, w_nbytes,
                                      w, 0, NULL, NULL));
        CL_CHECK(clFinish(dev->commandQueue));
        CL_CHECK(clEnqueueWriteBuffer(dev->commandQueue, b_memobj, CL_TRUE, 0, b_nbytes,
                                      b, 0, NULL, NULL));
    }
    void load_input(TYPE *i) {
        CL_CHECK(clFinish(dev->commandQueue));
        CL_CHECK(clEnqueueWriteBuffer(dev->commandQueue, i_memobj, CL_TRUE, 0, i_nbytes,
                                      i, 0, NULL, NULL));
    }

    void get_output(TYPE *o) {
        CL_CHECK(clFinish(dev->commandQueue));
        CL_CHECK(clEnqueueReadBuffer(dev->commandQueue, o_memobj, CL_TRUE, 0, o_nbytes, o,
                                     0, NULL, NULL));
    }

    void run() {

        char device_string[1024];
        clGetDeviceInfo(dev->device_id, CL_DEVICE_NAME, sizeof(device_string),
                        &device_string, NULL);
        printf("Executing convolution layer using device: %s\n", device_string);
        printf("Executing kernel with OutLen: %d, Output buffer size: %d\n", OUT,
               o_points);
        CL_CHECK(clEnqueueNDRangeKernel(dev->commandQueue, kernel, 2, NULL, global_size,
                                        local_size, 0, NULL, NULL));
    }
};

#endif
