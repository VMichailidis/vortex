#ifndef _CONVOLUTION_HPP
#define _CONVOLUTION_HPP

#include "Device.hpp"
#include <CL/opencl.h>
#include <vector>
// Convolution layer
// Input: C channels of IN samples
// Output: F channels of IN - K + 1 samples
// Parameters: K: kernel size

template <typename T, unsigned IN, unsigned K, unsigned C, unsigned F> class Convolution {
  public:
    const unsigned OUT = IN - K + 1;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem i_memobj = NULL;
    cl_mem w_memobj = NULL;
    cl_mem b_memobj = NULL;
    cl_mem o_memobj = NULL;
    uint8_t *kernel_bin = NULL;
    const size_t global_size[2] = {F, OUT};
    const size_t local_size[2] = {1, 1};
    Convolution(Device dev, const char *filename) {
        uint32_t o_points = F * OUT;
        uint32_t i_points = C * IN;
        uint32_t w_points = F * C * K;
        uint32_t b_points = F;
        size_t i_nbytes = i_points * sizeof(TYPE);
        size_t w_nbytes = w_points * sizeof(TYPE);
        size_t b_nbytes = b_points * sizeof(TYPE);
        size_t o_nbytes = o_points * sizeof(TYPE);
        size_t kernel_size;

        i_memobj = CL_CHECK2(
            clCreateBuffer(dev.context, CL_MEM_READ_ONLY, i_nbytes, NULL, &_err));
        dev.buffers.push_back(i_memobj);
        w_memobj = CL_CHECK2(
            clCreateBuffer(dev.context, CL_MEM_READ_ONLY, w_nbytes, NULL, &_err));
        dev.buffers.push_back(w_memobj);
        b_memobj = CL_CHECK2(
            clCreateBuffer(dev.context, CL_MEM_READ_ONLY, b_nbytes, NULL, &_err));
        dev.buffers.push_back(b_memobj);
        o_memobj = CL_CHECK2(
            clCreateBuffer(dev.context, CL_MEM_WRITE_ONLY, o_nbytes, NULL, &_err));
        dev.buffers.push_back(o_memobj);
        printf("Create program from kernel source\n");
        if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
            printf("Failed to read kernel file");
        program = CL_CHECK2(clCreateProgramWithSource(
            dev.context, 1, (const char **)&kernel_bin, &kernel_size, &_err));
        if (program == NULL) {
            printf("Failed to build program");
        }

        // Build program
        CL_CHECK(clBuildProgram(program, 1, &dev.device_id, NULL, NULL, NULL));

        // Create kernel
        kernel = CL_CHECK2(clCreateKernel(program, "conv1", &_err));

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&i_memobj));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&w_memobj));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&b_memobj));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&o_memobj));

        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(uint32_t), &IN));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(uint32_t), &K));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(uint32_t), &C));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(uint32_t), &F));
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
    int run(std::vector<TYPE> i, std::vector<TYPE> w, std::vector<TYPE> b,
            std::vector<TYPE> o) {}
};

#endif
