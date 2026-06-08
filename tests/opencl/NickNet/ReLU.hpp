
#ifndef _RELU_HPP
#define _RELU_HPP
#ifndef TYPE
#define TYPE float
#endif
#include "Device.hpp"
#include <CL/opencl.h>
#include <vector>

template <typename T, unsigned N, unsigned C> class ReLU {
  public:
    cl_mem port_in = NULL;
    uint32_t i_points = N * C;
    size_t i_nbytes;
    cl_mem port_out = NULL;
    uint32_t o_points = N * C;
    size_t o_nbytes;

    cl_kernel kernel = NULL;
    Device *dev;
    const size_t global_size[1] = {N * C};
    const size_t local_size[1] = {1};
    cl_int cl_err;
    cl_command_queue q;

    ReLU(Device &dev_h, cl_mem input = NULL) {
        i_nbytes = i_points * sizeof(T);
        o_nbytes = o_points * sizeof(T);
        dev = &dev_h;

        if (input == NULL) {
            port_in = CL_CHECK2(cl_err, clCreateBuffer(dev_h.context, CL_MEM_READ_WRITE,
                                                       i_nbytes, NULL, &cl_err));
            dev_h.buffers.push_back(port_in);
        } else {
            port_in = input;
        }
        port_out = CL_CHECK2(cl_err, clCreateBuffer(dev_h.context, CL_MEM_READ_WRITE,
                                                    o_nbytes, NULL, &cl_err));
        dev_h.buffers.push_back(port_out);

        // Create kernel
        kernel = CL_CHECK2(cl_err, clCreateKernel(dev_h.program, "relu", &cl_err));
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&port_in));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&port_out));
        q = dev_h.q;
    }
    ~ReLU() {
        if (kernel) {
            clReleaseKernel(kernel);
        }
    }
    void load_input(T *i) {
        CL_CHECK(
            clEnqueueWriteBuffer(q, port_in, CL_TRUE, 0, i_nbytes, i, 0, NULL, NULL));
    }

    void get_output(T *o) {
        CL_CHECK(clFinish(q));
        CL_CHECK(
            clEnqueueReadBuffer(q, port_out, CL_TRUE, 0, o_nbytes, o, 0, NULL, NULL));
    }

    void run() {
        if (1) {
            CL_CHECK(clEnqueueNDRangeKernel(q, kernel, 1, NULL, global_size, NULL, 0,
                                            NULL, NULL));
            CL_CHECK(clFinish(q));
        }
        if (0) { // Write a known sentinel into port_out BEFORE the kernel runs
            std::vector<float> sentinel(i_points, 1234.5678f);
            clEnqueueWriteBuffer(q, port_in, CL_TRUE, 0, o_nbytes, sentinel.data(), 0,
                                 NULL, NULL);
            clFinish(q);

            CL_CHECK(clEnqueueNDRangeKernel(q, kernel, 1, NULL, global_size, local_size,
                                            0, NULL, NULL));
            clFinish(q);

            // Read back port_out immediately
            std::vector<float> check(o_points);
            clEnqueueReadBuffer(q, port_out, CL_TRUE, 0, o_nbytes, check.data(), 0, NULL,
                                NULL);
            printf("port_out after kernel (first 5): ");
            for (int i = 0; i < 5; i++)
                printf("%f ", check[i]);
            printf("\n");

            // Also read port_in to confirm ReLU can see conv's output
            std::vector<float> check_in(i_points);
            clEnqueueReadBuffer(q, port_in, CL_TRUE, 0, i_nbytes, check_in.data(), 0,
                                NULL, NULL);
            printf("port_in seen by ReLU (first 5): ");
            for (int i = 0; i < 5; i++)
                printf("%f ", check_in[i]);
            printf("\n");
        }
    }
};

#endif
