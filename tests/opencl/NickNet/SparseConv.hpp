#ifndef __SPARSE_HPP
#define __SPARSE_HPP

#include "CL/cl.h"
#include "Device.hpp"
#include "Network.hpp"
#include "common.h"
#include "sparse_block.h"
#include <cstddef>

template <typename T, uint32_t IN, uint32_t K, uint32_t C, uint32_t F>
class SparseConvolution {
  private:
    uint32_t in = IN;
    uint32_t k = K;
    uint32_t c = C;
    uint32_t f = F;

  public:
    const unsigned OUT = IN - K + 1;
    cl_mem port_in = NULL;
    cl_mem w_memobj = NULL;
    cl_mem b_memobj = NULL;
    cl_mem blocks_memobj = NULL;
    cl_mem data_blocks_memobj = NULL;
    cl_mem port_out = NULL;
    uint32_t o_points = F * SPARSE_LEN;
    uint32_t i_points = C * SPARSE_LEN;
    uint32_t w_points = F * C * K;
    uint32_t b_points = F;
    uint32_t blocks_points = SPARSE_LEN;
    uint32_t data_blocks_points = F * SPARSE_LEN;
    size_t i_nbytes = sizeof(Sparse_block) + sizeof(T) * i_points;
    size_t w_nbytes = w_points * sizeof(T);
    size_t b_nbytes = b_points * sizeof(T);

    size_t blocks_nbytes = blocks_points * sizeof(T);
    size_t data_blocks_nbytes = data_blocks_points * sizeof(T);

    size_t o_nbytes = sizeof(Sparse_block) + sizeof(T) * o_points;
    cl_kernel kernel = NULL;
    cl_int cl_err;
    cl_command_queue q;
    Device *dev;
    const size_t x_size = 1;
    const size_t y_size = 64;
    const size_t OUT_l = ((OUT + y_size - 1) / y_size) * y_size;
    const size_t F_l = ((F + x_size - 1) / x_size) * x_size;
    const size_t global_size[2] = {F, OUT_l};
    const size_t local_size[2] = {1, 64};
    SparseConvolution(Device &dev_h, const char *kernel_name, cl_mem input = NULL) {
        dev = &dev_h;
        if (input == NULL) {
            port_in = CL_CHECK2(cl_err, clCreateBuffer(dev->context, CL_MEM_READ_WRITE,
                                                       i_nbytes, NULL, &cl_err));
            dev->buffers.push_back(port_in);
        } else {
            port_in = input;
        }

        port_out = CL_CHECK2(cl_err, clCreateBuffer(dev->context, CL_MEM_READ_WRITE,
                                                    o_nbytes, NULL, &cl_err));
        dev->buffers.push_back(port_out);

        w_memobj = CL_CHECK2(cl_err, clCreateBuffer(dev->context, CL_MEM_READ_ONLY,
                                                    w_nbytes, NULL, &cl_err));
        dev->buffers.push_back(w_memobj);

        b_memobj = CL_CHECK2(cl_err, clCreateBuffer(dev->context, CL_MEM_READ_ONLY,
                                                    b_nbytes, NULL, &cl_err));
        dev->buffers.push_back(b_memobj);

        blocks_memobj = CL_CHECK2(cl_err, clCreateBuffer(dev->context, CL_MEM_READ_WRITE,
                                                         blocks_nbytes, NULL, &cl_err));
        dev->buffers.push_back(blocks_memobj);

        data_blocks_memobj =
            CL_CHECK2(cl_err, clCreateBuffer(dev->context, CL_MEM_READ_WRITE,
                                             data_blocks_nbytes, NULL, &cl_err));
        dev->buffers.push_back(data_blocks_memobj);

        kernel = CL_CHECK2(cl_err, clCreateKernel(dev->program, kernel_name, &cl_err));

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&w_memobj)); // Weights
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_memobj)); // Biases
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(uint32_t), (void *)&k));      // K
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(uint32_t), (void *)&f));      // F
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&port_in));  // SRC
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&port_out)); // DST
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&blocks_memobj));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&data_blocks_memobj));
        q = dev_h.q;
    }
    ~SparseConvolution() {
        if (kernel) {
            printf("Releasing kernel\n");
            clReleaseKernel(kernel);
            kernel = NULL;
        }
    }
    void load_weights(TYPE *w, TYPE *b) {
        CL_CHECK(
            clEnqueueWriteBuffer(q, w_memobj, CL_TRUE, 0, w_nbytes, w, 0, NULL, NULL));
        CL_CHECK(
            clEnqueueWriteBuffer(q, b_memobj, CL_TRUE, 0, b_nbytes, b, 0, NULL, NULL));
    }
    void load_input(Sparse_block *i) {
        CL_CHECK(
            clEnqueueWriteBuffer(q, port_in, CL_TRUE, 0, i_nbytes, i, 0, NULL, NULL));
    }
    void get_output(Sparse_block *o) {
        CL_CHECK(
            clEnqueueReadBuffer(q, port_out, CL_TRUE, 0, o_nbytes, o, 0, NULL, NULL));
    }
    cl_event run() {
        cl_event target;
        size_t max_size = dev->max_work_group_sizes();
        if (local_size[0] * local_size[1] > max_size) {
            printf("\033[31mError:\033[0m Requested work group size (%d) is greater than "
                   "maximum work "
                   "group "
                   "size (%d)\n",
                   local_size[0] * local_size[1], max_size);
            exit(-1);
        }
        CL_CHECK(clEnqueueNDRangeKernel(q, kernel, 2, NULL, global_size, local_size, 0,
                                        NULL, NULL));
        return target;
    }
};

#endif
