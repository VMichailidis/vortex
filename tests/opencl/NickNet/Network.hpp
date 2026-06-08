#ifndef _NETWORK_HPP
#define _NETWORK_HPP
#include "CL/cl.h"
#include "common.h"
#ifndef TYPE
#define TYPE float
#endif

#include "Convolution.hpp"
#include "Device.hpp"
#include <CL/opencl.h>
#include <vector>
// KCF
template <typename T, unsigned IN, unsigned K0, unsigned C0, unsigned K1, unsigned C1,
          unsigned K2, unsigned C2, unsigned F>
class Network {
  public:
    static const unsigned OUT0 = IN - K0 + 1;
    static const unsigned OUT1 = OUT0 - K1 + 1;
    static const unsigned OUT2 = OUT1 - K2 + 1;

    Convolution<T, IN, K0, C0, C1> *c0;
    Convolution<T, OUT0, K1, C1, C2> *c1;
    Convolution<T, OUT1, K2, C2, F> *c2;

    Device *dev;
    cl_int cl_err;

    Network(Device &dev_h, cl_mem input = NULL) {
        c0 = new Convolution<T, IN, K0, C0, C1>(dev_h, "conv1_relu");
        c1 = new Convolution<T, OUT0, K1, C1, C2>(dev_h, "conv1_relu", c0->port_out);
        c2 = new Convolution<T, OUT1, K2, C2, F>(dev_h, "conv1", c1->port_out);
    }
    void load_weights(TYPE **w, TYPE **b) {
        c0->load_weights(w[0], b[0]);
        c1->load_weights(w[1], b[1]);
        c2->load_weights(w[2], b[2]);
    }
    void load_input(TYPE *i) { c0->load_input(i); }

    void get_output(TYPE *o) { c2->get_output(o); }
    void run() {
        cl_event layer0, layer1;
        layer0 = c0->run();
        layer1 = c1->run(layer0);
        c2->run(layer1);
        CL_CHECK(clReleaseEvent(layer0));
        CL_CHECK(clReleaseEvent(layer1));
    }
};

#endif
