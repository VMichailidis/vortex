#ifndef _NETWORK_HPP
#define _NETWORK_HPP
#ifndef TYPE
#define TYPE float
#endif

#include "Convolution.hpp"
#include "Device.hpp"
#include <CL/opencl.h>
#include <vector>
// KCF
template <typename T, unsigned IN, unsigned K0, unsigned C0, unsigned F0, unsigned K1,
          unsigned C1, unsigned F1, unsigned K2, unsigned C2, unsigned F2>
class Network {
  public:
    const unsigned OUT0 = IN - K0 + 1;
    const unsigned OUT1 = OUT0 - K1 + 1;
    const unsigned OUT2 = OUT1 - K2 + 1;
    Device *dev;
    Network(Device &dev_h) {
        Convolution<T, IN, K0, C0, F0> c0(dev_h);
        Convolution<T, OUT0, K1, C1, F1> c1(dev_h);
        Convolution<T, OUT1, K2, C2, F2> c2(dev_h);
    }
};

#endif
