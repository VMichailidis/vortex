#include "common.h"
#include "vortex.hpp"
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>

int main() {
    auto kernel = Kernel<KERNEL_ARGS>();
    print(kernel.apply((uint64_t)1, (uint64_t)2, (uint64_t)3, (uint64_t)4));
}
