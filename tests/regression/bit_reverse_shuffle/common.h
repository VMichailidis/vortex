#ifndef _COMMON_H_
#define _COMMON_H_

#include <cstdint>
#ifndef TYPE
#define TYPE float
#endif

typedef struct {
    uint64_t input;
    uint64_t pages;
    uint64_t scratch_pool;
    uint64_t log_n;
} kernel_arg_t;

template <unsigned char LOG_N>
void advance_index_and_reversed(unsigned long &index, unsigned long &reversed) {
    unsigned long temp = index + 1;
    unsigned long tail = (index ^ temp);
    // tail is of the form 00...011...1
    // It is a mask representing the number of bitflips
    index = temp;
    // create the reverse of tail, which is of form 11...100...0:
    //
    // Because when incrementing a number the number of bitflips
    // is a continuous string of 1's until cout is zero
    // We can simply shift the mask to the left and have the
    // reverse of the bitflip mask
    auto shift = __builtin_clzl(tail);
    tail <<= shift;
    tail >>= ((sizeof(unsigned long) * 8) - LOG_N);

    // xor reversed with reversed tail gives reversed of index+1:
    // The bitflips can be propageted via a simple xor
    reversed ^= tail;
}
template <const char W> int reverse(int x) {
    int reversed = 0;
    for (int j = 0; j < W; j++) // log2(N) = 3 bits needed to represent indices
        reversed = (reversed << 1) | (x >> j & 1);
    return reversed;
}
#endif
