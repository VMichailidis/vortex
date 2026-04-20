#ifndef BRA_HPP
#define BRA_HPP
#include <algorithm>
#include <cstring>
#include <iostream>
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

// TODO: re-write all methods to be more templated
template <typename T, unsigned char LOG_N> class Shuffle {
  public:
    inline static void apply(T *__restrict const v) {
        if (LOG_N % 2) {
            odd(v);
        } else {
            even(v);
        }
        return;
    }

  private:
    inline static void even(T *__restrict const v) {
        unsigned long rlo = 0;
        unsigned long ru = 0;
        for (unsigned long lo = 0; lo < (1 << (LOG_N / 2));) {

            for (unsigned long u = 0; u < rlo;) {
                unsigned long index = (u << (LOG_N / 2)) | lo;
                unsigned long xendi = (rlo << (LOG_N / 2)) | ru;

                std::swap(v[index], v[xendi]);
                advance_index_and_reversed<LOG_N / 2>(u, ru);
            }
            ru = 0;
            advance_index_and_reversed<LOG_N / 2>(lo, rlo);
        }
    }
    inline static void odd(T *__restrict const v) {
        unsigned long rlo = 0;
        unsigned long ru = 0;
        for (unsigned long lo = 0; lo < 1 << (LOG_N / 2);) {
            for (unsigned long u = 0; u < rlo;) {
                unsigned long index = (u << ((LOG_N / 2) + 1)) | lo;
                unsigned long xendi = (rlo << ((LOG_N / 2) + 1)) | ru;
                std::swap(v[index], v[xendi]);
                advance_index_and_reversed<LOG_N / 2>(u, ru);
            }
            ru = 0;
            advance_index_and_reversed<LOG_N / 2>(lo, rlo);
        }
        rlo = 0;
        ru = 0;
        for (unsigned long lo = 0; lo < 1 << (LOG_N / 2);) {
            for (unsigned long u = 0; u < rlo;) {
                unsigned long int index = (u << ((LOG_N / 2) + 1)) | 1 << LOG_N / 2 | lo;
                unsigned long int xendi =
                    (rlo << ((LOG_N / 2) + 1)) | 1 << LOG_N / 2 | ru;
                std::swap(v[index], v[xendi]);
                advance_index_and_reversed<LOG_N / 2>(u, ru);
            }
            ru = 0;
            advance_index_and_reversed<LOG_N / 2>(lo, rlo);
        }
    }
};
#endif
