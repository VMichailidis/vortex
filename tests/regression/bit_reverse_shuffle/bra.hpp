#ifndef BRA_HPP
#define BRA_HPP
#include "common.h"
#include <algorithm>
#include <cstring>
#include <iostream>
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
