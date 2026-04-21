#ifndef __COMMON_H__
#define __COMMON_H__
#include <cstdint>
#include <cstdio>
typedef struct {
    uint64_t input;
    uint64_t pages;
    uint64_t scratch_pool;
    uint64_t log_n;
} kernel_arg_t;

// Required macro since type reflections are not available
// This macro is simply the type name alongside the fields of the type
#define KERNEL_ARGS                                                                      \
    kernel_arg_t, &kernel_arg_t::input, &kernel_arg_t::pages,                            \
        &kernel_arg_t::scratch_pool, &kernel_arg_t::log_n

void print(kernel_arg_t p) {
    printf("input: %lul, pages: %lul, scratch pool: %lul, log_n: %lul\n", p.input, p.pages,
           p.scratch_pool, p.log_n);
}
#endif
