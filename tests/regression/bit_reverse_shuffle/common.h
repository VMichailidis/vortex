#ifndef _COMMON_H_
#define _COMMON_H_

#include <cstdint>
#ifndef TYPE
#define TYPE float
#endif

typedef struct {
    uint32_t input;
    uint64_t page;
    uint64_t scratch_pool;
} kernel_arg_t;

#endif
