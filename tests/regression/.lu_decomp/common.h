#ifndef _COMMON_H_
#define _COMMON_H_

#include <cstdint>
#ifndef TYPE
#define TYPE float
#endif

typedef struct {
    uint32_t size;
    uint64_t A_addr;
    uint64_t L_addr;
    uint64_t U_addr;
} kernel_arg_t;

#endif
