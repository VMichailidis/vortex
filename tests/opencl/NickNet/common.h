#ifndef _COMMON_H
#define _COMMON_H

#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define TYPE float
#define CL_CHECK(_expr)                                                                  \
    do {                                                                                 \
        cl_int _err = _expr;                                                             \
        if (_err == CL_SUCCESS)                                                          \
            break;                                                                       \
        printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);                  \
        exit(-1);                                                                        \
    } while (0)

#define CL_CHECK2(_expr)                                                                 \
    ({                                                                                   \
        cl_int _err = CL_INVALID_VALUE;                                                  \
        decltype(_expr) _ret = _expr;                                                    \
        if (_err != CL_SUCCESS) {                                                        \
            printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);              \
            exit(-1);                                                                    \
        }                                                                                \
        _ret;                                                                            \
    })

static int read_kernel_file(const char *filename, uint8_t **data, size_t *size) {
    if (nullptr == filename || nullptr == data || 0 == size)
        return -1;

    FILE *fp = fopen(filename, "r");
    if (NULL == fp) {
        fprintf(stderr, "Failed to load kernel.");
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    rewind(fp);

    *data = (uint8_t *)malloc(fsize);
    *size = fread(*data, 1, fsize, fp);

    fclose(fp);

    return 0;
}
#endif
