
#include "common.h"

__kernel void compute_pivot(__global TYPE *L, __global TYPE *U, __global TYPE *A,
                            const int N, const int i) {
    TYPE sum = 0;
    for (int j = 0; j < i; j++) {
        sum += L[i * N + j] * U[j * N + i];
    }
    U[i * N + i] = A[i * N + i] - sum;
    L[i * N + i] = 1.0;
}

// kernel 2: lu_decomp.cl
// Run with global_work_size = (size - i - 1), offset k starts at i+1
__kernel void lu_update(__global TYPE *L, __global TYPE *U, __global TYPE *A, const int N,
                        const int i) {
    int k = get_global_id(0) + i + 1; // k > i, skip pivot
    TYPE sum_u = 0;
    TYPE sum_l = 0;
    for (int j = 0; j < i; j++) {
        sum_l += L[i * N + j] * U[j * N + k];
        sum_u += L[k * N + j] * U[j * N + i];
    }
    U[i * N + k] = A[i * N + k] - sum_l; // upper triangle
    L[k * N + i] =
        (A[k * N + i] - sum_u) / U[i * N + i]; // lower triangle, pivot now safe to read
}
