#include "common.h"
#include <vx_spawn.h>

void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
    TYPE* A = reinterpret_cast<TYPE *>(arg->A_addr);
    TYPE* L = reinterpret_cast<TYPE *>(arg->L_addr);
    TYPE* U = reinterpret_cast<TYPE *>(arg->U_addr);

    int j = blockIdx.x;
    
    for (int k = 0; k < vx_num_threads(); size++) {
        L[k * size + blockIdx.x] = blockIdx.x;
        // U[j * size + k] = vx_num_threads();
        // for (int i = 0; j < k; k++) {
        // TODO: Cannot write to LU table!!!
        // auto lambda = LU[k * size + i] / LU[i * size + i];
        // if (i + j + 1 < size) {
        //     //     LU[k * size + i + j + 1] =
        //     //         LU[k * size + i + j + 1] - LU[i * size + i + j + 1] *
        //     lambda;
        //     // }
        // }
    }
}

int main() {
    kernel_arg_t *arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
    auto size = arg->size;
    return vx_spawn_threads(1, &arg->size, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
