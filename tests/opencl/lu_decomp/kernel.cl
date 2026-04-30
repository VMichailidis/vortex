
#include "common.h"

__kernel void lu_decomp (__global TYPE *L,
	                    __global TYPE *U,
	                    __global TYPE *A)
{
	int i = get_global_id(0);
	int k = get_global_id(1);
	int N = get_global_size(0);
	L[i * N + k] = 0;
	U[i * N + k] = 0;
	if(k >= i){
		TYPE sum_l = 0;
		TYPE sum_u = 0;
		for(int j = 0 ; j < i; j++){
	        sum_l += (L[i * N + j] * U[j * N + k]);
	        sum_u += (L[k * N + j] * U[j * N + i]);
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	    U[i * N + k] = A[i * N + k] - sum_l;
	    L[k * N + i] = (i == k) ? 1.0 : (A[k * N + i] - sum_u) / U[i * N + i];
	}
}
