
#include "common.h"

__kernel void lu_decomp (__global TYPE *A,
	                    __global TYPE *B,
	                    __global TYPE *C)
{
  int gid = get_global_id(0);
  C[gid] = A[gid] + B[gid];
}
