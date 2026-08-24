#ifndef TYPE
#define TYPE float
#endif
#ifndef LS0
#define LS0 1
#endif
#ifndef LS1
#define LS1 32
#endif

TYPE conv(__global TYPE *I, // [Num Channels, Seq Length]
          __global TYPE *W, // [Num Filters, Num Channels, K]
          __global TYPE *B, // [Num Filters]
          __global TYPE *O, // [Num Filters, Out Len]
          const int IN, const int K, const int C, const int F,
          __local TYPE *LI, // [Num Channels, local size + K]
          int f, int t) {

    int samples = get_local_size(1);
    int lt = get_local_id(1);
    int lf = get_local_id(0);
    if (lf == 0) {
        for (int c = 0; c < C; c++) {
            LI[c * (samples + K) + lt] = I[c * IN + t];
        }
        if (lt < K) {
            for (int c = 0; c < C; c++) {
                LI[c * (samples + K) + samples + lt] = I[c * IN + samples + t];
            }
        }
    }
    // for(int )
    barrier(CLK_LOCAL_MEM_FENCE);
    TYPE acc = B[f];
    for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
            acc += LI[c * (samples + K) + lt + k] * W[f * C * K + c * K + k];
            // acc += I[c * IN + t + k] * W[f * C * K + c * K + k];
        }
    }
    return acc;
}

TYPE conv_no_cache(__global TYPE *I, // [Num Channels, Seq Length]
                   __global TYPE *W, // [Num Filters, Num Channels, K]
                   __global TYPE *B, // [Num Filters]
                   __global TYPE *O, // [Num Filters, Out Len]
                   const int IN, const int K, const int C, const int F,
                   // __local TYPE *LW, // [Num Filters, Num Channels, K]
                   int f, int t) {

    TYPE acc = B[f];
    for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
            acc += I[(t + k) * C + c] * W[f * C * K + c * K + k];
        }
    }
    return acc;
}

__kernel void conv1(__global TYPE *I, // [Num Channels, Seq Length]
                    __global TYPE *W, // [Num Filters, Num Channels, K]
                    __global TYPE *B, // [Num Filters]
                    __global TYPE *O, // [Num Filters, Out Len]
                    // __local TYPE *L,  // [Num Channels, local size + K
                    const int IN, const int K, const int C, const int F) {
    int f = get_global_id(0); // filter
    int t = get_global_id(1); // timestamp

    int OUT = (IN - K) + 1;
    // TYPE result = conv(I, W, B, O, IN, K, C, F, L, f, t);
    TYPE result = conv_no_cache(I, W, B, O, IN, K, C, F, f, t);
    if (f >= F || t >= OUT)
        return;
    O[F * t + f] = result;
}
__kernel void conv1_relu(__global TYPE *I, // [Num Channels, Seq Length]
                         __global TYPE *W, // [Num Filters, Num Channels, K]
                         __global TYPE *B, // [Num Filters]
                         __global TYPE *O, // [Num Filters, Out Len]
                         // __local TYPE *L,  // [Num Channels, local size + K
                         const int IN, const int K, const int C, const int F) {
    int f = get_global_id(0); // filter
    int t = get_global_id(1); // timestamp
    int OUT = (IN - K) + 1;
    // TYPE result = conv(I, W, B, O, IN, K, C, F, L, f, t);
    TYPE result = conv_no_cache(I, W, B, O, IN, K, C, F, f, t);
    if (f >= F || t >= OUT)
        return;
    O[F * t + f] = result > 0.0f ? result : 0.0f;
}
