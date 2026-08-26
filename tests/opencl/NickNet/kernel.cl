#ifndef TYPE
#define TYPE float
#endif
#ifndef LS0
#define LS0 1
#endif
#ifndef LS1
#define LS1 32
#endif

TYPE conv(__global TYPE *I,   // [Batch Size, Seq Length, Num Channels]
          __constant TYPE *W, // [Num Filters, Num Channels, K]
          __constant TYPE *B, // [Num Filters]
          __global TYPE *O,   // [Num Filters, Out Len]
          __local TYPE *LI,   // [Batch Size, local size + K, Num Channels]
          const int IN, const int K, const int C, const int F, int f, int t, int b) {

    int samples = get_local_size(1);
    int lt = get_local_id(1);
    int lf = get_local_id(0);
    if (lf == 0) {
        for (int c = 0; c < C; c++) {
            LI[(samples + K) * C + lt * C + c] = I[b * IN * C + t * C + c];
        }
        if (lt < K) {
            for (int c = 0; c < C; c++) {
                LI[(samples + K) * C + (lt + samples) * C + c] =
                    I[b * IN * C + (t + samples) * C + c];
            }
        }
    }
    // for(int )
    barrier(CLK_LOCAL_MEM_FENCE);
    TYPE acc = B[f];
    for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
            acc += LI[C * (samples + K) + (lt + k) * C + c] * W[f * C * K + c * K + k];
        }
    }
    return acc;
}

TYPE conv_no_cache(__global TYPE *I,   // [Num Channels, Seq Length]
                   __constant TYPE *W, // [Num Filters, Num Channels, K]
                   __constant TYPE *B, // [Num Filters]
                   __global TYPE *O,   // [Out Len, Num filters]
                   const int IN, const int K, const int C, const int F,
                   // __local TYPE *LW, // [Num Filters, Num Channels, K]
                   int f, int t, int b) {

    TYPE acc = B[f];
    for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
            acc += I[b * IN * C + (t + k) * C + c] * W[f * C * K + c * K + k];
        }
    }
    return acc;
}

__kernel void conv1(__global TYPE *I,   // [Num Channels, Seq Length]
                    __constant TYPE *W, // [Num Filters, Num Channels, K]
                    __constant TYPE *B, // [Num Filters]
                    __global TYPE *O,   // [Out Len, Num filters]
                    __local TYPE *L,    // [Batch Size, local size + K, Num Channels]
                    const int IN, const int K, const int C, const int F,
                    const int BATCH) {
    int f = get_global_id(0); // filter
    int t = get_global_id(1); // timestamp
    int b = get_global_id(2); // batch

    int OUT = (IN - K) + 1;
    // TYPE result = conv(I, W, B, O, IN, K, C, F, L, f, t);
    TYPE result = conv(I, W, B, O, L, IN, K, C, F, f, t, b);
    if (b >= BATCH || f >= F || t >= OUT)
        return;
    O[b * OUT * F + F * t + f] = result;
}
__kernel void conv1_relu(__global TYPE *I,   // [Num Channels, Seq Length]
                         __constant TYPE *W, // [Num Filters, Num Channels, K]
                         __constant TYPE *B, // [Num Filters]
                         __global TYPE *O,   // [Out Len, Num filters]
                         __local TYPE *L,    // [Batch Size, local size + K, Num Channels]
                         const int IN, const int K, const int C, const int F,
                         const int BATCH) {
    int f = get_global_id(0); // filter
    int t = get_global_id(1); // timestamp
    int b = get_global_id(2); // batch
    int OUT = (IN - K) + 1;
    // TYPE result = conv(I, W, B, O, IN, K, C, F, L, f, t);
    TYPE result = conv(I, W, B, O, L, IN, K, C, F, f, t, b);
    if (b >= BATCH || f >= F || t >= OUT)
        return;
    O[b * OUT * F + F * t + f] = result > 0.0f ? result : 0.0f;
}
