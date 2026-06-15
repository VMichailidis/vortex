#ifndef TYPE
#define TYPE float
#endif
#ifndef LS0
#define LS0 1
#endif
#ifndef LS1
#define LS1 32
#endif

TYPE conv_no_cache(__global TYPE *I, // [Num Channels, Seq Length]
                   __global TYPE *W, // [Num Filters, Num Channels, K]
                   __global TYPE *B, // [Num Filters]
                   __global TYPE *O, // [Num Filters, Out Len]
                   const int IN, const int K, const int C, const int F, int f, int t) {

    TYPE acc = B[f];
    for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
            acc += I[c * IN + t + k] * W[f * C * K + c * K + k];
        }
    }
    return acc;
}
__kernel void conv1_sparse(__global TYPE *I, //[IN, C]
                           __global int *II, //[IN]
                           __global TYPE *W, //[F, C, K]
                           __global TYPE *B, //[F]
                           __global TYPE *O, //[OUT, F]
                           __global int *OO, //[OUT]
                           const int IN, const int K, const int C, const int F) {
    int f = get_global_id(0);
    int t = get_global_id(1);
    int gt = get_global_size(1);

    // OUT = IN - K + 1, iterations = (OUT + gt - 1) / gt
    for (int i = 0; i < (IN - K + gt) / gt; i++) {
        TYPE acc = B[f];
        int id = i * gt + t;
        int ii = id;
        int root = II[id];
        do {
            for (int c = 0; c < C; c++) {
                acc += W[f * C * K + c * K + root - II[ii]];
            }
            ii++;
        } while (II[ii] < root + K);
        O[id * F + f] = acc;
    }
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
    TYPE result = conv_no_cache(I, W, B, O, IN, K, C, F, f, t);
    if (f >= F || t >= OUT)
        return;
    O[f * OUT + t] = result;
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
    TYPE result = conv_no_cache(I, W, B, O, IN, K, C, F, f, t);
    if (f >= F || t >= OUT)
        return;
    O[f * OUT + t] = result > 0.0f ? result : 0.0f;
}
