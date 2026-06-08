#ifndef TYPE
#define TYPE float
#endif
__kernel void conv1(__global TYPE *I, // [Num Channels, Seq Length]
                    __global TYPE *W, // [Num Filters, Num Channels, K]
                    __global TYPE *B, // [Num Filters]
                    __global TYPE *O, // [Num Filters, Out Len]
                    const int SeqLength, const int K, const int NumChannels,
                    const int NumFilters) {
    int f = get_global_id(0); // filter
    int t = get_global_id(1); // timestamp

    int OutLen = (SeqLength - K) + 1;
    if (f >= NumFilters || t >= OutLen)
        return;
    TYPE acc = B[f];
    for (int c = 0; c < NumChannels; c++) {
        for (int k = 0; k < K; k++) {
            acc += I[c * SeqLength + t + k] * W[f * NumChannels * K + c * K + k];
        }
    }
    O[f * OutLen + t] = acc;
}
__kernel void conv1_relu(__global TYPE *I, // [Num Channels, Seq Length]
                         __global TYPE *W, // [Num Filters, Num Channels, K]
                         __global TYPE *B, // [Num Filters]
                         __global TYPE *O, // [Num Filters, Out Len]
                         const int SeqLength, const int K, const int NumChannels,
                         const int NumFilters) {
    int f = get_global_id(0); // filter
    int t = get_global_id(1); // timestamp

    int OutLen = (SeqLength - K) + 1;
    if (f >= NumFilters || t >= OutLen)
        return;
    TYPE acc = B[f];
    for (int c = 0; c < NumChannels; c++) {
        for (int k = 0; k < K; k++) {
            acc += I[c * SeqLength + t + k] * W[f * NumChannels * K + c * K + k];
        }
    }
    O[f * OutLen + t] = acc > 0.0f ? acc : 0.0f;
}

__kernel void relu(__global TYPE *I, // [Num Channels, Seq Length]
                   __global TYPE *O  // [Num Filters, Out Len]
) {
    int x = get_global_id(0); // filter
    O[x] = I[x] > 0.0f ? I[x] : 0.0f;
}
