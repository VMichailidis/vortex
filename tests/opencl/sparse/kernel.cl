#ifndef TYPE
#define TYPE float
#endif
__kernel void conv1(__global TYPE *I, // [Num Channels, Seq Length]
                    __global TYPE *W, // [Num Filters, Num Channels, K]
                    __global TYPE *B, // [Num Filters]
                    __global TYPE *O, // [Num Filters, Out Len]
                    const int SeqLength, const int K, const int Stride,
                    const int NumChannels, const int NumFilters) {
    int f = get_global_id(0); // filter
    int t = get_global_id(1); // timestamp

    int OutLen = (SeqLength - K) / Stride + 1;
    if (f >= NumFilters || t >= OutLen)
        return;

    TYPE acc = B[f];
    // if (f == 1 && t == 1)
    for (int c = 0; c < NumChannels; c++) {
        for (int k = 0; k < K; k++) {
            // printf("[%d,%d]: Accessing I:%x, W:%x, B:%x, O:%x\n", f, t,
            //        &(I[c * SeqLength + t * Stride + k]),
            //        &(W[f * NumChannels * K + c * K + k]), &(B[f]), &(O[f * out_len +
            //        t]));
            acc += I[c * SeqLength + t * Stride + k] * W[f * NumChannels * K + c * K + k];
        }
    }
    O[f * OutLen + t] = acc;
}
