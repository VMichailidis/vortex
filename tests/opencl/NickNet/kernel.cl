// #include <cstdio>
// #include <opencl-c-base.h>
#ifndef TYPE
#define TYPE float
#endif
#ifndef LS0
#define LS0 1
#endif
#ifndef LS1
#define LS1 32
#endif
#define SPARSE_LEN 64
typedef struct Sparse_block {
    unsigned C;
    unsigned char len;
    unsigned short size;
    unsigned samples[SPARSE_LEN];
    unsigned char mask[SPARSE_LEN];
    unsigned ptx[SPARSE_LEN];
    TYPE data[]; // Dynamically allocate val buffer based on channel count and samples
} Sparse_block;

typedef enum rob_status { FREE, PENDING, FINISHED } rob_status;

#define ROB_ENTRIES 10

__global typedef struct {
    bool lock_head;
    bool lock_tail;
    unsigned char head;
    unsigned char tail;

    rob_status status[ROB_ENTRIES]; // init to free
    Sparse_block entries[ROB_ENTRIES];
} ROB;

__global Sparse_block *pop(ROB *rob) {
    if (rob->status[rob->tail] == FINISHED) {
        rob->status[rob->tail] = FREE;
        rob->tail = (rob->tail + 1) % ROB_ENTRIES;
    }
}

unsigned char alloc(ROB *rob) {
    if (rob->status[rob->head] == FREE) {
        rob->status[rob->head] = PENDING;
        unsigned result = rob->head;
        rob->head = (rob->head + 1) % ROB_ENTRIES;
    }
}

__kernel void dispatcher(ROB *out, ROB *in) {
    __global Sparse_block *src = pop(in);
    __global Sparse_block *dst = &out->entries[alloc(out)];
    // conv(W, B, K, dest, src);
}

// __kernel void compress(ROB *r){
// }

TYPE conv(__global TYPE *W, __global TYPE *B, const int K, __global Sparse_block *src) {
    unsigned f = get_global_id(0);
    unsigned tr = get_global_id(1);
    unsigned char sample = tr;
    TYPE acc = B[f];

    while (sample < src->len - K + 1 && src->samples[sample] < (src->samples[tr] + K)) {
        unsigned char mask = src->mask[sample];
        while (mask) {
            unsigned char channel = __builtin_ffs(mask);
            unsigned char c = channel - 1;
            unsigned offset = __builtin_popcount(~(~0 << c) & src->mask[sample]);
            unsigned k_pos = src->samples[sample] - src->samples[tr];
            acc +=
                W[f * src->C * K + c * K + k_pos] * src->data[src->ptx[sample] + offset];

            mask = mask & (~0 << channel);
        }
        sample++;
    }
    return acc;
}

// TODO: implement merging of results
// IMPORTANT: merging cannot happen at the element level only. channels need to be
// compacted also
//
// Gets a block output by the conv kernel and generates the mask for each sample
void gen_mask(__global Sparse_block *block) {
    int f = get_global_id(0);
    int id = get_global_id(1);
    if (f == 0 && id < block->len) {
        unsigned char m = 0;
        unsigned pos = id * block->C;
        for (unsigned c = 0; c < block->C; c++)
            if (block->data[pos + c] != 0)
                m |= 1 << c;

        block->mask[id] = m;
    }
    return;
}
//
// gets the output of gen_mask and packs each element to the left hand side of its
// address subspace [c0, 0, c2] -> [c0, c2, c2]
void pack_samples(__global Sparse_block *block) {
    int f = get_global_id(0);
    int id = get_global_id(1);
    if (f != 0 || id >= block->len)
        return;
    unsigned char m = block->mask[id];
    unsigned pos = id * block->C;
    for (unsigned c = 0; c < block->C; c++) {
        unsigned char offset = __builtin_popcount(~(~0 << c) & m);
        block->data[pos + offset] = block->data[pos + c];
    }
}

// Pack blocks of 2^N samples
// Each block is assumed packed with blocks[i] elements
// len is the number of samples
void concat(__global Sparse_block *block, __global unsigned char *block_len,
            __global unsigned *block_size, unsigned N) {
    unsigned id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    id = get_global_id(1);
    if (id >= block->len)
        return;
    if (N == 0) {
        unsigned pop = __builtin_popcount(block->mask[id]);
        block_size[id] = pop;
        block_len[id] = pop > 0 ? 1 : 0;

        return;
    }
    unsigned b = block->len / (1 << N); // number of groups
    if (id >= b * (1 << (N - 1)))
        return;

    unsigned group = 2 * id / (1 << N); // each group employs 2^N / 2 work items
    unsigned group_size = 1 << (N - 1);
    unsigned item = id % group_size;
    unsigned group_offset_l = group << group_size;         // offset of left block
    unsigned group_offset_r = group_offset_l + group_size; // offset of right block
    unsigned len_l = block_len[group_offset_l];            // packed length of left block
    unsigned len_r = block_len[group_offset_r];            // packed length of right block
    unsigned len_new = len_l + len_r;
    unsigned size_l = group_offset_l > block->len
                          ? 0
                          : block_size[group_offset_l]; // packed size of left block
    unsigned size_r = group_offset_r > block->len
                          ? 0
                          : block_size[group_offset_r]; // packed size of right block
    unsigned size_new = size_l + size_r;
    unsigned shift_delta = (1 << (N - 1)) - size_l;

    if (item == 0 && group_offset_l < block->len) {
        block_len[group_offset_l] = len_new;
        block_size[group_offset_l] = size_new;
    }
    if (item < len_r) {
        block->samples[group_offset_l + len_l + item] =
            block->samples[group_offset_r + item];
        block->mask[group_offset_l + len_l + item] = block->mask[group_offset_r + item];

        block->ptx[group_offset_l + len_l + item] =
            block->ptx[group_offset_r + item] - shift_delta;
    }
    for (int i = 0; i + item < size_r; i += group_size) {
        block->data[group_offset_l + size_l + i + item] =
            block->data[group_offset_r + i + item];
    }
}

void compress(__global Sparse_block *block, __global unsigned char *blocks,
              __global unsigned *data_blocks) {

    unsigned iterations = 0;
    unsigned len = block->len;
    while (len != 0) {
        len >>= 1;
        iterations++;
    } // calculate the ceil(log_2(len))

    for (unsigned i = 0; i < iterations; i++) {
        concat(block, blocks, data_blocks, i);
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    if (get_global_id(0) == 0 && get_global_id(1) == 0) {
        block->size = data_blocks[0];
        block->len = blocks[0];
    }
}

__kernel void conv_relu(__global TYPE *W, __global TYPE *B, const int K, const unsigned F,
                        __global Sparse_block *src, __global Sparse_block *dst,
                        // helper buffers of size SPARSE_LEN
                        __global unsigned char *blocks, __global unsigned *data_blocks) {
    unsigned char f = get_global_id(0);
    unsigned char tr = get_global_id(1);
    if (f == 0 && tr == 0)
        printf("Starting Convolution with %d threads\n", src->len - K + 1);
    if (tr < src->len - K + 1) {
        TYPE result = conv(W, B, K, src);
        result = result < 0 ? 0.0f : result;
        dst->data[tr * F + f] = result;
        if (f == 0) {
            dst->ptx[tr] = tr * F;
            dst->samples[tr] = src->samples[tr];
        }
    }
    if (f == 0 && tr == 0) {
        dst->len = src->len - K + 1;
        dst->C = F;
        printf("Generating Masks\n");
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    gen_mask(dst);
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (f == 0 && tr == 0)
        printf("Packing Output\n");
    barrier(CLK_GLOBAL_MEM_FENCE);

    pack_samples(dst);
    if (f == 0 && tr == 0)
        printf("Compressing Output\n");
    barrier(CLK_GLOBAL_MEM_FENCE);
    compress(dst, blocks, data_blocks);
    barrier(CLK_GLOBAL_MEM_FENCE);
}
