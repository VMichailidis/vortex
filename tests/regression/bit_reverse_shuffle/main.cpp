/*
 * main.cpp  –  Vortex host driver for the libra shuffle
 *
 * Mirrors CULIBRAShuffle::apply() from the CUDA version:
 *
 *   1. Generate the list of off-diagonal page pairs and diagonal pages
 *      using the same gen_page / bit_reverse logic.
 *   2. Allocate device buffers (input data, page index array, scratch pool).
 *   3. For each of the two passes (off-diagonal, diagonal):
 *        a. Upload the page-index list.
 *        b. Upload a kernel_arg_t that describes this pass.
 *        c. Upload & start the kernel binary.
 *        d. Wait for completion.
 *   4. Download the result and verify the bit-reversal permutation.
 *
 * CUDA concept              →  Vortex equivalent
 * ──────────────────────────────────────────────
 * cudaMalloc                →  vx_mem_alloc
 * cudaMemcpy H→D            →  vx_copy_to_dev
 * cudaMemcpy D→H            →  vx_copy_from_dev
 * kernel<<<grid,block>>>    →  vx_upload_kernel_file + vx_start
 * cudaDeviceSynchronize     →  vx_ready_wait
 * cudaFree                  →  vx_mem_free
 */

#include "common.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <vortex.h>

/* ------------------------------------------------------------------ */
/*  Error-check macro (mirrors the vecadd example)                     */
/* ------------------------------------------------------------------ */
#define RT_CHECK(_expr)                                                                  \
    do {                                                                                 \
        int _ret = (_expr);                                                              \
        if (_ret == 0)                                                                   \
            break;                                                                       \
        printf("Error: '%s' returned %d!\n", #_expr, _ret);                              \
        cleanup();                                                                       \
        exit(-1);                                                                        \
    } while (false)

/* ------------------------------------------------------------------ */
/*  Bit-reversal helper (host-side, matches kernel-side implementation) */
/* ------------------------------------------------------------------ */
static int bit_reverse(int x, int W) {
    int r = 0;
    for (int j = 0; j < W; j++)
        r = (r << 1) | ((x >> j) & 1);
    return r;
}

/* ------------------------------------------------------------------ */
/*  gen_page – next page index in the traversal order                  */
/*  Direct port of CULIBRAShuffle::gen_page().                         */
/* ------------------------------------------------------------------ */
static unsigned int gen_page(unsigned int prev) {
    constexpr unsigned int is_odd = NUM_B_BITS % 2;
    constexpr unsigned int hmax = 1u << (NUM_B_BITS / 2);

    unsigned int mid_point = is_odd ? (hmax & prev) : 0u;
    unsigned int lo = prev & (hmax - 1);
    unsigned int rlo = bit_reverse(static_cast<int>(lo), NUM_B_BITS / 2);
    unsigned int up = prev >> ((NUM_B_BITS / 2) + is_odd);

    if (prev == (1u << NUM_B_BITS) - 1u)
        return 0; /* signal: no next value */

    if (rlo == up)
        return 0u | mid_point | (lo + 1); /* reached symmetric point → new row */
    else
        return ((up + 1) << ((NUM_B_BITS / 2) + is_odd)) | mid_point | lo;
}

/* ------------------------------------------------------------------ */
/*  Global handles (kept global so cleanup() can reach them)           */
/* ------------------------------------------------------------------ */
const char *kernel_file = "kernel.vxbin";

vx_device_h device = nullptr;
vx_buffer_h input_buf = nullptr;   /* the data array being permuted   */
vx_buffer_h pages_buf = nullptr;   /* page-index list for current pass */
vx_buffer_h scratch_buf = nullptr; /* scratch tile pool               */
vx_buffer_h krnl_buf = nullptr;
vx_buffer_h args_buf = nullptr;

static void show_usage() {
    std::cout << "Vortex Libra Shuffle Test.\n"
              << "Usage: [-k kernel_file] [-h]\n";
}

static void parse_args(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "k:h")) != -1) {
        switch (c) {
        case 'k':
            kernel_file = optarg;
            break;
        case 'h':
            show_usage();
            exit(0);
        default:
            show_usage();
            exit(-1);
        }
    }
}

void cleanup() {
    if (device) {
        vx_mem_free(input_buf);
        vx_mem_free(pages_buf);
        vx_mem_free(scratch_buf);
        vx_mem_free(krnl_buf);
        vx_mem_free(args_buf);
        vx_dev_close(device);
    }
}

/* ------------------------------------------------------------------ */
/*  run_pass – upload pages, build kernel_arg_t, launch, wait          */
/*                                                                      */
/*  Corresponds to the inner lambda `run(pages, is_diag)` in the CUDA  */
/*  CULIBRAShuffle::apply() implementation, minus the hw_concurrency   */
/*  tiling (Vortex's scheduler manages occupancy automatically).        */
/* ------------------------------------------------------------------ */
static void run_pass(const std::vector<unsigned> &pages, bool is_diag,
                     uint64_t input_addr, uint64_t scratch_addr) {
    if (pages.empty())
        return;

    uint32_t num_pages = static_cast<uint32_t>(pages.size());

    /* (Re-)allocate / resize the page-index device buffer if needed */
    if (pages_buf) {
        vx_mem_free(pages_buf);
        pages_buf = nullptr;
    }
    RT_CHECK(vx_mem_alloc(device, num_pages * sizeof(uint32_t), VX_MEM_READ, &pages_buf));

    /* Upload page indices */
    RT_CHECK(vx_copy_to_dev(pages_buf, pages.data(), 0, num_pages * sizeof(uint32_t)));

    /* Build and upload kernel argument */
    uint64_t pages_addr = 0;
    RT_CHECK(vx_mem_address(pages_buf, &pages_addr));

    kernel_arg_t karg = {};
    karg.input_addr = input_addr;
    karg.pages_addr = pages_addr;
    karg.scratch_addr = scratch_addr;
    karg.num_pages = num_pages;
    karg.is_diag = is_diag ? 1u : 0u;

    if (args_buf) {
        vx_mem_free(args_buf);
        args_buf = nullptr;
    }
    RT_CHECK(vx_upload_bytes(device, &karg, sizeof(kernel_arg_t), &args_buf));

    /* Upload kernel binary (once per pass; harmless to re-upload) */
    if (krnl_buf) {
        vx_mem_free(krnl_buf);
        krnl_buf = nullptr;
    }
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buf));

    /* Start and wait */
    RT_CHECK(vx_start(device, krnl_buf, args_buf));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
}

/* ------------------------------------------------------------------ */
/*  main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv) {
    parse_args(argc, argv);
    std::srand(42);

    /* ── 1. Open device ── */
    std::cout << "open device connection\n";
    RT_CHECK(vx_dev_open(&device));

    constexpr unsigned long N = N_ELEMS;
    const uint64_t data_bytes = N * sizeof(TYPE);

    /* ── 2. Generate page lists ── */
    std::vector<unsigned> off_diag_pages; /* pairs: src, dst, src, dst … */
    std::vector<unsigned> diag_pages;     /* self-symmetric pages         */

    {
        unsigned b = 0;
        do {
            unsigned rb = bit_reverse(static_cast<int>(b), NUM_B_BITS);
            if (b != rb) {
                off_diag_pages.push_back(b);
                off_diag_pages.push_back(rb);
            } else {
                diag_pages.push_back(b);
            }
            b = gen_page(b);
        } while (b != 0);
    }

    std::cout << "off-diagonal pages : " << off_diag_pages.size()
              << "  diagonal pages: " << diag_pages.size() << "\n";

    /* ── 3. Allocate host reference data ── */
    std::vector<TYPE> h_input(N);
    std::vector<TYPE> h_result(N);
    for (unsigned long i = 0; i < N; ++i)
        h_input[i] = static_cast<TYPE>(std::rand()) / RAND_MAX;

    /* ── 4. Allocate device buffers ── */
    std::cout << "allocate device memory\n";
    RT_CHECK(vx_mem_alloc(device, data_bytes, VX_MEM_READ_WRITE, &input_buf));

    /*
     * Scratch pool: one tile (BUF_ELEMS × sizeof(TYPE)) per concurrent block.
     * We size it for the larger of the two passes.
     */
    uint32_t max_pages =
        static_cast<uint32_t>(std::max(off_diag_pages.size(), diag_pages.size()));
    uint64_t scratch_bytes = static_cast<uint64_t>(max_pages) * BUF_ELEMS * sizeof(TYPE);
    RT_CHECK(vx_mem_alloc(device, scratch_bytes, VX_MEM_READ_WRITE, &scratch_buf));

    uint64_t input_addr = 0;
    uint64_t scratch_addr = 0;
    RT_CHECK(vx_mem_address(input_buf, &input_addr));
    RT_CHECK(vx_mem_address(scratch_buf, &scratch_addr));

    std::cout << "dev_input=0x" << std::hex << input_addr << "\n";
    std::cout << "dev_scratch=0x" << std::hex << scratch_addr << std::dec << "\n";

    /* ── 5. Upload input data ── */
    std::cout << "upload input data\n";
    RT_CHECK(vx_copy_to_dev(input_buf, h_input.data(), 0, data_bytes));

    /* ── 6. Off-diagonal pass (mirrors run(pages, false)) ── */
    std::cout << "run off-diagonal pass\n";
    run_pass(off_diag_pages, /*is_diag=*/false, input_addr, scratch_addr);

    /* ── 7. Diagonal pass (mirrors run(diag, true)) ── */
    std::cout << "run diagonal pass\n";
    run_pass(diag_pages, /*is_diag=*/true, input_addr, scratch_addr);

    /* ── 8. Download result ── */
    std::cout << "download result\n";
    RT_CHECK(vx_copy_from_dev(h_result.data(), input_buf, 0, data_bytes));

    /* ── 9. Verify: result[bit_reverse(i)] should equal input[i] ── */
    std::cout << "verify result\n";
    int errors = 0;
    for (unsigned long i = 0; i < N; ++i) {
        unsigned long ri = bit_reverse(static_cast<int>(i), LOG_N);
        if (h_result[ri] != h_input[i]) {
            if (errors < 20)
                printf("*** error: [%lu] expected input[%lu]=%f, "
                       "got result[%lu]=%f\n",
                       i, i, (double)h_input[i], ri, (double)h_result[ri]);
            ++errors;
        }
    }

    /* ── 10. Cleanup ── */
    std::cout << "cleanup\n";
    cleanup();

    if (errors) {
        std::cout << "Found " << std::dec << errors << " errors!\n";
        std::cout << "FAILED!\n";
        return 1;
    }

    std::cout << "PASSED!\n";
    return 0;
}
