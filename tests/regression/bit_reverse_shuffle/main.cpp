#include "./bra.hpp"
#define RT_CHECK(_expr)                                                                  \
    do {                                                                                 \
        int _ret = _expr;                                                                \
        if (0 == _ret)                                                                   \
            break;                                                                       \
        printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);                         \
        cleanup();                                                                       \
        exit(-1);                                                                        \
    } while (false)

template <unsigned char LOG_N> bool is_reversed(int a, int index) {
    return reverse<LOG_N>(index) == a;
}

template <unsigned char LOG_N> void generate(int *test_array) {
    for (int i = 0; i < 1 << LOG_N; i++) {
        test_array[i] = i;
    }
}
#define LOG_N_TEST 4
int main() {
    int test_array[1 << LOG_N_TEST];
    generate<LOG_N_TEST>(test_array);
    int errors = 0;
    Shuffle<int, LOG_N_TEST>::apply(test_array);
    for (int i = 0; i < 1 << LOG_N_TEST; i++) {
        if (!is_reversed<LOG_N_TEST>(test_array[i], i) && (errors < 100)) {
            printf("error at %d: expected %d, got %d\n", i, reverse<LOG_N_TEST>(i),
                   test_array[i]);
        }
    }
    if (errors == 0) {
        printf("PASSED!\n");
    }
    return errors;
}
