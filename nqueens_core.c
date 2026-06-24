/*
 * N-Queens bitmask solver in C.
 * Same algorithm as v4: three-integer constraint state, bit & -bit candidate
 * extraction. No symmetry here — that's handled by the Python caller.
 */
#include <stdint.h>

static long long g_count;
static int       g_full;

static void solve(int cols, int left, int right) {
    if (cols == g_full) { g_count++; return; }
    int candidates = g_full & ~(cols | left | right);
    while (candidates) {
        int bit = candidates & -candidates;
        candidates &= candidates - 1;
        solve(cols | bit, (left | bit) << 1, (right | bit) >> 1);
    }
}

/* Returns total solution count for an n×n board. */
long long nqueens_count(int n) {
    g_full  = (1 << n) - 1;
    g_count = 0;

    /* Bilateral symmetry: search left half of row 0, double each subtree. */
    for (int col = 0; col < n / 2; col++) {
        int bit = 1 << col;
        long long before = g_count;
        solve(bit, bit << 1, bit >> 1);
        g_count += (g_count - before);   /* mirror */
    }

    /* Center column for odd N — no mirror. */
    if (n % 2 == 1) {
        int bit = 1 << (n / 2);
        solve(bit, bit << 1, bit >> 1);
    }

    return g_count;
}
