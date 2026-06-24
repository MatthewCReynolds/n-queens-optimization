"""
N-Queens v9: unit propagation + Numba thread-parallel 2-row dispatch.

Two new techniques replacing v8's approach:

1. Unit propagation (singleton forcing):
   If the next row has exactly ONE valid column, that move is forced —
   no branching needed. Detect via x & (x-1) == 0 in O(1). Chain through
   consecutive forced moves without any recursive calls. Near-zero overhead,
   high payoff near the leaves of the search tree where rows get highly
   constrained. (Standard CSP unit propagation / DPLL unit clause rule.)

2. Numba @njit(parallel=True) with prange — no multiprocessing:
   Replace ProcessPoolExecutor with Numba's OpenMP thread pool. Tasks
   (first two queen positions) are dispatched to threads inside the JIT —
   zero process-spawn overhead, shared memory, starts in microseconds.
   For N=19: ~144 tasks across ~8 cores vs v7/v8's 9 tasks.
   This also eliminates the ~18s of process-spawn overhead eaten by N=1-16
   in v7/v8 — those now run single-threaded Numba in <0.01s each.
"""

import time
import numpy as np
from numba import njit, prange


@njit
def _solve(cols, left, right, full, depth, n):
    # Unit propagation: fast-path through forced placements
    while True:
        if cols == full:
            return 1
        cands = full & ~(cols | left | right)
        if not cands:
            return 0
        if cands & (cands - 1):  # more than one bit → must branch
            break
        # Exactly one candidate — forced, no branching
        cols  |= cands
        left   = (left  | cands) << 1
        right  = (right | cands) >> 1
        depth += 1

    count = 0
    while cands:
        bit    = cands & -cands          # lowest set bit
        cands &= cands - 1              # clear it
        nc = cols | bit
        nl = (left  | bit) << 1
        nr = (right | bit) >> 1
        # Forward checking: prune if any future row becomes empty
        ok = True
        fl, fr = nl, nr
        for _ in range(n - depth - 1):
            if not (full & ~(nc | fl | fr)):
                ok = False
                break
            fl <<= 1
            fr >>= 1
        if ok:
            count += _solve(nc, nl, nr, full, depth + 1, n)
    return count


@njit(parallel=True)
def _count_parallel(tasks, n):
    """Dispatch (col0, col1) pairs to Numba threads via prange."""
    full   = (1 << n) - 1
    counts = np.zeros(tasks.shape[0], dtype=np.int64)
    for i in prange(tasks.shape[0]):
        col0  = tasks[i, 0]
        col1  = tasks[i, 1]
        bit0  = 1 << col0
        bit1  = 1 << col1
        cols  = bit0 | bit1
        left  = ((bit0 << 1) | bit1) << 1
        right = ((bit0 >> 1) | bit1) >> 1
        counts[i] = _solve(cols, left, right, full, 2, n)
    return counts


def count_solutions(n):
    if n == 1:
        return 1
    full = (1 << n) - 1

    half_tasks   = []  # col0 < n//2, results doubled
    center_tasks = []  # col0 = n//2 for odd n, not doubled

    for col0 in range(n // 2):
        bit0  = 1 << col0
        avail = full & ~(bit0 | (bit0 << 1) | (bit0 >> 1))
        c = avail
        while c:
            bit1  = c & -c
            c    &= c - 1
            half_tasks.append((col0, bit1.bit_length() - 1))

    if n % 2 == 1:
        col0  = n // 2
        bit0  = 1 << col0
        avail = full & ~(bit0 | (bit0 << 1) | (bit0 >> 1))
        c = avail
        while c:
            bit1  = c & -c
            c    &= c - 1
            center_tasks.append((col0, bit1.bit_length() - 1))

    all_tasks = half_tasks + center_tasks
    if not all_tasks:
        return 0

    tasks_arr = np.array(all_tasks, dtype=np.int64)
    results   = _count_parallel(tasks_arr, n)

    total  = int(results[:len(half_tasks)].sum()) * 2
    total += int(results[len(half_tasks):].sum())
    return total


if __name__ == "__main__":
    # Warm up Numba compilation before the clock starts
    _count_parallel(np.array([[0, 2]], dtype=np.int64), 4)

    deadline = time.time() + 60.0
    n = 1
    while time.time() < deadline:
        start = time.time()
        count = count_solutions(n)
        elapsed = time.time() - start
        print(f"N={n:>2}  solutions={count:>10}  time={elapsed:.3f}s")
        if time.time() >= deadline:
            break
        n += 1
    print(f"\nReached N={n} within 60 seconds.")
