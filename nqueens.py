"""
N-Queens v10: MRV (Minimum Remaining Values) + Numba thread-parallel.

Core idea: instead of placing queens row-by-row (rows 0, 1, 2, ...),
at each step choose the UNPLACED ROW with the fewest valid columns
remaining. This is the "fail-first" MRV heuristic from constraint
satisfaction (Russell & Norvig §6.3, Haralick & Elliott 1980).

Why it matters: adjacent queens immediately constrain nearby rows.
MRV detects when those rows are empty (dead end) BEFORE reaching them
in row order — potentially skipping entire subtrees. For N-queens, two
queens placed in the middle of the board can block a distant row
completely; row-by-row backtracking wouldn't discover this until it
actually reached that row.

State per node: avail[r] = bitmask of still-valid columns for row r.
After placing queen at (r0, bit):
    delta = |r - r0|
    avail[r] &= ~(bit | (bit << delta) | (bit >> delta))
This propagates column, left-diagonal, and right-diagonal constraints
to ALL unplaced rows simultaneously, not just the next one.

Unit propagation is implicit: MRV will select a row with avail=1 (one
option) before any row with more options. Those forced placements chain
without any branch overhead.

No explicit forward checking needed — avail[r]==0 is checked during
MRV selection and propagation (prune immediately on domain wipeout).
"""

import time
import numpy as np
from numba import njit, prange


@njit
def _popcount(x):
    c = 0
    while x:
        c += 1
        x &= x - 1
    return c


@njit
def _solve_mrv(avail, placed, depth, n, full):
    if depth == n:
        return 1

    # MRV: find the unplaced row with the smallest domain
    best_row = -1
    best_cnt = n + 1
    for r in range(n):
        if not placed[r]:
            cnt = _popcount(avail[r])
            if cnt == 0:
                return 0      # domain wipeout — dead end
            if cnt < best_cnt:
                best_cnt = cnt
                best_row = r

    # Save avail state before we try candidates
    saved = avail.copy()
    placed[best_row] = True
    count = 0

    cands = saved[best_row]
    while cands:
        bit    = cands & -cands      # lowest candidate
        cands &= cands - 1

        # Reset avail to pre-branch state
        avail[:] = saved

        # Propagate: queen placed at (best_row, col=log2(bit))
        valid = True
        for r in range(n):
            if not placed[r]:
                d = best_row - r
                if d < 0:
                    d = -d
                removed = bit | ((bit << d) & full) | (bit >> d)
                avail[r] = avail[r] & ~removed & full
                if avail[r] == 0:
                    valid = False
                    break

        if valid:
            count += _solve_mrv(avail, placed, depth + 1, n, full)

    avail[:] = saved
    placed[best_row] = False
    return count


@njit(parallel=True)
def _count_parallel(tasks, n):
    """Dispatch (col0, col1) pre-placed pairs to Numba threads via prange."""
    full   = (1 << n) - 1
    counts = np.zeros(tasks.shape[0], dtype=np.int64)

    for i in prange(tasks.shape[0]):
        col0 = tasks[i, 0]
        col1 = tasks[i, 1]
        bit0 = np.int64(1) << col0
        bit1 = np.int64(1) << col1

        avail  = np.full(n, full, dtype=np.int64)
        placed = np.zeros(n, dtype=np.bool_)

        # Pre-place queen 0 at (row 0, col0)
        placed[0] = True
        avail[0]  = np.int64(0)
        for r in range(1, n):
            d       = r
            removed = bit0 | ((bit0 << d) & full) | (bit0 >> d)
            avail[r] = avail[r] & ~removed & full

        # Pre-place queen 1 at (row 1, col1) if still valid
        if avail[1] & bit1:
            placed[1] = True
            avail[1]  = np.int64(0)
            for r in range(n):
                if not placed[r]:
                    d       = 1 - r
                    if d < 0:
                        d = -d
                    removed = bit1 | ((bit1 << d) & full) | (bit1 >> d)
                    avail[r] = avail[r] & ~removed & full

            counts[i] = _solve_mrv(avail, placed, 2, n, full)

    return counts


def count_solutions(n):
    if n == 1:
        return 1
    full = (1 << n) - 1

    half_tasks   = []
    center_tasks = []

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
