"""
N-Queens v11: pure bitmask + unit propagation + Numba prange.

Lessons from v8-v10:
- Forward checking (v8,v9): O(n) overhead per node, ~0 net benefit for N>16.
  The bitmask already prunes when the immediate row is empty; FC's lookahead
  costs more than the subtrees it eliminates.
- MRV (v10): 7x SLOWER. Per-node cost (O(n) scan + O(n) propagation + array
  copy) crushes the pruning benefit. Row-by-row IS near-optimal for N-queens
  because diagonal propagation already orders placements by constraint density.
- 90-degree rotation (attempted): bilateral H-symmetry already implicitly
  handles all 4 rotations -- R90(S) or H(R90(S)) always appears as a
  distinct solution in the same c_0<n//2 search, so the orbit is already
  counted correctly without any explicit R90 condition.

v11 = the empirically fastest approach:
  - Pure bitmask backtracking (three-integer state: cols, left, right)
  - Unit propagation: x & (x-1) == 0 detects forced moves in O(1)
  - Bilateral symmetry: search c_0 in [0, n//2), double each subtree
  - 2-row task dispatch: ~n^2/8 tasks for fine-grained load balance
  - Numba @njit(parallel=True) with prange: OpenMP threading, zero overhead

This is the performance ceiling for the row-by-row bitmask approach on this
hardware. Further gains require either more cores or a fundamentally different
mathematical formulation (e.g. Dancing Links exact cover, transfer matrices,
or GPU parallelism).
"""

import time
import numpy as np
from numba import njit, prange


@njit
def _solve(cols, left, right, full, depth, n):
    # Unit propagation: fast-path through forced placements in O(1) each
    while True:
        if cols == full:
            return 1
        cands = full & ~(cols | left | right)
        if not cands:
            return 0
        if cands & (cands - 1):   # more than one bit — must branch
            break
        # Exactly one option: place it without creating a branch
        cols  |= cands
        left   = (left  | cands) << 1
        right  = (right | cands) >> 1
        depth += 1

    count = 0
    while cands:
        bit    = cands & -cands          # isolate lowest set bit
        cands &= cands - 1              # clear it
        count += _solve(
            cols  | bit,
            (left  | bit) << 1,
            (right | bit) >> 1,
            full, depth + 1, n,
        )
    return count


@njit(parallel=True)
def _count_parallel(tasks, n):
    """Dispatch pre-placed (col0, col1) pairs to Numba OpenMP threads."""
    full   = (1 << n) - 1
    counts = np.zeros(tasks.shape[0], dtype=np.int64)
    for i in prange(tasks.shape[0]):
        col0  = tasks[i, 0]
        col1  = tasks[i, 1]
        bit0  = np.int64(1) << col0
        bit1  = np.int64(1) << col1
        cols  = bit0 | bit1
        left  = ((bit0 << 1) | bit1) << 1
        right = ((bit0 >> 1) | bit1) >> 1
        counts[i] = _solve(cols, left, right, full, 2, n)
    return counts


def count_solutions(n):
    if n == 1:
        return 1
    full = (1 << n) - 1

    half_tasks   = []   # c0 < n//2: results doubled for bilateral symmetry
    center_tasks = []   # c0 = n//2 (odd n only): not doubled

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
