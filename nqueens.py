"""
N-Queens v8: forward checking + 2-row dispatch + Numba + multiprocessing.

Two simultaneous upgrades over v7:

1. Forward checking (Russell & Norvig §6.3 / arc consistency look-ahead):
   After tentatively placing each queen, scan ALL remaining rows to verify
   each still has at least one valid column. Any future row with zero options
   means this branch can never succeed — prune it immediately without
   recursing. Adds O(n) work per node but eliminates entire subtrees the
   pure bitmask solver would explore.

2. Two-row dispatch: fix BOTH row 0 and row 1 queen positions before
   dispatching to worker processes. For N=19 this yields ~144 tasks vs v7's
   9, near-perfect load balance across cores. The longest v7 task (central
   first-row column) now splits into ~16 sub-tasks of similar size.
"""

import time
from concurrent.futures import ProcessPoolExecutor
from numba import njit


@njit
def _solve_fc(cols, left, right, full, depth, n):
    if cols == full:
        return 1
    count = 0
    candidates = full & ~(cols | left | right)
    while candidates:
        bit = candidates & -candidates
        candidates &= candidates - 1
        nc = cols | bit
        nl = (left | bit) << 1
        nr = (right | bit) >> 1
        # Forward checking: verify every future row still has candidates
        ok = True
        fl, fr = nl, nr
        for _ in range(n - depth - 1):
            if not (full & ~(nc | fl | fr)):
                ok = False
                break
            fl <<= 1
            fr >>= 1
        if ok:
            count += _solve_fc(nc, nl, nr, full, depth + 1, n)
    return count


@njit
def _solve_from_two(col0, col1, n):
    """Solve with queens pre-placed at (row=0, col0) and (row=1, col1)."""
    full = (1 << n) - 1
    bit0 = 1 << col0
    bit1 = 1 << col1
    cols  = bit0 | bit1
    left  = ((bit0 << 1) | bit1) << 1
    right = ((bit0 >> 1) | bit1) >> 1
    return _solve_fc(cols, left, right, full, 2, n)


def _worker(args):
    col0, col1, n = args
    return int(_solve_from_two(col0, col1, n))


def count_solutions(n):
    if n == 1:
        return 1
    full = (1 << n) - 1

    tasks_half   = []  # col0 < n//2 — each subtree gets doubled
    tasks_center = []  # col0 = n//2 for odd n — no doubling

    for col0 in range(n // 2):
        bit0 = 1 << col0
        avail = full & ~(bit0 | (bit0 << 1) | (bit0 >> 1))
        c = avail
        while c:
            bit1 = c & -c
            c &= c - 1
            tasks_half.append((col0, bit1.bit_length() - 1, n))

    if n % 2 == 1:
        col0 = n // 2
        bit0 = 1 << col0
        avail = full & ~(bit0 | (bit0 << 1) | (bit0 >> 1))
        c = avail
        while c:
            bit1 = c & -c
            c &= c - 1
            tasks_center.append((col0, bit1.bit_length() - 1, n))

    all_tasks = tasks_half + tasks_center
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(_worker, all_tasks))

    total  = sum(results[:len(tasks_half)]) * 2
    total += sum(results[len(tasks_half):])
    return total


if __name__ == "__main__":
    _solve_from_two(0, 2, 4)  # warm up Numba JIT before the clock

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
