"""
N-Queens v7: multiprocessing + Numba JIT.

The n//2 first-row subtrees from bilateral symmetry are completely
independent. We dispatch each to a worker process via ProcessPoolExecutor,
where Numba re-JITs and runs natively. On an 8-12 core M-series Mac that's
another ~8-12x on top of v6's 33x JIT speedup.
"""

import time
from concurrent.futures import ProcessPoolExecutor
from numba import njit


@njit
def _solve(cols, left, right, full):
    if cols == full:
        return 1
    count = 0
    candidates = full & ~(cols | left | right)
    while candidates:
        bit = candidates & -candidates
        candidates &= candidates - 1
        count += _solve(cols | bit, (left | bit) << 1, (right | bit) >> 1, full)
    return count


@njit
def _solve_from(col, n):
    """Count solutions with the first queen fixed at (row=0, col=col)."""
    full = (1 << n) - 1
    bit  = 1 << col
    return _solve(bit, bit << 1, bit >> 1, full)


def _worker(args):
    col, n = args
    # Each worker process JITs on first call, then runs natively.
    return int(_solve_from(col, n))


def count_solutions(n):
    half = n // 2
    tasks = [(col, n) for col in range(half)]

    with ProcessPoolExecutor() as pool:
        subtree_counts = list(pool.map(_worker, tasks))

    total = sum(subtree_counts) * 2

    # Center column for odd N — no mirror partner.
    if n % 2 == 1:
        total += _worker((n // 2, n))

    return total


if __name__ == "__main__":
    # Warm up Numba in the main process (workers JIT independently).
    _solve_from(0, 4)

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
