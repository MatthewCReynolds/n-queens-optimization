"""
N-Queens v6: Numba JIT compilation via LLVM.

Numba compiles the bitmask solver to native machine code inside the Python
process — no unsigned binary issues, no ctypes overhead. The @njit decorator
triggers LLVM compilation on first call; subsequent calls run as native code.
Same bitmask algorithm as v4 + bilateral symmetry from v3.
"""

import time
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
def _count(n):
    full = (1 << n) - 1
    count = 0
    for col in range(n // 2):
        bit = 1 << col
        sub = _solve(bit, bit << 1, bit >> 1, full)
        count += sub * 2
    if n % 2 == 1:
        bit = 1 << (n // 2)
        count += _solve(bit, bit << 1, bit >> 1, full)
    return count


def count_solutions(n):
    return int(_count(n))


if __name__ == "__main__":
    # Warm up JIT compilation before the clock starts
    _count(1)

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
