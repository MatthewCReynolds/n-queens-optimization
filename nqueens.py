"""
N-Queens v3: bilateral symmetry halves the search tree.

Every solution with the first queen at column c has a mirror solution
with the first queen at column (n-1-c). So we only search the left half
of row 0 and double the count. For odd N, the center column is searched
separately without doubling.
"""

import time


def count_solutions(n):
    cols = set()
    diag1 = set()
    diag2 = set()
    count = 0

    def solve(row):
        nonlocal count
        if row == n:
            count += 1
            return
        for col in range(n):
            d1, d2 = row - col, row + col
            if col in cols or d1 in diag1 or d2 in diag2:
                continue
            cols.add(col)
            diag1.add(d1)
            diag2.add(d2)
            solve(row + 1)
            cols.remove(col)
            diag1.remove(d1)
            diag2.remove(d2)

    # Search only the left half of row 0, mirror each result
    for col in range(n // 2):
        d1, d2 = -col, col
        cols.add(col); diag1.add(d1); diag2.add(d2)
        before = count
        solve(1)
        count = before + (count - before) * 2  # each solution has a mirror
        cols.remove(col); diag1.remove(d1); diag2.remove(d2)

    # For odd N, the center column has no mirror — count it once
    if n % 2 == 1:
        col = n // 2
        d1, d2 = -col, col
        cols.add(col); diag1.add(d1); diag2.add(d2)
        solve(1)
        cols.remove(col); diag1.remove(d1); diag2.remove(d2)

    return count


if __name__ == "__main__":
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
