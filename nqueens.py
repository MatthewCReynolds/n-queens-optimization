"""
N-Queens v2: O(1) conflict checking with sets.

Track occupied columns, / diagonals, and \\ diagonals as sets.
No more walking up the board on every placement — just O(1) membership tests.
"""

import time


def count_solutions(n):
    cols = set()
    diag1 = set()  # (row - col) is constant along each / diagonal
    diag2 = set()  # (row + col) is constant along each \\ diagonal
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

    solve(0)
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
