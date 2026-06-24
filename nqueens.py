"""
N-Queens v4: bitmask constraint propagation.

Instead of sets, represent constraints as three integers:
  cols  — columns already attacked (bit i = column i is occupied)
  left  — left diagonals attacking downward (shift left each row)
  right — right diagonals attacking downward (shift right each row)

Available columns = all_ones & ~(cols | left | right)
Extract next candidate with: bit = candidates & -candidates (lowest set bit)
Clear it with: candidates &= candidates - 1

No hashing, no heap allocation — just integer bitwise ops per node.
Still uses bilateral symmetry from v3.
"""

import time


def count_solutions(n):
    full = (1 << n) - 1  # bitmask with all n columns set

    def solve(cols, left, right):
        if cols == full:
            return 1
        count = 0
        candidates = full & ~(cols | left | right)
        while candidates:
            bit = candidates & -candidates          # isolate lowest set bit
            candidates &= candidates - 1            # clear it
            count += solve(
                cols | bit,
                (left  | bit) << 1,
                (right | bit) >> 1,
            )
        return count

    # Bilateral symmetry: search left half of row 0, double each subtree
    count = 0
    for col in range(n // 2):
        bit = 1 << col
        sub = solve(bit, bit << 1, bit >> 1)
        count += sub * 2

    if n % 2 == 1:
        bit = 1 << (n // 2)
        count += solve(bit, bit << 1, bit >> 1)

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
