"""
N-Queens v5: iterative bitmask solver (explicit stack, no recursion).

Python function calls cost ~200ns each. For N=16 that's tens of millions
of calls. We replace the call stack with an explicit stack of tuples and
drive the search with a single while-loop — same bitmask math, zero call
overhead. Bilateral symmetry from v3 still halves the first-row work.
"""

import time


def count_solutions(n):
    full = (1 << n) - 1
    count = 0

    def search(cols0, left0, right0):
        nonlocal count
        if cols0 == full:
            count += 1
            return
        # Stack entries: (cols, left, right, candidates)
        candidates0 = full & ~(cols0 | left0 | right0)
        stack = [(cols0, left0, right0, candidates0)]
        while stack:
            cols, left, right, cands = stack[-1]
            if not cands:
                stack.pop()
                continue
            # Pick and clear the lowest candidate bit in-place
            bit = cands & -cands
            stack[-1] = (cols, left, right, cands & (cands - 1))
            new_cols = cols | bit
            if new_cols == full:
                count += 1
                continue
            new_left  = (left  | bit) << 1
            new_right = (right | bit) >> 1
            new_cands = full & ~(new_cols | new_left | new_right)
            if new_cands:
                stack.append((new_cols, new_left, new_right, new_cands))

    for col in range(n // 2):
        bit = 1 << col
        before = count
        search(bit, bit << 1, bit >> 1)
        count += (count - before)   # double the subtree count

    if n % 2 == 1:
        bit = 1 << (n // 2)
        search(bit, bit << 1, bit >> 1)

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
