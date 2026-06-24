"""
N-Queens Baseline Solver (v1)
Simple backtracking — place one queen per row, check for conflicts.
"""

import time


def is_safe(board, row, col, n):
    # Check column
    for r in range(row):
        if board[r] == col:
            return False

    # Check upper-left diagonal
    r, c = row - 1, col - 1
    while r >= 0 and c >= 0:
        if board[r] == c:
            return False
        r -= 1
        c -= 1

    # Check upper-right diagonal
    r, c = row - 1, col + 1
    while r >= 0 and c < n:
        if board[r] == c:
            return False
        r -= 1
        c += 1

    return True


def solve(board, row, n, solutions):
    if row == n:
        solutions.append(board[:])
        return

    for col in range(n):
        if is_safe(board, row, col, n):
            board[row] = col
            solve(board, row + 1, n, solutions)
            board[row] = -1


def count_solutions(n):
    board = [-1] * n
    solutions = []
    solve(board, 0, n, solutions)
    return len(solutions)


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
