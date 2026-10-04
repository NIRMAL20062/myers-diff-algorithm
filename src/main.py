"""Minimal line diff (Part A) and changed-character highlighting (Part B).

Usage:
    python src/main.py lines     A B
    python src/main.py highlight A B

Both commands use Myers' O(ND) algorithm (Myers 1986, "An O(ND) Difference
Algorithm and Its Variations"). The same `myers` function works on any
sequence: lists of lines for Part A, strings of characters for Part B.
"""

import sys
from array import array


# ---------------------------------------------------------------------------
# Reading input
# ---------------------------------------------------------------------------

def read_lines(path):
    """Read a file as raw bytes and split it into lines (without the \\n)."""
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    if lines[-1] == b"":
        lines.pop()  # a final \n creates no extra line; an empty file has no lines
    return lines


# ---------------------------------------------------------------------------
# Myers' algorithm
# ---------------------------------------------------------------------------

def myers(a, b):
    """Return the matched pairs (x, y), with a[x] == b[y], of a shortest edit
    script from a to b. Pairs are in increasing order of x and of y.

    Edit graph: a point (x, y) means "a[:x] and b[:y] are done". Moving right
    deletes a[x], moving down inserts b[y], and a diagonal move keeps a[x] ==
    b[y] for free. Diagonal k holds the points with x - y == k.

    Round d finds, for every diagonal k in -d, -d+2, ..., d, the furthest x
    reachable with exactly d edits. v[offset + k] stores that x.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []  # nothing can match; the caller deletes/inserts everything

    offset = n + m                 # shift so that v[offset + k] works for k < 0
    v = [0] * (2 * offset + 2)
    trace = []                     # trace[d][i] = v for diagonal -d + 2i after round d

    for d in range(n + m + 1):
        for k in range(-d, d + 1, 2):
            # Choose the neighbour diagonal that got further in round d-1.
            if k == -d or (k != d and v[offset + k - 1] < v[offset + k + 1]):
                x = v[offset + k + 1]       # move down from diagonal k+1 (insert)
            else:
                x = v[offset + k - 1] + 1   # move right from diagonal k-1 (delete)
            y = x - k

            # Follow the snake: take free diagonal moves while the items match.
            while x < n and y < m and a[x] == b[y]:
                x += 1
                y += 1
            v[offset + k] = x

            if x >= n and y >= m:
                return backtrack(trace, d, n, m)

        # Save only the diagonals written in this round, -d, -d+2, ..., d (not the
        # whole of v), as compact 4-byte C ints instead of Python int objects.
        trace.append(array("i", v[offset - d: offset + d + 1: 2]))

    raise AssertionError("unreachable: d = n + m always reaches (n, m)")


def backtrack(trace, last_d, x, y):
    """Walk back from (x, y) = (n, m) to (0, 0) through the saved rounds and
    collect the diagonal moves (the matched pairs)."""
    pairs = []
    for d in range(last_d, 0, -1):
        prev = trace[d - 1]         # round d-1: diagonals -(d-1), -(d-1)+2, ..., d-1
        k = x - y
        i = (k + d) // 2            # prev[i] is diagonal k+1, prev[i-1] is diagonal k-1

        # Make the same choice as the forward pass did in round d.
        if k == -d or (k != d and prev[i - 1] < prev[i]):
            prev_k, prev_x = k + 1, prev[i]       # we came down (an insert)
        else:
            prev_k, prev_x = k - 1, prev[i - 1]   # we came right (a delete)
        prev_y = prev_x - prev_k

        # Undo the snake of round d: every diagonal move is a matched pair.
        while x > prev_x and y > prev_y:
            x -= 1
            y -= 1
            pairs.append((x, y))
        # Undo the single edit of round d.
        x, y = prev_x, prev_y

    # Round 0 is a snake from (0, 0) with no edit before it.
    while x > 0 and y > 0:
        x -= 1
        y -= 1
        pairs.append((x, y))

    pairs.reverse()                 # we collected them from the end; one O(n) reverse
    return pairs


def matched_pairs(a, b):
    """Same result as myers(a, b) but faster on real files. Two shortcuts,
    neither of which changes the number of edits:

    1. A common prefix and a common suffix are always kept, so only the middle
       goes to Myers.
    2. An item that never appears in the other sequence can never be kept, so
       it is hidden from Myers and simply deleted or inserted.
    """
    n, m = len(a), len(b)

    start = 0
    while start < n and start < m and a[start] == b[start]:
        start += 1
    end = 0
    while end < n - start and end < m - start and a[n - 1 - end] == b[m - 1 - end]:
        end += 1

    in_a = set(a[start:n - end])
    in_b = set(b[start:m - end])
    a_index = [i for i in range(start, n - end) if a[i] in in_b]
    b_index = [j for j in range(start, m - end) if b[j] in in_a]

    middle = myers([a[i] for i in a_index], [b[j] for j in b_index])

    pairs = [(i, i) for i in range(start)]
    pairs += [(a_index[x], b_index[y]) for x, y in middle]
    pairs += [(n - end + t, m - end + t) for t in range(end)]
    return pairs


# ---------------------------------------------------------------------------
# Part B: changed characters inside a line pair
# ---------------------------------------------------------------------------

def changed_ranges(kept, length):
    """Turn the sorted kept positions into "start-end" ranges of the rest."""
    ranges = []
    start = 0
    for p in kept + [length]:
        if p > start:
            ranges.append(f"{start}-{p}")
        start = p + 1
    return ",".join(ranges) if ranges else "."


def highlight_line(old_line, new_line):
    """The "? old | new" line for one paired - and + line."""
    # Decode to str so that indexing counts Unicode code points, not bytes.
    old = old_line.decode("utf-8", "surrogateescape")
    new = new_line.decode("utf-8", "surrogateescape")
    pairs = matched_pairs(old, new)
    old_ranges = changed_ranges([x for x, _ in pairs], len(old))
    new_ranges = changed_ranges([y for _, y in pairs], len(new))
    return f"? {old_ranges} | {new_ranges}\n".encode()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def render(a, b, pairs, highlight):
    """Build the whole output. Between two kept lines there is one change
    block: print all its deletes first, then all its inserts."""
    out = []
    i = j = 0
    for x, y in pairs + [(len(a), len(b))]:   # the sentinel flushes the last block
        deleted = a[i:x]
        inserted = b[j:y]
        for line in deleted:
            out.append(b"-" + line + b"\n")
        for t, line in enumerate(inserted):
            out.append(b"+" + line + b"\n")
            if highlight and t < len(deleted):  # the t-th + pairs with the t-th -
                out.append(highlight_line(deleted[t], line))
        if x < len(a):
            out.append(b" " + a[x] + b"\n")
        i, j = x + 1, y + 1
    return b"".join(out)


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("lines", "highlight"):
        print("usage: main.py lines|highlight A B", file=sys.stderr)
        return 2

    command, path_a, path_b = sys.argv[1:]
    try:
        a = read_lines(path_a)
        b = read_lines(path_b)
    except OSError as e:
        print(f"error: cannot read {e.filename}: {e.strerror}", file=sys.stderr)
        return 2

    pairs = matched_pairs(a, b)
    sys.stdout.buffer.write(render(a, b, pairs, command == "highlight"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
