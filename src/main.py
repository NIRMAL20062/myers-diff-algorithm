"""Minimal line diff (Part A) and changed-character highlighting (Part B).

Usage:
    python src/main.py lines     A B
    python src/main.py highlight A B

Both commands use Myers' O(ND) algorithm (Myers 1986, "An O(ND) Difference
Algorithm and Its Variations"). The same `myers` function works on any
sequence: lists of lines for Part A, strings of characters for Part B.

Kept items are passed around as runs (x, y, length): a[x:x+length] equals
b[y:y+length]. One run per snake is much cheaper than one pair per item when
a file has 500,000 lines and only a few edits.
"""

import sys
from array import array
from itertools import compress

# Up to this many edits, plain Myers is faster than first hiding the items
# that occur in only one file (measured on 500,000-line files).
EDIT_BUDGET = 400


# Reading input

def read_lines(path):
    """Read a file as raw bytes and split it into lines (without the \\n)."""
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    if lines[-1] == b"":
        lines.pop()  # a final \n creates no extra line; an empty file has no lines
    return lines


# Myers' algorithm

def myers(a, b, max_d):
    """Return the kept runs (x, y, length) of a shortest edit script from a to
    b, in order. Inside a run, a[x + t] == b[y + t]. Return None instead if
    that script needs more than max_d edits.

    Edit graph: a point (x, y) means "a[:x] and b[:y] are done". Moving right
    deletes a[x], moving down inserts b[y], and a diagonal move keeps a[x] ==
    b[y] for free. Diagonal k holds the points with x - y == k.

    Round d finds, for every diagonal k in -d, -d+2, ..., d, the furthest x
    reachable with exactly d edits. v[offset + k] stores that x.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []  # nothing can match; the caller deletes/inserts everything

    max_d = min(max_d, n + m)      # n + m edits always suffice
    offset = max_d + 1             # shift so that v[offset + k] works for k < 0
    v = [0] * (2 * offset + 1)     # diagonals -(max_d + 1) .. max_d + 1
    trace = []                     # trace[d][i] = v for diagonal -d + 2i after round d

    for d in range(max_d + 1):
        first, last = offset - d, offset + d    # where diagonals -d and d live in v
        # Walls just outside -d..d: no path comes from there, so diagonal -d
        # always moves down and diagonal d always moves right.
        v[first - 1] = v[last + 1] = -1

        for i in range(first, last + 1, 2):     # i = offset + k: diagonal k lives at v[i]
            # Choose the neighbour diagonal that got further in round d-1.
            if v[i - 1] < v[i + 1]:
                x = v[i + 1]                    # move down from diagonal k+1 (insert)
            else:
                x = v[i - 1] + 1                # move right from diagonal k-1 (delete)
            y = x - (i - offset)                # y = x - k

            # Follow the snake: take free diagonal moves while the items match.
            while x < n and y < m and a[x] == b[y]:
                x += 1
                y += 1
            v[i] = x

            if x >= n and y >= m:
                return backtrack(trace, d, n, m)

        # Save only the diagonals written in this round, -d, -d+2, ..., d (not the
        # whole of v), as compact 4-byte C ints instead of Python int objects.
        trace.append(array("i", v[first: last + 1: 2]))

    return None                     # more than max_d edits are needed


def backtrack(trace, last_d, x, y):
    """Walk back from (x, y) = (n, m) to (0, 0) through the saved rounds.
    Every round ends with one snake; return those snakes as runs."""
    runs = []
    for d in range(last_d, 0, -1):
        prev = trace[d - 1]         # round d-1: diagonals -(d-1), -(d-1)+2, ..., d-1
        k = x - y
        i = (k + d) // 2            # prev[i] is diagonal k+1, prev[i-1] is diagonal k-1

        # Make the same choice as the forward pass did in round d (there the
        # walls handled k == -d and k == d; here we test them directly).
        if k == -d or (k != d and prev[i - 1] < prev[i]):
            prev_k, prev_x = k + 1, prev[i]       # we came down (an insert)
        else:
            prev_k, prev_x = k - 1, prev[i - 1]   # we came right (a delete)
        prev_y = prev_x - prev_k

        # The snake of round d starts just after that edit and ends at (x, y).
        # The edit moved one step in x or in y, so the snake is the smaller gap.
        length = min(x - prev_x, y - prev_y)
        if length:
            runs.append((x - length, y - length, length))
        x, y = prev_x, prev_y       # undo the edit of round d

    if x:                           # round 0 is a snake from (0, 0), so here x == y
        runs.append((0, 0, x))

    runs.reverse()                  # we collected them from the end
    return runs


def add_run(runs, x, y, length):
    """Append a run, merging it into the previous run when the two touch."""
    if runs:
        last_x, last_y, last_length = runs[-1]
        if last_x + last_length == x and last_y + last_length == y:
            runs[-1] = (last_x, last_y, last_length + length)
            return
    runs.append((x, y, length))


def map_run(runs, a_index, b_index, x, y, length):
    """Map a run of visible items back to positions in a and b. Hidden items
    inside it split it into parts, so check it whole, else check each half."""
    first_a, first_b = a_index[x], b_index[y]
    last_a, last_b = a_index[x + length - 1], b_index[y + length - 1]
    if last_a - first_a == length - 1 and last_b - first_b == length - 1:
        add_run(runs, first_a, first_b, length)   # no hidden item inside
    else:
        half = length // 2                        # length >= 2 here
        map_run(runs, a_index, b_index, x, y, half)
        map_run(runs, a_index, b_index, x + half, y + half, length - half)


def common_prefix(a, b):
    """Length of the longest common prefix of a and b. Compares slices whose
    size doubles while they match and halves when they don't, so C code does
    the comparing instead of one Python step per item."""
    limit = min(len(a), len(b))
    i, step = 0, 1                  # always a[:i] == b[:i]
    while step:
        if i + step <= limit and a[i:i + step] == b[i:i + step]:
            i += step
            step *= 2
        else:
            step //= 2
    return i


def common_runs(a, b):
    """Same number of edits as myers(a, b, len(a) + len(b)), but faster on
    real files. Three shortcuts, none of which changes the number of edits:

    1. A common prefix and a common suffix are always kept, so only the middle
       goes to Myers.
    2. Most real diffs have few edits, so first try plain Myers with a small
       edit budget.
    3. If that is not enough: an item that never appears in the other sequence
       can never be kept, so it is hidden from Myers and simply deleted or
       inserted. Fewer edits are left for Myers, and its cost grows with the
       square of the edits.
    """
    n, m = len(a), len(b)
    start = common_prefix(a, b)
    end = common_prefix(a[start:][::-1], b[start:][::-1])  # common suffix of what is left

    runs = []
    if start:
        runs.append((0, 0, start))

    a_mid, b_mid = a[start:n - end], b[start:m - end]
    middle = myers(a_mid, b_mid, EDIT_BUDGET)
    if middle is not None:
        for x, y, length in middle:
            add_run(runs, start + x, start + y, length)
    else:
        in_a, in_b = set(a_mid), set(b_mid)
        a_visible = [item in in_b for item in a_mid]   # True if Myers should see it
        b_visible = [item in in_a for item in b_mid]
        # compress(items, flags) keeps the items whose flag is True.
        a_index = list(compress(range(start, n - end), a_visible))
        b_index = list(compress(range(start, m - end), b_visible))
        a_seen = list(compress(a_mid, a_visible))
        b_seen = list(compress(b_mid, b_visible))

        middle = myers(a_seen, b_seen, len(a_seen) + len(b_seen))
        for x, y, length in middle:
            map_run(runs, a_index, b_index, x, y, length)   # back to positions in a and b

    if end:
        add_run(runs, n - end, m - end, end)
    return runs


# Part B: changed characters inside a line pair

def changed_ranges(kept, length):
    """Turn the sorted kept (start, length) spans into "start-end" ranges of
    the positions in between."""
    ranges = []
    pos = 0
    for start, span in kept + [(length, 0)]:
        if start > pos:
            ranges.append(f"{pos}-{start}")
        pos = start + span
    return ",".join(ranges) if ranges else "."


def highlight_line(old_line, new_line):
    """The "? old | new" line for one paired - and + line."""
    # Decode to str so that indexing counts Unicode code points, not bytes.
    old = old_line.decode("utf-8", "surrogateescape")
    new = new_line.decode("utf-8", "surrogateescape")
    runs = common_runs(old, new)
    old_ranges = changed_ranges([(x, length) for x, _, length in runs], len(old))
    new_ranges = changed_ranges([(y, length) for _, y, length in runs], len(new))
    return f"? {old_ranges} | {new_ranges}\n".encode()


# Output

def prefixed(prefix, lines):
    """Output lines in one join, e.g. b"-a\\n-b\\n" for b"-" and [b"a", b"b"]."""
    return prefix + (b"\n" + prefix).join(lines) + b"\n"


def render(a, b, runs, highlight):
    """Build the whole output. Between two runs of kept lines there is one
    change block: print all its deletes first, then all its inserts."""
    out = []
    i = j = 0
    for x, y, length in runs + [(len(a), len(b), 0)]:  # the sentinel flushes the last block
        deleted = a[i:x]
        inserted = b[j:y]
        if deleted:
            out.append(prefixed(b"-", deleted))
        if highlight:
            for t, line in enumerate(inserted):
                out.append(b"+" + line + b"\n")
                if t < len(deleted):        # the t-th + pairs with the t-th -
                    out.append(highlight_line(deleted[t], line))
        elif inserted:
            out.append(prefixed(b"+", inserted))
        if length:
            out.append(prefixed(b" ", a[x:x + length]))
        i, j = x + length, y + length
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

    runs = common_runs(a, b)
    sys.stdout.buffer.write(render(a, b, runs, command == "highlight"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
