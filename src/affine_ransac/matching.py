"""Pair up points from two sets (e.g. design contacts and SEM contacts).

- match_points(): mutual nearest neighbours within a gate. Correct only while the offset between
  the two sets is well below half the pitch: the gate rejects pairs, it never chooses them.
- match_with_shift(): for offsets of any size up to search_nm. Finds the shift b − a at which
  most points line up (within tolerance_nm), then pairs with match_points at that shift (D48).
- consistent_choice(): for many neighbouring tiles at once: picks each tile's shift among its
  scored_shifts so that neighbours agree, and the lattice offset of the whole group by the total
  score (D49). Replaces the per-tile smallest-shift tie-break, which failed on real data.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


def match_points(a: np.ndarray, b: np.ndarray, max_distance: float) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one pairs between point sets a (N, 2) and b (M, 2).

    a[i] and b[j] are paired when each is the other's nearest point and they are
    closer than max_distance. Points without such a partner are left out, e.g. a
    design contact whose SEM contact was cut off by the image edge.

    Returns:
        (ia, ib): index arrays of equal length; a[ia[k]] is paired with b[ib[k]].
    """
    if len(a) == 0 or len(b) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)  # (N, M)
    nearest_b = distances.argmin(axis=1)  # for each a, its nearest b
    nearest_a = distances.argmin(axis=0)  # for each b, its nearest a

    ia = np.arange(len(a))
    mutual = nearest_a[nearest_b] == ia
    close = distances[ia, nearest_b] < max_distance
    keep = mutual & close
    return ia[keep], nearest_b[keep]


@dataclass
class ShiftMatch:
    ia: np.ndarray      # pairs: a[ia[k]] is paired with b[ib[k]]
    ib: np.ndarray
    shift: np.ndarray   # (2,) shift b − a used for pairing, nm (NaN if nothing matched)
    score: int          # number of pairs at that shift
    ambiguous: bool     # another candidate shift matched almost as many points (see choose_shift)


def candidate_shifts(a: np.ndarray, b: np.ndarray, search_nm: float, tolerance_nm: float,
                     max_candidates: int = 9) -> np.ndarray:
    """Up to max_candidates shifts b − a (K, 2) that line up many points, most votes first.

    Every pair closer than search_nm votes for its difference b[j] − a[i]; a shift's votes are the
    differences within tolerance_nm of it. Peaks are taken greedily, each suppressing the
    differences within 2 × tolerance_nm. On a periodic array the true shift and the shifts one
    pitch away all get many votes, so several candidates are kept and scored by choose_shift.
    """
    near = cKDTree(b).query_ball_point(a, search_nm)
    diffs = np.array([b[j] - a[i] for i, js in enumerate(near) for j in js]).reshape(-1, 2)
    if len(diffs) == 0:
        return np.empty((0, 2))
    tree = cKDTree(diffs)
    votes = tree.query_ball_point(diffs, tolerance_nm, return_length=True)
    taken = np.zeros(len(diffs), bool)
    peaks = []
    for k in np.argsort(-votes, kind="stable"):
        if taken[k]:
            continue
        taken[tree.query_ball_point(diffs[k], 2 * tolerance_nm)] = True
        peaks.append(diffs[k])
        if len(peaks) == max_candidates:
            break
    return np.array(peaks)


def choose_shift(shifts: np.ndarray, scores: np.ndarray, tie_fraction: float = 0.1) -> tuple[int, bool]:
    """Index of the shift to use, and whether the choice was ambiguous.

    Tie-break for ONE set on its own (user, D48): shifts scoring at least (1 − tie_fraction) × the
    best score count as equally good matches; of those, the SMALLEST shift is taken. This is right
    only while the true offset is below half the pitch; on real data neighbouring tiles then jumped
    one row, so tiles are now chosen together by consistent_choice (D49). Ambiguous = more than
    one shift counted as a match.
    """
    near_ties = np.flatnonzero(scores >= (1 - tie_fraction) * scores.max())
    chosen = near_ties[np.argmin(np.linalg.norm(shifts[near_ties], axis=1))]
    return int(chosen), len(near_ties) > 1


def scored_shifts(a: np.ndarray, b: np.ndarray, search_nm: float, tolerance_nm: float) -> list[ShiftMatch]:
    """Every candidate shift of a (N, 2) against b (M, 2), refined and scored, best score first.

    1. candidate_shifts: shifts at which many points line up (up to search_nm).
    2. Each is refined (mean b − a of its pairs) and scored: the number of mutual nearest pairs
       within tolerance_nm after the shift (match_points on a + shift).
    Candidates without any pair are left out; ambiguous is False here (set by the chooser).
    """
    if len(a) == 0 or len(b) == 0:
        return []
    matches = []
    for shift in candidate_shifts(a, b, search_nm, tolerance_nm):
        ia, ib = match_points(a + shift, b, tolerance_nm)
        if len(ia) == 0:
            continue
        shift = (b[ib] - a[ia]).mean(axis=0)  # refine on the pairs, then pair again
        ia, ib = match_points(a + shift, b, tolerance_nm)
        matches.append(ShiftMatch(ia, ib, shift, len(ia), False))
    return sorted(matches, key=lambda m: -m.score)


def match_with_shift(a: np.ndarray, b: np.ndarray, search_nm: float, tolerance_nm: float,
                     tie_fraction: float = 0.1) -> ShiftMatch:
    """One-to-one pairs between a (N, 2) and b (M, 2) offset by an unknown shift up to search_nm,
    for ONE set on its own: the best-scoring shift of scored_shifts, near-ties broken by the
    smallest shift (choose_shift). Several neighbouring tiles: use consistent_choice instead.
    The shift only chooses the pairs; b[ib] − a[ia] still holds the full difference.
    tolerance_nm must stay well below half the pitch; search_nm above the largest expected shift.
    """
    matches = scored_shifts(a, b, search_nm, tolerance_nm)
    if not matches:
        return ShiftMatch(np.empty(0, int), np.empty(0, int), np.full(2, np.nan), 0, False)
    k, ambiguous = choose_shift(np.array([m.shift for m in matches]), np.array([m.score for m in matches]),
                                tie_fraction)
    chosen = matches[k]
    return ShiftMatch(chosen.ia, chosen.ib, chosen.shift, chosen.score, ambiguous)


def consistent_choice(candidates: list[list[ShiftMatch]], neighbours: list[tuple[int, int]],
                      min_score_fraction: float = 0.5, passes: int = 20) -> np.ndarray:
    """One candidate shift per tile such that neighbouring tiles agree (D49).

    candidates: per tile, its scored_shifts (best first; [] = no candidate); neighbours: (i, j)
    pairs of overlapping tiles. A tile may only use candidates scoring at least
    min_score_fraction × its best score. On a periodic array a tile's candidates lie about one
    pitch apart, while neighbours' true shifts differ by far less, so agreeing with the neighbours
    fixes the choice up to ONE lattice offset per connected group of tiles ("branch"):

    1. Per group, start from the tile with the most matched points (the seed), once for each of
       its allowed candidates, and spread outwards: each tile takes the candidate closest to the
       mean of its already decided neighbours; then every tile re-takes the candidate closest to
       the median of all its neighbours until nothing changes.
    2. Of these solutions the one with the highest TOTAL score wins: at the wrong lattice offset
       every tile loses an edge row, so the sum over many tiles decides clearly where one tile
       alone could not. Equal totals: the smaller median shift.
    Tiles without neighbours keep their best-scoring candidate. Returns (n_tiles,) indices into
    each tile's candidates; -1 for tiles without candidates.
    """
    n = len(candidates)
    allowed = [[k for k, m in enumerate(c) if m.score >= min_score_fraction * c[0].score] if c else []
               for c in candidates]
    adjacent = [[] for _ in range(n)]
    for i, j in neighbours:
        if allowed[i] and allowed[j]:
            adjacent[i].append(j)
            adjacent[j].append(i)

    def shift(t, k):
        return candidates[t][k].shift

    def closest(t, target):
        return min(allowed[t], key=lambda k: np.linalg.norm(shift(t, k) - target))

    def spread(group, seed, seed_choice):
        choice = {seed: seed_choice}
        queue = [seed]
        while queue:
            t = queue.pop(0)
            for u in adjacent[t]:
                if u not in choice:
                    decided = [shift(v, choice[v]) for v in adjacent[u] if v in choice]
                    choice[u] = closest(u, np.mean(decided, axis=0))
                    queue.append(u)
        for _ in range(passes):
            changed = False
            for t in group:
                if adjacent[t]:
                    best = closest(t, np.median([shift(v, choice[v]) for v in adjacent[t]], axis=0))
                    changed |= best != choice[t]
                    choice[t] = best
            if not changed:
                break
        return choice

    result = np.full(n, -1)
    seen = np.zeros(n, bool)
    for start in range(n):
        if seen[start] or not allowed[start]:
            continue
        group, queue = [], [start]  # connected group of tiles with candidates
        seen[start] = True
        while queue:
            t = queue.pop()
            group.append(t)
            for u in adjacent[t]:
                if not seen[u]:
                    seen[u] = True
                    queue.append(u)
        seed = max(group, key=lambda t: candidates[t][0].score)
        solutions = []
        for k in allowed[seed]:
            choice = spread(group, seed, k)
            total = sum(candidates[t][choice[t]].score for t in group)
            median_shift = np.linalg.norm(np.median([shift(t, choice[t]) for t in group], axis=0))
            solutions.append((-total, median_shift, choice))
        _, _, best = min(solutions, key=lambda s: (s[0], s[1]))
        for t, k in best.items():
            result[t] = k
    return result
