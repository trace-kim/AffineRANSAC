"""Pair up points from two sets (e.g. design contacts and SEM contacts).

- match_points(): mutual nearest neighbours within a gate. Correct only while the offset between
  the two sets is well below half the pitch: the gate rejects pairs, it never chooses them.
- scored_shifts(): for offsets of any size up to search_nm: candidate shifts b − a, each scored
  by its MISMATCHES, the points without a partner where both sets are complete (D50).
- match_with_shift(): one set on its own: the candidate with the fewest mismatches (D48, D50).
- consistent_choice(): many neighbouring tiles at once: each tile keeps a clear winner; only
  tiles whose best candidates are tied (periodic content) follow their neighbours (D49, D50).
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
    mismatches: int     # points without a partner inside the common box (see scored_shifts)
    compared: int       # points (of a and b) inside the common box

    @property
    def mismatch_fraction(self) -> float:
        return self.mismatches / self.compared if self.compared else 1.0


def candidate_shifts(a: np.ndarray, b: np.ndarray, search_nm: float, tolerance_nm: float,
                     max_candidates: int = 9) -> np.ndarray:
    """Up to max_candidates shifts b − a (K, 2) that line up many points, most votes first.

    Every pair closer than search_nm votes for its difference b[j] − a[i]; a shift's votes are the
    differences within tolerance_nm of it. Peaks are taken greedily, each suppressing the
    differences within 2 × tolerance_nm. On a periodic array the true shift and the shifts one
    pitch away all get many votes, so several candidates are kept and scored by scored_shifts.
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


def _in_box(points: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.all((points >= low) & (points <= high), axis=1)


def _score(a: np.ndarray, b: np.ndarray, shift: np.ndarray, tolerance_nm: float) -> ShiftMatch:
    """Pairs at this shift and the mismatches inside the common box: the overlap of the bounding
    boxes of a + shift and b, shrunk by tolerance_nm, where both sets should be complete."""
    moved = a + shift
    ia, ib = match_points(moved, b, tolerance_nm)
    low = np.maximum(moved.min(axis=0), b.min(axis=0)) + tolerance_nm
    high = np.minimum(moved.max(axis=0), b.max(axis=0)) - tolerance_nm
    inside_a, inside_b = _in_box(moved, low, high), _in_box(b, low, high)
    paired_a, paired_b = np.zeros(len(a), bool), np.zeros(len(b), bool)
    paired_a[ia], paired_b[ib] = True, True
    mismatches = int((inside_a & ~paired_a).sum() + (inside_b & ~paired_b).sum())
    return ShiftMatch(ia, ib, np.asarray(shift, float), len(ia), mismatches,
                      int(inside_a.sum() + inside_b.sum()))


def scored_shifts(a: np.ndarray, b: np.ndarray, search_nm: float, tolerance_nm: float) -> list[ShiftMatch]:
    """Every candidate shift of a (N, 2) against b (M, 2), refined and scored, best first.

    1. candidate_shifts: shifts at which many points line up (up to search_nm).
    2. Each is refined (mean b − a of its pairs, then paired again within tolerance_nm).
    3. Scored by MISMATCHES (D50): inside the box where both sets are complete at that shift,
       the points of either set without a partner. At the true shift that is ~0 (only failed
       detections); one row off, every missing contact makes a point without a partner. Counting
       pairs instead would reward how much area the two sets share, which depends on where the
       offset comes from and can favour the wrong row.
    Order: lowest mismatch fraction first, then most pairs. Candidates without pairs are left out.
    """
    if len(a) == 0 or len(b) == 0:
        return []
    matches = []
    for shift in candidate_shifts(a, b, search_nm, tolerance_nm):
        first = _score(a, b, shift, tolerance_nm)
        if first.score == 0:
            continue
        refined = (b[first.ib] - a[first.ia]).mean(axis=0)
        matches.append(_score(a, b, refined, tolerance_nm))
    return sorted(matches, key=lambda m: (m.mismatch_fraction, -m.score))


def near_ties(matches: list[ShiftMatch], min_extra: float = 2.0) -> list[int]:
    """Indices of the candidates (of a scored_shifts list, best first) not clearly worse than the
    best: their extra mismatch fraction over the best, in points of the best's compared set, is
    below min_extra. One index = a clear winner; several = tied (e.g. a fully periodic tile)."""
    best = matches[0]
    return [k for k, m in enumerate(matches)
            if (m.mismatch_fraction - best.mismatch_fraction) * best.compared < min_extra]


def choose_shift(matches: list[ShiftMatch], min_extra: float = 2.0) -> tuple[int, bool]:
    """Index of the candidate to use for ONE set on its own, and whether it was tied.

    The fewest mismatches; among near_ties the SMALLEST shift (user's rule, D48), which is right
    only while the true offset is below half the pitch. Several tiles: use consistent_choice.
    """
    tied = near_ties(matches, min_extra)
    chosen = min(tied, key=lambda k: np.linalg.norm(matches[k].shift))
    return chosen, len(tied) > 1


def match_with_shift(a: np.ndarray, b: np.ndarray, search_nm: float, tolerance_nm: float,
                     min_extra: float = 2.0) -> tuple[ShiftMatch, bool]:
    """Pairs between a (N, 2) and b (M, 2) offset by an unknown shift up to search_nm, for ONE set
    on its own (scored_shifts, then choose_shift). Returns (match, tied); match.score == 0 and a
    NaN shift if nothing matched. The shift only chooses the pairs; b[ib] − a[ia] keeps the full
    difference. tolerance_nm must stay well below half the pitch."""
    matches = scored_shifts(a, b, search_nm, tolerance_nm)
    if not matches:
        return ShiftMatch(np.empty(0, int), np.empty(0, int), np.full(2, np.nan), 0, 0, 0), False
    k, tied = choose_shift(matches, min_extra)
    return matches[k], tied


def consistent_choice(candidates: list[list[ShiftMatch]], neighbours: list[tuple[int, int]],
                      min_extra: float = 2.0, passes: int = 20) -> np.ndarray:
    """One candidate shift per tile (D49, D50).

    candidates: per tile, its scored_shifts (best first; [] = no candidate); neighbours: (i, j)
    pairs of overlapping tiles.
    1. A tile whose best candidate is a clear winner (near_ties has one entry) keeps it: its own
       content, e.g. missing contacts, decides; neighbours never overrule it.
    2. Tied tiles (periodic content) choose among their tied candidates the one closest to the
       mean of their already decided neighbours, spreading out from the decided tiles; then they
       re-choose (closest to the median of all neighbours) until nothing changes.
    3. A connected group of tied tiles without any decided tile is solved from its tile with the
       most pairs, once for each of its tied candidates; the solution with the fewest total
       mismatches wins, equal: the smallest median shift (the user's rule, for the whole group).
    Tiles without neighbours: choose_shift. Returns (n_tiles,) indices into each tile's
    candidates; -1 for tiles without candidates.
    """
    n = len(candidates)
    allowed = [near_ties(c, min_extra) if c else [] for c in candidates]
    adjacent = [[] for _ in range(n)]
    for i, j in neighbours:
        if allowed[i] and allowed[j]:
            adjacent[i].append(j)
            adjacent[j].append(i)

    def shift(t, k):
        return candidates[t][k].shift

    def closest(t, target):
        return min(allowed[t], key=lambda k: np.linalg.norm(shift(t, k) - target))

    def spread(group, fixed):
        """fixed: {tile: candidate} decided up front; the other tiles follow their neighbours."""
        choice = dict(fixed)
        queue = list(fixed)
        while queue:
            t = queue.pop(0)
            for u in adjacent[t]:
                if u not in choice:
                    choice[u] = closest(u, np.mean([shift(v, choice[v]) for v in adjacent[u] if v in choice], axis=0))
                    queue.append(u)
        for _ in range(passes):
            changed = False
            for t in group:
                if t not in fixed:
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
        if len(group) == 1:
            result[start] = choose_shift(candidates[start], min_extra)[0]
            continue
        decided = {t: allowed[t][0] for t in group if len(allowed[t]) == 1}
        if decided:
            best = spread(group, decided)
        else:
            seed = max(group, key=lambda t: candidates[t][0].score)
            solutions = []
            for k in allowed[seed]:
                choice = spread(group, {seed: k})
                total = sum(candidates[t][choice[t]].mismatches for t in group)
                median_shift = np.linalg.norm(np.median([shift(t, choice[t]) for t in group], axis=0))
                solutions.append((total, median_shift, choice))
            best = min(solutions, key=lambda s: (s[0], s[1]))[2]
        for t, k in best.items():
            result[t] = k
    return result
