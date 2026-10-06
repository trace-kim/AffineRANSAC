"""Pair up points from two sets (e.g. design contacts and SEM contacts).

- match_points(): mutual nearest neighbours within a gate. Correct only while the offset between
  the two sets is well below half the pitch: the gate rejects pairs, it never chooses them.
- match_with_shift(): for offsets of any size up to search_nm. Finds the shift b − a at which
  most points line up (within tolerance_nm), then pairs with match_points at that shift (D48).
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

    Tie-break (user, D48): shifts scoring at least (1 − tie_fraction) × the best score count as
    equally good matches; of those, the SMALLEST shift is taken. This is right while the true
    offset is below half the pitch. Ambiguous = more than one shift counted as a match. Kept as a
    separate function so the rule can be replaced (e.g. by consistency with the neighbour tiles).
    """
    near_ties = np.flatnonzero(scores >= (1 - tie_fraction) * scores.max())
    chosen = near_ties[np.argmin(np.linalg.norm(shifts[near_ties], axis=1))]
    return int(chosen), len(near_ties) > 1


def match_with_shift(a: np.ndarray, b: np.ndarray, search_nm: float, tolerance_nm: float,
                     tie_fraction: float = 0.1) -> ShiftMatch:
    """One-to-one pairs between a (N, 2) and b (M, 2) offset by an unknown shift up to search_nm.

    1. candidate_shifts: shifts at which many points line up.
    2. Each candidate is refined (mean b − a of its pairs) and scored: the number of mutual
       nearest pairs within tolerance_nm after the shift (match_points on a + shift).
    3. choose_shift picks one (best score; near-ties: the smallest shift).
    The shift only chooses the pairs; b[ib] − a[ia] still holds the full difference.
    tolerance_nm must stay well below half the pitch; search_nm above the largest expected shift.
    """
    nothing = ShiftMatch(np.empty(0, int), np.empty(0, int), np.full(2, np.nan), 0, False)
    if len(a) == 0 or len(b) == 0:
        return nothing
    candidates = candidate_shifts(a, b, search_nm, tolerance_nm)
    if len(candidates) == 0:
        return nothing

    matches = []
    for shift in candidates:
        ia, ib = match_points(a + shift, b, tolerance_nm)
        if len(ia):
            shift = (b[ib] - a[ia]).mean(axis=0)  # refine on the pairs, then pair again
            ia, ib = match_points(a + shift, b, tolerance_nm)
        matches.append((ia, ib, shift))
    scores = np.array([len(ia) for ia, _, _ in matches])
    if scores.max() == 0:
        return nothing
    k, ambiguous = choose_shift(np.array([m[2] for m in matches]), scores, tie_fraction)
    ia, ib, shift = matches[k]
    return ShiftMatch(ia, ib, np.asarray(shift, float), int(scores[k]), ambiguous)
