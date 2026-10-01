"""Pair up points from two sets (e.g. design contacts and SEM contacts)."""

import numpy as np


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
