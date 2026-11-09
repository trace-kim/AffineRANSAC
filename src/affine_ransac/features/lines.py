"""Find horizontal lines in an SEM image and measure their edges (docs/SPEC.md S2, D153).

A line edge says where it is across itself, here its height y, and nothing along itself. Each line has an upper
edge (the space above it: the edge faces up) and a lower edge (it faces down), measured at every column of the image
as y(x). Where a line runs out of the image (it touches the top or bottom border) that edge is missing in that column.

- detect_lines(): the same threshold as for contacts (features.contact.contact_mask; dark_lines as dark_contacts). A
  line is a region of the mask that runs from the image's left border to its right border; its edges are the topmost
  and bottommost row of the region in each column. Other regions are left out (counted).
- refine_line_edges(): each edge point moves up or down its own column to the edge by the rule used for contacts
  (features.edges: mid-level, steepest gradient, threshold; each line has its own levels, as each contact), on the
  image blurred across the lines only. So each column's edge comes from its own pixels: the profile along the edge
  is not smoothed, and the noise of neighbouring columns stays independent, which column_noise_px relies on.
- edge_profiles(), roughness_px(), column_noise_px(): the profile along each edge (the edge minus the straight line
  fitted along it: the edge roughness, with the noise), its rms over the image, and the noise of one column's edge.
- in_image_lines(): lines found in an upscaled image (features.contact.upscaled_image) at the image's own columns and
  in its own pixels.

Pixel frame: (x, y) = (column, row), y pointing down, pixel centres at whole numbers (docs/SPEC.md §3).
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from affine_ransac.features.contact import contact_mask
from affine_ransac.features.edges import RULES, crossings, edge_offsets, level_offsets, sample_profiles, \
    without_isolated


@dataclass
class LineEdges:
    """The lines found in one image, from the top down. upper[k, x], lower[k, x]: y (px) of line k's upper and lower
    edge at column x of the image; NaN where the line runs out of the image in that column."""
    upper: np.ndarray  # (L, C)
    lower: np.ndarray  # (L, C)
    threshold: float   # the detection's threshold (grey level)
    left_out: int = 0  # regions of the mask that do not run from the left border to the right border

    @property
    def edges(self) -> np.ndarray:
        """(2L, C) every edge: the upper ones, then the lower ones."""
        return np.vstack([self.upper, self.lower])


def detect_lines(image: np.ndarray, dark_lines: bool = True, blur_sigma: float = 1.5,
                 method: str = "otsu") -> LineEdges:
    """The lines of a grayscale uint8 image, at the pixel rows of the threshold mask (module docstring)."""
    mask, threshold = contact_mask(image, dark_lines, blur_sigma, method)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4 if method == "band" else 8)
    rows, cols = mask.shape
    left, width = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_WIDTH]
    across = [label for label in range(1, count) if left[label] == 0 and left[label] + width[label] == cols]
    upper, lower, middle = [], [], []
    for label in across:
        top, height = stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_HEIGHT]
        region = labels[top:top + height] == label  # a connected region from the left to the right border has
        first = top + region.argmax(axis=0)         # pixels in every column: its topmost
        last = top + height - 1 - region[::-1].argmax(axis=0)  # and bottommost row in each
        upper.append(np.where(first > 0, first, np.nan))  # on the image's first or last row: no edge there
        lower.append(np.where(last < rows - 1, last, np.nan))
        middle.append(np.mean(first + last))
    order = np.argsort(middle)  # from the top down
    return LineEdges(np.array(upper, float).reshape(-1, cols)[order], np.array(lower, float).reshape(-1, cols)[order],
                     float(threshold), count - 1 - len(across))


def line_interior(signed: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> float:
    """Median of `signed` over the pixels of one line: in each column from its upper to its lower edge (the image's
    border where an edge is missing)."""
    rows = np.arange(signed.shape[0])[:, None]
    inside = (rows >= np.nan_to_num(upper, nan=0)) & (rows <= np.nan_to_num(lower, nan=signed.shape[0] - 1))
    return float(np.median(signed[inside]))


def refine_line_edges(image: np.ndarray, lines: LineEdges, dark_lines: bool = True, search_px: float = 5.0,
                      step_px: float = 0.25, blur_sigma: float = 1.0, rule: str = "mid-level") -> LineEdges:
    """The lines with each edge point moved along its column (up from an upper edge, down from a lower one, searched
    from -search_px to +search_px in steps of step_px) to its edge by `rule`, on the image blurred across the lines
    only (Gaussian along y, sigma blur_sigma; module docstring)."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, not {rule!r}")
    image = image.astype(np.float32)
    smoothed = gaussian_filter1d(image, blur_sigma, axis=0) if blur_sigma > 0 else image
    sign = 1.0 if dark_lines else -1.0  # bright lines: the rules on the negative image, as for contacts
    signed = sign * smoothed
    offsets = np.arange(-search_px, search_px + step_px / 2, step_px)
    refined = {"upper": [], "lower": []}
    for upper, lower in zip(lines.upper, lines.lower):
        if not (np.isfinite(upper).any() or np.isfinite(lower).any()):  # the line fills the image's height: no edge
            refined["upper"].append(upper)
            refined["lower"].append(lower)
            continue
        found = {"upper": (upper, -1.0), "lower": (lower, 1.0)}  # each edge and its way out of the line (in y)
        profiles = {side: outward_profiles(signed, y, outward, offsets) for side, (y, outward) in found.items()}
        if rule == "steepest gradient":
            shifts = {side: edge_offsets(p, offsets) for side, p in profiles.items()}
        elif rule == "threshold":
            shifts = {side: crossings(p, offsets, sign * lines.threshold) for side, p in profiles.items()}
        else:  # mid-level: half-way between the line's interior and its band (one band over both of its edges)
            both = level_offsets(np.vstack([profiles["upper"], profiles["lower"]]), offsets,
                                 line_interior(signed, upper, lower))
            shifts = {"upper": both[:len(profiles["upper"])], "lower": both[len(profiles["upper"]):]}
        for side, (y, outward) in found.items():
            shift = shifts[side] if rule == "steepest gradient" else without_isolated(shifts[side], closed=False)
            moved = y.copy()
            moved[np.isfinite(y)] += outward * shift
            refined[side].append(moved)
    shape = (-1, image.shape[1])
    return LineEdges(np.array(refined["upper"]).reshape(shape), np.array(refined["lower"]).reshape(shape),
                     lines.threshold, lines.left_out)


def outward_profiles(signed: np.ndarray, y: np.ndarray, outward: float, offsets: np.ndarray) -> np.ndarray:
    """(M, T) profiles of one edge y (C,) at the M columns where it is found, along its column, going out of the
    line (outward -1: up, from an upper edge; +1: down)."""
    columns = np.flatnonzero(np.isfinite(y))
    if not len(columns):
        return np.empty((0, len(offsets)), np.float32)
    points = np.column_stack([columns, y[columns]]).astype(float)
    return sample_profiles(signed, points, np.tile([0.0, outward], (len(columns), 1)), offsets)


def in_image_lines(lines: LineEdges, factor: int) -> LineEdges:
    """Lines found in an image upscaled by `factor`, at the columns of the image itself and in its pixels. Column x
    of the image lies at factor * x + (factor - 1) / 2 of the upscaled one: there (odd factor), or half-way between
    the two columns beside it (even factor)."""
    if factor == 1:
        return lines
    x = factor * np.arange(lines.upper.shape[1] // factor)
    left, right = x + (factor - 1) // 2, x + factor // 2

    def back(y):
        return ((y[:, left] + y[:, right]) / 2 + 0.5) / factor - 0.5
    return LineEdges(back(lines.upper), back(lines.lower), lines.threshold, lines.left_out)


def edge_profiles(lines: LineEdges) -> np.ndarray:
    """(2L, C) each edge (LineEdges.edges) minus the straight line fitted along it: its profile along, the edge
    roughness with the noise; NaN where the edge is missing (and all of an edge found in fewer than 2 columns)."""
    edges = lines.edges
    profiles = np.full_like(edges, np.nan)
    x = np.arange(edges.shape[1])
    for k, y in enumerate(edges):
        have = np.isfinite(y)
        if have.sum() >= 2:
            tilt, height = np.polyfit(x[have], y[have], 1)
            profiles[k] = y - (height + tilt * x)
    return profiles


def roughness_px(lines: LineEdges) -> float:
    """The rms of every edge's profile along (edge_profiles), over the image: the edge roughness with the noise
    (px); NaN without any edge."""
    profiles = edge_profiles(lines)
    profiles = profiles[np.isfinite(profiles)]
    return float(np.sqrt(np.mean(profiles ** 2))) if len(profiles) else np.nan


def column_noise_px(lines: LineEdges) -> float:
    """The noise of one column's edge position (px), from the edges found in neighbouring columns: the rms of their
    differences / sqrt(2), over the image. Two neighbouring columns see nearly the same roughness and independent
    noise; roughness finer than a few columns adds to it (an upper bound), and a denoised or upscaled image, whose
    neighbouring columns share noise, makes it read low. NaN without two neighbouring edge points."""
    steps = np.diff(edge_profiles(lines), axis=1)
    steps = steps[np.isfinite(steps)]
    return float(np.sqrt(np.mean(steps ** 2) / 2)) if len(steps) else np.nan
