"""Matplotlib plots for overlap_flow.ipynb.

Positions are drawn in µm; differences and residuals in nm. Raw B − A differences are drawn
magnified by `arrow_scale`; residuals (what is left after a correction) are much smaller and
are drawn magnified by `residual_scale`. Every arrow plot has a key arrow showing 1 nm at its
own magnification, and its title says which magnification it uses.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from affine_ransac.overlap_image import shift_image

TILE_COLORS = plt.cm.tab10.colors


# ---------- small building blocks ----------

def draw_arrows(ax, xy_nm, vectors_nm, arrow_scale, color="black"):
    """Arrows at xy_nm (drawn in µm) for vectors in nm, magnified by arrow_scale."""
    return ax.quiver(xy_nm[:, 0] / 1000, xy_nm[:, 1] / 1000,
                     vectors_nm[:, 0] / 1000 * arrow_scale, vectors_nm[:, 1] / 1000 * arrow_scale,
                     angles="xy", scale_units="xy", scale=1, color=color, width=0.003)


def draw_scale_key(ax, quiver, arrow_scale, length_nm=1.0):
    """A key arrow in the lower-right corner of the plot showing how long length_nm is drawn."""
    ax.quiverkey(quiver, 0.97, 0.05, length_nm / 1000 * arrow_scale, f"{length_nm:g} nm (x{arrow_scale})",
                 labelpos="N", coordinates="axes", labelsep=0.05,
                 fontproperties={"size": 8})


def fit_view(ax, xy_nm, vectors_nm_list, arrow_scale, min_pad_um=0.1):
    """Set the view to the points and their (magnified) arrow tips, plus padding. Needed because
    matplotlib does not include arrows when autoscaling, and an overlap strip can be a single
    row of contacts, which would leave a zero-height plot."""
    points = xy_nm / 1000
    tips = [points + v / 1000 * arrow_scale for v in vectors_nm_list]
    everything = np.vstack([points, *tips])
    low, high = everything.min(axis=0), everything.max(axis=0)
    pad = max(min_pad_um, 0.05 * (high - low).max())
    ax.set_xlim(low[0] - pad, high[0] + pad)
    ax.set_ylim(low[1] - pad, high[1] + pad)


def circle_outliers(ax, xy_nm, inliers):
    outliers = xy_nm[~inliers] / 1000
    ax.scatter(outliers[:, 0], outliers[:, 1], s=200, facecolors="none", edgecolors="tab:red", lw=1.5, label="outlier")


def finish_position_axes(ax, title):
    ax.set_aspect("equal")
    ax.set_xlabel("mask x (µm)")
    ax.set_ylabel("mask y (µm)")
    ax.set_title(title, loc="left", fontsize=10)


def difference_scatter(ax, diffs, inliers, marks):
    """Each contact's B − A difference as a point (nm); marks: list of (xy, style dict) to add."""
    ax.scatter(diffs[inliers, 0], diffs[inliers, 1], color="tab:blue", s=25, label="contact (inlier)")
    ax.scatter(diffs[~inliers, 0], diffs[~inliers, 1], color="tab:red", marker="x", s=50, label="contact (outlier)")
    for xy, style in marks:
        ax.scatter([xy[0]], [xy[1]], **style)
    ax.set_aspect("equal")
    ax.set_xlabel("dx = B − A (nm)")
    ax.set_ylabel("dy = B − A (nm)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def along_strip(points_nm):
    """For an overlap strip: (position along it, axis index along, axis index across)."""
    horizontal = np.ptp(points_nm[:, 0]) >= np.ptp(points_nm[:, 1])
    return (points_nm[:, 0], 0, 1) if horizontal else (points_nm[:, 1], 1, 0)


# ---------- step 1: robust translation ----------

def plot_translation_fit(a, b, fit, arrow_scale, residual_scale, outlier_factor):
    """Before (raw differences) and after (residuals) arrows, and the difference scatter."""
    diffs = b - a
    fig = plt.figure(figsize=(13, 6.5), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[2.2, 1])
    ax_before = fig.add_subplot(grid[0, 0])
    ax_after = fig.add_subplot(grid[1, 0], sharex=ax_before, sharey=ax_before)
    ax_scatter = fig.add_subplot(grid[:, 1])

    q = draw_arrows(ax_before, a, diffs, arrow_scale)
    draw_scale_key(ax_before, q, arrow_scale)
    circle_outliers(ax_before, a, fit.inliers)
    finish_position_axes(ax_before, f"Before: B − A differences, arrows x{arrow_scale} "
                                    f"(fitted shift {fit.shift[0]:+.2f}, {fit.shift[1]:+.2f} nm)")

    q = draw_arrows(ax_after, a, fit.residuals, residual_scale)
    draw_scale_key(ax_after, q, residual_scale)
    circle_outliers(ax_after, a, fit.inliers)
    finish_position_axes(ax_after, f"After translation: residuals, arrows x{residual_scale}, "
                                   f"RMS {fit.rms():.2f} nm inliers / {fit.rms(include_outliers=True):.2f} nm all")
    fit_view(ax_before, a, [diffs * arrow_scale, fit.residuals * residual_scale], 1)

    threshold = outlier_factor * np.median(np.linalg.norm(fit.residuals, axis=1))
    difference_scatter(ax_scatter, diffs, fit.inliers,
                       [(fit.shift, dict(marker="+", s=300, color="black", lw=2, label="fitted shift"))])
    ax_scatter.add_patch(plt.Circle(fit.shift, threshold, fill=False, ls="--", color="gray", label="outlier threshold"))
    ax_scatter.legend(fontsize=8)
    ax_scatter.set_title(f"Differences of {len(a)} contacts", fontsize=10)
    plt.show()


# ---------- step 2: image registration ----------

def overlay_rgb(crop_a, crop_b):
    """A in magenta, B in green: where they line up the result is grey, misalignment shows colour."""
    def normalised(image):
        low, high = np.percentile(image, [1, 99])
        return np.clip((image - low) / (high - low), 0, 1)
    a, b = normalised(crop_a), normalised(crop_b)
    return np.dstack([a, b, a])


def plot_image_alignment(crop_a, crop_b, move_px, box, diffs, inliers, contact_shift, image_shift):
    """Overlap pixels overlaid before/after the image shift, and both shift estimates."""
    extent = np.array(box) / 1000  # (x_min, x_max, y_min, y_max) µm; row 0 of the crop is the top
    fig = plt.figure(figsize=(13, 6.5), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[2.2, 1])
    ax_before = fig.add_subplot(grid[0, 0])
    ax_after = fig.add_subplot(grid[1, 0], sharex=ax_before, sharey=ax_before)
    ax_scatter = fig.add_subplot(grid[:, 1])

    ax_before.imshow(overlay_rgb(crop_a, crop_b), extent=extent, interpolation="nearest")
    finish_position_axes(ax_before, "Before: A (magenta) and B (green) at nominal placement")
    ax_after.imshow(overlay_rgb(crop_a, shift_image(crop_b, -move_px)), extent=extent, interpolation="nearest")
    finish_position_axes(ax_after, f"After moving B back by the image shift ({move_px[0]:+.2f}, {move_px[1]:+.2f} px)")

    difference_scatter(ax_scatter, diffs, inliers, [
        (contact_shift, dict(marker="+", s=300, color="black", lw=2,
                             label=f"contacts ({contact_shift[0]:+.2f}, {contact_shift[1]:+.2f})")),
        (image_shift, dict(marker="x", s=200, color="tab:green", lw=2,
                           label=f"image ({image_shift[0]:+.2f}, {image_shift[1]:+.2f})")),
    ])
    ax_scatter.set_title("Contact differences and both shift estimates (nm)", fontsize=10)
    plt.show()


# ---------- step 3: translation + rotation ----------

def plot_rotation_fit(a, fit_t, fit_r, residual_scale):
    """Residuals after translation vs after translation + rotation, and the across-strip residual
    along the strip, where a rotation shows up as a straight-line trend."""
    position, i_along, i_across = along_strip(a)
    fig = plt.figure(figsize=(13, 9), layout="constrained")
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.2])
    ax_t = fig.add_subplot(grid[0])
    ax_r = fig.add_subplot(grid[1], sharex=ax_t, sharey=ax_t)
    ax_line = fig.add_subplot(grid[2])

    for ax, fit, name in [(ax_t, fit_t, "translation"), (ax_r, fit_r, "translation + rotation")]:
        q = draw_arrows(ax, a, fit.residuals, residual_scale)
        draw_scale_key(ax, q, residual_scale)
        circle_outliers(ax, a, fit.inliers)
        finish_position_axes(ax, f"Residuals after {name}, arrows x{residual_scale}: RMS {fit.rms():.2f} nm "
                                 f"inliers / {fit.rms(include_outliers=True):.2f} nm all")
    fit_view(ax_t, a, [fit_t.residuals, fit_r.residuals], residual_scale)

    across = fit_t.residuals[:, i_across]
    ax_line.scatter(position[fit_t.inliers] / 1000, across[fit_t.inliers], color="tab:blue", label="contact (inlier)")
    ax_line.scatter(position[~fit_t.inliers] / 1000, across[~fit_t.inliers], color="tab:red", marker="x", label="outlier")
    # A rotation θ about c moves a point at along-position s across the strip by ±θ (s − c).
    sign = 1 if i_along == 0 else -1
    s = np.linspace(position.min(), position.max(), 2)
    ax_line.plot(s / 1000, sign * fit_r.rotation * (s - fit_r.center[i_along]), color="black",
                 label=f"fitted rotation {fit_r.rotation * 1e6:+.1f} µrad")
    ax_line.axhline(0, color="gray", lw=0.8)
    ax_line.set_xlabel(f"position along the strip, mask {'xy'[i_along]} (µm)")
    ax_line.set_ylabel(f"residual across the strip, d{'xy'[i_across]} (nm)")
    ax_line.set_title("After translation: residual across the strip vs position (rotation = straight line)", loc="left", fontsize=10)
    ax_line.grid(alpha=0.3)
    ax_line.legend(fontsize=8)
    plt.show()


# ---------- all pairs ----------

def plot_all_pairs(summary):
    """Per-pair comparison of the three methods (summary: DataFrame from the notebook)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
    pair = np.arange(len(summary))

    ax = axes[0, 0]
    ax.scatter(summary.rms_translation_nm, summary.rms_rigid_nm)
    for k in pair:
        ax.annotate(str(k), (summary.rms_translation_nm[k], summary.rms_rigid_nm[k]), fontsize=7)
    top = max(summary.rms_translation_nm.max(), summary.rms_rigid_nm.max()) * 1.1
    ax.plot([0, top], [0, top], color="gray", ls="--", label="no improvement")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.set_xlabel("RMS, translation only (nm)")
    ax.set_ylabel("RMS, translation + rotation (nm)")
    ax.set_title("Does rotation help? (points well below the line = yes)", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(summary.rotation_urad, bins=20, color="tab:blue")
    ax.axvline(0, color="gray")
    ax.set_xlabel("fitted rotation (µrad)")
    ax.set_ylabel("pairs")
    ax.set_title(f"Rotation: mean {summary.rotation_urad.mean():+.1f}, std {summary.rotation_urad.std():.1f} µrad", fontsize=10)

    ax = axes[1, 0]
    ax.scatter(summary.image_minus_contact_x_nm, summary.image_minus_contact_y_nm)
    for k in pair:
        ax.annotate(str(k), (summary.image_minus_contact_x_nm[k], summary.image_minus_contact_y_nm[k]), fontsize=7)
    ax.add_patch(plt.Circle((0, 0), 0.3, fill=False, ls="--", color="gray", label="±0.3 nm (expected)"))
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("image − contact shift, x (nm)")
    ax.set_ylabel("image − contact shift, y (nm)")
    ax.set_title("Agreement of image and contact shifts", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.bar(pair, summary.matched, color="lightgray", label="matched contacts")
    ax.bar(pair, summary.outliers, color="tab:red", label="outliers")
    ax.set_xlabel("pair (row of the table)")
    ax.set_ylabel("contacts")
    ax.set_title("Matched contacts and outliers per pair", fontsize=10)
    ax.legend(fontsize=8)
    plt.show()


# ---------- stitching: before / after correction ----------

def draw_tile_outlines(ax, centers_nm, fovs_nm, tile_ids):
    for k, (center, fov, tile_id) in enumerate(zip(centers_nm, fovs_nm, tile_ids)):
        color = TILE_COLORS[k % len(TILE_COLORS)]
        corner = (center - fov / 2) / 1000
        ax.add_patch(plt.Rectangle(corner, *(fov / 1000), fill=False, color=color, lw=1))
        ax.text(*(center / 1000), tile_id, color=color, ha="center", va="center", fontsize=9)


def plot_stitch_residuals(centers, fovs, tile_ids, pair_data, corrections, arrow_scale, residual_scale):
    """All pairs' B − A differences at their mask positions, before and after the per-tile
    corrections. pair_data: list of (i, j, a, b, inliers)."""
    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(14, 7.5), sharex=True, sharey=True, layout="constrained")
    panels = [(ax_before, arrow_scale, "Before correction (nominal placement)"),
              (ax_after, residual_scale, "After per-tile translation correction")]
    vectors_by_panel = {ax_before: [], ax_after: []}
    for i, j, a, b, inliers in pair_data:
        before = b - a
        after = (b + corrections[j]) - (a + corrections[i])
        for (ax, scale, _), vectors in zip(panels, [before, after]):
            quiver = draw_arrows(ax, a, vectors, scale)
            circle_outliers(ax, a, inliers)
            vectors_by_panel[ax].append(vectors[inliers])
    for ax, scale, name in panels:
        rms = np.sqrt((np.concatenate(vectors_by_panel[ax]) ** 2).sum(axis=1).mean())
        draw_tile_outlines(ax, centers, fovs, tile_ids)
        draw_scale_key(ax, ax.collections[0], scale)
        finish_position_axes(ax, f"{name}\nB − A at every overlap contact, arrows x{scale}, RMS {rms:.2f} nm (inliers)")
    corners = np.vstack([centers - fovs / 2, centers + fovs / 2])
    fit_view(ax_before, corners, [], arrow_scale)
    plt.show()


def plot_mosaic(ax, mosaic, extent_nm, contours_by_tile, centers_by_tile, boxes, title):
    """Stitched image with each tile's contours and contact centres (one colour per tile) and
    the overlap boxes (dashed). Contours and centres are in mask nm."""
    ax.imshow(mosaic, cmap="gray", extent=np.array(extent_nm) / 1000, interpolation="nearest")
    for k, (contours, centers) in enumerate(zip(contours_by_tile, centers_by_tile)):
        color = TILE_COLORS[k % len(TILE_COLORS)]
        ax.add_collection(LineCollection([np.vstack([c, c[:1]]) / 1000 for c in contours], colors=[color], linewidths=0.6))
        ax.scatter(centers[:, 0] / 1000, centers[:, 1] / 1000, marker="+", s=20, color=color, lw=0.8)
    for x_min, x_max, y_min, y_max in boxes:
        ax.add_patch(plt.Rectangle((x_min / 1000, y_min / 1000), (x_max - x_min) / 1000, (y_max - y_min) / 1000,
                                   fill=False, ls="--", color="yellow", lw=0.8))
    finish_position_axes(ax, title)
