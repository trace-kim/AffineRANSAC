import cv2
import klayout.db as kdb
import numpy as np
import pytest
from scipy.spatial import cKDTree

from affine_ransac.io.design import load_layout
from affine_ransac.io.sem_image import load_sem_image
from affine_ransac.features.contact import (DetectedContacts, denoised_image, detect_contacts, flatten_background,
                                            in_image_pixels, upscaled_image)
from affine_ransac.features.edges import RULES, refine_edges
from affine_ransac.geometry.affine import apply_affine
from affine_ransac.overlap_fit import fit_overlap
from affine_ransac.pipeline import (MIN_AGREEMENT, DesignContacts, TileResult, capped_rms, choose_design_contacts,
                                    contact_spacing, design_outlines, find_contacts, inside_frame, measurement_pairs,
                                    overlap_checks, pair_kind, pair_residuals, placed_by_fingerprint,
                                    placed_image_by_image, prepared_image, process_tile, slipped_pairs, stitch_design,
                                    stitch_image_by_image, stitch_tiles, stitch_tiles_affine, stitch_tiles_rigid,
                                    tie_residuals, tile_stripes)
from sample_sem import render_tile
from test_affine import known_affine
from test_stitching import about, rotation

FOV, SIZE = 720.0, 512
LATTICE = np.array([(x, y) for x in np.arange(-1530, 1530, 90.0) for y in np.arange(-1530, 1530, 90.0)])
# 2 x 2 tiles, 600 nm apart: ~120 nm overlap strips. True position = nominal + stage error.
NOMINAL = np.array([[0.0, 0.0], [0.0, -600.0], [600.0, 0.0], [600.0, -600.0]])
ERRORS = np.array([[1.2, -0.7], [-3.1, 2.4], [0.5, 1.5], [2.0, -2.0]])


def write_tile(path, true_center, seed):
    ok, encoded = cv2.imencode(".jpg", render_tile(LATTICE, true_center, FOV, SIZE, seed=seed),
                               [cv2.IMWRITE_JPEG_QUALITY, 95])
    encoded.tofile(str(path))


def write_design(path, centers, size_nm=40, reversed=False, fov=FOV):
    """.oas on layer 1/0 (dbu 1 nm) with square contacts at centers (tile-local nm). Tone reversed:
    the drawn shape is the FOV frame minus the contacts (one polygon with holes)."""
    layout = kdb.Layout()
    layout.dbu = 0.001
    top = layout.create_cell("TOP")
    half = round(size_nm / 2)
    holes = kdb.Region()
    for x, y in np.round(centers).astype(int).tolist():  # KLayout needs Python ints
        holes.insert(kdb.Box(x - half, y - half, x + half, y + half))
    if reversed:
        frame = kdb.Region(kdb.Box(-round(fov / 2), -round(fov / 2), round(fov / 2), round(fov / 2)))
        holes = frame - holes
    top.shapes(layout.layer(1, 0)).insert(holes)
    layout.write(str(path))
    return load_layout(path)


def visible(points, margin=30):
    return points[np.all(np.abs(points) < FOV / 2 - margin, axis=1)]


def test_process_tile_finds_sem_and_design_contacts(tmp_path):
    write_tile(tmp_path / "tile.jpg", true_center=(0.0, 0.0), seed=1)
    write_design(tmp_path / "tile.oas", visible(LATTICE, margin=0))

    tile = process_tile(tmp_path / "tile.jpg", tmp_path / "tile.oas", (5000.0, -2000.0), (FOV, FOV), layer=(1, 0))

    assert tile.pixel_size_nm == FOV / SIZE
    assert len(tile.refined.centers) == len(tile.otsu.centers) > 20
    assert tile.design.ok and not tile.design.reversed
    np.testing.assert_allclose(sorted(map(tuple, tile.design_points_nm() - (5000.0, -2000.0))),
                               sorted(map(tuple, visible(LATTICE, margin=0))), atol=1e-9)
    # SEM points in mask nm: near lattice positions shifted by the tile centre. (render_tile's
    # wide rims leave small background gaps that are detected too; only real contacts count.)
    points = (tile.sem_points_nm() - (5000.0, -2000.0))[tile.refined.areas > 200]
    assert len(points) > 20
    nearest = np.linalg.norm(points[:, None] - LATTICE[None], axis=2).min(axis=1)
    assert nearest.max() < 1.0


def test_tile_contact_sizes_of_sem_discs_and_design_squares(tmp_path):
    write_tile(tmp_path / "tile.jpg", true_center=(0.0, 0.0), seed=1)  # discs of 40 nm diameter
    write_design(tmp_path / "tile.oas", visible(LATTICE, margin=0))  # squares of 40 nm
    tile = process_tile(tmp_path / "tile.jpg", tmp_path / "tile.oas", (0.0, 0.0), (FOV, FOV), layer=(1, 0))

    assert tile.oas_path == tmp_path / "tile.oas"
    sem, design = tile.contact_sizes((1, 0), roi_width_nm=0.0)
    assert sem.shape == (len(tile.refined.centers), 3) and design.shape == (len(tile.design.centers), 3)
    np.testing.assert_allclose(design, np.tile([1600.0, 40.0, 40.0], (len(design), 1)))
    contacts = tile.refined.areas > 200  # real contacts, not the background gaps (see above)
    np.testing.assert_allclose(np.median(sem[contacts], axis=0), [np.pi * 20 ** 2, 40, 40], rtol=0.1)
    banded, _ = tile.contact_sizes((1, 0), roi_width_nm=10.0)  # chords off the centre of a disc are shorter
    assert np.median(banded[contacts, 1]) < np.median(sem[contacts, 1])


def test_process_tile_can_read_the_design_without_matching_it(tmp_path):
    write_tile(tmp_path / "tile.jpg", true_center=(0.0, 0.0), seed=1)
    write_design(tmp_path / "tile.oas", visible(LATTICE, margin=0) + 45.0)  # half a pitch off: no match

    tile = process_tile(tmp_path / "tile.jpg", tmp_path / "tile.oas", (0.0, 0.0), (FOV, FOV), layer=(1, 0),
                        match_design=False)

    assert tile.design.ok and np.isnan(tile.design.match_fraction)
    assert len(tile.design.centers) > 0 and np.all(np.abs(tile.design.centers) < FOV / 2)  # frame contacts dropped


def test_process_tile_refines_with_the_edge_rule_asked_for(tmp_path):
    write_tile(tmp_path / "tile.jpg", true_center=(0.0, 0.0), seed=1)
    image = load_sem_image(tmp_path / "tile.jpg")

    for rule in RULES:
        tile = process_tile(tmp_path / "tile.jpg", None, (0.0, 0.0), (FOV, FOV), layer=(1, 0), refinement=rule)
        np.testing.assert_array_equal(tile.refined.centers, refine_edges(image, tile.otsu, rule=rule).centers)
    default = process_tile(tmp_path / "tile.jpg", None, (0.0, 0.0), (FOV, FOV), layer=(1, 0))
    np.testing.assert_array_equal(default.refined.centers, refine_edges(image, default.otsu, rule="mid-level").centers)
    # none: no refinement, the detected contours are what the tile is measured with (D124)
    plain = process_tile(tmp_path / "tile.jpg", None, (0.0, 0.0), (FOV, FOV), layer=(1, 0), refinement="none")
    assert plain.refined is plain.otsu and plain.seconds["refine"] < 0.01
    np.testing.assert_array_equal(plain.sem_points_nm(), plain.sem_points_nm(refined=False))


def test_find_contacts_passes_every_option_on(tmp_path):
    """D128: find_contacts gives what detect_contacts and refine_edges give with the same options; min_solidity
    applies to band only; max_area_px 0 is no limit; dark_contacts reaches both steps."""
    write_tile(tmp_path / "tile.jpg", true_center=(0.0, 0.0), seed=1)
    image = load_sem_image(tmp_path / "tile.jpg")

    detected, refined, dropped = find_contacts(
        image, detection="otsu3", detection_blur_px=1.0, min_area_px=30, max_area_px=5000,
        refinement="steepest gradient", search_px=4.0, edge_step_px=0.5, edge_blur_px=0.7, normal_span=2)
    expected = detect_contacts(image, True, 1.0, 30, 5000, "otsu3")
    np.testing.assert_array_equal(detected.centers, expected.centers)
    np.testing.assert_array_equal(refined.centers,
                                  refine_edges(image, expected, True, 4.0, 0.5, 0.7, 2, "steepest gradient").centers)
    assert dropped == 0 and len(detected.centers) > 40

    every = len(find_contacts(image, max_area_px=0)[0].centers)
    none = find_contacts(image, max_area_px=1)[0]
    assert len(none.centers) == 0 and none.left_out["large"] == every
    assert len(find_contacts(image, min_solidity=1.5)[0].centers) == every  # otsu: no solidity check
    assert len(find_contacts(image, detection="band", min_solidity=1.5)[0].centers) == 0  # unreachable for band
    _, negative, _ = find_contacts(255 - image, dark_contacts=False)
    np.testing.assert_allclose(negative.centers, find_contacts(image)[1].centers, atol=0.01)


def test_find_contacts_flattens_the_background_before_every_step():
    """D130: with background_px every step works on the flattened image (prepared_image), so one threshold, and the
    threshold edge rule, fit a background that brightens across the image; bright contacts likewise."""
    from test_contact import ramp_image
    image, centres = ramp_image()
    assert prepared_image(image, 0.0) is image
    np.testing.assert_array_equal(prepared_image(image, 20.0), flatten_background(image, 20.0))

    plain = find_contacts(image, detection="otsu")[1]
    flat = find_contacts(image, background_px=20.0, detection="otsu", refinement="threshold")[1]
    assert len(plain.centers) < len(centres) == len(flat.centers)
    bright = find_contacts(255 - image, background_px=20.0, detection="otsu", refinement="threshold",
                           dark_contacts=False)[1]
    # every contact where it is, those near the image's edges too (a mirrored ramp moved them by 0.1 px)
    for found in (flat, bright):
        assert len(found.centers) == len(centres) and cKDTree(found.centers).query(centres)[0].max() < 0.03


def test_find_contacts_upscales_the_image_with_its_options_in_pixels_of_the_image():
    """D131: with upscale the contacts are found in the image upscaled (prepared_image), with the options in pixels of
    the image as given (areas: its square), and come back in its pixel frame: the same contacts, the detected ones
    (pixel outlines) closer to where they are. Contacts 6 px across."""
    from test_edges import no_band_image
    grid = np.array([(x, y) for y in range(12, 70, 13) for x in range(12, 130, 13)], float)
    centres = grid + np.random.default_rng(3).uniform(-0.5, 0.5, grid.shape)
    image = no_band_image(centres, 3.0, noise=2)
    assert prepared_image(image, 0.0, upscale=1) is image
    np.testing.assert_array_equal(prepared_image(image, 0.0, upscale=4), upscaled_image(image, 4))

    detected, refined, _ = find_contacts(image, upscale=4)
    expected = in_image_pixels(detect_contacts(upscaled_image(image, 4), True, 1.5 * 4, 20 * 4 ** 2), 4)
    np.testing.assert_array_equal(detected.centers, expected.centers)
    plain = find_contacts(image)[0]
    assert len(detected.centers) == len(plain.centers) == len(centres)
    # the same areas on average; finer: a pixel outline's area moves in whole pixels (found: std 0.5 and 1.0 px)
    assert abs(detected.areas.mean() - plain.areas.mean()) < 0.02 * plain.areas.mean()
    assert detected.areas.std() < 0.7 * plain.areas.std()

    def distances(found):
        return cKDTree(found.centers).query(centres)[0]
    assert distances(detected).max() < 0.1 < distances(plain).max()  # found: 0.056 and 0.276 px
    assert distances(refined).max() < 0.1
    # an area limit of the image as given, at every factor
    assert len(find_contacts(image, upscale=4, min_area_px=2 * int(plain.areas.max()))[0].centers) == 0


def test_find_contacts_denoises_the_image_before_it_upscales_it(tmp_path):
    """D132: the contacts are found in the image denoised, with each method's own option, then upscaled
    (prepared_image); the contrast check sees the denoised image too."""
    write_tile(tmp_path / "tile.jpg", true_center=(0.0, 0.0), seed=1)
    image = load_sem_image(tmp_path / "tile.jpg")
    for method, option in (("median", {"median_px": 5}), ("non-local means", {"nlm_h": 15.0}),
                           ("total variation", {"tv_weight": 0.2})):
        denoised = denoised_image(image, method, **option)
        np.testing.assert_array_equal(prepared_image(image, denoise=method, upscale=2, **option),
                                      upscaled_image(denoised, 2))
        expected = find_contacts(denoised, upscale=2, min_contrast=3.0)
        found = find_contacts(image, denoise=method, upscale=2, min_contrast=3.0, **option)
        np.testing.assert_array_equal(found[1].centers, expected[1].centers)
        assert found[2] == expected[2] and len(found[1].centers) > 40


def test_with_refinement_none_the_design_tone_is_chosen_with_the_detected_centres(tmp_path, monkeypatch):
    write_tile(tmp_path / "tile.jpg", true_center=(0.0, 0.0), seed=1)
    write_design(tmp_path / "tile.oas", visible(LATTICE, margin=0))
    seen = []
    import affine_ransac.pipeline as pipeline
    real = pipeline.choose_design_contacts
    monkeypatch.setattr(pipeline, "choose_design_contacts", lambda *a, **k: seen.append(a[3]) or real(*a, **k))

    tile = process_tile(tmp_path / "tile.jpg", tmp_path / "tile.oas", (0.0, 0.0), (FOV, FOV), layer=(1, 0),
                        refinement="none")

    np.testing.assert_array_equal(seen[0], tile.sem_points_nm(refined=False))  # tile centre (0, 0): local = mask


def test_process_tile_without_oas(tmp_path):
    write_tile(tmp_path / "tile.jpg", true_center=(0.0, 0.0), seed=1)
    tile = process_tile(tmp_path / "tile.jpg", None, (0.0, 0.0), (FOV, FOV), layer=(1, 0))
    assert tile.design.centers.shape == (0, 2) and not tile.design.ok


@pytest.mark.parametrize("reversed", [False, True])
def test_design_tone_is_chosen_by_matching_the_sem(tmp_path, reversed):
    design = visible(LATTICE, margin=0)
    sem = visible(LATTICE) + (4.0, -3.0)  # stage offset; edge contacts cut off in the SEM
    layout = write_design(tmp_path / "d.oas", design, reversed=reversed)

    found = choose_design_contacts(layout, (1, 0), (FOV, FOV), sem)

    assert found.ok and found.reversed == reversed and found.match_fraction == 1.0
    np.testing.assert_allclose(sorted(map(tuple, found.centers)), sorted(map(tuple, design)), atol=1e-9)
    outlines = design_outlines(layout, (1, 0), (FOV, FOV), found.reversed)  # one per centre, same order
    np.testing.assert_allclose([o.mean(axis=0) for o in outlines], found.centers, atol=1e-9)


def test_design_tone_flagged_when_neither_tone_matches(tmp_path):
    layout = write_design(tmp_path / "d.oas", visible(LATTICE, margin=0))
    sem = np.random.default_rng(3).uniform(-300, 300, (60, 2))  # no lattice: no shift lines it up

    found = choose_design_contacts(layout, (1, 0), (FOV, FOV), sem)

    assert not found.ok and found.match_fraction < 0.5


def test_design_tone_check_finds_offsets_beyond_the_old_gate(tmp_path):
    # 47 nm offset (beyond the former 40 nm gate; pitch 90 nm, 40 nm < P / 2 along x): found by the shift search.
    layout = write_design(tmp_path / "d.oas", visible(LATTICE, margin=0))
    sem = visible(LATTICE, margin=60) + (40.0, -25.0)

    found = choose_design_contacts(layout, (1, 0), (FOV, FOV), sem)

    assert found.ok and found.match_fraction == 1.0
    np.testing.assert_allclose(found.shift_nm, (40.0, -25.0), atol=1e-6)


def test_stitch_tiles_recovers_stage_errors(tmp_path):
    points = []
    for k, (nominal, error) in enumerate(zip(NOMINAL, ERRORS)):
        write_tile(tmp_path / f"{k}.jpg", nominal + error, seed=k)
        tile = process_tile(tmp_path / f"{k}.jpg", None, nominal, (FOV, FOV), layer=(1, 0))
        points.append(tile.sem_points_nm())

    result = stitch_tiles(points, NOMINAL, np.full((4, 2), FOV), max_match_nm=10)

    # Neighbours along x and y overlap in strips; the diagonal pairs only touch at a corner.
    assert sorted((p.i, p.j) for p in result.pairs) == [(0, 1), (0, 2), (1, 3), (2, 3)]
    # A tile seen at nominal + error needs the correction +error (gauge: corrections average 0).
    np.testing.assert_allclose(result.corrections, ERRORS - ERRORS.mean(axis=0), atol=0.3)


def seen_points(errors, noise_nm=0.2, seed=0):
    """Per tile, the lattice contacts it sees (fully inside its true FOV), at nominal placement:
    a tile whose true centre is nominal + error reports contact p at p − error."""
    rng = np.random.default_rng(seed)
    points = []
    for nominal, error in zip(NOMINAL, errors):
        inside = np.all(np.abs(LATTICE - (nominal + error)) < FOV / 2 - 10, axis=1)  # cut-off contacts dropped
        points.append(LATTICE[inside] - error + rng.normal(0, noise_nm, (inside.sum(), 2)))
    return points


def test_default_gate_pairs_offsets_beyond_10_nm():
    errors = np.array([[0.0, 0.0], [12.0, -8.0], [-10.0, 6.0], [4.0, 9.0]])  # B − A up to ~20 nm

    result = stitch_tiles(seen_points(errors), NOMINAL, np.full((4, 2), FOV))

    assert len(result.pairs) == 4 and not any(r.failed for r in result.rejected)
    np.testing.assert_allclose(result.corrections, errors - errors.mean(axis=0), atol=0.3)


def test_unmatched_tile_is_reported_never_silently_zero():
    errors = np.array([[0.0, 0.0], [3.0, -2.0], [-2.0, 1.0], [40.0, 0.0]])  # tile 3: 40 nm off, beyond the gate

    with pytest.warns(UserWarning, match="SEM stitching incomplete"):
        result = stitch_tiles(seen_points(errors), NOMINAL, np.full((4, 2), FOV))

    assert sorted((r.i, r.j) for r in result.rejected if r.failed) == [(1, 3), (2, 3)]
    assert result.unplaced.tolist() == [3]
    assert np.isnan(result.corrections[3]).all()
    np.testing.assert_allclose(result.corrections[:3], errors[:3] - errors[:3].mean(axis=0), atol=0.3)


def test_inside_frame_drops_contacts_touching_the_fov():
    centers = np.array([[0.0, 0.0], [340.0, 0.0], [0.0, -330.0]])
    sizes = np.full((3, 2), 40.0)  # half size 20: 340 + 20 reaches the 360 frame edge
    np.testing.assert_array_equal(inside_frame(centers, sizes, (FOV, FOV)), [True, False, True])


def design_tile(nominal, offset, ok=True):
    """A tile whose .oas is drawn `offset` nm away from where it should be (no SEM data needed)."""
    local = LATTICE - nominal
    local = local[np.all(np.abs(local) < FOV / 2 - 10, axis=1)] - offset
    empty = DetectedContacts(np.empty((0, 2)), np.empty(0), [], 0.0)
    return TileResult(image=np.zeros((SIZE, SIZE), np.uint8), center_nm=np.asarray(nominal, float),
                      fov_nm=np.array([FOV, FOV]), pixel_size_nm=FOV / SIZE, otsu=empty, refined=empty,
                      design=DesignContacts(local, False, 1.0, ok))


def test_stitch_design_recovers_offsets_between_oas_files():
    offsets = np.array([[0.0, 0.0], [6.0, -4.0], [-5.0, 3.0], [2.0, 7.0]])

    result = stitch_design([design_tile(n, o) for n, o in zip(NOMINAL, offsets)])

    # A file drawn `offset` too far left needs +offset to line up with its neighbours.
    np.testing.assert_allclose(result.corrections, offsets - offsets.mean(axis=0), atol=1e-9)


def test_stitch_design_leaves_out_flagged_tiles():
    tiles = [design_tile(n, (0.0, 0.0), ok=(k != 3)) for k, n in enumerate(NOMINAL)]
    with pytest.warns(UserWarning, match="design stitching incomplete"):
        result = stitch_design(tiles)
    assert result.unplaced.tolist() == [3]


def test_tile_neighbours_are_the_overlapping_pairs():
    from affine_ransac.pipeline import tile_neighbours
    assert sorted(tile_neighbours(NOMINAL, np.full((4, 2), FOV))) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert tile_neighbours(np.array([[0.0, 0.0], [5000.0, 0.0]]), np.full((2, 2), FOV)) == []


def test_affine_stitching_removes_tile_scale_and_rotation_that_shifts_cannot():
    # Each tile measures with its own scale / rotation (~1 nm at the tile edge) and stage error.
    rng = np.random.default_rng(4)
    truth = [about(known_affine(*rng.normal(0, 3000, 4), t=e), c) for e, c in zip(ERRORS, NOMINAL)]
    points = []
    for t, c in zip(truth, NOMINAL):
        inside = np.all(np.abs(apply_affine(np.linalg.inv(t), LATTICE) - c) < FOV / 2 - 10, axis=1)
        points.append(apply_affine(np.linalg.inv(t), LATTICE[inside]))  # tile reports truth^-1(p)
    stitch = stitch_tiles(points, NOMINAL, np.full((4, 2), FOV))

    affines = stitch_tiles_affine(points, NOMINAL, stitch)

    rms = {name: np.sqrt((tie_residuals(points, stitch, c) ** 2).sum(axis=1).mean())
           for name, c in (("shift", stitch.corrections), ("affine", affines))}
    assert rms["shift"] > 0.2 and rms["affine"] < 1e-3
    assert len(tie_residuals(points, stitch, affines)) == sum(p.fit.inliers.sum() for p in stitch.pairs)


def test_stitch_tiles_uses_only_the_given_pairs():
    errors = np.array([[0.0, 0.0], [3.0, -2.0], [-2.0, 1.0], [1.0, 2.0]])
    with pytest.warns(UserWarning, match="could not be stitched"):
        result = stitch_tiles(seen_points(errors), NOMINAL, np.full((4, 2), FOV), only_pairs={(0, 1), (0, 2)})
    assert sorted((p.i, p.j) for p in result.pairs) == [(0, 1), (0, 2)] and result.rejected == []
    assert result.unplaced.tolist() == [3]  # its overlaps were not allowed


def serpentine(stripes=3, per_stripe=4):
    """Tile centres of `stripes` stripes of `per_stripe` tiles (2 um apart in x, 1.8 um in y), stripe by
    stripe from the bottom up, with some stage scatter in x."""
    centers = np.array([(s * 2000.0, k * 1800.0) for s in range(stripes) for k in range(per_stripe)])
    centers[:, 0] += np.random.default_rng(0).normal(0, 20, len(centers))
    return centers, np.full((len(centers), 2), 2500.0)


from affine_ransac.overlap import overlapping_pairs as overlapping  # noqa: E402


def drifting_serpentine(step_nm=17.0, pitch=100.0, fov=1200.0):
    """2 stripes x 3 tiles of a contact grid (pitch 100 nm, 300 nm overlaps), measured as a serpentine
    (stripe 0 upward: tiles 0, 1, 2; stripe 1 downward: 5, 4, 3); the stage drifts up by step_nm per image,
    so the bottom tiles of the two stripes (0 and 3, five images apart) are 85 nm apart: within
    max_match_nm (25) of one pitch. Returns the points at the nominal placement, centres, FOVs and the
    true corrections (each tile's drift removed, mean 0)."""
    centers = np.array([(x, y) for x in (0.0, 900.0) for y in (0.0, 900.0, 1800.0)])
    drift = np.zeros((6, 2))
    drift[[0, 1, 2, 5, 4, 3], 1] = step_nm * np.arange(6)  # in measurement order
    grid = np.array([(x, y) for x in np.arange(-650, 1600, pitch) for y in np.arange(-650, 2500, pitch)])
    rng = np.random.default_rng(0)
    points = []
    for center, d in zip(centers, drift):
        seen = grid[np.all(np.abs(grid - center - d) < fov / 2 - 20, axis=1)]
        points.append(seen - d + rng.normal(0, 0.3, seen.shape))  # placed at the nominal centre
    return points, centers, np.full((6, 2), fov), drift - drift.mean(axis=0)


def test_an_offset_near_one_pitch_slips_a_row_unless_guided_by_the_measurement_order():
    points, centers, fovs, truth = drifting_serpentine()
    order = measurement_pairs(centers, tile_stripes(centers))
    first = stitch_tiles(points, centers, fovs, only_pairs=order)
    np.testing.assert_allclose(first.corrections, truth, atol=0.5)  # 17 nm between consecutive images

    with pytest.warns(UserWarning):  # the middle (51 nm) and corner pairs fail, the bottom one (85 nm) slips
        plain = stitch_tiles(points, centers, fovs)
    bottom = next(p for p in plain.pairs if (p.i, p.j) == (0, 3))
    assert abs(bottom.fit.shift[1] - 15) < 1  # paired one row (100 nm) off: +15 nm instead of -85 nm
    assert slipped_pairs(plain, first.corrections, 25.0) == [(0, 3)]

    guided = stitch_tiles(points, centers, fovs, guide=first.corrections)  # every overlap pairs, no warning
    assert sorted((p.i, p.j) for p in guided.pairs) == sorted((i, j) for i, j, _ in overlapping(centers, fovs))
    np.testing.assert_allclose(guided.corrections, truth, atol=0.5)
    assert slipped_pairs(guided, first.corrections, 25.0) == []


def mosaic(errors, contacts, fov=1000.0, step=880.0, cols=3, rows=4, seed=0, noise=0.3):
    """cols x rows tiles `step` apart (overlaps fov - step = 120 nm wide). Tile k really sits at its centre +
    errors[k], so it reports the contacts it sees (fully inside its true FOV) at p − errors[k], with `noise` nm noise.
    Returns the points at the nominal placement, centres, FOVs and the true corrections (mean 0)."""
    centers = np.array([(c * step, r * step) for c in range(cols) for r in range(rows)])
    rng = np.random.default_rng(seed)
    points = []
    for center, e in zip(centers, errors):
        seen = contacts[np.all(np.abs(contacts - center - e) < fov / 2 - 10, axis=1)]
        points.append(seen - e + rng.normal(0, noise, seen.shape))
    return points, centers, np.full((len(centers), 2), fov), errors - errors.mean(axis=0)


GRID_50 = np.array([(x, y) for x in np.arange(-600, 2400, 50.0) for y in np.arange(-600, 3300, 50.0)])


def whole_pitches(corrections, truth, pitch):
    """Whether every tile is placed right up to whole pitches (the same for any two tiles' contacts)."""
    off = (corrections - truth) - (corrections - truth)[0]
    return np.allclose(off / pitch, np.round(off / pitch), atol=0.02)


def test_image_by_image_lines_up_overlaps_that_pairing_at_the_stage_position_cannot():
    errors = np.random.default_rng(1).normal(0, 20, (12, 2))  # neighbours up to ~60 nm apart: beyond half the pitch
    points, centers, fovs, truth = mosaic(errors, GRID_50)

    with pytest.warns(UserWarning):
        plain = stitch_tiles(points, centers, fovs, max_match_nm=100)
    worst = max(np.abs(r).max() for r in pair_residuals(points, plain, plain.corrections))
    assert worst > 10 and not whole_pitches(plain.corrections, truth, 50)  # overlaps disagree: the images do not line up

    guide = placed_image_by_image(points, centers, fovs, search_nm=100)
    stitch = stitch_image_by_image(points, centers, fovs, guide)
    assert all(pair_kind(centers, r.i, r.j) == "corner" and not r.failed for r in stitch.rejected)  # too few contacts
    assert len(stitch.unplaced) == 0
    worst = max(np.abs(r).max() for r in pair_residuals(points, stitch, stitch.corrections))
    assert worst < 2  # every overlap agrees (noise 0.3 nm per contact)
    assert whole_pitches(stitch.corrections, truth, 50)  # on a repeating pattern a whole pitch is not seen
    assert all(c.lines_up for c in overlap_checks(points, centers, fovs, stitch.corrections, tolerance_nm=0.5, search_nm=100))


def test_image_by_image_changes_nothing_while_the_images_lie_within_half_a_pitch():
    errors = np.random.default_rng(2).normal(0, 4, (12, 2))
    points, centers, fovs, truth = mosaic(errors, GRID_50)

    guide = placed_image_by_image(points, centers, fovs, search_nm=100)
    stitch = stitch_image_by_image(points, centers, fovs, guide)
    plain = stitch_tiles(points, centers, fovs)

    np.testing.assert_allclose(guide - guide.mean(axis=0), truth, atol=0.5)  # no image moved by a pitch
    np.testing.assert_allclose(stitch.corrections, truth, atol=0.3)
    np.testing.assert_allclose(stitch.corrections, plain.corrections, atol=0.2)


def test_image_by_image_finds_large_offsets_exactly_where_the_pattern_does_not_repeat():
    contacts = np.random.default_rng(3).uniform((-600, -600), (2400, 3300), (3000, 2))  # no two alike
    errors = np.zeros((12, 2))
    errors[5] = (70.0, -45.0)  # beyond any pitch-like spacing
    points, centers, fovs, truth = mosaic(errors, contacts)

    stitch = stitch_image_by_image(points, centers, fovs, placed_image_by_image(points, centers, fovs, search_nm=150))

    np.testing.assert_allclose(stitch.corrections, truth, atol=0.3)


def test_stitching_image_by_image_fits_the_overlaps_robustly_whatever_the_noise():
    """D139: with 1 nm contact noise (more than the 0.5 nm within which the stitching paired once, D135), every
    overlap that holds enough contacts is used and the images line up to a fraction of a nanometre."""
    errors = np.random.default_rng(1).normal(0, 20, (12, 2))
    points, centers, fovs, truth = mosaic(errors, GRID_50, noise=1.0)

    stitch = stitch_image_by_image(points, centers, fovs, placed_image_by_image(points, centers, fovs, search_nm=100))

    assert not any(r.failed for r in stitch.rejected) and whole_pitches(stitch.corrections, truth, 50)
    off = stitch.corrections - truth
    off -= np.round(off / 50) * 50  # the whole pitches a repeating pattern does not show
    assert np.abs(off - off.mean(axis=0)).max() < 1.0  # 0.5 nm measured; it scales with the noise


def test_stitching_image_by_image_leaves_out_an_overlap_that_does_not_line_up_where_the_placement_puts_it():
    """D139: tile 5 placed 20 nm off (beyond the gate): its overlaps do not line up there; they are left out (not
    matched), so the tile is not stitched, instead of a few contacts that happen to pair pulling it."""
    points, centers, fovs, truth = mosaic(np.zeros((12, 2)), GRID_50)
    guide = placed_image_by_image(points, centers, fovs, search_nm=100)
    guide[5] += (20.0, 0.0)

    with pytest.warns(UserWarning, match="stitching incomplete"):
        stitch = stitch_image_by_image(points, centers, fovs, guide)

    assert stitch.unplaced.tolist() == [5]
    assert {(r.i, r.j) for r in stitch.rejected if r.failed} == {(1, 5), (4, 5), (5, 6), (5, 9)}


def test_the_contact_spacing_is_the_pitch_of_the_array():
    points, _, _, _ = mosaic(np.zeros((12, 2)), GRID_50)
    assert abs(contact_spacing(points) - 50) < 1
    assert np.isnan(contact_spacing([np.zeros((1, 2)), np.empty((0, 2))]))


def test_the_overlap_check_names_every_overlap_of_an_image_placed_wrong():
    """D137: whatever paired them, the overlaps of an image placed wrong do not line up: by more than the gate (a
    quarter of the 50 nm pitch) its contacts fall between the other's, by less they pair but lie apart."""
    errors = np.random.default_rng(4).normal(0, 4, (12, 2))
    points, centers, fovs, truth = mosaic(errors, GRID_50)
    right = overlap_checks(points, centers, fovs, truth, tolerance_nm=1.0, search_nm=100)
    assert len(right) > 12 and all(c.lines_up for c in right)  # the 120 nm overlaps hold two rows each
    assert min(c.agreement for c in right) > 0.95 and max(c.offset_nm for c in right) < 0.5

    wrong = truth.copy()
    wrong[5] += (20.0, 0.0)
    wrong[6] += (0.0, 5.0)
    checks = {(c.i, c.j): c for c in overlap_checks(points, centers, fovs, wrong, tolerance_nm=1.0, search_nm=100)}
    assert {ij for ij, c in checks.items() if not c.lines_up} == {(c.i, c.j) for c in right if {5, 6} & {c.i, c.j}}
    assert checks[4, 5].agreement < 0.1  # 20 nm off: nothing pairs
    assert checks[2, 6].agreement > 0.95 and abs(checks[2, 6].offset_nm - 5) < 0.5  # 5 nm off: every contact pairs

    pitch = truth.copy()
    pitch[5] += (50.0, 0.0)  # a whole pitch on a repeating pattern: nothing to see, and nothing changes (D105)
    assert all(c.lines_up for c in overlap_checks(points, centers, fovs, pitch, tolerance_nm=1.0, search_nm=100))


def test_image_by_image_takes_the_shift_pairing_every_contact_not_a_smaller_one_pairing_most():
    """D138: where the pattern does not repeat (a fifth of the contacts missing), the shift one pitch smaller than the
    true one pairs most contacts too (80 %); it no longer wins for being smaller. Every image is placed exactly, also
    tile 11, whose overlap with tile 10 is one row that happens to be complete: it waits until its other neighbour has
    decided (placed clear images first)."""
    contacts = GRID_50[np.random.default_rng(5).random(len(GRID_50)) > 0.2]
    errors = np.zeros((12, 2))
    errors[5] = (40.0, 0.0)  # beyond half a pitch from its neighbours: the shift one pitch off is -10 nm
    points, centers, fovs, truth = mosaic(errors, contacts)

    guide = placed_image_by_image(points, centers, fovs, search_nm=100)

    np.testing.assert_allclose(guide - guide.mean(axis=0), truth, atol=0.3)


def test_image_by_image_does_not_depend_on_a_tolerance_near_the_noise():
    """D138: contacts measured with 1 nm noise (the stitching within 0.5 nm left tiles out) are paired within a
    quarter of the pitch and fitted robustly: every image placed, right up to whole pitches."""
    errors = np.random.default_rng(1).normal(0, 20, (12, 2))
    points, centers, fovs, truth = mosaic(errors, GRID_50, noise=1.0)

    guide = placed_image_by_image(points, centers, fovs, search_nm=100)

    assert not np.isnan(guide).any() and whole_pitches(guide - guide.mean(axis=0), truth, 50)


def test_the_overlap_check_sees_a_whole_pitch_slip_where_the_pattern_does_not_fully_repeat():
    """D140: a fifth of the contacts missing, tile 5 placed one pitch off: most of its overlap contacts still pair
    (about 80 %, above MIN_AGREEMENT), but the shift one pitch back pairs clearly more, so they do not line up. On the
    full grid the same slip lines up (every whole-pitch shift ties)."""
    contacts = GRID_50[np.random.default_rng(5).random(len(GRID_50)) > 0.2]
    points, centers, fovs, truth = mosaic(np.zeros((12, 2)), contacts)
    slipped = truth.copy()
    slipped[5] += (50.0, 0.0)

    checks = overlap_checks(points, centers, fovs, slipped, tolerance_nm=1.0, search_nm=100)

    bad = [c for c in checks if not c.lines_up]
    assert {(c.i, c.j) for c in bad} == {(1, 5), (4, 5), (5, 6), (5, 9)}  # its sides; a corner holds too few contacts
    assert all(c.better and c.agreement > MIN_AGREEMENT for c in bad)
    assert all(c.lines_up for c in overlap_checks(points, centers, fovs, truth, tolerance_nm=1.0, search_nm=100))


def test_the_fingerprint_places_every_image_exactly_where_the_mask_has_placement_errors():
    """D144: contacts 1 nm (sigma) off a 50 nm grid, measured with 0.2 nm noise. At the true shift each pair is one
    contact seen twice (they differ by the noise); a pitch off, two neighbours (they differ also by their placement
    errors). Every image is placed exactly, by its residual, where image by image is whole pitches off (they tie)."""
    errors = np.random.default_rng(1).normal(0, 20, (12, 2))  # neighbours up to ~60 nm apart: beyond half the pitch
    contacts = GRID_50 + np.random.default_rng(2).normal(0, 1.0, GRID_50.shape)
    points, centers, fovs, truth = mosaic(errors, contacts, noise=0.2)

    guide, choices = placed_by_fingerprint(points, centers, fovs, search_nm=100)
    stitch = stitch_image_by_image(points, centers, fovs, guide)
    plain = stitch_image_by_image(points, centers, fovs, placed_image_by_image(points, centers, fovs, search_nm=100))

    np.testing.assert_allclose(stitch.corrections, truth, atol=0.3)
    assert len(choices) == 11 and all(c.options > 1 and c.fitting == 1 and c.tied == 1 for c in choices)
    assert min(c.ratio for c in choices) > 3  # a pitch off: sqrt(2 (0.2² + 1²)) / (sqrt(2) 0.2), about 5 times more
    assert whole_pitches(plain.corrections, truth, 50) and np.abs(plain.corrections - truth).max() > 40
    assert all(c.lines_up for c in overlap_checks(points, centers, fovs, stitch.corrections, tolerance_nm=1.0,
                                                  search_nm=100, by_fit=True))


def test_without_placement_errors_on_the_mask_the_fingerprint_places_as_image_by_image():
    """D144: on a perfect grid every whole-pitch shift fits as well (the noise only): every image is tied, placed as
    image by image, and its overlaps line up (whole pitches do not matter there, D105)."""
    errors = np.random.default_rng(1).normal(0, 20, (12, 2))
    points, centers, fovs, truth = mosaic(errors, GRID_50)

    guide, choices = placed_by_fingerprint(points, centers, fovs, search_nm=100)
    stitch = stitch_image_by_image(points, centers, fovs, guide)

    assert len(choices) == 11 and all(c.fitting > 1 and c.tied > 1 for c in choices)
    assert whole_pitches(stitch.corrections, truth, 50)
    assert all(c.lines_up for c in overlap_checks(points, centers, fovs, stitch.corrections, tolerance_nm=1.0,
                                                  search_nm=100, by_fit=True))


def test_the_fingerprint_check_is_not_misled_by_a_contact_row_cut_by_an_image_frame():
    """D144: with placement errors on the mask, a contact row can lie along an image's frame, seen in part: at the true
    placement a contact or two of the other image have no partner, while a pitch off all pair. Counted, that shift
    pairs clearly more (image by image would not place it there); by the residuals it fits far worse."""
    errors = np.random.default_rng(3).normal(0, 20, (12, 2))
    contacts = GRID_50 + np.random.default_rng(0).normal(0, 1.0, GRID_50.shape)
    points, centers, fovs, truth = mosaic(errors, contacts, noise=0.2)

    counted = overlap_checks(points, centers, fovs, truth, tolerance_nm=1.0, search_nm=100)
    fitted = overlap_checks(points, centers, fovs, truth, tolerance_nm=1.0, search_nm=100, by_fit=True)

    assert [(c.i, c.j) for c in counted if not c.lines_up] == [(0, 1)] and all(c.better for c in counted
                                                                             if not c.lines_up)
    assert all(c.lines_up for c in fitted)


def test_the_fingerprint_residual_counts_every_pair_an_outlier_at_most_at_the_outlier_limit():
    """D144: the residual of a fit (capped_rms) keeps the pairs the robust fit left out, each at most at its outlier
    limit (3 times the median residual), so it is not made small by leaving out the largest residuals of a few pairs,
    and one gross outlier does not dominate it."""
    a = np.zeros((40, 2))
    b = np.random.default_rng(0).normal(0, 1.0, (40, 2))  # rms length sqrt(2)
    b[0] = (500.0, 0.0)
    fit = fit_overlap(a, b)

    assert not fit.inliers[0]
    assert fit.rms() < capped_rms(fit) < 1.7  # the outlier counted, but at its limit: not 500 nm / sqrt(40)


def test_image_by_image_leaves_an_image_without_matching_contacts_unplaced():
    points, centers, fovs, _ = mosaic(np.zeros((12, 2)), GRID_50)
    points[7] = np.empty((0, 2))

    guide = placed_image_by_image(points, centers, fovs, search_nm=100)

    assert np.flatnonzero(np.isnan(guide[:, 0])).tolist() == [7]


def test_tiles_are_grouped_into_stripes_by_their_centre_x():
    centers, fovs = serpentine()
    np.testing.assert_array_equal(tile_stripes(centers), np.repeat([0, 1, 2], 4))


def test_stripes_break_at_the_x_steps_whatever_the_overlap():
    width = 2650.0
    x = np.repeat([0.0, 0.3, 0.6], 4) * width  # stripes overlap by 70 % of the tile width
    y = np.tile(np.arange(4) * 2000.0, 3)
    centers = np.column_stack([x + 1e-9 * np.arange(12), y])  # equal x up to rounding
    np.testing.assert_array_equal(tile_stripes(centers), np.repeat([0, 1, 2], 4))
    np.testing.assert_array_equal(tile_stripes(centers[:4]), np.zeros(4))  # one stripe

    # Contact bounding boxes: centre x differs by up to a pitch within a stripe; the image centres decide.
    boxes = centers + np.column_stack([np.tile([0.0, 3.0, 130.0, 0.0], 3), np.zeros(12)])
    assert tile_stripes(boxes).max() > 2  # split wrongly
    np.testing.assert_array_equal(tile_stripes(boxes, centers), np.repeat([0, 1, 2], 4))
    np.testing.assert_array_equal(tile_stripes(boxes, np.full((12, 2), np.nan)), tile_stripes(boxes))  # unknown


def test_measurement_pairs_follow_the_serpentine():
    centers, fovs = serpentine()  # tiles 0-3: stripe 0 bottom to top, 4-7: stripe 1, 8-11: stripe 2
    stripe = tile_stripes(centers)
    within = {(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11)}
    # Stripe 0 measured upward ends at its top (3) and stripe 1 starts at its top (7); stripe 1 ends
    # at its bottom (4), where stripe 2 starts (8).
    assert measurement_pairs(centers, stripe) == within | {(3, 7), (4, 8)}
    assert measurement_pairs(centers, stripe, first_upward=False) == within | {(0, 4), (7, 11)}


def test_rigid_stitching_removes_tile_rotations_that_shifts_cannot():
    # Each tile measures with its own small rotation (~1 nm at the tile edge) and stage error.
    rng = np.random.default_rng(5)
    truth = [about(rotation(rng.normal(0, 3000e-6), e), c) for e, c in zip(ERRORS, NOMINAL)]
    points = []
    for t, c in zip(truth, NOMINAL):
        inside = np.all(np.abs(apply_affine(np.linalg.inv(t), LATTICE) - c) < FOV / 2 - 10, axis=1)
        points.append(apply_affine(np.linalg.inv(t), LATTICE[inside]))
    stitch = stitch_tiles(points, NOMINAL, np.full((4, 2), FOV))

    rigid = stitch_tiles_rigid(points, NOMINAL, stitch)

    rms = {name: np.sqrt((tie_residuals(points, stitch, c) ** 2).sum(axis=1).mean())
           for name, c in (("shift", stitch.corrections), ("rigid", rigid))}
    assert rms["shift"] > 0.2 and rms["rigid"] < 0.005  # first-order rotation: ~θ²/2 · r ~ 0.002 nm at 3 mrad
    per_pair = pair_residuals(points, stitch, rigid)
    assert len(per_pair) == len(stitch.pairs) and sum(len(r) for r in per_pair) == len(tie_residuals(points, stitch, rigid))


def test_pair_kind_from_the_tile_centres():
    centers = np.array([[0.0, 0.0], [10.0, 600.0], [600.0, -5.0], [600.0, 600.0]])
    assert [pair_kind(centers, 0, k) for k in (1, 2, 3)] == ["vertical", "horizontal", "corner"]
