import numpy as np
import pytest
from scipy.spatial import cKDTree

from affine_ransac.analysis import Settings
from affine_ransac.features.contact import DetectedContacts
from affine_ransac.geometry.frames import tile_nm_to_pixel
from affine_ransac.matching import match_points
from affine_ransac.pipeline import DesignContacts, TileResult
from affine_ransac.realign import merge_close, realign_tiles

FOV, SIZE, PITCH = 1000.0, 500, 100.0
CENTERS = np.array([(x, y) for x in np.arange(6) * 800.0 for y in np.arange(6) * 800.0])  # 200 nm overlaps
OFFSET = np.array([-250.0, -1730.0])  # design − SEM: every image shows the design this far away
STRIPE = np.array([(0.0, y) for y in np.arange(12) * 800.0])  # one stripe, one image wide, 12 images long


def mask_contacts(missing: float, seed: int = 0) -> np.ndarray:
    """The contacts on the mask: a square lattice with a random fraction missing (aperiodic content)."""
    lattice = np.array([(x, y) for x in np.arange(-40, 60) * PITCH for y in np.arange(-60, 150) * PITCH])
    return lattice[np.random.default_rng(seed).random(len(lattice)) >= missing]


def tile(center, design_local, sem_local) -> TileResult:
    pixels = tile_nm_to_pixel(sem_local, (SIZE, SIZE), FOV / SIZE)
    found = DetectedContacts(pixels, np.ones(len(pixels)), [], 0.0)
    return TileResult(image=np.zeros((SIZE, SIZE), np.uint8), center_nm=np.asarray(center, float),
                      fov_nm=np.array([FOV, FOV]), pixel_size_nm=FOV / SIZE, otsu=found, refined=found,
                      design=DesignContacts(design_local, False, np.nan, True))


def inside(points, center, margin=30.0):
    """Points within the field of view at center, away from its edge, tile-local."""
    local = points - center
    return local[np.all(np.abs(local) < FOV / 2 - margin, axis=1)]


def misaligned_tiles(contacts, seed=1, blank_stripe=False, centers=CENTERS, offset=OFFSET) -> list[TileResult]:
    """Each .oas covers its nominal field (centers), drawn up to ~8 nm off (files offset against each other, more
    than group_radius_nm); each image shows the field `offset` away, with a stage error and noise. blank_stripe: the
    images of the first stripe show no pattern, only false centres at random places."""
    rng = np.random.default_rng(seed)
    tiles = []
    for center in centers:
        seen = center + offset + rng.normal(0, 3, 2)  # where the image really is
        sem = inside(contacts, seen) + rng.normal(0, 0.3, (len(inside(contacts, seen)), 2))
        if blank_stripe and center[0] == 0:
            sem = rng.uniform(-FOV / 2 + 30, FOV / 2 - 30, (40, 2))
        tiles.append(tile(center, inside(contacts, center, margin=20) + rng.uniform(-6, 6, 2), sem))
    return tiles


def test_realign_tiles_finds_the_offset_and_gives_each_image_its_design():
    contacts = mask_contacts(missing=0.15)
    lines = []

    realigned = realign_tiles(misaligned_tiles(contacts), Settings(realign=True), lines.append)

    shifts = np.array([t.center_nm for t in realigned]) - CENTERS
    np.testing.assert_allclose(shifts, np.tile(shifts[0], (len(CENTERS), 1)))  # one offset for all
    np.testing.assert_allclose(shifts[0], OFFSET, atol=5)  # within the stage errors' mean
    assert not any(line.startswith("!!") for line in lines)
    measured = [t for t in realigned if t.design.ok]
    assert len(measured) >= 24  # bottom rows look below every .oas file: nothing to match there
    for t in measured:
        inner = inside(t.design_points_nm(), t.center_nm, margin=40) + t.center_nm  # the SEM sees these
        ia, _ = match_points(inner, t.sem_points_nm(), 15.0)
        assert len(ia) == len(inner) > 0
        assert t.oas_center_nm is not None


def test_realign_tiles_merges_the_same_contact_of_offset_oas_files():
    realigned = realign_tiles(misaligned_tiles(mask_contacts(missing=0.15)), Settings(realign=True), lambda line: None)

    for t in realigned:
        if len(t.design.centers) > 1:  # one design contact per contact: none closer than half the pitch
            assert cKDTree(t.design.centers).query(t.design.centers, k=2)[0][:, 1].min() > PITCH / 2


def test_realign_tiles_finds_the_offset_with_a_stripe_of_blank_images():
    lines = []

    realigned = realign_tiles(misaligned_tiles(mask_contacts(missing=0.15), blank_stripe=True),
                              Settings(realign=True), lines.append)

    np.testing.assert_allclose(realigned[0].center_nm - CENTERS[0], OFFSET, atol=5)
    assert not any(line.startswith("!!") for line in lines)


def test_realign_tiles_finds_an_offset_of_any_size_on_a_single_stripe():
    """D104: the images sit 0.6 of an image across the stripe and 4 images along it from their .oas. The old search
    reached only half the field: half an image across a single stripe."""
    offset = np.array([600.0, -3300.0])
    lines = []

    realigned = realign_tiles(misaligned_tiles(mask_contacts(missing=0.15), centers=STRIPE, offset=offset),
                              Settings(realign=True), lines.append)

    np.testing.assert_allclose(realigned[0].center_nm - STRIPE[0], offset, atol=5)
    assert sum(t.design.ok for t in realigned) >= 8  # the images past the end of the .oas files find nothing


def test_realign_tiles_refuses_a_fully_periodic_pattern():
    """Every whole-pitch shift fits a repeating pattern as well: the offset cannot be told from the contacts, so the
    run stops instead of guessing (D104; it used to go on with the best by chance)."""
    with pytest.raises(ValueError, match="no offset is clearly the best"):
        realign_tiles(misaligned_tiles(mask_contacts(missing=0.0)), Settings(realign=True), lambda line: None)


def test_realign_tiles_refuses_images_outside_the_area_of_the_oas_files():
    """Images 1.5 images across a single stripe from their .oas files show nothing those files hold: some wrong offsets
    fit partly (a lattice with missing contacts), none clearly: the run stops instead of using one."""
    with pytest.raises(ValueError, match="no offset is clearly the best"):
        realign_tiles(misaligned_tiles(mask_contacts(missing=0.15), centers=STRIPE, offset=np.array([1500.0, -3300.0])),
                      Settings(realign=True), lambda line: None)


def test_merge_close_keeps_one_mean_point_per_group():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [100.0, 0.0], [100.0, 3.0], [500.0, 0.0]])

    merged = merge_close(points, 5.0)

    np.testing.assert_allclose(sorted(merged.tolist()), [[1.0, 0.0], [100.0, 1.5], [500.0, 0.0]])
