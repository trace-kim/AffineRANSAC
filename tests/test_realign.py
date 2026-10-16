import numpy as np

from affine_ransac.analysis import Settings
from affine_ransac.features.contact import DetectedContacts
from affine_ransac.geometry.frames import tile_nm_to_pixel
from affine_ransac.matching import match_points
from affine_ransac.pipeline import DesignContacts, TileResult
from affine_ransac.realign import merge_close, realign_tiles

FOV, SIZE, PITCH = 1000.0, 500, 100.0
CENTERS = np.array([(x, y) for x in np.arange(6) * 800.0 for y in np.arange(6) * 800.0])  # 200 nm overlaps
OFFSET = np.array([-250.0, -1730.0])  # design − SEM: every image shows the design this far away


def mask_contacts(missing: float, seed: int = 0) -> np.ndarray:
    """The contacts on the mask: a square lattice with a random fraction missing (aperiodic content)."""
    steps = np.arange(-40, 60) * PITCH
    lattice = np.array([(x, y) for x in steps for y in steps])
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


def misaligned_tiles(contacts, seed=1) -> list[TileResult]:
    """Each .oas covers its nominal field; each image shows the field OFFSET away, with a stage error and noise."""
    rng = np.random.default_rng(seed)
    tiles = []
    for center in CENTERS:
        seen = center + OFFSET + rng.normal(0, 3, 2)  # where the image really is
        sem = inside(contacts, seen) + rng.normal(0, 0.3, (len(inside(contacts, seen)), 2))
        tiles.append(tile(center, inside(contacts, center, margin=20), sem))
    return tiles


def test_realign_tiles_finds_the_offset_and_gives_each_image_its_design():
    contacts = mask_contacts(missing=0.15)
    lines = []

    realigned = realign_tiles(misaligned_tiles(contacts), Settings(realign_search_um=3.0), lines.append)

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
        assert t.oas_path is None


def test_realign_tiles_warns_on_a_fully_periodic_pattern():
    lines = []

    realign_tiles(misaligned_tiles(mask_contacts(missing=0.0)), Settings(realign_search_um=3.0), lines.append)

    assert any(line.startswith("!! Realignment") for line in lines)


def test_merge_close_keeps_one_mean_point_per_group():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [100.0, 0.0], [100.0, 3.0], [500.0, 0.0]])

    merged = merge_close(points, 5.0)

    np.testing.assert_allclose(sorted(merged.tolist()), [[1.0, 0.0], [100.0, 1.5], [500.0, 0.0]])
