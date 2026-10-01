import cv2
import numpy as np

from affine_ransac.features.contact import detect_contacts
from affine_ransac.features.edges import edge_offset, outward_normals, refine_edges, sample_profiles
from affine_ransac.io.sem_image import load_sem_image
from sample_sem import write_sem_like


def test_outward_normals_point_outwards_for_either_order():
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    circle = np.column_stack([10 + 5 * np.cos(angles), 20 + 5 * np.sin(angles)])
    for contour in (circle, circle[::-1]):
        normals = outward_normals(contour, span=1)
        radial = (contour - [10, 20]) / 5
        np.testing.assert_allclose((normals * radial).sum(axis=1), 1, atol=0.01)


def test_sample_profiles_follows_each_direction():
    image = np.zeros((50, 50), dtype=np.uint8)
    image[20, 30] = 100  # 10 px right of (20, 20)
    points = np.array([[20.0, 20.0], [20.0, 20.0]])
    directions = np.array([[1.0, 0.0], [0.0, 1.0]])  # right, and down (y points down)
    profile = sample_profiles(image, points, directions, np.array([0.0, 10.0]))
    np.testing.assert_allclose(profile, [[0, 100], [0, 0]])


def test_edge_offset_finds_steepest_rise_between_samples():
    offsets = np.arange(-5, 5, 0.5)
    profile = 40 + 160 / (1 + np.exp(-(offsets - 1.3) * 2))  # smooth step at offset 1.3
    assert abs(edge_offset(profile, offsets, rising=True) - 1.3) < 0.05
    assert abs(edge_offset(-profile, offsets, rising=False) - 1.3) < 0.05


def exact_ellipse_image(center, semi_axes, size=80, dark=40, bright=180, blur=2.0, oversample=16):
    """Blurred dark ellipse with exact sub-pixel area coverage (cv2's anti-aliased drawing is
    slightly asymmetric, which would bias the test by ~0.1 px)."""
    fine_y, fine_x = (np.indices((size * oversample, size * oversample)) + 0.5) / oversample - 0.5
    inside = ((fine_x - center[0]) / semi_axes[0]) ** 2 + ((fine_y - center[1]) / semi_axes[1]) ** 2 < 1
    coverage = inside.reshape(size, oversample, size, oversample).mean(axis=(1, 3))
    image = (bright - (bright - dark) * coverage).astype(np.float32)
    return cv2.GaussianBlur(image, (0, 0), blur).round().astype(np.uint8)


def distance_to_ellipse(points, center, semi_axes):
    """Distance (px) from each point to the nearest point of the ellipse outline."""
    t = np.linspace(0, 2 * np.pi, 20000)
    outline = np.column_stack([center[0] + semi_axes[0] * np.cos(t), center[1] + semi_axes[1] * np.sin(t)])
    return np.array([np.linalg.norm(outline - p, axis=1).min() for p in points])


def test_refined_contour_follows_an_ellipse():
    # Elongated contact: semi-axes 24 x 10 px. The steepest rise of the blurred edge is on the
    # true boundary, all the way around, so the refined contour should follow the ellipse.
    center, axes = (40.3, 37.6), (24, 10)
    image = exact_ellipse_image(center, axes)

    refined = refine_edges(image, detect_contacts(image, blur_sigma=0), blur_sigma=0)

    np.testing.assert_allclose(refined.centers, [center], atol=0.05)
    distance = distance_to_ellipse(refined.contours[0], center, axes)
    assert distance.mean() < 0.3
    # At the sharp tips of the long axis (curvature radius ~4 px) the blur pulls the steepest
    # gradient slightly inward; that is a property of the blurred image, not of the search.
    assert distance.max() < 1.5


def test_search_window_is_a_fixed_pixel_distance():
    image = exact_ellipse_image((40, 40), (20, 12))
    otsu = detect_contacts(image, blur_sigma=0)

    refined = refine_edges(image, otsu, search_px=0.5, blur_sigma=0)

    moved = np.linalg.norm(refined.contours[0] - otsu.contours[0], axis=1)
    assert moved.max() <= 0.5 + 1e-9


def test_refine_keeps_centres_accurate_on_sem_like_image(tmp_path):
    path = tmp_path / "tile.jpg"
    truth = write_sem_like(path, databar_px=0)
    image = load_sem_image(path)
    found = detect_contacts(image)

    refined = refine_edges(image, found)

    assert len(refined.centers) == len(found.centers) == len(truth)
    errors = np.linalg.norm(refined.centers[:, None] - truth[None], axis=2).min(axis=1)
    assert errors.max() < 0.2  # pixels
    # The refined edge sits further out than the Otsu edge (on the rise towards the bright rim).
    assert refined.areas.mean() > found.areas.mean()
