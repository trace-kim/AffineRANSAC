import numpy as np

from affine_ransac.features.contact import detect_contacts
from affine_ransac.features.edges import refine_edges
from affine_ransac.geometry.frames import pixel_to_tile_nm
from affine_ransac.overlap import match_overlap, overlap_box
from affine_ransac.overlap_fit import fit_overlap
from affine_ransac.overlap_image import image_overlap_shift
from sample_sem import render_tile

FOV, SIZE = 720.0, 512
PIXEL = FOV / SIZE
LATTICE = np.array([(x, y) for x in np.arange(-900, 900, 90.0) for y in np.arange(-900, 900, 90.0)])


def two_tiles():
    """Tile B is nominally 600.3 nm below tile A (a fractional pixel step), so they overlap in a
    ~120 nm strip. Their true positions differ from nominal by different stage errors."""
    nominal_a, nominal_b = np.array([0.0, 0.0]), np.array([0.0, -600.3])
    error_a, error_b = np.array([1.2, -0.7]), np.array([-3.1, 2.4])
    image_a = render_tile(LATTICE, nominal_a + error_a, FOV, SIZE, seed=1)
    image_b = render_tile(LATTICE, nominal_b + error_b, FOV, SIZE, seed=2)
    expected = -(error_b - error_a)  # B − A for the same contact, in nominal mask coordinates
    box = overlap_box(nominal_a, (FOV, FOV), nominal_b, (FOV, FOV))
    return image_a, nominal_a, image_b, nominal_b, box, expected


def test_phase_correlation_shift_matches_the_stage_errors():
    image_a, center_a, image_b, center_b, box, expected = two_tiles()

    shift, error = image_overlap_shift(image_a, center_a, image_b, center_b, PIXEL, box)

    np.testing.assert_allclose(shift, expected, atol=0.4)  # nm (~0.3 px; pixel is 1.4 nm)
    assert 0 <= error < 1


def test_contact_and_image_methods_agree():
    image_a, center_a, image_b, center_b, box, expected = two_tiles()

    def contacts_mask_nm(image, center):
        found = refine_edges(image, detect_contacts(image))
        return pixel_to_tile_nm(found.centers, image.shape, PIXEL) + center

    a, b = contacts_mask_nm(image_a, center_a), contacts_mask_nm(image_b, center_b)
    ia, ib = match_overlap(a, b, box, max_distance=10)
    fit = fit_overlap(a[ia], b[ib])
    image_shift, _ = image_overlap_shift(image_a, center_a, image_b, center_b, PIXEL, box)

    assert len(ia) >= 3
    np.testing.assert_allclose(fit.shift, expected, atol=0.3)
    np.testing.assert_allclose(image_shift, fit.shift, atol=0.5)
