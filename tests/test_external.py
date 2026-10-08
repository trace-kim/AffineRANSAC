import numpy as np
import pytest

from affine_ransac.io.external import read_external


@pytest.mark.parametrize("separator", [" ", "\t", ",", "  "])
def test_reads_positions_in_um_as_nm_and_errors_in_nm(tmp_path, separator):
    rows = [(-36003.5385, -19248.0, 0.25, -1.5), (-36000.0, -19100.125, -0.75, 2.0)]
    lines = [separator.join(str(v) for v in row) for row in rows]
    (tmp_path / "ref.txt").write_text("\n".join(lines) + "\n\n", encoding="utf-8")  # a trailing empty line

    xy_nm, error_nm = read_external(tmp_path / "ref.txt")

    np.testing.assert_allclose(xy_nm, [[-36003538.5, -19248000.0], [-36000000.0, -19100125.0]])
    np.testing.assert_allclose(error_nm, [[0.25, -1.5], [-0.75, 2.0]])


def test_flip_sign_negates_only_the_errors(tmp_path):
    (tmp_path / "ref.txt").write_text("1 2 0.5 -0.25\n", encoding="utf-8")
    xy_nm, error_nm = read_external(tmp_path / "ref.txt", flip_sign=True)
    np.testing.assert_allclose(xy_nm, [[1000.0, 2000.0]])
    np.testing.assert_allclose(error_nm, [[-0.5, 0.25]])


def test_extra_columns_are_ignored_and_too_few_columns_raise(tmp_path):
    (tmp_path / "wide.txt").write_text("1 2 3 4 99\n5 6 7 8 99\n", encoding="utf-8")
    assert read_external(tmp_path / "wide.txt")[1].tolist() == [[3.0, 4.0], [7.0, 8.0]]
    (tmp_path / "narrow.txt").write_text("1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="need 4 columns"):
        read_external(tmp_path / "narrow.txt")
