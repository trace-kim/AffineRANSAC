"""Write small synthetic OASIS files for tests and for trying the viewer.

Run directly to create a sample file:
    python tests/sample_design.py sample.oas
"""

import sys

import klayout.db as kdb


def write_contact_array(
    path,
    nx=3,
    ny=2,
    pitch_x_nm=200,
    pitch_y_nm=300,
    size_nm=100,
    dbu_nm=1.0,
):
    """Write a TOP cell with an nx-by-ny array of square contacts on layer 1/0.

    The first contact is centred at (0, 0). Also adds a text label on 1/0
    (which the reader must skip) and one box on layer 2/0.
    """
    layout = kdb.Layout()
    layout.dbu = dbu_nm / 1000.0  # KLayout dbu is in micrometres
    to_dbu = 1.0 / dbu_nm

    top = layout.create_cell("TOP")
    contact = layout.create_cell("CONTACT")
    layer1 = layout.layer(1, 0)
    layer2 = layout.layer(2, 0)

    half = round(size_nm / 2 * to_dbu)
    contact.shapes(layer1).insert(kdb.Box(-half, -half, half, half))

    step_x = kdb.Vector(round(pitch_x_nm * to_dbu), 0)
    step_y = kdb.Vector(0, round(pitch_y_nm * to_dbu))
    top.insert(kdb.CellInstArray(contact.cell_index(), kdb.Trans(), step_x, step_y, nx, ny))

    top.shapes(layer1).insert(kdb.Text("label", kdb.Trans()))
    top.shapes(layer2).insert(kdb.Box(0, 0, round(50 * to_dbu), round(50 * to_dbu)))

    layout.write(str(path))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sample.oas"
    write_contact_array(out, nx=20, ny=15, pitch_x_nm=500, pitch_y_nm=500, size_nm=150)
    print(f"Wrote {out}")
