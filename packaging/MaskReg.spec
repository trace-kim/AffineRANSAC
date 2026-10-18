# PyInstaller spec of MaskReg (docs/SPEC.md D100): one folder, dist/MaskReg/, holding MaskReg.exe (no console
# window) and everything it needs, Python included.
# Build from the repository root:  python -m PyInstaller packaging/MaskReg.spec --noconfirm
import importlib.util
import os

import klayout

# Image folders need the metadata reader, which exists only on the data machine (CLAUDE.md, task T001): a MaskReg
# built without it opens contour-CSV folders only. MASKREG_TEST_BUILD=1 builds such a copy anyway, for tests.
if importlib.util.find_spec("affine_ransac.io.metadata") is None and os.environ.get("MASKREG_TEST_BUILD") != "1":
    raise SystemExit("affine_ransac/io/metadata.py is missing: build MaskReg on the data machine "
                     "(or set MASKREG_TEST_BUILD=1 for a test build that cannot open image folders)")

# KLayout loads its file-format readers (OASIS, GDS2, ...) at run time from db_plugins/, which imports do not show.
plugins = os.path.join(os.path.dirname(klayout.__file__), "db_plugins", "*.dll")

a = Analysis([os.path.join(SPECPATH, "maskreg.py")], pathex=[os.path.join(SPECPATH, "..", "src")],
             binaries=[(plugins, "klayout/db_plugins")],
             excludes=["matplotlib", "pandas", "IPython", "pytest", "tkinter", "pyftpdlib"])  # dev only
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="MaskReg", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="MaskReg")
