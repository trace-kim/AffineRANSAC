"""Entry script of the packaged MaskReg.exe (packaging/MaskReg.spec). Each tile-loading worker is a MaskReg.exe too:
it runs this script as far as multiprocessing.freeze_support() in main(), marking the stages of its start-up on the way
for the detailed tile-loading log (docs/SPEC.md D150)."""

from affine_ransac.machine import mark_startup

mark_startup("Python running")  # Windows has started MaskReg.exe and Python
from affine_ransac.release.app import main  # noqa: E402  (every library MaskReg uses)

mark_startup("libraries loaded")

if __name__ == "__main__":
    main()
