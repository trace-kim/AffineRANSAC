"""Build MaskReg for users (docs/SPEC.md D100): the program folder dist/MaskReg/ (PyInstaller, MaskReg.spec), then
the installer dist/MaskReg-setup-<version>.exe (Inno Setup, MaskReg.iss), for the version in affine_ransac.__version__.

Run on the data machine, from the repository root:  python packaging/build.py
Needs Inno Setup 6 (free; e.g. winget install JRSoftware.InnoSetup).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from affine_ransac import __version__

HERE = Path(__file__).parent


def inno_compiler() -> str:
    """ISCC.exe, Inno Setup's compiler: on the PATH or in its usual install folders."""
    folders = [Path(os.environ.get("LOCALAPPDATA", ""), "Programs"),  # installed for this user only
               Path(os.environ.get("ProgramFiles(x86)", "")), Path(os.environ.get("ProgramFiles", ""))]
    candidates = [shutil.which("ISCC")] + [str(folder / "Inno Setup 6" / "ISCC.exe") for folder in folders]
    found = [c for c in candidates if c and Path(c).is_file()]
    if not found:
        sys.exit("Inno Setup 6 not found: install it (winget install JRSoftware.InnoSetup) and run this again")
    return found[0]


def main():
    compiler = inno_compiler()
    print(f"Building MaskReg {__version__}")
    subprocess.run([sys.executable, "-m", "PyInstaller", str(HERE / "MaskReg.spec"), "--noconfirm",
                    "--distpath", str(HERE.parent / "dist"), "--workpath", str(HERE.parent / "build")], check=True)
    subprocess.run([compiler, f"/DVersion={__version__}", str(HERE / "MaskReg.iss")], check=True)
    print(f"Done: {HERE.parent / 'dist' / f'MaskReg-setup-{__version__}.exe'}")


if __name__ == "__main__":
    main()
