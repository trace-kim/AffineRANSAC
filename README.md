# AffineRANSAC

Photomask registration-error measurement: OASIS design vs. stitched SEM images.
Design and decisions: [`docs/SPEC.md`](docs/SPEC.md).

## Setup (once per machine)

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

In VS Code, select `.venv` as the Python interpreter and as the notebook kernel.

## After every `git pull`

```
pip install -e ".[dev]"
```

- Always include `".[dev]"`. Plain `pip install -e .` skips the development packages
  (pytest, ipykernel, pandas).
- The `-e` (editable) install means code changes take effect without reinstalling. Re-running
  the command only matters when dependencies change, and it finishes in seconds when nothing
  is new, so it's simplest to always run it.
- If a notebook was open, **restart its kernel**. Python keeps already-imported modules in
  memory, so a running kernel won't see the new code.

## Run

```
pytest                                                     # tests
python -m affine_ransac.view_design tile.oas               # design viewer
python -m affine_ransac.view_sem tile.jpg --detect         # SEM viewer with detected contacts
```

## Releasing MaskReg

MaskReg (`maskreg`, or `python -m affine_ransac.release.app`) has one version number,
`__version__` in `src/affine_ransac/__init__.py`. The rules are in `docs/SPEC.md`, "Versions and run files".

**In every change** to what MaskReg computes or saves (the tests in `tests/test_release_versions.py` fail until this
is done):
- other results for the same folder → raise `ANALYSIS` in `src/affine_ransac/release/analysis.py` and record the new
  numbers in the test;
- run files saved before can no longer be read → raise `FORMAT` in `src/affine_ransac/release/cache.py` and record
  the new layout in the test;
- add a line under *Unreleased* in `CHANGELOG.md`.

**To release a version:**
1. Choose the number from the last release: MAJOR if `FORMAT` went up, else MINOR if `ANALYSIS` went up or something
   was added, else PATCH (fixes only).
2. Set `__version__`; in `CHANGELOG.md` rename *Unreleased* to `X.Y.Z (YYYY-MM-DD)` and start a new empty
   *Unreleased* above it.
3. `pytest` must pass.
4. On the data machine (image folders need its metadata reader; the build refuses without it), with Inno Setup 6
   installed once (`winget install JRSoftware.InnoSetup`), build the installer (~2 min):
   ```
   python packaging/build.py
   ```
   It makes `dist\MaskReg\` (the program, ~400 MB) and `dist\MaskReg-setup-X.Y.Z.exe` (~100 MB). Install it, start
   MaskReg from the Start menu and open a folder to check it.
5. Commit, tag and push:
   ```
   git commit -am "Release X.Y.Z"
   git tag -a vX.Y.Z -m "MaskReg X.Y.Z"
   git push origin main vX.Y.Z
   ```
6. On GitHub: *Releases → Draft a new release*, choose the tag, paste its `CHANGELOG.md` section and attach
   `MaskReg-setup-X.Y.Z.exe`. Users install it by running it (no admin rights needed); a newer version installs over
   the older one.

Planned (`docs/SPEC.md` D100): automatic updates from a read-only FTP folder.

## Notebooks

Notebooks in `notebooks/` test the functions on real data. **Work on a copy** named
`<name>.local.ipynb` (gitignored): set the paths in its first cell and run all cells. Editing the
tracked notebook itself would make the next `git pull` fail.
