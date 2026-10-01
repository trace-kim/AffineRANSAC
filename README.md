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

Notebooks in `notebooks/` test the functions on real data. **Work on a copy** named
`<name>.local.ipynb` (gitignored): set the paths in its first cell and run all cells. Editing the
tracked notebook itself would make the next `git pull` fail.
