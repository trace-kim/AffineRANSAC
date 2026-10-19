# MaskReg changelog

What changed in MaskReg, the release app, per version. The rules for the version number and for the run-file
numbers `FORMAT` and `ANALYSIS` are in [`docs/SPEC.md`](docs/SPEC.md), "Versions and run files" (D99); how to
release a version is in [`README.md`](README.md), "Releasing MaskReg".

Every change goes under **Unreleased** in the commit that makes it. Say when saved runs are affected:
- **Format N:** `FORMAT` went up; runs saved by older versions are analysed again when opened (kept as backups).
- **Analysis N:** `ANALYSIS` went up; runs saved by older versions are shown flagged as outdated, with an offer to
  analyse them again.

## Unreleased (will be 1.0.0)

First numbered version. Saved runs: format 3, analysis 3. Runs saved by earlier copies of MaskReg carry no version:
they open, flagged as outdated.

- *File → Folders...*: choose the folder of `maskreg.ini` and the folder where FTP downloads are kept (D96).
- *File → Open recent*: the last 10 folders (D97).
- Analysing a folder again keeps the run it replaces in `MaskReg_backups/` (the last 5); an interrupted save no
  longer leaves a damaged file (D98).
- Every run records the MaskReg version and analysis that made it, and opening a folder checks them (D99). A run
  saved by a newer MaskReg is never replaced by an older one without the user's restart.
- MaskReg comes as an installer, `MaskReg-setup-X.Y.Z.exe`: a program of its own, Python included, installed per
  user without admin rights, with a Start menu entry and an uninstall entry (D100). `MaskReg <folder>` opens that
  folder at start.
- Automatic updates: at every start MaskReg looks for a newer version on the update server and offers it; Yes installs
  it and restarts MaskReg, No asks again next time. *Help → Check for updates...* (D101).
- A run that fails for too few measured contacts now shows what it found on the Stitching tab: the tiles, their
  centres, the SEM images and design on request, and the tiles not measured with the reason (layer *Tiles not
  measured*; hover a tile). Nothing is saved (D102).
- **Analysis 3.** Fix: realignment (`realign`) no longer limits the offset to half the field. Across a single stripe
  that was half an image, so offsets of several µm could not be found, and a run could move every image to the limit
  (e.g. −166 µm). It now searches offsets of any size. On a repeating pattern, where many offsets fit equally and give
  the same errors (the design is ideal), it takes the one that pairs the most contacts and logs that the others fit as
  well. It stops only when no offset fits (D104, D105). Runs saved with an earlier analysis open flagged as outdated;
  analysed again, only those made with `realign` on give other numbers.
- The contrast check (`min_contrast`, 5 by default since D93) is now off by default (0): on noisy images it can drop
  every real contact (seen on noisy synthetic images). When turned on, the log warns (`!!`) if it removes every
  contact of a tile. Parameters saved before keep their value (D103).
- Fix: *Load design* showed a solid block of colour for `.oas` files drawn tone reversed (the area around the
  holes). It now shows their contacts, in the tone the analysis read each file. Runs saved before show it too (D106).
- Stitching tab: the failure markers can be told apart, also when drawn with the GPU, where dashed lines came out
  solid. Overlaps not matched are filled boxes, orange for the measured stitching and blue for the design; tiles not
  stitched have a thick red outline. Each has its own check box (D107).
- Stitching tab: tile names are dark text on white labels, readable over the SEM images (D108).
- Everything before (SPEC D77–D95): Stitching and Registration tabs, FTP browser and download, autosave, contact
  size tab.
