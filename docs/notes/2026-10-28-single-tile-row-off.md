# Diagnosis: single tiles paired one row off (40 nm staggered data, image by image)

*2026-10-28, local agent. Answer to: "Stitched + affine, red arrows, but I don't fully buy your explanation ...
Analyze properly." Diagnosis only, no code changed. All numbers below come from synthetic staggered mosaics run
through the release's own functions, not from your data.*

You were right on three points. The error is an abrupt jump at one photo, not a ripple: the ripple came from the
small fitting windows in my test, not your 40 µm ones. Your arrays are staggered, so I redid everything on staggered
grids (four layouts). And "lucky" was the wrong word. Here is the analysis again, with only what I could prove.

## What your observations mean together

- **Red arrows on that tile:** each of its holes is paired with a design hole in the next row. On a staggered array
  that's the next row, half a pitch to the side.
- **Only that tile, neighbours fine:** that one photo picked a different design row from its neighbours.
- **The picture still looks right:** its holes sit on design holes because the pattern repeats, but every error on
  that tile is one row long.
- **No flag on the Stitching tab:** its check compares photos with each other, and those overlaps do line up. It
  never looks at the photo-to-design pairing.

## Why "it fits in any row" still matters

On its own, a photo fits its design at many rows. But the stitching has already fixed where the photo sits next to
its neighbours, and only one row agrees with that: the one its neighbours use. The pairing step has to pick that row
for every photo. If it picks another, that photo and only that photo jumps by a row.

## How the row is picked, and the two flaws

`matching.consistent_choice`, not changed by the update.

1. Each photo gets the 9 rows nearest its stage position. On a clean repeating array they all fit equally well.
2. **Flaw 1, the anchor** (`src/affine_ransac/matching.py:253`–`260`):
   - When no photo is clearly decided, the program makes the photo with the most holes the anchor.
   - It tries each of the anchor's rows and lets the other photos follow their neighbours.
   - It keeps the answer with the fewest unmatched holes, then the smallest "median shift".
   - The anchor never re-chooses (`:224`), and agreement between neighbours is never counted.
3. **Flaw 2, the drop rule** (`:153`): a row with just 2 more unmatched holes than the best is struck from a photo's
   options. With a few missed detections, the row that agrees with the neighbours can be struck. The photo then
   takes the nearest row left, and comes out alone, a row off.

## Evidence

Staggered test mosaics, the release's own functions, one change at a time.

- **The program's own run** (I only added print statements): 84 photos, none decided, and the anchor was photo 79.
  One answer had every photo agreeing, with a median shift of 3.2196 nm. The program chose the answer with photo 79
  alone a row off, at 3.2190 nm: it won by **0.0006 nm**. The right row was among photo 79's options.
- **Stitching:** both versions put every photo within about 0.1 nm of its true position, apart from whole rows, which
  do no harm on a repeating array.
- **Clean grids:** photos were paired off in 8 of 16 mosaics with the old version and 5 of 16 with the new. Picking
  the answer where neighbours agree first fixed all 10 affected mosaics, in both versions.
- **0.3 % missed holes:** all 8 mosaics were affected, in both versions. Agreement first still left 1–3 lone photos
  off in 4 of them (flaw 2). Also not striking rows for 2 holes left 0 in all 8.

## Why it flipped with the update

The code is deterministic: the same data always gives the same answer, which is why your runs are consistent. But
the update changed this step's inputs. The two versions put 15–48 of 84 photos a whole row apart and otherwise
differ by hundredths of a nm. Flaw 1 decides on margins like 0.0006 nm and flaw 2 on 2 holes, so that is enough to
change which photo fails. I found nothing that makes the new version systematically worse. If several datasets
broke with the update, not just this one, please tell me: that would point to something I haven't found.

> **Insight**
> - On a repeating array every row choice scores 0 unmatched holes, so the tie-break alone decides, and the median
>   shift says nothing about which answer is right.
> - The fix ranks answers by "do neighbours agree?" first, which is exactly the property that fails here.

## Still open

The 5–10 nm gap in Stitched + affine. By my estimate, not measured, one photo a row off moves a 40 µm window fit by
only about 1 nm, so that gap needs another cause.

## Quick checks on your data

1. In plain **Stitched**, do that tile's overlaps line up, with the same hole within about 1 nm? Is that overlap in
   the log's `!! … do not line up` list?
2. Do all the red arrows on the bad tile have the same length and direction?
3. Is there one bad tile, or several isolated ones?
4. What is the stagger: spacing along a row, spacing between rows, and how far every other row is shifted?

I haven't changed any code. If you want proof on your own data, I can write a small task for the remote agent. It
would print which photo was the anchor and the two numbers it compared, and run the agreement-first test.

As I said mid-way: stopping my tests force-closed every Python program on the local PC (not the remote), so please
check MaskReg or any notebook you had open there. The picture page still shows the wrong ripple and square grids; I
can correct it if you'd like.
