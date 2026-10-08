# Coordinate / orientation validation — 2025-09-10 quiet-Sun synthesis (40 / 60 / 80 MHz)

Validates the Carrington↔sky coordinate handling behind the 40/60/80 MHz maps produced by
`script/run_quietsun_20250910.py` (`out/raytrace_{40,60,80}MHz.npz`): the latitude (theta)
axis and the Carrington-rotation (longitude) axis.

**Verdict: the orientation is correct.** Three independent checks pass — a sky-projection
check against sunpy, an empirical check against real HMI data (daily + synoptic, mapped
through sunpy only), and feature-position matching. No mirror, axis swap, sign flip or
rotation was found beyond the known, deliberate omission of the solar B0 tilt (quantified
below). The remaining few-degree residuals are processing/epoch scatter between the PSI
boundary map and the HMI products, not an orientation error.

## Data used

| item | source |
|---|---|
| Daily ground truth | `hmi.M_720s[2025.09.10_12:00:00_TAI]` — JSOC export via drms (automated in `daily_anchor_test.py`) |
| Synoptic reference | `hmi.B_synoptic_Br.2302.fits` — `http://jsoc2.stanford.edu/data/hmi/synoptic/hmi.B_synoptic_Br.2302.fits` (browse PNG and `_small.2302.fits` live in the same tree) |
| Model boundary | `corona/br_r0.hdf` — PSI MAS CR 2302 bundle (`cr2302-high/hmi_mast_mas_std_0201`) |
| Sky↔Carrington transforms | sunpy (Helioprojective ↔ HeliographicCarrington); L0 = 342.5205°, B0 = +7.2480° at 2025-09-10 12:00 UT |

All comparisons are done on a common regular (Carrington lon, lat) grid (0.6° step) built
with sunpy transforms **on both sides** — no hand-rolled FITS-WCS reading. (A hand-rolled
reader for the CRLN/CEA synoptic format initially produced the wrong wrap direction because
of its `CDELT1 = −0.1` and `CRVAL1 ≈ 828540`; that was the source of an initial false
"orientation broken" alarm. It is superseded — see the lineage note in Files.)

## Test 1 — display mapping vs sunpy (sky projection)

Production formula (`cart_to_sph` with `phi0 = L0 = 342.52°`, B0 ignored) vs sunpy
Helioprojective over a ±80° lon × ±60° lat grid:

| variant | RMS vs sunpy |
|---|---|
| **as-is (production)** | **71.8″** |
| longitude-mirrored | 1074″ |
| latitude-mirrored | 1095″ |

The 71.8″ residual is exactly the B0 signature: `dTy` at disk center = **+120.7″** =
`R_sun·sin(B0)` = 952.7″ × sin(7.248°); the residual pattern correlates **+0.99** with the
`cos(lat)·cos(lon−L0)` term that B0 multiplies. So the west/north senses and the phi0 = L0
assignment are right; the only sky-vs-formula deviation is the documented B0 omission.

## Test 2 — model grid vs real HMI data (empirical)

**Control — daily HMI vs HMI synoptic** (validates the whole sunpy sampling chain):

| variant | r (full grid, smoothed) | r (\|B\| > 100 G) |
|---|---|---|
| **as-is** | **+0.214** (best shift = 0.0°, 0.0°) | **+0.857** |
| lon-mirror | +0.002 | — |
| lat-mirror | +0.055 | — |
| both / sign-flip | ≤ 0 / negative | — |

Strong-field agreement r ≈ 0.86 proves the (lon, lat) sampling of both real maps is
correct — any orientation error would cap this far lower.

**Model boundary vs HMI synoptic** (same epoch):

| smoothing | as-is | lon-mirror | lat-mirror | sign-flip |
|---|---|---|---|---|
| 1×1 (raw 1°) | +0.163 | −0.009 | +0.026 | −0.163 |
| 3×3 (≈1.8°) | +0.256 | −0.014 | +0.041 | −0.256 |
| 5×5 (≈3°) | +0.321 | −0.017 | +0.051 | −0.321 |
| 7×7 (≈4.2°) | +0.380 | −0.019 | +0.061 | −0.380 |

Best residual shift: **dlon = −0.8°, dlat = −1.9°** (r: 0.256 → 0.265). Correlation rising
with smoothing + near-zero shift + all flipped variants dead ⇒ same orientation, same zero
point (≲ 2°), same polarity sense. The moderate amplitude is explained by the model boundary
being a heavily smoothed/processed version of the HMI map: max |B| ≈ 70 G in `br_r0` vs
1333 G (daily) and 2703 G (synoptic).

**Model boundary vs daily HMI** (instantaneous): as-is r = +0.135 (smoothed); a small shift
(−3.8°, +4.0°) recovers r = 0.203; strong-field subsets r = 0.22–0.31. Consistent with
smoothing + epoch differences (ARs evolve during the rotation); no systematic flip.

## Test 3 — feature positions

The 7 strongest synoptic features (|B| up to 2703 G) vs their daily/model counterparts
(box = ±5°; offsets in degrees):

| synoptic | daily | model |
|---|---|---|
| 2703 G @ (51.3, +3.2) | 155 G @ (+1.8, −3.6) | 53 G @ (+0.6, +1.2) |
| 2420 G @ (249.9, +10.4) | off-disk at 12:00 | 63 G @ (−1.8, +3.6) |
| 2071 G @ (357.9, −14.8) | 1241 G @ (−0.6, 0.0) | 68 G @ (0.0, −4.8) |
| 1926 G @ (48.9, +26.6) | 961 G @ (+1.8, +0.6) | 32 G @ (+4.8, +2.4) |
| 1893 G @ (62.7, +9.2) | 393 G @ (+1.8, −1.2) | 63 G @ (+0.6, +4.8) |
| 1593 G @ (2.7, −10.0) | 1333 G @ (+2.4, −4.2) | 50 G @ (−3.6, −4.8) |
| 1590 G @ (56.1, +11.6) | 194 G @ (−4.8, −1.8) | 51 G @ (+4.8, +1.2) |
| 1257 G @ (8.1, −11.8) | 1333 G @ (−3.0, −2.4) | 33 G @ (−4.8, −4.2) |

The main AR complex (lon ≈ 355–13, lat ≈ −15) matches across all three products within
~2–4°; every feature stays within the ±5° box with mixed-sign small offsets — processing
scatter, not misalignment.

## What this means for the radio maps

* The 40/60/80 MHz maps sample the model through the same validated chain
  (image x,y → lon,lat,r → model), so their east–west and north–south orientations are
  correct; AR/CH-scale features sit in the right places.
* The pipeline ignores the +7.248° B0 tilt (by design). Measured effect on reference
  positions (see `orientation_on_radio_map.png`): east–west identical; north–south differs
  by up to **0.11 R_sun (~2′)** near disk center, shrinking toward the limb:
  * AR-A (5.0°, −14.5°): (+0.37, −0.25) → (+0.37, −0.36)
  * AR-B (50.7°, +27.2°): (+0.83, +0.46) → (+0.83, +0.41)
  * syn-max (51.3°, +3.2°): (+0.93, +0.06) → (+0.93, +0.01)
* **If pixel-level comparison against real (LWA) images is the goal**, add a B0-aware
  sampling step (rotate the sampling frame about the x-axis by B0). It is a small change
  and can be validated with the same sunpy harness used here (expected: RMS → few arcsec).

## Reproduce

Scripts assume the inti project layout (`ROOT = /data1/pzhang/dev/QuietSunPol`; venv at
`$ROOT/venv`); outputs are written to `$ROOT/out/orient_validation/` under the same names
as the plots/JSON here.

```bash
cd validation/coord
PY="env -u PYTHONPATH ../../../venv/bin/python"
$PY daily_anchor_test.py   # exports the daily HMI FITS via drms + first sunpy cross-checks
$PY daily_anchor_v2.py     # regular HGC grid, sunpy on both sides (control + model tests)
$PY daily_anchor_v3.py     # model-vs-synoptic, strong-field subsets, shift scans (+ HGC figure)
$PY daily_anchor_v4.py     # feature table + radio-map overlay figure
```

Expected numbers: control daily-vs-synoptic strong-field r ≈ 0.86; model-vs-synoptic as-is
r = 0.163 → 0.380 (1° → 4.2° smoothing); residual shift (−0.8°, −1.9°); formula-vs-sunpy
RMS 71.8″ (= the B0 term only).

## Files

* `coord_validation.md` — this memo.
* `daily_anchor_v2.py`, `daily_anchor_v3.py`, `daily_anchor_v4.py` — final validation suite
  (sunpy transforms only).
* `validate_orientation.py`, `daily_anchor_test.py` — first-pass harnesses, superseded
  (they contain the hand-rolled CRLN/CEA reader that caused the initial false alarm); kept
  for the record.
* `orientation_hgc_compare.png` — daily / synoptic / model / longitude-mirrored control on
  the common Carrington grid.
* `orientation_on_radio_map.png` — Tb 60 MHz with AR reference positions: production
  formula (green circles) vs sunpy B0-aware (white crosses).
* `daily_anchor_results.json`, `anchor_v2_results.json`, `anchor_v3_results.json`,
  `anchor_v4_results.json` — machine-readable results.
* Data files (FITS/npz) are not committed; fetch with the commands above.
