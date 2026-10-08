#!/usr/bin/env python
"""Anchor v3: model boundary vs HMI synoptic directly + strong-field subsets + figure."""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

ROOT = "/data1/pzhang/dev/QuietSunPol"
WORK = f"{ROOT}/out/orient_validation"

d = np.load(f"{WORK}/hgc_grids.npz")
LL, BB, Dg, Sg, Mg = d["LL"], d["BB"], d["Dg"], d["Sg"], d["Mg"]
lams, bets = d["lams"], d["bets"]
RES = {}


def box(a, k):
    a = np.asarray(a, float)
    b = uniform_filter(np.nan_to_num(a, nan=0.0), size=k)
    w = uniform_filter(np.isfinite(a).astype(float), size=k)
    return np.where(w > 0.5, b / np.maximum(w, 1e-9), np.nan)


def cc(a, b, sel):
    a = np.asarray(a, float)[sel]
    b = np.asarray(b, float)[sel]
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 50 else np.nan


base = np.isfinite(Dg) & np.isfinite(Sg) & np.isfinite(Mg)
dstep = float(lams[1] - lams[0])

print("== model br_r0 vs HMI synoptic (same-epoch, both smoothed) ==")
for k in (1, 3, 5, 7):
    Ms, Ss = box(Mg, k), box(Sg, k)
    res = dict(as_is=cc(Ms, Ss, base), lon_mirror=cc(np.fliplr(Ms), Ss, base),
               lat_mirror=cc(np.flipud(Ms), Ss, base),
               sign_flip=cc(-Ms, Ss, base))
    print("  box%dx%d: as_is=%+.3f lon_mirror=%+.3f lat_mirror=%+.3f signflip=%+.3f" %
          (k, k, res["as_is"], res["lon_mirror"], res["lat_mirror"], res["sign_flip"]))
    RES[f"model_vs_synoptic_box{k}"] = res


def shift_scan(A, B, tag, rng=20.0, step=1.0):
    best = None
    n = int(round(step / dstep))
    for ilon in range(int(-rng / step), int(rng / step) + 1):
        for jbet in range(int(-4 / step), int(4 / step) + 1):
            As = np.roll(np.roll(A, ilon * n, axis=1), jbet * n, axis=0)
            c = cc(As, B, base)
            if np.isfinite(c) and (best is None or c > best[0]):
                best = (c, ilon * step, jbet * step)
    for ilon in np.arange(best[1] - 2, best[1] + 2.01, 0.3):
        for jbet in np.arange(best[2] - 2, best[2] + 2.01, 0.3):
            As = np.roll(np.roll(A, int(round(ilon / dstep)), axis=1), int(round(jbet / dstep)), axis=0)
            c = cc(As, B, base)
            if np.isfinite(c) and c > best[0]:
                best = (c, ilon, jbet)
    print("  %s shift scan: dlon=%+.2f dlat=%+.2f -> r=%.3f" % (tag, best[1], best[2], best[0]))
    RES[f"shift_{tag}"] = dict(r=best[0], dlon=best[1], dlat=best[2])


print("\n== shift scans ==")
shift_scan(box(Mg, 3), box(Sg, 3), "model_vs_synoptic")
shift_scan(box(Mg, 5), box(Sg, 5), "model_vs_synoptic_box5")
shift_scan(box(Mg, 3), box(Dg, 3), "model_vs_daily")

print("\n== strong-field subsets ==")
for thresh in (50.0, 100.0, 200.0):
    mS = base & (np.abs(Sg) > thresh)
    mD = base & (np.abs(Dg) > thresh)
    mSD = base & ((np.abs(Dg) > thresh) | (np.abs(Sg) > thresh))
    mSDa = base & ((np.abs(Dg) > thresh) & (np.abs(Sg) > thresh))
    row = dict(
        n=mS.sum(),
        corr_model_synoptic=cc(box(Mg, 3), box(Sg, 3), mS),
        corr_daily_synoptic=cc(box(Dg, 3), box(Sg, 3), mSD),
        corr_daily_synoptic_both=cc(box(Dg, 3), box(Sg, 3), mSDa),
        corr_daily_model=cc(box(Dg, 3), box(Mg, 3), mSD),
    )
    print("  |B|>%5.0f: n=%5d  model~synoptic=%+.3f  daily~synoptic=%+.3f (both) ; daily~model=%+.3f" %
          (thresh, row["n"], row["corr_model_synoptic"], row["corr_daily_synoptic_both"], row["corr_daily_model"]))
    RES[f"strongfield_{int(thresh)}"] = row

print("\n== feature check: strongest synoptic maxima and their daily/model counterparts ==")
from scipy.ndimage import maximum_filter, label
Sv = np.where(np.isfinite(Sg), np.abs(Sg), 0.0)
loc = (maximum_filter(Sv, size=9) == Sv) & (Sv > 500)
lab, n = label(loc)
feats = []
for i in range(1, n + 1):
    jj, ii = np.where(lab == i)
    k = int(np.argmax(Sv[jj, ii]))
    feats.append((float(Sv[jj[k], ii[k]]), float(LL[jj[k], ii[k]]), float(BB[jj[k], ii[k]])))
feats.sort(reverse=True)


def safe_off(A, lam, bet, box=5.0):
    sel = (np.abs(BB - bet) < box) & (np.abs(((LL - lam + 180) % 360) - 180) < box)
    if sel.sum() < 4:
        return None
    Aa = np.where(np.isfinite(A), np.abs(A), np.nan)
    if np.all(~np.isfinite(Aa[sel])):
        return None
    idx = np.nanargmax(np.where(sel, Aa, np.nan))
    jj, ii = np.unravel_index(idx, Aa.shape)
    return float(Aa[jj, ii]), float(((LL[jj, ii] - lam + 180) % 360) - 180), float(BB[jj, ii] - bet)


feat_rows = []
for val, lam, bet in feats[:5]:
    rD = safe_off(Dg, lam, bet)
    rM = safe_off(Mg, lam, bet)
    feat_rows.append(dict(syn_val=val, lam=lam, bet=bet, daily=rD, model=rM))
    print("  synoptic |B|=%6.0f at (%6.1f,%+5.1f): daily %s | model %s" % (
        val, lam, bet,
        "n/a (off-disk)" if rD is None else "%5.0f G at dl=%+4.1f db=%+4.1f" % rD,
        "n/a" if rM is None else "%5.0f G at dl=%+4.1f db=%+4.1f" % rM))
RES["features"] = feat_rows

# ---------------- figure: same HGC grid, three fields + control
fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
ext = [lams[0], lams[-1], bets[0], bets[-1]]
panels = [
    ("SDO/HMI daily LOS 2025-09-10 12:00 UT", Dg, 300, "RdBu_r", None),
    ("HMI synoptic Br (CR 2302)", Sg, 300, "RdBu_r", None),
    ("model MAS boundary br (CR 2302 run)", Mg, 60, "RdBu_r", None),
    ("control: model mirrored in longitude (wrong)", np.fliplr(Mg), 60, "RdBu_r", None),
]
for ax, (title, A, vlim, cmap, _) in zip(axes.ravel(), panels):
    im = ax.imshow(A, origin="lower", extent=ext, cmap=cmap, vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Carrington lon [deg]"); ax.set_ylabel("lat [deg]")
    fig.colorbar(im, ax=ax, label="G", shrink=0.8)
    # mark the main AR complex (lambda ~ 3, beta ~ -14.5) and the secondary (lambda ~ 50, beta ~ +27)
    for lam0, bet0 in [(3.0, -14.5), (2.5, 55.0), (50.7, 27.2)]:
        ax.plot(lam0, bet0, marker="o", ms=9, mfc="none", mec="lime", mew=1.6)
fig.suptitle("Orientation check on a common (Carrington lon, lat) grid - 2025-09-10", fontsize=14)
fig.savefig(f"{WORK}/orientation_hgc_compare.png", dpi=140)
plt.close(fig)
print("figure: orientation_hgc_compare.png")

with open(f"{WORK}/anchor_v3_results.json", "w") as fh:
    json.dump(RES, fh, indent=2, default=float)
print("DONE")
