#!/usr/bin/env python
"""Anchor v4: guarded feature table, HGC comparison figure, and radio-map overlay.

Produces:
  orientation_hgc_compare.png   - daily / synoptic / model / mirrored-control on HGC grid
  orientation_on_radio_map.png  - Tb 60 MHz with AR positions under production formula vs sunpy(B0-aware)
  anchor_v4_results.json
"""
import json

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from scipy.ndimage import maximum_filter, label, uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/data1/pzhang/dev/QuietSunPol"
WORK = f"{ROOT}/out/orient_validation"

d = np.load(f"{WORK}/hgc_grids.npz")
LL, BB, Dg, Sg, Mg = d["LL"], d["BB"], d["Dg"], d["Sg"], d["Mg"]
lams, bets = d["lams"], d["bets"]
RES = {}

# reference sky frame
import sunpy.map
from sunpy.coordinates import sun, frames
EPOCH = Time("2025-09-10T12:00:00", scale="utc")
L0 = float(sun.L0(EPOCH).to_value("deg"))
B0 = float(sun.B0(EPOCH).to_value("deg"))
R_ANG = float(sun.angular_radius(EPOCH).to_value(u.arcsec))
PHI0 = 342.52
print("L0=%.4f B0=%.4f R_ang=%.1f arcsec" % (L0, B0, R_ANG))


# ---------- feature table (guarded)
def local_max_at(A, lam, bet, box=5.0):
    sel = (np.abs(BB - bet) < box) & (np.abs(((LL - lam + 180) % 360) - 180) < box)
    if sel.sum() < 4:
        return None
    Aa = np.where(np.isfinite(A), np.abs(A), np.nan)
    if np.all(~np.isfinite(Aa[sel])):
        return None
    idx = np.nanargmax(np.where(sel, Aa, np.nan))
    jj, ii = np.unravel_index(idx, Aa.shape)
    dlam = float(((LL[jj, ii] - lam + 180) % 360) - 180)
    return float(Aa[jj, ii]), dlam, float(BB[jj, ii] - bet)


print("\n== strongest synoptic features vs daily & model (box=5 deg) ==")
Sv = np.where(np.isfinite(Sg), np.abs(Sg), 0.0)
loc = (maximum_filter(Sv, size=9) == Sv) & (Sv > 300)
lab, n = label(loc)
feats = []
for i in range(1, n + 1):
    jj, ii = np.where(lab == i)
    k = int(np.argmax(Sv[jj, ii]))
    feats.append((float(Sv[jj[k], ii[k]]), float(LL[jj[k], ii[k]]), float(BB[jj[k], ii[k]])))
feats.sort(reverse=True)
rows = []
for val, lam, bet in feats[:8]:
    rD = local_max_at(Dg, lam, bet)
    rM = local_max_at(Mg, lam, bet)
    row = dict(syn_val=val, lam=lam, bet=bet,
               daily=None if rD is None else dict(v=rD[0], dl=rD[1], db=rD[2]),
               model=None if rM is None else dict(v=rM[0], dl=rM[1], db=rM[2]))
    rows.append(row)
    print("  syn %6.0f G @ (%6.1f,%+5.1f):  daily %s   model %s" % (
        val, lam, bet,
        "n/a (off-disk)" if rD is None else "%5.0f G @ dl=%+4.1f db=%+4.1f" % (rD[0], rD[1], rD[2]),
        "n/a" if rM is None else "%5.0f G @ dl=%+4.1f db=%+4.1f" % (rM[0], rM[1], rM[2])))
RES["feature_table"] = rows

# ---------- figure 1: HGC compare
fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
ext = [lams[0], lams[-1], bets[0], bets[-1]]
panels = [
    ("SDO/HMI daily LOS  2025-09-10 12:00 UT", Dg, 300),
    ("HMI synoptic Br (CR 2302)", Sg, 300),
    ("model MAS boundary br (CR 2302)", Mg, 60),
    ("control: model mirrored in longitude", np.fliplr(Mg), 60),
]
markers = [(5.0, -14.5, "AR-A"), (50.7, 27.2, "AR-B"), (51.3, 3.2, "syn-max")]
for ax, (title, A, vlim) in zip(axes.ravel(), panels):
    im = ax.imshow(A, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Carrington lon [deg]")
    ax.set_ylabel("lat [deg]")
    fig.colorbar(im, ax=ax, label="G", shrink=0.8)
    for lam0, bet0, nm in markers:
        ax.plot(lam0, bet0, marker="o", ms=10, mfc="none", mec="lime", mew=1.6)
        ax.annotate(nm, (lam0, bet0), textcoords="offset points", xytext=(8, 6), color="lime", fontsize=9)
for ax in axes.ravel():
    ax.set_xlim(L0 - 95, L0 + 95)
fig.suptitle("Orientation check on a common (Carrington lon, lat) grid - 2025-09-10 (L0=%.1f)" % L0, fontsize=14)
fig.savefig(f"{WORK}/orientation_hgc_compare.png", dpi=140)
plt.close(fig)
print("saved orientation_hgc_compare.png")

# ---------- figure 2: radio map overlay
rd = np.load(f"{ROOT}/raytracingGRFF/out/raytrace_60MHz.npz")
tb = rd["emission_cube"][:, :, 0]
xr = rd["x_coords"] / 6.957e8
yr = rd["y_coords"] / 6.957e8
ext2 = [xr[0], xr[-1], yr[0], yr[-1]]

fig, ax = plt.subplots(figsize=(7.6, 6.6), constrained_layout=True)
im = ax.imshow(tb, origin="lower", extent=ext2, cmap="inferno", aspect="equal")
ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, edgecolor="cyan", lw=0.8, ls=":"))
legend_lines = []
for lam0, bet0, nm in [(5.0, -14.5, "AR-A"), (50.7, 27.2, "AR-B"), (51.3, 3.2, "syn-max")]:
    b = np.deg2rad(bet0)
    xp = np.cos(b) * np.sin(np.deg2rad(lam0 - PHI0))
    yp = np.sin(b)
    sc = SkyCoord(lam0 * u.deg, bet0 * u.deg, 1 * u.R_sun,
                  frame=frames.HeliographicCarrington(obstime=EPOCH, observer="earth"))
    hp = sc.transform_to(frames.Helioprojective(obstime=EPOCH, observer="earth"))
    xs_ = float(hp.Tx.to_value(u.arcsec)) / R_ANG
    ys_ = float(hp.Ty.to_value(u.arcsec)) / R_ANG
    ax.plot(xp, yp, marker="o", ms=11, mfc="none", mec="lime", mew=1.8)
    ax.plot(xs_, ys_, marker="x", ms=9, mec="white", mew=1.8)
    ax.annotate(nm, (xp, yp), textcoords="offset points", xytext=(9, 7), color="lime", fontsize=10)
    legend_lines.append("%s: production (formula) (%+.2f,%+.2f) vs sunpy(B0) (%+.2f,%+.2f)" % (nm, xp, yp, xs_, ys_))
ax.set_xlabel("x west (R_sun)"); ax.set_ylabel("y north (R_sun)")
ax.set_title("Tb 60 MHz 2025-09-10 with AR reference positions\n(o = production formula (B0 ignored), x = sunpy with B0=+%.2f deg)" % B0)
fig.colorbar(im, ax=ax, label="Tb (K)", shrink=0.85)
fig.savefig(f"{WORK}/orientation_on_radio_map.png", dpi=150)
plt.close(fig)
for l in legend_lines:
    print(l)
print("saved orientation_on_radio_map.png")

with open(f"{WORK}/anchor_v4_results.json", "w") as fh:
    json.dump(RES, fh, indent=2, default=float)
print("DONE")
