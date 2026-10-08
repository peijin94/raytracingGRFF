#!/usr/bin/env python
"""Orientation anchor v2: regular HGC grid, sunpy transforms on BOTH sides.

- daily HMI M_720s 2025-09-10 12:00 TAI (already downloaded)
- HMI CR2302 synoptic FITS
- model br_r0 boundary
All compared on a regular (lambda, beta) grid around disk center; uses sunpy
world<->pixel transforms only (no manual WCS). Wide shift scans + feature matching.
"""
import json
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits as afits
from pyhdf.SD import SD, SDC
from scipy.interpolate import RegularGridInterpolator

ROOT = "/data1/pzhang/dev/QuietSunPol"
WORK = f"{ROOT}/out/orient_validation"
import sys
sys.path.insert(0, f"{ROOT}/raytracingGRFF")
from raytracingGRFF.coords import cart_to_sph

import sunpy.map
from sunpy.coordinates import frames, sun
from astropy.time import Time

DFILE = f"{WORK}/hmi.M_720s.20250910_1200.fits"
SYN = f"{WORK}/hmi.B_synoptic_Br.2302.fits"
RES = {}

m = sunpy.map.Map(DFILE)
hm = sunpy.map.Map(SYN)
EPOCH = Time("2025-09-10T12:00:00", scale="utc")
L0 = float(sun.L0(EPOCH).to_value("deg"))
B0 = float(sun.B0(EPOCH).to_value("deg"))
print("L0=%.4f B0=%.4f  daily date=%s" % (L0, B0, m.date))

# ---- model boundary
f = SD(f"{ROOT}/corona/br_r0.hdf", SDC.READ)
marr = np.asarray(f.select("Data-Set-2")[:]).astype(np.float64)
f.end()
model_ll = RegularGridInterpolator((np.arange(361) * 1.0, np.linspace(-90, 90, 181)),
                                   marr[:, ::-1], bounds_error=False, fill_value=np.nan)


def model_br(lon, lat):
    pts = np.column_stack([np.ravel(np.mod(lon, 360.0)), np.ravel(np.clip(lat, -90, 90))])
    return model_ll(pts).reshape(np.shape(lon))


def bilinear(field, x, y):
    x = np.clip(np.asarray(x, float), 0.0, field.shape[1] - 1.000001)
    y = np.clip(np.asarray(y, float), 0.0, field.shape[0] - 1.000001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = x - x0
    fy = y - y0
    return (field[y0, x0] * (1 - fx) * (1 - fy) + field[y0, x0 + 1] * fx * (1 - fy)
            + field[y0 + 1, x0] * (1 - fx) * fy + field[y0 + 1, x0 + 1] * fx * fy)


# ---- regular HGC grid around disk center
dstep = 0.6
lams = L0 + np.arange(-95.0, 95.001, dstep)
bets = np.arange(-85.0, 85.001, dstep)
LL, BB = np.meshgrid(np.mod(lams, 360.0), bets)   # (nbet, nlam)
print("HGC grid:", LL.shape)

# ---- daily map sampled through sunpy (HGC -> HPC(daily) -> pixel)
sc = SkyCoord(LL * u.deg, BB * u.deg, np.ones_like(LL) * u.R_sun,
              frame=frames.HeliographicCarrington(obstime=m.date, observer="earth"))
hp = sc.transform_to(m.coordinate_frame)
xq, yq = m.world_to_pixel(hp)
xd = np.asarray(xq.value if hasattr(xq, "value") else xq, float)
yd = np.asarray(yq.value if hasattr(yq, "value") else yq, float)
okpix = (xd >= 0) & (xd < m.data.shape[1] - 1) & (yd >= 0) & (yd < m.data.shape[0] - 1)
rho_pix = np.hypot(xd - (float(m.reference_pixel.x.value) - 1.0),
                   yd - (float(m.reference_pixel.y.value) - 1.0))
r_sun_pix = float(m.rsun_obs.value) / float(m.scale.axis1.value)
vis = okpix & (rho_pix < 0.99 * r_sun_pix)
Dg = np.full(LL.shape, np.nan)
Dg[vis] = bilinear(m.data.astype(np.float64), xd[vis], yd[vis])
print("daily sampled: %d/%d visible; D range %.0f..%.0f  (p1,p99=%.0f,%.0f)" % (
    vis.sum(), vis.size, np.nanmin(Dg), np.nanmax(Dg),
    np.nanpercentile(Dg, 1), np.nanpercentile(Dg, 99)))

# ---- synoptic sampled through sunpy (no manual reader)
sc2 = SkyCoord(LL * u.deg, BB * u.deg, frame=hm.coordinate_frame)
xq2, yq2 = hm.world_to_pixel(sc2)
xs = np.asarray(xq2.value, float)
ys = np.asarray(yq2.value, float)
ok2 = (xs >= 0) & (xs < hm.data.shape[1] - 1) & (ys >= 0) & (ys < hm.data.shape[0] - 1)
Sg = np.full(LL.shape, np.nan)
Sg[ok2] = bilinear(hm.data.astype(np.float64), xs[ok2], ys[ok2])
print("synoptic sampled: %d/%d" % (ok2.sum(), ok2.size))

Mg = model_br(LL, BB)

base = np.isfinite(Dg) & np.isfinite(Sg) & np.isfinite(Mg)
print("common valid:", base.sum())


def cc(a, b, sel):
    a = np.asarray(a, float)[sel]
    b = np.asarray(b, float)[sel]
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 100 else np.nan


def box(a, k=3):
    from scipy.ndimage import uniform_filter
    a = np.asarray(a, float)
    b = uniform_filter(np.nan_to_num(a, nan=0.0), size=k)
    w = uniform_filter(np.isfinite(a).astype(float), size=k)
    return np.where(w > 0.5, b / np.maximum(w, 1e-9), np.nan)


print("\n== control: daily vs HMI synoptic (sunpy both sides) ==")
for nm, S in [("as_is", Sg), ("lon_mirror", np.fliplr(Sg)), ("lat_mirror", np.flipud(Sg)),
              ("both", np.flipud(np.fliplr(Sg))), ("as_is_smoothed", box(Sg))]:
    print("   %-16s r = %+.3f" % (nm, cc(Dg if not nm.endswith("smoothed") else box(Dg), S, base)))

print("\n== daily vs MODEL br_r0 ==")
for nm, M in [("as_is", Mg), ("lon_mirror", np.fliplr(Mg)), ("lat_mirror", np.flipud(Mg)),
              ("both", np.flipud(np.fliplr(Mg)))]:
    print("   %-12s r = %+.3f   (sign-flip %+.3f)" % (nm, cc(M, Dg, base), cc(-M, Dg, base)))
print("   as-is smoothed: r = %+.3f" % cc(box(Mg), box(Dg), base))

print("\n== daily vs synoptic, smoothed ==")
for nm, S in [("as_is", Sg), ("lon_mirror", np.fliplr(Sg)), ("lat_mirror", np.flipud(Sg)),
              ("both", np.flipud(np.fliplr(Sg)))]:
    print("   %-12s r = %+.3f" % (nm, cc(box(S), box(Dg), base)))


# ---- wide shift scan: find (dlam, dbet) maximizing corr after smoothing
def shift_scan(A, B, tag, lo=-180.0, hi=180.0, step=2.0, bstep=2.0):
    best = None
    nl = A.shape[1]
    shift_lon = int(round(step / dstep))
    shift_bet = int(round(bstep / dstep))
    for ilon in range(int(round(lo / step)), int(round(hi / step)) + 1):
        for jbet in range(int(round(-30 / bstep)), int(round(30 / bstep)) + 1):
            As = np.roll(A, ilon * shift_lon, axis=1)
            As = np.roll(As, jbet * shift_bet, axis=0)
            c = cc(As, B, base)
            if np.isfinite(c) and (best is None or c > best[0]):
                best = (c, ilon * step, jbet * bstep)
    print("   %s wide scan: best dlon=%+.1f dlat=%+.1f -> r=%.3f" % (tag, best[1], best[2], best[0]))
    RES[f"widescan_{tag}"] = dict(r=best[0], dlon=best[1], dlat=best[2])
    # fine around best
    b2 = None
    for ilon in np.arange(best[1] - 3, best[1] + 3.01, 0.6):
        for jbet in np.arange(best[2] - 3, best[2] + 3.01, 0.6):
            As = np.roll(np.roll(A, int(round(ilon / dstep)), axis=1), int(round(jbet / dstep)), axis=0)
            c = cc(As, B, base)
            if np.isfinite(c) and (b2 is None or c > b2[0]):
                b2 = (c, ilon, jbet)
    print("   %s fine scan: best dlon=%+.2f dlat=%+.2f -> r=%.3f" % (tag, b2[1], b2[2], b2[0]))
    RES[f"finescan_{tag}"] = dict(r=b2[0], dlon=b2[1], dlat=b2[2])
    return b2


shift_scan(box(Mg), box(Dg), "model_vs_daily")
shift_scan(box(Sg), box(Dg), "synoptic_vs_daily")

# ---- feature matching: strongest visible features
print("\n== strongest features ==")


def top_clusters(A, k=3, thresh=400.0):
    from scipy.ndimage import maximum_filter, label
    a = np.abs(A)
    a2 = np.where(np.isfinite(a), a, 0.0)
    loc = (maximum_filter(a2, size=15) == a2) & (a2 > thresh)
    lab, n = label(loc)
    out = []
    for i in range(1, n + 1):
        jj, ii = np.where(lab == i)
        val = a2[jj, ii].max()
        out.append((val, float(lams[ii[0]] % 360), float(bets[jj[0]]), ll_of(ii[0], jj[0]), bb_of(ii[0], jj[0])))
    out.sort(reverse=True)
    return out[:k]


def ll_of(i, j):
    return float(LL[j, i])


def bb_of(i, j):
    return float(BB[j, i])


print("-- daily strongest (VISIBLE):")
tops = top_clusters(Dg, k=5, thresh=300.0)
for val, lam, bet, _, _ in tops:
    print("   |B|=%6.0f at lam=%6.1f bet=%+5.1f" % (val, lam, bet))
for val, lam, bet, _, _ in tops[:3]:
    i = int(np.argmin(np.abs((lams % 360) - lam)))
    # model & synoptic local max within +-6 deg of this point
    def local(A):
        sel = (np.abs(BB - bet) < 6) & (np.abs(((LL - lam + 180) % 360) - 180) < 6)
        if sel.sum() == 0 or np.all(~np.isfinite(A[sel])):
            return None
        idx = np.nanargmax(np.abs(A[sel]))
        jj, ii = np.where(sel)
        return float(A[jj[idx], ii[idx]]), float(LL[jj[idx], ii[idx]]), float(BB[jj[idx], ii[idx]])

    for nm, A in (("model", Mg), ("synoptic", Sg)):
        r = local(A)
        if r is None:
            print("   [%6.1f,%+5.1f] -> %s: none within 6 deg" % (lam, bet, nm))
        else:
            dlam = (r[1] - lam + 180) % 360 - 180
            print("   [%6.1f,%+5.1f] -> %s local max %+6.1f G at dlam=%+.2f dlat=%+.2f" % (lam, bet, nm, r[0], dlam, r[2] - bet))

# save grids
np.savez(f"{WORK}/hgc_grids.npz", LL=LL, BB=BB, Dg=Dg, Sg=Sg, Mg=Mg, lams=lams, bets=bets)
with open(f"{WORK}/anchor_v2_results.json", "w") as fh:
    json.dump(RES, fh, indent=2, default=float)
print("\nDONE")
