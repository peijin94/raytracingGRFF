#!/usr/bin/env python
"""Orientation validation for the QuietSunPol 2025-09-10 synthesis.

Tests:
  T1  model boundary map br_r0 (Carrington synoptic) vs real HMI Br synoptic for CR 2302:
      as-is correlation + deliberate phi/lat/sign flips + best (dlon,dlat) shift search.
  T2  production display formula (cart_to_sph with phi0) vs sunpy Helioprojective
      coordinate transforms: residual structure + flip controls.
  T3  end-to-end disk view: model vs real field both sampled through the production
      formula's (lon,lat) assignment; plus top active-region position comparison.

Run: env -u PYTHONPATH <venv>/bin/python validate_orientation.py   (cwd anywhere)

NOTE (superseded): the hand-rolled synoptic WCS reader in Section 1 (used by T1/T3) was
later found to have a wrap-direction bug for the CRLN/CEA format (CDELT1=-0.1,
CRVAL1~828540). The final validation uses sunpy world_to_pixel for BOTH maps and lives in
daily_anchor_v2/v3/v4.py; this first-pass harness is kept for the record only.
"""
import json
import sys
import numpy as np

ROOT = "/data1/pzhang/dev/QuietSunPol"
WORK = f"{ROOT}/out/orient_validation"
sys.path.insert(0, f"{ROOT}/raytracingGRFF")

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from pyhdf.SD import SD, SDC
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import maximum_filter

from raytracingGRFF.coords import cart_to_sph  # production mapping (same code the pipeline uses)

EPOCH = Time("2025-09-10T12:00:00", scale="utc")
PHI0 = 342.52
RESULTS = {}

# ----------------------------------------------------------------------------
# 0. Epoch reference (sunpy)
# ----------------------------------------------------------------------------
from sunpy.coordinates import sun, frames
L0 = float(sun.L0(EPOCH).to_value("deg"))
B0 = float(sun.B0(EPOCH).to_value("deg"))
PANG = float(sun.P(EPOCH).to_value("deg"))
D_RSUN = float(sun.earth_distance(EPOCH).to_value(u.R_sun))
ANG_ARCSEC = float(sun.angular_radius(EPOCH).to_value(u.arcsec))
print("== epoch 2025-09-10T12:00 UT ==")
print("L0=%.4f  B0=%.4f  P=%.4f  dist=%.3f R_sun  ang.radius=%.1f arcsec" % (L0, B0, PANG, D_RSUN, ANG_ARCSEC))
RESULTS["epoch"] = dict(L0=L0, B0=B0, P=PANG, dist_rsun=D_RSUN, ang_arcsec=ANG_ARCSEC)

# ----------------------------------------------------------------------------
# 1. HMI CR 2302 Br synoptic: manual WCS sampler (+ sunpy cross-check)
# ----------------------------------------------------------------------------
hdul = fits.open(f"{WORK}/hmi.B_synoptic_Br.2302.fits")
hdr = hdul[0].header
Bmap = hdul[0].data.astype(np.float64)
NY, NX = Bmap.shape
CRVAL1, CDELT1, CRPIX1 = float(hdr["CRVAL1"]), float(hdr["CDELT1"]), float(hdr["CRPIX1"])
CRVAL2, CDELT2, CRPIX2 = float(hdr["CRVAL2"]), float(hdr["CDELT2"]), float(hdr["CRPIX2"])
print("synoptic: %dx%d  CRVAL1=%.1f CDELT1=%.4f  CRVAL2=%.1f CDELT2=%.9f" % (NX, NY, CRVAL1, CDELT1, CRVAL2, CDELT2))


def hmi_lonlat_to_xy(lon, lat):
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    lon_eff = np.mod(lon, 360.0) + 360.0 * np.floor(CRVAL1 / 360.0)
    x = (lon_eff - CRVAL1) / CDELT1 + (CRPIX1 - 1.0)
    y = (np.sin(np.deg2rad(lat)) - CRVAL2) / CDELT2 + (CRPIX2 - 1.0)
    return x, y


def bilinear(field, x, y):
    x = np.clip(np.asarray(x, float), 0.0, field.shape[1] - 1.000001)
    y = np.clip(np.asarray(y, float), 0.0, field.shape[0] - 1.000001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = x - x0
    fy = y - y0
    return (field[y0, x0] * (1 - fx) * (1 - fy) + field[y0, x0 + 1] * fx * (1 - fy)
            + field[y0 + 1, x0] * (1 - fx) * fy + field[y0 + 1, x0 + 1] * fx * fy)


def hmi_sample(lon, lat):
    x, y = hmi_lonlat_to_xy(lon, lat)
    return bilinear(Bmap, x, y)


try:
    import sunpy.map
    hm = sunpy.map.Map(f"{WORK}/hmi.B_synoptic_Br.2302.fits")
    rng = np.random.default_rng(0)
    tl = rng.uniform(0, 360, 40)
    tb = rng.uniform(-80, 80, 40)
    sx, sy = hm.world_to_pixel(SkyCoord(tl * u.deg, tb * u.deg, frame=hm.coordinate_frame))
    mx, my = hmi_lonlat_to_xy(tl, tb)
    dpx = float(np.nanmax(np.abs(np.asarray(sx) - mx)))
    dpy = float(np.nanmax(np.abs(np.asarray(sy) - my)))
    print("manual WCS vs sunpy pixels: max |dx|=%.3g |dy|=%.3g  (0 => identical)" % (dpx, dpy))
    RESULTS["wcs_crosscheck_px"] = [dpx, dpy]
except Exception as e:
    print("sunpy map cross-check unavailable:", repr(e))
    RESULTS["wcs_crosscheck_px"] = None

# ----------------------------------------------------------------------------
# 2. model boundary br_r0 (Carrington grid; phi 0..360, theta 0..180 -> lat +90..-90)
# ----------------------------------------------------------------------------
f = SD(f"{ROOT}/corona/br_r0.hdf", SDC.READ)
marr = np.asarray(f.select("Data-Set-2")[:]).astype(np.float64)
f.end()
phi_nodes = np.arange(361) * 1.0
lat_nodes = np.linspace(-90.0, 90.0, 181)
mF = marr[:, ::-1]  # theta j -> lat 90-j ; reversed axis -> ascending lat -90..90
model_lonlat = RegularGridInterpolator((phi_nodes, lat_nodes), mF, bounds_error=False, fill_value=np.nan)


def model_sample_brr0(lon, lat):
    pts = np.column_stack([np.ravel(np.mod(lon, 360.0)), np.ravel(np.clip(lat, -90, 90))])
    return model_lonlat(pts).reshape(np.shape(lon))


print("br_r0 range: %.1f..%.1f  (HMI Br range: %.1f..%.1f)" % (marr.min(), marr.max(), Bmap.min(), Bmap.max()))


def corr(a, b):
    a = np.ravel(np.asarray(a, float))
    b = np.ravel(np.asarray(b, float))
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


# ----------------------------------------------------------------------------
# T1: synoptic correlation + flips + shifts
# ----------------------------------------------------------------------------
lons = np.arange(0.5, 360.0, 2.0)
lats = np.arange(-76.0, 76.01, 2.0)
LL, BB = np.meshgrid(lons, lats, indexing="ij")
Mv = model_sample_brr0(LL, BB)
R0 = hmi_sample(LL, BB)
t1 = {}
t1["as_is"] = corr(Mv, R0)
t1["phi_flip"] = corr(Mv[::-1, :], R0)
t1["lat_flip"] = corr(Mv[:, ::-1], R0)
t1["both_flip"] = corr(Mv[::-1, ::-1], R0)
t1["sign_flip"] = corr(-Mv, R0)
best = None
for dlon in np.arange(-40.0, 40.01, 2.0):
    for dlat in np.arange(-10.0, 10.01, 2.0):
        c = corr(Mv, hmi_sample(LL + dlon, BB + dlat))
        if best is None or c > best[0]:
            best = (c, dlon, dlat)
c0, dl0, db0 = best
for dlon in np.arange(dl0 - 2.0, dl0 + 2.01, 0.5):
    for dlat in np.arange(db0 - 2.0, db0 + 2.01, 0.5):
        c = corr(Mv, hmi_sample(LL + dlon, BB + dlat))
        if c > best[0]:
            best = (c, dlon, dlat)
t1["best_shift"] = dict(corr=best[0], dlon_deg=best[1], dlat_deg=best[2])
# polarity slope
mv = np.ravel(Mv); rv = np.ravel(R0)
mm = np.isfinite(mv) & np.isfinite(rv) & (np.abs(rv) > 20)
slope = float(np.polyfit(mv[mm], rv[mm], 1)[0]) if mm.sum() > 50 else float("nan")
t1["polarity_slope"] = slope
RESULTS["T1_synoptic"] = t1
print("\n== T1 synoptic correlation (model br_r0 vs HMI Br CR2302) ==")
for k, v in t1.items():
    if k != "best_shift":
        print("   %-10s r = %+.3f" % (k, v))
print("   best shift: dlon=%+.1f dlat=%+.1f deg -> r=%.3f" % (t1["best_shift"]["dlon_deg"], t1["best_shift"]["dlat_deg"], t1["best_shift"]["corr"]))
print("   polarity slope (HMI vs model) = %+.4f" % slope)

# ----------------------------------------------------------------------------
# T2: production formula vs sunpy Helioprojective
# ----------------------------------------------------------------------------
lam_g, bet_g = np.meshgrid(L0 + np.arange(-80, 80.01, 10.0), np.arange(-60, 60.01, 10.0))
c = SkyCoord(lam_g * u.deg, bet_g * u.deg, np.ones_like(lam_g) * u.R_sun,
             frame=frames.HeliographicCarrington(obstime=EPOCH, observer="earth"))
hpc = c.transform_to(frames.Helioprojective(obstime=EPOCH, observer="earth"))
Tx_s = hpc.Tx.to_value(u.arcsec)
Ty_s = hpc.Ty.to_value(u.arcsec)


def formula_sky(lon_deg, lat_deg):
    dlon = np.deg2rad(lon_deg - PHI0)
    b = np.deg2rad(lat_deg)
    x = np.cos(b) * np.sin(dlon)
    y = np.sin(b)
    z = np.cos(b) * np.cos(dlon)
    txf = np.rad2deg(np.arctan2(x, D_RSUN - z)) * 3600.0
    tyf = np.rad2deg(np.arctan2(y, D_RSUN - z)) * 3600.0
    return txf, tyf


Tx_f, Ty_f = formula_sky(lam_g, bet_g)
dTx = Tx_f - Tx_s
dTy = Ty_f - Ty_s
# control: formulas with lon and lat flips
Txf_flip, Tyf_flip = formula_sky(2 * L0 - lam_g, bet_g)   # lon mirrored about L0
Txb_flip, Tyb_flip = formula_sky(lam_g, -bet_g)           # lat mirrored
rms = lambda a, b: float(np.sqrt(np.nanmean((a - b) ** 2)))
t2 = dict(
    rms_as_is=float(np.sqrt(np.nanmean(dTx ** 2 + dTy ** 2))),
    rms_lon_flip=float(np.sqrt(np.nanmean((Txf_flip - Tx_s) ** 2 + (Tyf_flip - Ty_s) ** 2))),
    rms_lat_flip=float(np.sqrt(np.nanmean((Txb_flip - Tx_s) ** 2 + (Tyb_flip - Ty_s) ** 2))),
    dTy_center_arcsec=float(dTy[np.argmin(np.abs(bet_g) + np.abs(lam_g - L0))]),
    corr_dTy_with_z=float(np.corrcoef(np.ravel(dTy), np.ravel(np.cos(np.deg2rad(bet_g)) * np.cos(np.deg2rad(lam_g - PHI0))))[0, 1]),
)
RESULTS["T2_formula_vs_sunpy"] = t2
print("\n== T2 formula vs sunpy (arcsec RMS over lon=L0+-80, lat=+-60 grid) ==")
print("   as-is    RMS = %.1f" % t2["rms_as_is"])
print("   lon-flip RMS = %.1f   lat-flip RMS = %.1f" % (t2["rms_lon_flip"], t2["rms_lat_flip"]))
print("   dTy at disk center = %+.1f arcsec   (B0 = %+.4f deg, sunpy convention)" % (t2["dTy_center_arcsec"], B0))
print("   corr(dTy, cosb*cos(dlon)) = %+.3f  (1 => pure B0-tilt pattern)" % t2["corr_dTy_with_z"])

# ----------------------------------------------------------------------------
# T3: disk view through the production formula, model vs real
# ----------------------------------------------------------------------------
n = 241
g = np.linspace(-1.25, 1.25, n)
XX, YY = np.meshgrid(g, g)
rr2 = XX ** 2 + YY ** 2
disk = rr2 <= 1.0
ZZ = np.sqrt(np.maximum(1.0 - rr2, 0.0))
r_, colat, lon_rad = cart_to_sph(XX, YY, ZZ, phi0_offset=PHI0)
lam = np.rad2deg(lon_rad)
bet = 90.0 - np.rad2deg(colat)
Mdisk = model_sample_brr0(lam, bet)
Rdisk = hmi_sample(lam, bet)
md = disk & np.isfinite(Mdisk) & np.isfinite(Rdisk)
t3 = dict(
    corr_as_is=corr(Mdisk[md], Rdisk[md]),
    corr_model_flipx=corr(model_sample_brr0(2 * PHI0 - lam, bet)[md], Rdisk[md]),
    corr_model_flipy=corr(model_sample_brr0(lam, -bet)[md], Rdisk[md]),
    corr_signflip=corr(-Mdisk[md], Rdisk[md]),
)
RESULTS["T3_diskview"] = t3
print("\n== T3 disk view (production formula coords; model vs HMI synoptic mapped to sky) ==")
for k, v in t3.items():
    print("   %-16s r = %+.3f" % (k, v))

# top ARs in the real synoptic, visible at 12:00, and their model counterparts
absB = np.abs(Bmap)
loc = (maximum_filter(absB, size=(61, 61)) == absB) & (absB > 200.0)
jj, ii = np.where(loc)
cand = [(float(absB[jj[k], ii[k]]), int(ii[k]), int(jj[k])) for k in range(len(jj))]
cand.sort(reverse=True)
ar_rows = []
for mag, i0, j0 in cand:
    lon_e = CRVAL1 + (i0 - (CRPIX1 - 1.0)) * CDELT1
    lon_ar = float(np.mod(lon_e, 360.0))
    slat = CRVAL2 + (j0 - (CRPIX2 - 1.0)) * CDELT2
    lat_ar = float(np.rad2deg(np.arcsin(np.clip(slat, -1, 1))))
    dlon_vis = (lon_ar - L0 + 180.0) % 360.0 - 180.0
    if abs(lat_ar) > 78 or abs(dlon_vis) > 88:
        continue
    xf = np.cos(np.deg2rad(lat_ar)) * np.sin(np.deg2rad(lon_ar - PHI0))
    yf = np.sin(np.deg2rad(lat_ar))
    # search model peak within 0.35 R_sun of predicted position
    sel = ((XX - xf) ** 2 + (YY - yf) ** 2) < 0.35 ** 2
    if sel.sum() == 0 or np.all(~np.isfinite(Mdisk[sel])):
        ar_rows.append((mag, lon_ar, lat_ar, xf, yf, None, None))
        continue
    idx = np.nanargmax(np.abs(Mdisk[sel]))
    xs = XX[sel][idx]; ys = YY[sel][idx]
    dx = xs - xf; dy = ys - yf
    ar_rows.append((mag, lon_ar, lat_ar, xf, yf, dx, dy))
    if len(ar_rows) >= 6:
        break
RESULTS["T3_AR_pairs"] = [dict(mag=r[0], lon=r[1], lat=r[2], x_pred=r[3], y_pred=r[4], dx=r[5], dy=r[6]) for r in ar_rows]
print("\n== top visible ARs: real synoptic (lon,lat) -> formula-sky (x,y) -> model counterpart ==")
for r in ar_rows:
    if r[5] is None:
        print("   |B|=%6.0f at (%6.1f,%+5.1f) -> (x=%+.3f,y=%+.3f): no model peak within 0.35 R_sun" % (r[0], r[1], r[2], r[3], r[4]))
    else:
        sep = float(np.hypot(r[5], r[6]))
        print("   |B|=%6.0f at (%6.1f,%+5.1f) -> (x=%+.3f,y=%+.3f): model peak offset dx=%+.3f dy=%+.3f (%.0f arcsec)  [%+.2f R_sun]"
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], sep * ANG_ARCSEC, sep))

# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
ax = axes[0, 0]
im = ax.imshow(Mv.T, origin="lower", extent=[0, 360, -76, 76], aspect="auto", cmap="seismic", vmin=-150, vmax=150)
ax.set_title("model br_r0 (CR 2302 synoptic)")
ax.set_xlabel("Carrington lon [deg]"); ax.set_ylabel("lat [deg]"); fig.colorbar(im, ax=ax, label="G")
ax = axes[0, 1]
im = ax.imshow(R0.T, origin="lower", extent=[0, 360, -76, 76], aspect="auto", cmap="seismic", vmin=-150, vmax=150)
ax.set_title("HMI Br synoptic (CR 2302, real)")
ax.set_xlabel("Carrington lon [deg]"); fig.colorbar(im, ax=ax, label="G")
ax = axes[0, 2]
im = ax.imshow(Mdisk, origin="lower", extent=[-1.25, 1.25, -1.25, 1.25], cmap="seismic", vmin=-80, vmax=80)
ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, edgecolor="k", lw=0.6))
ax.set_title("model disk view (phi0=%.1f)" % PHI0); ax.set_xlabel("x west [R_sun]"); ax.set_ylabel("y north [R_sun]")
fig.colorbar(im, ax=ax, label="G")
ax = axes[1, 0]
im = ax.imshow(Rdisk, origin="lower", extent=[-1.25, 1.25, -1.25, 1.25], cmap="seismic", vmin=-80, vmax=80)
ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, edgecolor="k", lw=0.6))
ax.set_title("real synoptic mapped via production coords")
ax.set_xlabel("x west [R_sun]"); fig.colorbar(im, ax=ax, label="G")
ax = axes[1, 1]
try:
    jpg = plt.imread(f"{WORK}/20250910_111500_2048_HMIB.jpg")
    ax.imshow(jpg, cmap="gray")
    ax.set_title("SDO/HMI magnetogram 2025-09-10 11:15 UT")
except Exception as e:
    ax.set_title("daily magnetogram missing")
ax = axes[1, 2]
Mflip = model_sample_brr0(2 * PHI0 - lam, bet)
im = ax.imshow(Mflip, origin="lower", extent=[-1.25, 1.25, -1.25, 1.25], cmap="seismic", vmin=-80, vmax=80)
ax.set_title("control: model disk with lon mirror (wrong by construction)")
ax.set_xlabel("x west [R_sun]")
fig.suptitle("Orientation validation - 2025-09-10 (CR 2302)")
fig.savefig(f"{WORK}/orientation_validation.png", dpi=140)
plt.close(fig)

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 6.5), constrained_layout=True)
im = axes2[0].imshow(Mdisk, origin="lower", extent=[-1.25, 1.25, -1.25, 1.25], cmap="seismic", vmin=-80, vmax=80)
axes2[0].add_patch(plt.Circle((0, 0), 1.0, fill=False, edgecolor="k", lw=0.7))
axes2[0].set_title("model boundary field, disk view (production mapping)")
axes2[0].set_xlabel("x west [R_sun]"); axes2[0].set_ylabel("y north [R_sun]")
fig2.colorbar(im, ax=axes2[0], label="G")
im = axes2[1].imshow(Rdisk, origin="lower", extent=[-1.25, 1.25, -1.25, 1.25], cmap="seismic", vmin=-80, vmax=80)
axes2[1].add_patch(plt.Circle((0, 0), 1.0, fill=False, edgecolor="k", lw=0.7))
axes2[1].set_title("HMI CR2302 synoptic, same mapping")
axes2[1].set_xlabel("x west [R_sun]")
fig2.colorbar(im, ax=axes2[1], label="G")
fig2.savefig(f"{WORK}/orientation_disk_compare.png", dpi=150)
plt.close(fig2)

with open(f"{WORK}/orientation_results.json", "w") as fh:
    json.dump(RESULTS, fh, indent=2, default=float)
print("\nFigure + results written to", WORK)
