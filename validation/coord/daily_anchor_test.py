#!/usr/bin/env python
"""Anchor the MODEL Carrington grid to real HMI data (decisive orientation test).

NOTE (superseded): the `hmi_lonlat_to_xy` manual WCS reader below has a wrap-direction
bug for the CRLN/CEA synoptic format; daily_anchor_v2/v3/v4.py use sunpy world_to_pixel
for both maps and supersede the synoptic comparisons here. Kept for the record.

Uses the daily HMI M_720s magnetogram at 2025-09-10 12:00 TAI:
  * pixel -> HPC -> HGC (lon, lat) through sunpy (independently validated chain)
  * correlate daily field against the model boundary br_r0 sampled under several
    grid-interpretation variants (as-is / lon-mirror / lat-mirror / both / sign flips)
  * same comparison for the HMI CR2302 synoptic map sampled with the manual WCS reader
  * numeric check: production display formula vs sunpy Helioprojective (T2, fixed)
"""
import io
import json
import os
import sys
import tarfile

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pyhdf.SD import SD, SDC
from scipy.interpolate import RegularGridInterpolator

ROOT = "/data1/pzhang/dev/QuietSunPol"
WORK = f"{ROOT}/out/orient_validation"
sys.path.insert(0, f"{ROOT}/raytracingGRFF")
from raytracingGRFF.coords import cart_to_sph

EMAIL = "pjer1316@icloud.com"
RECORD = "hmi.M_720s[2025.09.10_12:00:00_TAI]"
DFILE = f"{WORK}/hmi.M_720s.20250910_1200.fits"

# ------------------------------------------------------------------ fetch daily
if not os.path.exists(DFILE):
    import drms
    import requests
    c = drms.Client()
    r = c.export(RECORD, method="url-tar", protocol="fits", email=EMAIL)
    print("export status:", getattr(r, "status", None))
    uo = getattr(r, "urls", None)
    print("urls object type:", type(uo))
    print(uo if uo is None or isinstance(uo, str) else str(uo)[:600])
    if isinstance(uo, str):
        url = uo
    elif hasattr(uo, "iloc") and getattr(uo, "shape", (0,))[0] > 0:
        cols = list(uo.columns)
        col = "url" if "url" in cols else cols[-1]
        url = str(uo.iloc[0][col])
    elif uo is not None:
        url = list(uo)[0]
    else:
        raise RuntimeError("no export urls")
    print("downloading:", url[:140])
    blob = requests.get(url, timeout=600).content
    print("downloaded", len(blob), "bytes")
    if blob[:2] == b"MZ" or b"<html" in blob[:400].lower():
        raise RuntimeError("unexpected export payload (login page?): " + blob[:200].decode("latin-1", "replace"))
    if blob[:2] == b"\x1f\x8b" or b"ustar" in blob[:1000] or b"tar" in blob[:300]:
        tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:*")
        names = [n for n in tf.getnames() if n.endswith(".fits")]
        print("tar members:", names)
        with open(DFILE, "wb") as fh:
            fh.write(tf.extractfile(names[0]).read())
    else:
        with open(DFILE, "wb") as fh:
            fh.write(blob)
print("daily file:", DFILE, os.path.getsize(DFILE), "bytes")

import sunpy.map
from sunpy.coordinates import frames

m = sunpy.map.Map(DFILE)
print("map date:", m.date, " shape:", m.data.shape, " scale:", m.scale, " rsun_obs:", m.rsun_obs)
print("observer:", m.observer_coordinate)

# ------------------------------------------------------------------ model grid
f = SD(f"{ROOT}/corona/br_r0.hdf", SDC.READ)
marr = np.asarray(f.select("Data-Set-2")[:]).astype(np.float64)
f.end()
phi_nodes = np.arange(361) * 1.0
lat_nodes = np.linspace(-90.0, 90.0, 181)
model_lonlat = RegularGridInterpolator((phi_nodes, lat_nodes), marr[:, ::-1], bounds_error=False, fill_value=np.nan)


def model_br(lon, lat):
    pts = np.column_stack([np.ravel(np.mod(lon, 360.0)), np.ravel(np.clip(lat, -90, 90))])
    return model_lonlat(pts).reshape(np.shape(lon))


# HMI synoptic (manual reader)
from astropy.io import fits as afits
hdul = afits.open(f"{WORK}/hmi.B_synoptic_Br.2302.fits")
hdr = hdul[0].header
Bmap = hdul[0].data.astype(np.float64)
CRVAL1, CDELT1, CRPIX1 = float(hdr["CRVAL1"]), float(hdr["CDELT1"]), float(hdr["CRPIX1"])
CRVAL2, CDELT2, CRPIX2 = float(hdr["CRVAL2"]), float(hdr["CDELT2"]), float(hdr["CRPIX2"])


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


def syn(lon, lat):
    x, y = hmi_lonlat_to_xy(lon, lat)
    return bilinear(Bmap, x, y)


def corr(a, b, sel):
    a = np.asarray(a, float)[sel]
    b = np.asarray(b, float)[sel]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 50:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


# ------------------------------------------------------------------ daily pixels -> HGC
ny, nx = m.data.shape
step = 16
ys, xs = np.mgrid[0:ny:step, 0:nx:step]
xp = xs.ravel().astype(float)
yp = ys.ravel().astype(float)
print("grid points:", xp.size)
cx = float(m.reference_pixel.x.value - 1.0)
cy = float(m.reference_pixel.y.value - 1.0)
scale_arc = float(m.scale.axis1.value)
rsun_arc = float(m.rsun_obs.value)
rho_arc = np.hypot((xp - cx) * scale_arc, (yp - cy) * scale_arc)
keep0 = rho_arc < 0.75 * rsun_arc
print("kept for transform:", keep0.sum(), "of", xp.size)

coords_hpc = m.pixel_to_world(xp[keep0] * u.pix, yp[keep0] * u.pix)
hgc = coords_hpc.transform_to(frames.HeliographicCarrington(observer="earth", obstime=m.date))
lam = hgc.lon.deg.astype(float)
bet = hgc.lat.deg.astype(float)
D = m.data[ys.ravel()[keep0], xs.ravel()[keep0]].astype(float)
print("lam range %.2f..%.2f  bet range %.2f..%.2f" % (lam.min(), lam.max(), bet.min(), bet.max()))

sel = np.isfinite(D) & (np.abs(D) < 2000.0)
print("finite daily samples:", sel.sum())

res = {}
print("\n== daily HMI vs MODEL br_r0 (sampled at sunpy HGC) ==")
for name, L, B in [("as_is", lam, bet), ("lon_mirror", 360 - lam, bet),
                   ("lat_mirror", lam, -bet), ("both_mirror", 360 - lam, -bet)]:
    Mv = model_br(L, B)
    res[f"model_{name}"] = corr(Mv, D, sel)
    res[f"model_{name}_signflip"] = corr(-Mv, D, sel)
    print("   %-12s r(as-is field)=%+.3f   r(sign-flipped)=%+.3f" % (name, res[f"model_{name}"], res[f"model_{name}_signflip"]))

print("\n== daily HMI vs HMI synoptic (manual reader variants) ==")
for name, L, B in [("as_is", lam, bet), ("lon_mirror", 360 - lam, bet),
                   ("lat_mirror", lam, -bet), ("both_mirror", 360 - lam, -bet)]:
    Sv = syn(L, B)
    res[f"synoptic_{name}"] = corr(Sv, D, sel)
    res[f"synoptic_{name}_signflip"] = corr(-Sv, D, sel)
    print("   %-12s r(as-is field)=%+.3f   r(sign-flipped)=%+.3f" % (name, res[f"synoptic_{name}"], res[f"synoptic_{name}_signflip"]))

# shift scan for the best model variant and best synoptic variant
def shift_scan(fn, tag):
    best = None
    for dlon in np.arange(-25.0, 25.01, 1.0):
        for dlat in np.arange(-6.0, 6.01, 1.0):
            c = corr(fn(lam + dlon, bet + dlat), D, sel)
            if np.isfinite(c) and (best is None or c > best[0]):
                best = (c, dlon, dlat)
    if best:
        c0, dl, db = best
        for dlon in np.arange(dl - 1.0, dl + 1.01, 0.25):
            for dlat in np.arange(db - 1.0, db + 1.01, 0.25):
                c = corr(fn(lam + dlon, bet + dlat), D, sel)
                if np.isfinite(c) and c > best[0]:
                    best = (c, dlon, dlat)
    print("   %s best shift: dlon=%+.2f dlat=%+.2f -> r=%.3f" % (tag, best[1], best[2], best[0]))
    res[f"{tag}_best_shift"] = dict(r=best[0], dlon=best[1], dlat=best[2])
    return best

print("\n== shift scans ==")
shift_scan(lambda L, B: model_br(L, B), "model_as_is")
shift_scan(lambda L, B: syn(L, B), "synoptic_as_is")

# ------------------------------------------------------------------ T2 formula vs sunpy (fixed)
from sunpy.coordinates import sun, frames as fr
EPOCH = Time("2025-09-10T12:00:00", scale="utc")
PHI0 = 342.52
L0v = float(sun.L0(EPOCH).to_value("deg"))
B0v = float(sun.B0(EPOCH).to_value("deg"))
D_RSUN = float(sun.earth_distance(EPOCH).to_value(u.R_sun))
lam_g, bet_g = np.meshgrid(L0v + np.arange(-80, 80.01, 10.0), np.arange(-60, 60.01, 10.0))
c2 = SkyCoord(lam_g * u.deg, bet_g * u.deg, np.ones_like(lam_g) * u.R_sun,
              frame=fr.HeliographicCarrington(obstime=EPOCH, observer="earth"))
hpc2 = c2.transform_to(fr.Helioprojective(obstime=EPOCH, observer="earth"))
Tx_s = hpc2.Tx.to_value(u.arcsec)
Ty_s = hpc2.Ty.to_value(u.arcsec)


def formula_sky(lon_deg, lat_deg):
    dlon = np.deg2rad(lon_deg - PHI0)
    b = np.deg2rad(lat_deg)
    x = np.cos(b) * np.sin(dlon)
    y = np.sin(b)
    z = np.cos(b) * np.cos(dlon)
    return (np.rad2deg(np.arctan2(x, D_RSUN - z)) * 3600.0,
            np.rad2deg(np.arctan2(y, D_RSUN - z)) * 3600.0)


Tx_f, Ty_f = formula_sky(lam_g, bet_g)
dTx = Tx_f - Tx_s
dTy = Ty_f - Ty_s
Txf2, Tyf2 = formula_sky(2 * L0v - lam_g, bet_g)
Txb2, Tyb2 = formula_sky(lam_g, -bet_g)
rms = lambda a, b: float(np.sqrt(np.nanmean((a - b) ** 2)))
i_center = np.argmin(np.abs(np.ravel(bet_g)) + np.abs(np.ravel(lam_g - L0v)))
zb = np.cos(np.deg2rad(bet_g)) * np.cos(np.deg2rad(lam_g - PHI0))
print("\n== T2 formula vs sunpy ==")
print("   as-is RMS = %.1f arcsec" % np.sqrt(np.nanmean(dTx ** 2 + dTy ** 2)))
print("   lon-mirror RMS = %.1f ; lat-mirror RMS = %.1f" % (
    np.sqrt(np.nanmean((Txf2 - Tx_s) ** 2 + (Tyf2 - Ty_s) ** 2)),
    np.sqrt(np.nanmean((Txb2 - Tx_s) ** 2 + (Tyb2 - Ty_s) ** 2))))
print("   dTy at disk center = %+.1f arcsec (B0=%+.3f deg)" % (np.ravel(dTy)[i_center], B0v))
print("   corr(dTy, cosb*cos(dlon)) = %+.3f" % float(np.corrcoef(np.ravel(dTy), np.ravel(zb))[0, 1]))
res["T2"] = dict(rms_as_is=float(np.sqrt(np.nanmean(dTx ** 2 + dTy ** 2))),
                 dTy_center=float(np.ravel(dTy)[i_center]), B0=B0v)

with open(f"{WORK}/daily_anchor_results.json", "w") as fh:
    json.dump(res, fh, indent=2, default=float)
np.savez(f"{WORK}/daily_anchor_samples.npz", lam=lam, bet=bet, D=D, sel=sel)
print("\nDONE. saved daily_anchor_results.json")
