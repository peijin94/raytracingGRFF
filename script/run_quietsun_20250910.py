#!/usr/bin/env python
"""CPU synthesis and plots for 2025-09-10; use the project's verified venv.

The audit wrappers only observe existing function inputs, outputs, and errors.
They preserve the pipeline's numerical behavior and count its silent zero paths.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "mplconfig"))
os.environ.setdefault("SUNPY_CONFIGDIR", str(OUT / "config" / "sunpy"))
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.utils import iers
iers.conf.auto_download = False
from sunpy.coordinates import sun
from psipy.model.variable import Variable
from script.pub.TbSpectra_gen import _lowband_params
from script import resample_with_ray_tracing as pipeline

R_SUN_M = 6.957e8
EPOCH = "2025-09-10T12:00:00"
MODEL = ROOT.parent / "corona"
LIB = ROOT.parent / "GRFF/binaries/GRFF_DEM_Transfer.so"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def audit_pipeline(meta):
    """Install process-local observers; return a restoration callback."""
    originals = []
    bad = {key: set() for key in ("no_valid", "grff_status", "grff_exception",
                                "invalid_stokes", "nonfinite_tb", "invalid_vi")}
    counts = Counter()
    state = {}

    def patch(obj, name, replacement):
        originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    original_sample = Variable.sample_at_coords

    def sample(self, *args, **kwargs):
        try:
            result = original_sample(self, *args, **kwargs)
            counts[f"{self.name}_sample_nonfinite"] += int(np.count_nonzero(~np.isfinite(result.value)))
            return result
        except Exception as exc:
            counts[f"{self.name}_sample_exceptions"] += 1
            meta.setdefault("sample_errors", []).append(f"{self.name}: {type(exc).__name__}: {exc}")
            raise
    patch(Variable, "sample_at_coords", sample)

    original_prepare = pipeline.prepare_ray_voxels_for_grff

    def prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        valid = result["valid_mask"] & np.isfinite(result["ne"]) & np.isfinite(result["te"]) & np.isfinite(result["b"])
        empty = ~np.any(valid, axis=0)
        bad["no_valid"].update(np.flatnonzero(empty).tolist())
        state["pixels"] = iter(np.flatnonzero(~empty).tolist())
        counts["recorded_points"] = int(valid.size)
        counts["rejected_points"] = int(np.count_nonzero(~valid))
        for key, mask in (("zero_ne", result["ne"] == 0), ("fill_te_1e4", result["te"] == 1e4), ("zero_b", result["b"] == 0)):
            counts[f"pixels_with_valid_{key}"] = int(np.count_nonzero(np.any(valid & mask, axis=0)))
        meta["valid_points_per_pixel_min_max"] = [int(valid.sum(axis=0).min()), int(valid.sum(axis=0).max())]
        return result
    patch(pipeline, "prepare_ray_voxels_for_grff", prepare)

    original_init = pipeline.initGET_MW

    def init(*args, **kwargs):
        get_mw = original_init(*args, **kwargs)

        def call(*call_args):
            p = next(state["pixels"])
            state["pixel"] = p
            counts["grff_calls"] += 1
            try:
                rc = get_mw(*call_args)
                counts[f"grff_rc_{rc}"] += 1
                if rc:
                    bad["grff_status"].add(p)
                return rc
            except Exception:
                bad["grff_exception"].add(p)
                raise
        return call
    patch(pipeline, "initGET_MW", init)

    original_convert = pipeline.rl_stokes_to_tb_vi

    def convert(rl, ifreq, *args, **kwargs):
        p = state["pixel"]
        denom = float(rl[5, ifreq] + rl[6, ifreq])
        if not np.isfinite(denom) or denom <= 0:
            bad["invalid_stokes"].add(p)
        tb, vi = original_convert(rl, ifreq, *args, **kwargs)
        if not np.isfinite(tb):
            bad["nonfinite_tb"].add(p)
        if not np.isfinite(vi):
            bad["invalid_vi"].add(p)
        return tb, vi
    patch(pipeline, "rl_stokes_to_tb_vi", convert)

    # Phase timings provide progress even where the integrator has no tqdm.
    for name in ("resample_to_xyz_cube", "resample_var_to_cube", "ray_trace", "sample_model_with_rays"):
        original = getattr(pipeline, name)

        def timed(*args, _original=original, _name=name, **kwargs):
            label = _name + (f":{args[1]}" if _name.startswith("resample") else "")
            print(f"{stamp()} START {label}", flush=True)
            start = time.perf_counter()
            result = _original(*args, **kwargs)
            seconds = time.perf_counter() - start
            meta.setdefault("phase_wall_seconds", {})[label] = seconds
            print(f"{stamp()} END {label} wall_seconds={seconds:.3f}", flush=True)
            return result
        patch(pipeline, name, timed)

    def restore():
        meta["audit_counts"] = dict(counts)
        meta["error_path_pixels"] = {key: len(value) for key, value in bad.items()}
        zero_keys = ("no_valid", "grff_status", "grff_exception", "invalid_stokes", "nonfinite_tb")
        meta["zeroed_or_error_pixels"] = len(set().union(*(bad[k] for k in zero_keys)))
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)
    return restore


def run(freq_mhz, n_pix):
    freq = freq_mhz * 1e6
    p = _lowband_params(freq)
    tag = f"{freq_mhz}MHz" + (f"_smoke{n_pix}" if n_pix != 128 else "")
    target = OUT / f"raytrace_{tag}.npz"
    temporary = OUT / f".raytrace_{tag}.partial.npz"
    config = dict(model_path=str(MODEL), grff_lib=str(LIB), N_pix=n_pix,
                  X_fov=p["x_fov"], freq_hz=freq, grid_n=p["grid_n"],
                  grid_extent=p["grid_extent"], z_observer=p["z_observer"],
                  dt=p["dt"], n_steps=p["n_steps"], record_stride=p["record_stride"],
                  n_workers=1, s_input_on=False, Nfreq=1, freq0=freq,
                  freq_log_step=0.0, save_plots=False, verbose=True, device="cpu",
                  fallback_to_cpu=False, raytrace_device="cpu", grff_backend="get_mw",
                  beam_fwhm_rsun=None, beam_diameter_m=None, phi0_offset=342.52,
                  grff_dist_e=0.0, grff_kappa=0.0, mech_flag=4.0)
    print("COMMAND:", "env -u PYTHONPATH", shlex.join([sys.executable, "-u", *sys.argv]), flush=True)
    if target.exists():
        try:
            with np.load(target, allow_pickle=False) as old:
                metadata = json.loads(str(old["run_metadata_json"]))
                assert metadata["status"] == "complete" and metadata["config"] == config
                assert old["emission_cube"].shape == old["emission_polVI_cube"].shape == (n_pix, n_pix, 1)
                assert np.array_equal(old["frequencies_Hz"], [freq])
                assert old["x_coords"].shape == old["y_coords"].shape == (n_pix,)
            print(f"SKIP existing complete {target}", flush=True)
            return
        except (ValueError, KeyError, AssertionError, OSError):
            print(f"REBUILD incomplete or mismatched {target}", flush=True)
    meta = dict(status="running", started_utc=stamp(), config=config,
                executable=sys.executable, python=sys.version, pid=os.getpid(),
                L0_deg=float(sun.L0(Time(EPOCH, scale="utc")).to_value("deg")),
                B0_deg=float(sun.B0(Time(EPOCH, scale="utc")).to_value("deg")), epoch_utc=EPOCH,
                thread_environment={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    print("CONFIG:", json.dumps(meta, sort_keys=True), flush=True)
    restore = audit_pipeline(meta)
    start = time.perf_counter()
    try:
        result = pipeline.run_ray_tracing_emission(**config, out_path=str(temporary))
        meta.update(status="complete", finished_utc=stamp(), pipeline_wall_seconds=time.perf_counter()-start)
        restore()
        meta["output_zero_tb_pixels"] = int(np.count_nonzero(result["emission_cube"] == 0))
        result["run_metadata_json"] = np.array(json.dumps(meta, sort_keys=True))
        np.savez_compressed(temporary, **result)
        temporary.replace(target)
        print("COMPLETE:", json.dumps(meta, sort_keys=True), flush=True)
        print(f"Saved complete {target} ({target.stat().st_size} bytes)", flush=True)
    except Exception:
        restore()
        error = traceback.format_exc()
        (ROOT / "FAILED.md").write_text(f"# Failed {tag}\n\nUTC: {stamp()}\n\n```\n{error}\n```\n")
        raise


def plot_all():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    profile_fig, profile_ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    profiles = {}
    for col, f in enumerate((40, 60, 80)):
        with np.load(OUT / f"raytrace_{f}MHz.npz", allow_pickle=False) as data:
            tb, vi = data["emission_cube"][:, :, 0], data["emission_polVI_cube"][:, :, 0]
            x, y = data["x_coords"] / R_SUN_M, data["y_coords"] / R_SUN_M
        disk = np.hypot(x[None, :], y[:, None]) <= 1
        vmax_vi = float(np.percentile(np.abs(vi[disk & np.isfinite(vi)]), 99))
        # Average the rows bracketing zero: linear interpolation exactly at y=0.
        hi = int(np.searchsorted(y, 0.0))
        weight = -y[hi-1] / (y[hi] - y[hi-1])
        profile = (1-weight)*tb[hi-1, :] + weight*tb[hi, :]
        profile_ax.plot(x, profile, label=f"{f} MHz")
        profiles[f"Tb_{f}MHz_K"] = profile
        profiles["x_Rsun"] = x
        for row, (name, arr, cmap, low, high, unit) in enumerate((
                ("Tb", tb, "inferno", 0, float(np.nanmax(tb)), r"$T_b$ (K)"),
                ("VI", vi, "RdBu_r", -vmax_vi, vmax_vi, "V/I"))):
            def draw(ax):
                image = ax.imshow(np.ma.masked_invalid(arr), origin="lower",
                                  extent=[x[0], x[-1], y[0], y[-1]], aspect="equal",
                                  interpolation="nearest", cmap=cmap, vmin=low, vmax=high)
                ax.add_patch(plt.Circle((0, 0), 1, fill=False, edgecolor="white", linewidth=0.8))
                ax.set(xlabel=r"x ($R_\odot$, west)", ylabel=r"y ($R_\odot$, north)",
                       title=f"{f} MHz — 2025-09-10 — {name}")
                ax.figure.colorbar(image, ax=ax, label=unit, shrink=0.85)
            draw(axes[row, col])
            single, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
            draw(ax)
            single.savefig(OUT / f"{name}_{f}MHz.png", dpi=160)
            plt.close(single)
    fig.savefig(OUT / "quiet_20250910_summary.png", dpi=160)
    plt.close(fig)
    profile_ax.axvline(-1, color="gray", linewidth=0.8, linestyle="--")
    profile_ax.axvline(1, color="gray", linewidth=0.8, linestyle="--")
    profile_ax.set(xlabel=r"Signed radius x at y=0 ($R_\odot$)", ylabel=r"$T_b$ (K)",
                   title="2025-09-10 — equatorial brightness profiles (no beam)")
    profile_ax.legend()
    profile_ax.grid(alpha=0.25)
    profile_fig.savefig(OUT / "quiet_20250910_radial_profiles.png", dpi=160)
    plt.close(profile_fig)
    np.savez_compressed(OUT / "quiet_20250910_radial_profiles.npz", **profiles)
    print("Saved six maps, summary, and y=0 profiles.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freq-mhz", type=int, choices=(40, 60, 80))
    parser.add_argument("--n-pix", type=int, choices=(16, 128), default=128)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot_all()
    elif args.freq_mhz is None:
        parser.error("--freq-mhz is required for synthesis")
    else:
        run(args.freq_mhz, args.n_pix)
