#!/usr/bin/env python
"""
Resample MAS model along ray-traced paths and compute GRFF emission.

Similar to resampling_MAS_LOS.py but uses ray tracing (build_rays.ray_trace)
instead of straight LOS. For a N_pix x N_pix image (default 64x64), each pixel
has one ray from the observer (large z) backward along -z. At each ray point we
sample Ne, Te, B from the model and pass segment lengths ds and cross-section
ratio S to GRFF. Parms[14,k] is set to S_accumulated (cross-section ratio at
that point, as source area factor for GRFF).

Output emission_cube is brightness temperature T_b in K: GRFF returns flux in
SFU; we convert via Rayleigh-Jeans (I = 2*k_B*T_b*nu^2/c^2) and solid angle
Omega = pixel_area / AU^2 to get T_b.
"""

import argparse
import warnings
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import astropy.units as u

from psipy.model import MASOutput
from psipy.io.mas import _read_mas
import xarray as xr
from psipy.model.variable import Variable

import sys
from raytracingGRFF.build_rays import ray_trace, resample_to_xyz_cube, load_mas_var_filtered
from raytracingGRFF.gpu_raytrace import sample_model_with_rays, trace_ray

warnings.filterwarnings('ignore')


def _ray_trace_chunk(args):
    """Worker: run ray_trace for a chunk of rays. Used by ProcessPoolExecutor."""
    (chunk_start, chunk_end, x_flat, y_flat, z_start, omega_pe_3d, xg, yg, zg,
     freq_hz, dt, n_steps, record_stride) = args
    x_chunk = x_flat[chunk_start:chunk_end]
    y_chunk = y_flat[chunk_start:chunk_end]
    z_chunk = z_start[chunk_start:chunk_end]
    n_chunk = len(x_chunk)
    kvec = np.tile([[0, 0, -1]], (n_chunk, 1))
    r_record, crosssection_record = ray_trace(
        omega_pe_3d=omega_pe_3d,
        x_grid=xg, y_grid=yg, z_grid=zg,
        freq_hz=freq_hz,
        x_start=x_chunk, y_start=y_chunk, z_start=z_chunk,
        kvec_in_norm=kvec,
        dt=dt, n_steps=n_steps, record_stride=record_stride,
        trace_crosssections=True, perturb_ratio=2,
    )
    S_chunk = np.array(crosssection_record)
    return (r_record, S_chunk)


# ============================================================================
# CONSTANTS
# ============================================================================

R_sun_cm = 6.957e10   # cm
R_sun_m = 6.957e8     # meters
PHI0_OFFSET = 24.0
R_MIN = 0.999999

# GRFF
try:
    from GRFFcodes import initGET_MW
except ImportError:
    from numpy.ctypeslib import ndpointer
    import ctypes
    def initGET_MW(libname):
        _intp = ndpointer(dtype=ctypes.c_int32, flags='F')
        _doublep = ndpointer(dtype=ctypes.c_double, flags='F')
        libc_mw = ctypes.CDLL(libname)
        mwfunc = libc_mw.PyGET_MW
        mwfunc.argtypes = [_intp, _doublep, _doublep, _doublep, _doublep, _doublep, _doublep]
        mwfunc.restype = ctypes.c_int
        return mwfunc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRFF_LIB = str(PROJECT_ROOT / "GRFF" / "binaries" / "GRFF_DEM_Transfer.so")
R_sun = 6.957e10  # cm (for synthetic_FF_map compatibility)
c = 2.998e10     # speed of light, cm/s
kb = 1.38065e-16 # Boltzmann constant, erg/K
sfu2cgs = 1e-19
AU_cm = 1.49599e13
# Brightness temperature T_b (K): GRFF returns flux in SFU; Rayleigh-Jeans gives
# I = 2*k_B*T_b*nu^2/c^2 (I in erg/s/cm^2/Hz/sr). Flux F at 1 AU from one pixel
# subtends Omega = pixel_area_cm2/AU_cm^2, so I = F/Omega => T_b = F_cgs * (AU_cm^2/pixel_area_cm2) * c^2/(2*k_B*nu^2).


def cart_to_sph(x, y, z, phi0_offset=0.0):
    """Convert Cartesian to spherical (r, colat, lon). Same convention as build_rays / resampling_MAS_LOS."""
    r = np.sqrt(x**2 + y**2 + z**2)
    colat = np.arccos(np.clip(z / r, -1.0, 1.0))
    lon = np.arctan2(y, x)
    lon = lon + phi0_offset * np.pi / 180.0
    lon = np.where(lon < 0, lon + 2 * np.pi, lon)
    return r, colat, lon


def resample_var_to_cube(model, var_name, x_grid, y_grid, z_grid, target_unit=None,
                         phi0_offset=0.0, fill_nan=0.0, verbose=True):
    """Resample a MAS variable onto xyz cube. target_unit: e.g. u.cm**-3, u.K, u.G."""

    var = load_mas_var_filtered(model, var_name)
    ny, nz = len(y_grid), len(z_grid)
    out = np.full((len(x_grid), ny, nz), np.nan, dtype=float)
    y_mesh, z_mesh = np.meshgrid(y_grid, z_grid, indexing='ij')

    x_iter = tqdm(list(enumerate(x_grid)), desc=f"Resample {var_name}", disable=not verbose, unit="slice")
    for ix, x_val in x_iter:
        x_mesh = np.full_like(y_mesh, x_val)
        r, colat, lon = cart_to_sph(x_mesh, -z_mesh, y_mesh, phi0_offset=phi0_offset)
        lat = np.pi / 2 - colat
        r_mask = np.isfinite(r) & (r >= R_MIN)
        if not np.any(r_mask):
            continue
        lat_deg = np.rad2deg(lat)
        lon_deg = np.rad2deg(lon)
        lon_deg = np.where(lon_deg < 0, lon_deg + 360.0, lon_deg)
        vals = np.full_like(r, np.nan, dtype=float)
        r_arr = r[r_mask] * u.R_sun
        lat_arr = lat_deg[r_mask] * u.deg
        lon_arr = lon_deg[r_mask] * u.deg
        try:
            sampled = var.sample_at_coords(lon_arr, lat_arr, r_arr)
            if target_unit is not None:
                try:
                    sampled_vals = np.asarray(sampled.to(target_unit).value)
                except Exception:
                    sampled_vals = np.asarray(sampled.value)
            else:
                sampled_vals = np.asarray(sampled.value)
            vals[r_mask] = sampled_vals
        except Exception:
            pass
        vals[~r_mask] = np.nan
        out[ix, :, :] = vals

    if fill_nan is not None:
        out = np.where(np.isfinite(out), out, fill_nan)
    return out


def run_ray_tracing_emission(model_path, N_pix=64, X_fov=1.44, freq_hz=75e6,
                              grid_n=256, grid_extent=3.0, z_observer=3.0,
                              dt=6e-3, n_steps=5000, record_stride=10,
                              n_workers=1, s_input_on=False,
                              out_path='ray_tracing_emission.npz', grff_lib=None,
                              Nfreq=1, freq0=None, freq_log_step=0.0,
                              save_plots=True, verbose=True,
                              device='cpu', fallback_to_cpu=True,
                              raytrace_device='cpu',
                              grff_backend='get_mw'):
    """
    Run ray tracing for each pixel, sample Ne/Te/B along rays, and compute GRFF emission.

    Parameters
    ----------
    model_path : str
        Path to MAS model (e.g. ./corona).
    N_pix : int
        Image size N_pix x N_pix (default 64).
    X_fov : float
        Half FOV in R_sun; x,y in [-X_fov, X_fov].
    freq_hz : float
        Ray tracing frequency (Hz).
    grid_n : int
        Number of grid points per axis for 3D cubes.
    grid_extent : float
        ȳxyz grid extent in R_sun (e.g. [-grid_extent, grid_extent]).
    z_observer : float
        Ray start z in R_sun (observer side); rays go in -z.
    dt, n_steps, record_stride : float, int, int
        Ray integrator and recording.
    n_workers : int
        Number of processes for parallel ray tracing (1 = serial).
    s_input_on : bool
        If True, pass cross-section ratio S in Parms[14,k]; if False, put 0.
    out_path : str
        Output npz path.
    grff_lib : str or None
        Path to GRFF_DEM_Transfer.so.
    Nfreq, freq0, freq_log_step : int, float, float
        GRFF frequency setup (default single freq at freq_hz).
    save_plots : bool
        Save emission map plot.
    verbose : bool
        Print progress.
    device : str
        LOS sampling device: 'cpu' (default) or 'cuda'.
    fallback_to_cpu : bool
        If True and CUDA sampling is unavailable, fall back to CPU sampler.
    raytrace_device : str
        Ray integration device: 'cpu' (default) or 'cuda'.
    grff_backend : str
        'get_mw' (default CPU library call) or 'fastgrff' (GPU get_mw_slice).

    Returns
    -------
    dict
        emission_cube, emission_polVI_cube, frequencies_Hz, x_coords, y_coords,
        Ne_LOS, Te_LOS, B_LOS, ds_LOS, S_LOS (for first-frequency compatibility).
    """
    if grff_lib is None:
        grff_lib = GRFF_LIB
    if freq0 is None:
        freq0 = freq_hz
    GET_MW = None
    get_mw_slice = None
    cp = None
    backend = grff_backend.lower()
    if backend == 'get_mw':
        lib_path = Path(grff_lib)
        if not lib_path.is_file():
            raise FileNotFoundError(f"GRFF library not found: {grff_lib}")
        if verbose:
            print("Loading GRFF library...")
        GET_MW = initGET_MW(str(lib_path))
    elif backend == 'fastgrff':
        try:
            import cupy as cp
            sys.path.insert(0, str((PROJECT_ROOT / "fastGRFF").resolve()))
            from fastGRFF import get_mw_slice as fast_get_mw_slice
            get_mw_slice = fast_get_mw_slice
        except Exception as e:
            raise RuntimeError("Failed to initialize fastGRFF backend. Ensure fastGRFF and CuPy are available.") from e
        if verbose:
            print("Using fastGRFF GPU backend (get_mw_slice)...")
    else:
        raise ValueError(f"Unsupported grff_backend '{grff_backend}'. Use 'get_mw' or 'fastgrff'.")

    if verbose:
        print(f"Loading MAS model from {model_path}...")
    model = MASOutput(str(model_path))
    if "te" in model.variables:
        temp_var = "te"
    elif "t" in model.variables:
        temp_var = "t"
    else:
        raise ValueError("No electron temperature variable (te or t) found.")
    if "br" not in model.variables or "bt" not in model.variables or "bp" not in model.variables:
        raise ValueError("Magnetic field components (br, bt, bp) not all found.")

    xg = np.linspace(-grid_extent, grid_extent, grid_n)
    yg = np.linspace(-grid_extent, grid_extent, grid_n)
    zg = np.linspace(-grid_extent, grid_extent, grid_n)

    if verbose:
        print("Resampling rho -> omega_pe (for ray tracing)...")
    rhoxyz = resample_to_xyz_cube(model, 'rho', xg, yg, zg, phi0_offset=PHI0_OFFSET,
                                  fill_nan=0.0, verbose=verbose)
    omega_pe_3d = 8.93e3 * np.sqrt(np.maximum(rhoxyz, 0.0)) * 2 * np.pi
    # Ne in cm^-3: same as resampling_MAS_LOS — sample rho at coords and convert to u.cm**-3
    if verbose:
        print("Resampling rho -> Ne (cm^-3, as in resampling_MAS_LOS)...")
    Ne_xyz = resample_var_to_cube(model, 'rho', xg, yg, zg, target_unit=u.cm**-3,
                                  phi0_offset=PHI0_OFFSET, fill_nan=0.0, verbose=verbose)
    Ne_xyz = np.maximum(Ne_xyz, 0.0)
    if verbose:
        print("Resampling Te...")
    Te_xyz = resample_var_to_cube(model, temp_var, xg, yg, zg, target_unit=u.K,
                                  phi0_offset=PHI0_OFFSET, fill_nan=np.nan, verbose=verbose)
    Te_xyz = np.where(np.isfinite(Te_xyz), Te_xyz, 1e4)
    if verbose:
        print("Resampling B components...")
    br_xyz = resample_var_to_cube(model, 'br', xg, yg, zg, target_unit=u.G,
                                  phi0_offset=PHI0_OFFSET, fill_nan=0.0, verbose=verbose)
    bt_xyz = resample_var_to_cube(model, 'bt', xg, yg, zg, target_unit=u.G,
                                  phi0_offset=PHI0_OFFSET, fill_nan=0.0, verbose=verbose)
    bp_xyz = resample_var_to_cube(model, 'bp', xg, yg, zg, target_unit=u.G,
                                  phi0_offset=PHI0_OFFSET, fill_nan=0.0, verbose=verbose)
    B_xyz = np.sqrt(br_xyz**2 + bt_xyz**2 + bp_xyz**2)

    # Image grid (R_sun)
    x_coords_Rsun = np.linspace(-X_fov, X_fov, N_pix)
    y_coords_Rsun = np.linspace(-X_fov, X_fov, N_pix)
    X_img, Y_img = np.meshgrid(x_coords_Rsun, y_coords_Rsun)
    x_flat = X_img.ravel()
    y_flat = Y_img.ravel()
    n_rays = len(x_flat)
    z_start = np.full(n_rays, z_observer)
    kvec_in_norm = np.tile([[0, 0, -1]], (n_rays, 1))

    if raytrace_device == 'cuda':
        if verbose:
            print(f"Ray tracing {n_rays} rays on CUDA...")
        r_record, crosssection_record = trace_ray(
            device='cuda',
            omega_pe_3d=omega_pe_3d,
            x_grid=xg, y_grid=yg, z_grid=zg,
            freq_hz=freq_hz,
            x_start=x_flat, y_start=y_flat, z_start=z_start,
            kvec_in_norm=kvec_in_norm,
            dt=dt, n_steps=n_steps, record_stride=record_stride,
            trace_crosssections=True, perturb_ratio=5,
        )
        S_arr = np.array(crosssection_record)
    else:
        if n_workers <= 1:
            if verbose:
                print(f"Ray tracing {n_rays} rays (serial)...")
            r_record, crosssection_record = ray_trace(
                omega_pe_3d=omega_pe_3d,
                x_grid=xg, y_grid=yg, z_grid=zg,
                freq_hz=freq_hz,
                x_start=x_flat, y_start=y_flat, z_start=z_start,
                kvec_in_norm=kvec_in_norm,
                dt=dt, n_steps=n_steps, record_stride=record_stride,
                trace_crosssections=True, perturb_ratio=2,
            )
            S_arr = np.array(crosssection_record)
        else:
            if verbose:
                print(f"Ray tracing {n_rays} rays in parallel ({n_workers} workers)...")
            chunk_size = (n_rays + n_workers - 1) // n_workers
            chunk_args = []
            for w in range(n_workers):
                start = w * chunk_size
                end = min(start + chunk_size, n_rays)
                if start >= end:
                    continue
                chunk_args.append((
                    start, end, x_flat, y_flat, z_start, omega_pe_3d, xg, yg, zg,
                    freq_hz, dt, n_steps, record_stride,
                ))
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(_ray_trace_chunk, chunk_args))
            r_record_list = [r for r, s in results]
            S_list = [s for r, s in results]
            r_record = np.concatenate(r_record_list, axis=1)
            S_arr = np.concatenate(S_list, axis=1)

    Nf = Nfreq
    frequencies_Hz = freq0 * (10.0 ** (freq_log_step * np.arange(Nf)))
    Lparms = np.zeros(5, dtype='int32')
    Lparms[0] = 0  # set per pixel
    Lparms[1] = Nf
    Rparms = np.zeros(3, dtype='double')
    pixel_size_Rsun = (2 * X_fov) / N_pix
    pixel_size_cm = pixel_size_Rsun * R_sun_cm
    pixel_area_cm2 = pixel_size_cm * pixel_size_cm
    Rparms[0] = pixel_area_cm2
    Rparms[1] = freq0
    Rparms[2] = freq_log_step

    x_coords = x_coords_Rsun * R_sun_m
    y_coords = y_coords_Rsun * R_sun_m
    emission_cube = np.zeros((N_pix, N_pix, Nf), dtype='double')
    emission_polVI_cube = np.zeros((N_pix, N_pix, Nf), dtype='double')

    if verbose:
        print(f"Sampling Ne/Te/B along rays on device='{device}' and calling GRFF...")
    ray_start = np.column_stack([x_flat, y_flat, z_start])  # (n_rays, 3)
    sampled = sample_model_with_rays(
        device=device,
        x_grid=xg,
        y_grid=yg,
        z_grid=zg,
        ne_xyz=Ne_xyz,
        te_xyz=Te_xyz,
        b_xyz=B_xyz,
        r_record=r_record,
        s_arr=S_arr,
        ray_start=ray_start,
        r_sun_cm=R_sun_cm,
        fill_ne=0.0,
        fill_te=1e4,
        fill_b=0.0,
        fallback_to_cpu=fallback_to_cpu,
        verbose=verbose,
    )
    ne_all = sampled['ne']
    te_all = sampled['te']
    b_all = sampled['b']
    ds_all = sampled['ds']
    valid_all = sampled['valid_mask']
    s_all = sampled['s']

    if backend == 'fastgrff':
        n_rec = ne_all.shape[0]
        if verbose:
            print(f"Running fastGRFF get_mw_slice for {n_rays} pixels, Nz={n_rec}, Nf={Nf}...")
        Parms_M = np.zeros((15, n_rec, n_rays), dtype=np.float64, order='F')
        Parms_M[4, :, :] = 90.0
        Parms_M[6, :, :] = 1 + 4
        Parms_M[7, :, :] = 30
        for p in range(n_rays):
            valid = valid_all[:, p]
            if not np.any(valid):
                continue
            cnt = int(np.count_nonzero(valid))
            Parms_M[0, :cnt, p] = ds_all[:, p][valid]
            Parms_M[1, :cnt, p] = te_all[:, p][valid]
            Parms_M[2, :cnt, p] = ne_all[:, p][valid]
            Parms_M[3, :cnt, p] = b_all[:, p][valid]
            if s_input_on:
                Parms_M[14, :cnt, p] = s_all[:, p][valid] * pixel_area_cm2

        Lparms_M = cp.zeros(6, dtype=cp.int32, order='F')
        Lparms_M[0] = n_rays
        Lparms_M[1] = n_rec
        Lparms_M[2] = Nf
        Lparms_M[3] = 1

        Rparms_M = cp.zeros((3, n_rays), dtype=cp.float64, order='F')
        Rparms_M[0, :] = pixel_area_cm2
        Rparms_M[1, :] = freq0
        Rparms_M[2, :] = freq_log_step

        Parms_M_cp = cp.array(np.asfortranarray(Parms_M), dtype=cp.float64, order='F', copy=True)
        dummy = cp.asarray(0, dtype=cp.float64)
        RL_M = cp.zeros((7, Nf, n_rays), dtype=cp.float64, order='F')

        status = get_mw_slice(
            Lparms_M, Rparms_M, Parms_M_cp, dummy, dummy, dummy, RL_M,
            tile_pixels=256, heap_bytes=2 * 1024 * 1024 * 1024,
        )
        if np.any(status != 0) and verbose:
            bad = np.where(status != 0)[0]
            print(f"fastGRFF: warning {bad.size} pixels returned non-zero status")

        RL_M_np = cp.asnumpy(RL_M)
        intensity = (RL_M_np[5] + RL_M_np[6]).T  # (Npix, Nf)
        denom = RL_M_np[5] + RL_M_np[6]
        pol_vi = np.where(denom != 0, (RL_M_np[5] - RL_M_np[6]) / (denom + 1e-30), 0.0).T
        nu_ghz = RL_M_np[0].T  # (Npix, Nf)

        emission_flat = np.zeros((n_rays, Nf), dtype=float)
        for ifreq in range(Nf):
            nu_hz = np.where(nu_ghz[:, ifreq] > 0, nu_ghz[:, ifreq] * 1e9, frequencies_Hz[ifreq])
            conversion_factor = (sfu2cgs * c * c / (2.0 * kb * nu_hz * nu_hz) / pixel_area_cm2) * (AU_cm * AU_cm)
            emission_flat[:, ifreq] = intensity[:, ifreq] * conversion_factor

        emission_cube[:, :, :] = emission_flat.reshape(N_pix, N_pix, Nf)
        emission_polVI_cube[:, :, :] = pol_vi.reshape(N_pix, N_pix, Nf)
    else:
        p_iter = tqdm(range(n_rays), desc="GRFF pixels", disable=not verbose, unit="px")
        for p in p_iter:
            i, j = p // N_pix, p % N_pix
            valid = valid_all[:, p]
            if not np.any(valid):
                emission_cube[i, j, :] = 0.0
                continue
            ne_ray = ne_all[:, p][valid]
            te_ray = te_all[:, p][valid]
            b_ray = b_all[:, p][valid]
            ds_ray = ds_all[:, p][valid]
            S_valid = s_all[:, p][valid]
            n_pts = len(ne_ray)

            N_valid = n_pts
            Parms = np.zeros((15, N_valid), dtype='double', order='F')
            for k in range(N_valid):
                Parms[0, k] = ds_ray[k]
                Parms[1, k] = te_ray[k]
                Parms[2, k] = ne_ray[k]
                Parms[3, k] = b_ray[k]
                Parms[4, k] = 90.0
                Parms[5, k] = 0.0
                Parms[6, k] = 1 + 4
                Parms[7, k] = 30
                Parms[8, k] = Parms[9, k] = Parms[10, k] = 0.0
                Parms[11, k] = Parms[12, k] = Parms[13, k] = 0.0
                Parms[14, k] = S_valid[k]*pixel_area_cm2 if s_input_on else 0.0  # S (cross-section) or 0
            Lparms_local = Lparms.copy()
            Lparms_local[0] = N_valid
            dummy_T = np.array(0, dtype='double')
            dummy_DEM = np.array(0, dtype='double')
            dummy_DDM = np.array(0, dtype='double')
            RL = np.zeros((7, Nf), dtype='double', order='F')
            try:
                res = GET_MW(Lparms_local, Rparms, Parms, dummy_T, dummy_DEM, dummy_DDM, RL)
                if res != 0:
                    emission_cube[i, j, :] = 0.0
                    continue
                for ifreq in range(Nf):
                    intensity_sfu = RL[5, ifreq] + RL[6, ifreq]  # total intensity from GRFF (SFU)
                    circularpol_VI = (RL[5, ifreq] - RL[6, ifreq]) / (RL[5, ifreq] + RL[6, ifreq] + 1e-30)
                    nu_GHz = RL[0, ifreq]
                    nu_Hz = frequencies_Hz[ifreq] if nu_GHz <= 0 else nu_GHz * 1e9
                    conversion_factor = (sfu2cgs * c * c / (2.0 * kb * nu_Hz * nu_Hz) / Rparms[0]) * (AU_cm * AU_cm)
                    emission_cube[i, j, ifreq] = intensity_sfu * conversion_factor  # brightness temperature in K
                    emission_polVI_cube[i, j, ifreq] = circularpol_VI
            except Exception as e:
                if verbose:
                    print(f"  Error pixel ({i},{j}): {e}")
                emission_cube[i, j, :] = 0.0

    if verbose:
        print("Ray-tracing emission complete.")
    # emission_cube: brightness temperature T_b in K (CGS conversion applied)
    result = {
        'emission_cube': emission_cube,       # T_b (K), shape (N_pix, N_pix, Nf)
        'emission_polVI_cube': emission_polVI_cube,
        'frequencies_Hz': frequencies_Hz,
        'x_coords': x_coords,
        'y_coords': y_coords,
    }
    np.savez_compressed(out_path, **result)
    if verbose:
        print(f"Saved {out_path}")

    if save_plots:
        _save_emission_plot(result, N_pix, X_fov, R_sun_m, out_path, verbose)
        _save_center_pixel_plots(
            sampled, N_pix, out_path, verbose,
        )
    return result


def _save_center_pixel_plots(sampled, N_pix, out_path, verbose):
    """Plot Ne, Te, B, and S along the ray for the center pixel (inspection)."""
    p_center = (int(N_pix*0.7) // 2) * N_pix + ((N_pix-1) // 2)
    valid = sampled['valid_mask'][:, p_center]
    if not np.any(valid):
        if verbose:
            print("Center pixel has no valid ray points; skipping center-pixel plot.")
        return
    ne_c = sampled['ne'][:, p_center][valid]
    te_c = sampled['te'][:, p_center][valid]
    b_c = sampled['b'][:, p_center][valid]
    S_valid = sampled['s'][:, p_center][valid]
    ds_c = sampled['ds'][:, p_center][valid]
    dist_cm = np.cumsum(ds_c.astype(float))
    dist_Rsun = dist_cm / R_sun_cm

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(dist_Rsun, ne_c, 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Distance along ray (R_sun)')
    axes[0, 0].set_ylabel('N_e (cm$^{-3}$)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title('Center pixel: N_e along ray')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(dist_Rsun, te_c, 'r-', linewidth=1.5)
    axes[0, 1].set_xlabel('Distance along ray (R_sun)')
    axes[0, 1].set_ylabel('T_e (K)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Center pixel: T_e along ray')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(dist_Rsun, b_c, 'green', linewidth=1.5)
    axes[1, 0].set_xlabel('Distance along ray (R_sun)')
    axes[1, 0].set_ylabel('|B| (G)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Center pixel: |B| along ray')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(dist_Rsun, S_valid, 'k-', linewidth=1.5)
    axes[1, 1].axhline(1.0, color='gray', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Distance along ray (R_sun)')
    axes[1, 1].set_ylabel('S (cross-section ratio)')
    axes[1, 1].set_title('Center pixel: S along ray')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(out_path).with_name(
        Path(out_path).stem + '_center_pixel.png'
    )
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"Center-pixel inspection plot saved to {plot_path}")


def _save_emission_plot(result, N_pix, X_fov, R_sun_m, out_path, verbose):
    emission_cube = result['emission_cube']
    x_coords = result['x_coords']
    y_coords = result['y_coords']
    frequencies_Hz = result['frequencies_Hz']
    emission_map = emission_cube[:, :, 0]
    emission_map[emission_map == 0] = np.nan
    x_range = [x_coords[0] / R_sun_m, x_coords[-1] / R_sun_m]
    y_range = [y_coords[0] / R_sun_m, y_coords[-1] / R_sun_m]
    fig, ax = plt.subplots(figsize=(6, 4.8))
    im = ax.imshow(emission_map, origin='lower',
                   extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                   aspect='equal', cmap='hot', interpolation='bilinear')
    ax.set_xlabel('x (R_sun)')
    ax.set_ylabel('y (R_sun)')
    ax.set_title(f'Ray-tracing emission T_b at {frequencies_Hz[0]/1e9:.3f} GHz')
    plt.colorbar(im, ax=ax, label='T_b (K)')
    plt.tight_layout()
    plot_path = Path(out_path).with_suffix('.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"Plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Ray-tracing emission map: resample MAS along rays and run GRFF.')
    parser.add_argument('--model-path', '-m', type=str, default='./corona',
                        help='MAS model directory (default: ./corona)')
    parser.add_argument('--N-pix', '-n', type=int, default=32,
                        help='Image size N_pix x N_pix (default: 64)')
    parser.add_argument('--X-FOV', '-f', type=float, default=1.44,
                        help='Half FOV in R_sun (default: 1.44)')
    parser.add_argument('--freq', type=float, default=75e6,
                        help='Ray frequency in Hz (default: 75e6)')
    parser.add_argument('--grid-n', type=int, default=128,
                        help='3D grid points per axis (default: 128)')
    parser.add_argument('--grid-extent', type=float, default=3.0,
                        help='3D grid extent in R_sun (default: 3)')
    parser.add_argument('--z-observer', type=float, default=3.0,
                        help='Ray start z in R_sun (default: 3)')
    parser.add_argument('--dt', type=float, default=6e-3,
                        help='Ray integrator dt (default: 6e-3)')
    parser.add_argument('--n-steps', type=int, default=5000,
                        help='Ray integration steps (default: 5000)')
    parser.add_argument('--record-stride', type=int, default=10,
                        help='Record every N steps (default: 10)')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='Number of processes for parallel ray tracing (default: 1)')
    parser.add_argument('--out-path', '-o', type=str, default='ray_tracing_emission.npz',
                        help='Output npz path (default: ray_tracing_emission.npz)')
    parser.add_argument('--grff-lib', type=str, default=GRFF_LIB,
                        help=f'GRFF library path (default: {GRFF_LIB})')
    parser.add_argument('--grff-backend', type=str, default='get_mw', choices=['get_mw', 'fastgrff'],
                        help="GRFF backend: 'get_mw' (default) or 'fastgrff' (GPU)")
    parser.add_argument('--s-input-on', action='store_true',
                        help='Pass cross-section ratio S in Parms[14]; otherwise use 0')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help="LOS sampling device: 'cpu' (default) or 'cuda'")
    parser.add_argument('--raytrace-device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help="Ray integration device: 'cpu' (default) or 'cuda'")
    parser.add_argument('--no-fallback', action='store_true',
                        help='If --device cuda fails, do not fall back to cpu')
    parser.add_argument('--no-plots', action='store_true', help='Do not save plot')
    parser.add_argument('--quiet', '-q', action='store_true', help='Less output')
    args = parser.parse_args()

    run_ray_tracing_emission(
        model_path=args.model_path,
        N_pix=args.N_pix,
        X_fov=args.X_FOV,
        freq_hz=args.freq,
        grid_n=args.grid_n,
        grid_extent=args.grid_extent,
        z_observer=args.z_observer,
        dt=args.dt,
        n_steps=args.n_steps,
        record_stride=args.record_stride,
        n_workers=args.workers,
        s_input_on=args.s_input_on,
        out_path=args.out_path,
        grff_lib=args.grff_lib,
        Nfreq=1,
        freq0=args.freq,
        freq_log_step=0.0,
        save_plots=not args.no_plots,
        verbose=not args.quiet,
        device=args.device,
        fallback_to_cpu=not args.no_fallback,
        raytrace_device=args.raytrace_device,
        grff_backend=args.grff_backend,
    )


if __name__ == '__main__':
    main()
