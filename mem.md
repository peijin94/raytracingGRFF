# GRFFradioSun — project memory

Notes for future sessions (human + agent). Last updated after kappa / limb-quality work.

## Purpose

Synthetic radio **free–free** brightness maps and spectra from **MAS** coronal MHD models, using the **GRFF** Fortran library (`PyGET_MW` via `GRFF_DEM_Transfer.so`). Two main geometry modes:

1. **Straight LOS** — `LOS/resample_MAS_LOS.py` samples Ne, Te, B along vertical columns; `script/synthetic_FF_map_single_thread.py` or `LOS/grff_image_from_LOS.py` integrates with GRFF.
2. **Ray tracing** — `script/resample_with_ray_tracing.py` traces rays through an ω_pe grid, samples the model on curved paths, optional cross-section weighting (`--s-input-on`).

## Repo layout (sibling tree)

Typical checkout under `dev/`:

```
dev/
  GRFF/              # GRFF sources + binaries/GRFF_DEM_Transfer.so
  GRFFradioSun/      # this repo
  corona2298/        # MAS HDF snapshots (not in git)
  kappatest/         # Maxwellian vs κ=4 tests (separate folder)
```

`PROJECT_ROOT` in scripts = parent of `GRFFradioSun` → `dev/`. GRFF `.so` path: `dev/GRFF/binaries/GRFF_DEM_Transfer.so`.

## Environment

- **Conda env `lwa`** is the working Python stack (psipy, astropy, GRFF ctypes).
- Loading `GRFF_DEM_Transfer.so` often needs **`LD_PRELOAD=$CONDA_PREFIX/lib/libgomp.so.1`** because the `.so` uses OpenMP but may not link `libgomp` in `DT_NEEDED` → otherwise `undefined symbol: omp_get_thread_num`.

## GRFF external parameters (important)

Updated GRFF uses **17 doubles per voxel** (Fortran `(17, Nz)`):

| Row | Name | Usage here |
|-----|------|------------|
| 0–7 | plasma, angles, mechanism | `ds`, `T_e`, `N_e`, `B`, viewing 90°, flags |
| 14 | `S_loc` | Ray workflow: `S × pixel_area` when `--s-input-on` |
| 15 | `Dist_E` | **0** Maxwellian, **1** kappa, **2** n-dist |
| 16 | `kappa` | Index when `Dist_E` ≠ 0 |

Python: `raytracingGRFF.grff_parms.fill_grff_parms_ext_column`, `rl_stokes_to_tb_vi`, `vi_plot_vmax`.  
Loader: `raytracingGRFF.grff_ctypes.initGET_MW`, `default_grff_lib_path()`.

**fastGRFF** GPU path is still **15-parameter** — use `--grff-backend get_mw` for non-Maxwellian `Dist_E` / `kappa`.

## MAS / coordinates

- LOS resampling uses **`cart_to_sph(x, -z, y, phi0_offset)`** (not raw `y` as vertical in spherical conversion).
- **`phi0_offset`** must match the model / observation geometry (often **−140°** for `corona2298` in kappatest; ray script default was 90° — override with `--phi0-offset`).
- **`r_min ≈ 1 R☉`** masks points inside the photosphere; invalid samples → `NaN` in LOS cubes.
- **Limb stability**: use `np.hypot(x,y)` and `max(radicand, 0)` for `z_start` so float noise at ρ ≈ R☉ does not produce NaN LOS.

## Brightness temperature & V/I

- GRFF returns Stokes-related flux in **SFU**; conversion to **T_b (K)** uses Rayleigh–Jeans + solid angle `pixel_area / AU²` — see `rl_stokes_to_tb_vi`.
- **V/I** at low signal or failed pixels: set **NaN** when `RL5+RL6 ≤ 0` or |V/I| > 1; plot scale uses **99th percentile** of |V/I| so outliers do not dominate the colorbar.

## CLI flags (kappa)

- `--grff-dist-e` / `--grff-kappa` on `synthetic_FF_map_single_thread.py`, `grff_image_from_LOS.py`, `resample_with_ray_tracing.py`.

## Quality knobs (limb / resolution)

- Finer maps: increase **`N_pix`**, **`N_z`**, decrease **`dz0`** (irregular z grid starts at `dz0` in R☉; README warns `7e4` is wrong — use ~`2e-4`–`4e-4`).
- kappatest realistic defaults (external): 192×192, N_z=480, dz0=2e-4 on `corona2298`.

## What not to commit

- MAS directories (`corona2298/`), `*.npz`, `*.png`, `*.hdf`, CASA logs, `fastGRFF/` submodule clone, `build/`, `.pytest_cache/` — see `.gitignore`.

## Related work outside this repo

- **`kappatest/`** — `run_realistic_corona2298_kappa60MHz.py` + `run_realistic_lwa.sh` for Maxwellian vs κ=4 at 60 MHz.
- **`corhel/synth_radio_CME.py`** (if present in dev) forwards `--grff-dist-e` / `--grff-kappa` into ray tracing.
