# Quiet-Sun MAS radio synthesis: 2025-09-10

Execution on `inti.solar.pvt`, 2026-10-06 UTC. Base repository revision:
`6495d8a866e9ca4095d4846393715d6d4aef14f8`. **All requested products were generated.
The polarization percentile sanity gate fails at all three frequencies; the
Tb and |V/I|≤1 gates pass.** These are reported outputs, not an all-gates-passing result.

## Pipeline comprehension

- Read `README.md`, `mem.md`, the synthesis script, production presets, the coordinate,
  ray tracing, sampling, GRFF parameter and utility modules, and all three test files.
- The HCC image frame is +x solar west, +y north, +z observerward. MAS is sampled with
  `lon = atan2(x,z) + phi0_offset`, `colat = acos(y/r)`, and psipy latitude = 90° − colat.
- `load_mas_var_filtered` selects numeric HDF snapshot suffixes, excluding `br_r0.hdf`.
  It loads rho, temperature (`t` here), and staggered br/bt/bp with physical units.
- MAS density is interpolated onto a uniform 256³ cube, used to form
  `omega_pe = 2*pi*8930*sqrt(Ne[cm^-3])`; separate cubes hold Ne, Te and magnetic components.
  Inside the photosphere, density/B are filled with zero and temperature with 10⁴ K.
- CPU rays use SciPy regular-grid interpolation and RK4 for position and wavevector;
  density gradients refract rays. Initial propagation is toward −z. The actual start is
  `z = sqrt(abs((2*z_observer)^2 - x^2 - y^2))/2`, not a flat plane.
- The existing CPU path also traces two perturbed rays (perturb_ratio=2) to estimate S.
  Sampling accepts finite positions and finite positive S, interpolates in float32, and
  projects the sampled spherical magnetic components into the image frame.
- Samples are reversed for transfer toward the observer; segment lengths and local
  propagation/B viewing angles are recomputed. GRFF receives Fortran-order (17,Nz)
  voxels: ds/Te/Ne/B/angles/flags, zero S_loc, Maxwellian Dist_E=0 and kappa=0.
- One `PyGET_MW` call per valid pixel supplies flux. The helper converts RL[5]+RL[6]
  from SFU to Tb using Rayleigh–Jeans and pixel solid angle at 1 AU, and returns
  V/I=(RL[5]−RL[6])/(RL[5]+RL[6]). In this GRFF source those rows are Le and Re.
- The result contains (y,x,frequency) Tb and V/I cubes and x/y coordinates in meters.
  No-valid-sample pixels, GRFF errors and invalid Stokes sums can become zero Tb;
  final nonfinite Tb is also zeroed. The driver observes these paths and preserves
  their numerical behavior. Plots mask nonfinite values and apply no smoothing.

## Inputs and exact configuration

Python: `/data1/pzhang/dev/QuietSunPol/venv/bin/python` (always `env -u PYTHONPATH`).
Data: `/data1/pzhang/dev/QuietSunPol/corona`, supplied as the CR2302 thermodynamic MAS
solution, valid 2025-09-09 through 2025-10-06, snapshot 002. The local `omas` confirms
COROTATING, phishift=0, domain 1–30 R_sun, and records the simulation execution on
2025-10-22; that execution date is distinct from the supplied model rotation validity.
GRFF: `/data1/pzhang/dev/QuietSunPol/GRFF/binaries/GRFF_DEM_Transfer.so`.
The repository loader preloads libgomp. Source/data/library checksums are in
`logs/quiet_inputs_sha256.log`.

SunPy `sun.L0(Time("2025-09-10T12:00:00", scale="utc"))` returned
**342.5204783480181 deg**. Synthesis uses the requested **342.52 deg**.

| Frequency | dt | n_steps | record_stride |
|---|---:|---:|---:|
| 40 MHz | 0.00948683298050514 | 2530 | 8 |
| 60 MHz | 0.007745966692414834 | 3098 | 6 |
| 80 MHz | 0.00670820393249937 | 3578 | 6 |

All three reuse `_lowband_params(f_Hz)` from `script/pub/TbSpectra_gen.py`.
Common parameters: grid_n=256, grid_extent=4 R_sun, z_observer=4 R_sun,
X_fov=2.8 R_sun (half FOV), N_pix=128, n_workers=1, Nfreq=1, freq0=f_Hz,
freq_log_step=0, s_input_on=False, device=cpu, raytrace_device=cpu,
grff_backend=get_mw, fallback_to_cpu=False, mech_flag=4 (thermal FF + GR,
H/He off), grff_dist_e=0, grff_kappa=0, beam_fwhm_rsun=None,
beam_diameter_m=None, save_plots=False, verbose=True.
The 60 MHz smoke run changes only N_pix to 16.
Other library defaults: beam_fwhm_factor=1.22 (unused), plot_log_norm=False,
plot_vmin/plot_vmax=None, plot_beam=True (unused), prepared_samples_path=None.
GRFF s_max=30, neutral/DEM/DDM/abundance rows 8–13=0; global Lparms has
Nfreq=1 and NT=0. Pixel area is `(5.6/128 * 6.957e10)^2 cm^2`; distance is
1.49599e13 cm. Output x/y coordinates use R_sun=6.957e8 m.

## Commands and validation

Working directory: `/data1/pzhang/dev/QuietSunPol/raytracingGRFF`.

```bash
env -u PYTHONPATH ../venv/bin/python -m pytest -q tests/
```

Result: **14 passed, 1 skipped in 9.17 s**. The skipped test requires CuPy.

The background runs use `nohup setsid nice -n 10`, one native thread per process,
and project-local cache/config directories. `setsid` keeps the background process
independent of the short-lived execution shell. No packages were installed.

The executed synthesis command template (expanded exact commands are also at the
top of each full-frequency log):

```bash
nohup setsid nice -n 10 env -u PYTHONPATH \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  XDG_CACHE_HOME="$PWD/out/cache" XDG_CONFIG_HOME="$PWD/out/config" \
  MPLCONFIGDIR="$PWD/out/mplconfig" SUNPY_CONFIGDIR="$PWD/out/config/sunpy" \
  /data1/pzhang/dev/QuietSunPol/venv/bin/python -u \
  script/run_quietsun_20250910.py --freq-mhz 60 --n-pix 16 \
  > logs/quiet_60MHz_smoke.log 2>&1 < /dev/null &

for freq in 40 60 80; do
  nohup setsid nice -n 10 env -u PYTHONPATH \
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    XDG_CACHE_HOME="$PWD/out/cache" XDG_CONFIG_HOME="$PWD/out/config" \
    MPLCONFIGDIR="$PWD/out/mplconfig" SUNPY_CONFIGDIR="$PWD/out/config/sunpy" \
    /data1/pzhang/dev/QuietSunPol/venv/bin/python -u \
    script/run_quietsun_20250910.py --freq-mhz "$freq" --n-pix 128 \
    >> "logs/quiet_${freq}MHz.log" 2>&1 < /dev/null &
done
```

The logs were initialized with their expanded `LAUNCH:` command before launching.
Three full synthesis processes were launched (PIDs 1697387, 1697388, 1697389),
after the smoke process had exited. Each has nice=10 and approximately one core of
CPU activity. The driver writes a temporary NPZ, then atomically renames it only
after saving complete metadata. The library's actual `out_path` is
`out/.raytrace_<frequency>MHz.partial.npz`, renamed to the requested final filename.
Matching complete outputs are skipped on rerun. A repeat 60 MHz invocation printed
`SKIP existing complete`; its SHA256 remained
`1cdd6cb73b10499fe4092ea5bcd2aa6cc5c99b3199eac564b7217529333f7185`
before and after (`logs/quiet_idempotence.log`).

Smoke result: `out/raytrace_60MHz_smoke16.npz`, 16×16×1, **97.113714 s** pipeline
wall time, all 256 GRFF calls return 0, no error/zeroed pixels, 24 on-disk pixels,
disk finite-Tb fraction=1, central 2×2 mean=414141.101833 K, disk mean=379099.399785 K,
Tb range=7.671174–587776.126316 K, disk 99.9th-percentile |V/I|=0.251517875342,
whole-map max |V/I|=0.255784865186. Thus smoke polarization already exceeds 0.2;
no parameter or data adjustment was made. Details: `logs/quiet_smoke_validation.log`.

Plot command (same thread/cache environment as synthesis, expanded in
`logs/quiet_plots.log`):

```bash
env -u PYTHONPATH /data1/pzhang/dev/QuietSunPol/venv/bin/python -u \
  script/run_quietsun_20250910.py --plot-only
```

Plots use origin lower, axes in R_sun, a thin white r=1 circle, Tb colorbars in K,
and frequency/date titles. V/I uses RdBu_r and symmetric limits equal to the disk
99th-percentile absolute V/I. Nonfinite values are masked. Neither maps nor plots
are beam convolved or repaired. The 2×3 summary has Tb above V/I, with frequency
columns 40/60/80 MHz. The separate equatorial profile plot shows both limbs versus
signed x (rho=|x|); y=0 is linearly interpolated between the two central rows at
y=±0.0220472440945 R_sun. Its numerical profiles are also archived.

Recompute the principal sanity values directly from the unmodified output arrays:

```bash
env -u PYTHONPATH /data1/pzhang/dev/QuietSunPol/venv/bin/python - <<'PY'
import json
import numpy as np
for f in (40, 60, 80):
    with np.load(f'out/raytrace_{f}MHz.npz', allow_pickle=False) as d:
        tb, vi = d['emission_cube'][:, :, 0], d['emission_polVI_cube'][:, :, 0]
        x, y = d['x_coords']/6.957e8, d['y_coords']/6.957e8
        disk = np.hypot(x[None, :], y[:, None]) <= 1
        finite_vi = disk & np.isfinite(vi)
        m = json.loads(str(d['run_metadata_json']))
        print(f, 'finite Tb fraction:', np.isfinite(tb[disk]).mean(),
              'center 8x8 Tb:', tb[60:68, 60:68].mean(),
              'disk |V/I| p99.9:', np.percentile(np.abs(vi[finite_vi]), 99.9),
              'whole-map |V/I| > 1:', np.count_nonzero(np.abs(vi) > 1),
              'zero/error pixels:', m['zeroed_or_error_pixels'],
              'wall seconds:', m['pipeline_wall_seconds'])
PY
```

## Results and sanity gates

All three required NPZs exist and pass schema/config checks: Tb and V/I shapes
128×128×1, a single matching frequency, and ascending x/y coordinates in meters.
Each NPZ contains the five requested pipeline arrays plus `run_metadata_json`.
All times below are UTC on 2026-10-06; wall seconds measure the existing pipeline
call, including its initial NPZ save. Per-stage timings are in the NPZ metadata.

| MHz | Pipeline wall (s) | Minutes | Ray tracing (s) | Start UTC | Finish UTC |
|---:|---:|---:|---:|---|---|
| 40 | 1349.343312 | 22.489 | 1170.302293 | 20:21:58.507010 | 20:44:27.877422 |
| 60 | 1451.197399 | 24.187 | 1304.098517 | 20:21:58.569531 | 20:46:09.801005 |
| 80 | 1769.166985 | 29.486 | 1590.370115 | 20:21:58.577130 | 20:51:27.780268 |

Disk = hypot(x/R_sun,y/R_sun)≤1, **1624 pixels** per map. The central 8×8
uses rows/columns 60:68. Percentiles use finite disk V/I; its finite fraction is
1.0 at every frequency. Disk positive-Tb fractions are also 1.0 at all frequencies.

| MHz | Finite disk Tb fraction (>0.99) | Center 8×8 Tb (K; 1e4–2e6) | Disk abs(V/I) p99.9 (<0.2) | Whole-map max abs(V/I) | Pixels abs(V/I)>1 | Zero/error pixels |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 1.000000 PASS | 246208.922410 PASS | **0.435842124793 FAIL** | 0.553376799963 | 0 | 0 |
| 60 | 1.000000 PASS | 420834.256243 PASS | **0.403644794060 FAIL** | 0.426582071533 | 0 | 0 |
| 80 | 1.000000 PASS | 567108.645670 PASS | **0.383569192587 FAIL** | 0.476337143063 | 0 | 0 |

**Gate 3 fails at 40, 60 and 80 MHz.** Its separate bound check passes: no saved
V/I value exceeds 1 in absolute value. The outputs were not tuned or clipped.
Gate 1 passes (all files and finite fractions), gate 2 passes, gate 4 counts are
reported here, and gate 5 is fulfilled by the saved and inspected y=0 profiles.

Per frequency: **16384 GRFF calls, all rc=0**; no-valid-sample pixels=0;
nonzero-GRFF-status pixels=0; GRFF-exception pixels=0; invalid-Stokes-sum
pixels=0; nonfinite-Tb conversion pixels=0; invalid-V/I pixels=0; final zero-Tb
pixels=0; unique pixels in emission zero/error paths=0. MAS sampling exceptions=0
and returned nonfinite sample values=0 for each of rho/t/br/bt/bp.

The following **sample** fills/rejections are separate from zeroing a whole map
pixel. Counts are recorded by driver observers; core numerical behavior is preserved.
The fill-count column is the number of rays with at least one accepted voxel
matching each of Ne=0, B=0 and Te=10^4 K (the three counts are equal here).

| MHz | V/I plot limits (±disk p99) | Recorded ray samples | Rejected ray samples | Pixels with accepted zero/fill samples | Valid samples/pixel min–max |
|---:|---:|---:|---:|---:|---|
| 40 | 0.340945827070 | 5193728 | 1285147 | 2049 | 172–297 |
| 60 | 0.241868926953 | 8470528 | 2066569 | 2668 | 297–484 |
| 80 | 0.193443781116 | 9781248 | 2376212 | 2663 | 349–571 |

All eight PNGs decoded successfully. The saved profile arrays agree with the
mean of the two rows bracketing y=0 and are finite. Full-precision metrics and
audit metadata: `out/quiet_20250910_sanity.json` and `logs/quiet_validation.log`.
Generated product and driver checksums: `logs/quiet_products_sha256.log`.

## Caveats and observed issues

- The existing coordinate transform ignores solar B0. SunPy B0 for the epoch is
  **+7.247969172088899 deg**. These are Earth central-meridian aligned with the
  requested phi0; they do not include the solar-axis inclination toward Earth.
- A read-only preflight using unfiltered `MASOutput["br"]` raised
  `ValueError: different number of dimensions on data and dims: 3 vs 4`, because
  psipy's prefix matching includes the boundary map `br_r0.hdf`. The existing
  pipeline's filtered loader avoids this; no core code change was needed.
- GRFF's default mechanism flag is preserved: these maps include thermal free-free
  and thermal gyroresonant contributions. V/I uses the repository's Le−Re sign.
- The cube has finite extent and the integrator stalls outside it. Subphotospheric
  fills, S validity, sample spacing, cutoff/reflection handling and absence of
  scattering are properties of the existing pipeline, not retuned here.
- Output finite fractions alone cannot prove healthy rays, because core synthesis
  replaces bad Tb with zero. Zero/error counts and positive-Tb fractions are therefore
  also recorded. Sanity failures are published without tuning.
- The observed disk polarization tail exceeds the requested 0.2 threshold at all
  frequencies. The summary shows localized strong polarization and the y=0 profiles
  contain asymmetric, jagged structure. These may include resolution/ray effects;
  no convergence study or mode-specific refracted-ray comparison was performed, so
  the numerical gates do not establish their physical accuracy. Settings were not tuned.
- An initial smoke launch without `setsid` did not persist after the execution shell
  exited and produced no output or traceback. The detached relaunch completed normally.

Only the driver and this report are tracked changes (added to the index, uncommitted). Generated products
are locally ignored using `/out/` in `.git/info/exclude`; logs match `*.log`.
No commit is made and no core module or model data is modified.

## Final file inventory

Sizes are bytes. Runtime cache/config directories under `out/cache/`,
`out/config/` and `out/mplconfig/` are excluded from this deliverable inventory.

| Path | Bytes |
|---|---:|
| `QUIETSUN_20250910_REPORT.md` | 15667 |
| `script/run_quietsun_20250910.py` | 12099 |
| `out/Tb_40MHz.png` | 82811 |
| `out/Tb_60MHz.png` | 76639 |
| `out/Tb_80MHz.png` | 73975 |
| `out/VI_40MHz.png` | 48682 |
| `out/VI_60MHz.png` | 46984 |
| `out/VI_80MHz.png` | 51968 |
| `out/quiet_20250910_radial_profiles.npz` | 4644 |
| `out/quiet_20250910_radial_profiles.png` | 134533 |
| `out/quiet_20250910_sanity.json` | 10372 |
| `out/quiet_20250910_summary.png` | 287311 |
| `out/raytrace_40MHz.npz` | 254653 |
| `out/raytrace_60MHz.npz` | 254584 |
| `out/raytrace_60MHz_smoke16.npz` | 6944 |
| `out/raytrace_80MHz.npz` | 254414 |
| `logs/quiet_40MHz.log` | 50822 |
| `logs/quiet_60MHz.log` | 60953 |
| `logs/quiet_60MHz_smoke.log` | 25828 |
| `logs/quiet_80MHz.log` | 63306 |
| `logs/quiet_environment.log` | 172 |
| `logs/quiet_idempotence.log` | 403 |
| `logs/quiet_inputs_sha256.log` | 1467 |
| `logs/quiet_monitor.log` | 3500 |
| `logs/quiet_pids.log` | 57 |
| `logs/quiet_plots.log` | 402 |
| `logs/quiet_preflight.log` | 320 |
| `logs/quiet_products_sha256.log` | 1364 |
| `logs/quiet_pytest.log` | 107 |
| `logs/quiet_smoke_validation.log` | 3153 |
| `logs/quiet_validation.log` | 10372 |
