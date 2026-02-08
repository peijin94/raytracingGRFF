# GRFFradioSun

Synthetic radio free-free emission from a MAS coronal model using GRFF.

## Requirements

- **Python**: `psipy`, `numpy`, `matplotlib`, `astropy`, `xarray`
- **Data**: MAS model in `corona/` (e.g. `rho002.hdf`, `t002.hdf`, `br002.hdf`, etc.)
- **GRFF**: `GRFF/binaries/GRFF_DEM_Transfer.so` (or set path in `synthetic_FF.py`)
- **Optional GPU**: NVIDIA CUDA + CuPy (example: `pip install cupy-cuda12x`)

## Layout

- `raytracingGRFF/`: installable Python package (ray tracing and LOS sampling code)
- `script/`: runnable workflows
  - `script/resample_with_ray_tracing.py` (ray-tracing pipeline)
  - `script/resampling_MAS_LOS.py` (non-raytracing LOS resampling, comparison baseline)
  - `script/synthetic_FF_map_single_thread.py` (non-raytracing synthetic map, comparison baseline)
- `fastGRFF/`: external module; not included in `raytracingGRFF` packaging

## How to run

**1. Resample MAS along lines of sight** (writes LOS \(N_e\), \(T_e\), \(B\), \(ds\)):

```bash
python script/resampling_MAS_LOS.py -m ./corona -o LOS_data_300MHz.npz
```

Options: `-n` pixels, `-f` X-FOV (half-extent in R_sun, e.g. 2.1 → [-2.1, 2.1]), `-z` LOS points, `-d` dz0, `--no-plots` to skip plots. Run `python resampling_MAS.py -h` for details.

**2. Compute brightness temperature and V/I** (reads LOS npz, calls GRFF, writes maps):

```bash
python script/synthetic_FF_map_single_thread.py
```

Expects `LOS_data_300MHz.npz` in the current directory (use the same `-o` name in step 1). Outputs: `emission_map.npz`, `emission_map.png`, `emission_map_Tb_VI.png`, `emission_map_log.png`.

## GPU acceleration (ray-traced LOS sampling)

`script/resample_with_ray_tracing.py` supports CPU/CUDA LOS sampling dispatch:

```bash
python script/resample_with_ray_tracing.py --device cpu   # default
python script/resample_with_ray_tracing.py --device cuda  # CUDA via CuPy RawKernel
```

If `--device cuda` is requested and CUDA/CuPy is unavailable, the script falls back to CPU unless `--no-fallback` is passed.

Ray integration (trajectory RK4) can also be moved to CUDA:

```bash
python script/resample_with_ray_tracing.py --raytrace-device cuda --device cuda
python script/resample_with_ray_tracing.py --raytrace-device cuda --device cuda --grff-backend fastgrff
```

CUDA-first entry point:

```bash
python resample_with_ray_tracing_GPU.py
```

Validation and benchmark helpers:

```bash
pytest -q tests/test_gpu_raytrace.py
python bench_raytrace.py --n-pix 256 --n-steps 256
```
