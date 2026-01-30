# GRFFradioSun

Synthetic radio free-free emission from a MAS coronal model using GRFF.

## Requirements

- **Python**: `psipy`, `numpy`, `matplotlib`, `astropy`, `xarray`
- **Data**: MAS model in `corona/` (e.g. `rho002.hdf`, `t002.hdf`, `br002.hdf`, etc.)
- **GRFF**: `GRFF/binaries/GRFF_DEM_Transfer.so` (or set path in `synthetic_FF.py`)

## How to run

**1. Resample MAS along lines of sight** (writes LOS \(N_e\), \(T_e\), \(B\), \(ds\)):

```bash
python resampling_MAS.py -m ./corona -o LOS_data_300MHz.npz
```

Options: `-n` pixels, `-f` X-FOV (half-extent in R_sun, e.g. 2.1 → [-2.1, 2.1]), `-z` LOS points, `-d` dz0, `--no-plots` to skip plots. Run `python resampling_MAS.py -h` for details.

**2. Compute brightness temperature and V/I** (reads LOS npz, calls GRFF, writes maps):

```bash
python synthetic_FF.py
```

Expects `LOS_data_300MHz.npz` in the current directory (use the same `-o` name in step 1). Outputs: `emission_map.npz`, `emission_map.png`, `emission_map_Tb_VI.png`, `emission_map_log.png`.
