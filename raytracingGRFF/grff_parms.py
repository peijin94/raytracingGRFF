"""
GRFF external voxel parameter layout (PyGET_MW / GET_MW).

Matches GRFF `InSize_ext` (see GRFF/source/MWtransfer.h): 17 doubles per voxel,
Fortran column-major shape (17, Nz).

Indices follow GRFF getparms `arr3` (see GRFF/source/getparms.cpp):
  0  dR (segment length, cm)
  1  T_0 (K)
  2  n_0 (cm^-3)
  3  B (G)
  4  theta (deg)
  5  phi (deg)
  6  mech_flag
  7  s_max
  8  n_p
  9  n_HI
  10 n_HeI
  11 DEM_key_loc
  12 DDM_key_loc
  13 abund_key
  14 S_loc (source area, cm^2)
  15 Dist_E (0 Maxwellian, 1 kappa, 2 n-distribution)
  16 kappa (index for kappa- or n-distribution; Maxwellian: use 0)
"""

from __future__ import annotations

import numpy as np

GRFF_PARMS_EXT_SIZE = 17


def fill_grff_parms_ext_column(
    Parms: np.ndarray,
    k: int,
    ds_cm: float,
    te_k: float,
    ne_cm3: float,
    b_g: float,
    *,
    s_cm2: float = 0.0,
    dist_e: float = 0.0,
    kappa: float = 0.0,
    theta_deg: float = 90.0,
    phi_deg: float = 0.0,
    mech_flag: float = 5.0,
    s_max: float = 30.0,
) -> None:
    """Fill one column k of Parms (shape (17, Nz), Fortran order)."""
    if Parms.shape[0] != GRFF_PARMS_EXT_SIZE:
        raise ValueError(f"Parms first dim must be {GRFF_PARMS_EXT_SIZE}, got {Parms.shape[0]}")
    Parms[0, k] = ds_cm
    Parms[1, k] = te_k
    Parms[2, k] = ne_cm3
    Parms[3, k] = b_g
    Parms[4, k] = theta_deg
    Parms[5, k] = phi_deg
    Parms[6, k] = mech_flag
    Parms[7, k] = s_max
    Parms[8, k] = 0.0
    Parms[9, k] = 0.0
    Parms[10, k] = 0.0
    Parms[11, k] = 0.0
    Parms[12, k] = 0.0
    Parms[13, k] = 0.0
    Parms[14, k] = s_cm2
    Parms[15, k] = float(dist_e)
    Parms[16, k] = float(kappa)


# Brightness-temperature conversion (matches synthetic_FF_map / ray-tracing workflows)
C_LIGHT_CM_S = 2.998e10
KB_ERG_K = 1.38065e-16
SFU_TO_CGS = 1e-19
AU_CM = 1.49599e13


def rl_stokes_to_tb_vi(
    RL: np.ndarray,
    ifreq: int,
    nu_hz: float,
    pixel_area_cm2: float,
    *,
    distance_cm: float = AU_CM,
) -> tuple[float, float]:
    """
    Convert GRFF ``RL`` Stokes row to brightness temperature (K) and circular V/I.

    Returns ``(tb_k, vi)``; ``vi`` is ``nan`` when the Stokes sum is invalid or |V/I| > 1.
    """
    denom = float(RL[5, ifreq] + RL[6, ifreq])
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0, np.nan
    conv = (
        SFU_TO_CGS
        * C_LIGHT_CM_S**2
        / (2.0 * KB_ERG_K * nu_hz * nu_hz)
        / pixel_area_cm2
    ) * (distance_cm * distance_cm)
    tb_k = denom * conv
    vi = (RL[5, ifreq] - RL[6, ifreq]) / denom
    if not np.isfinite(vi) or abs(vi) > 1.0 + 1e-9:
        vi = np.nan
    return tb_k, float(vi)


def vi_plot_vmax(pol_vi: np.ndarray, tb: np.ndarray, percentile: float = 99.0) -> float:
    """Robust |V/I| color-scale limit; masks zero-Tb pixels like legacy plotting."""
    plot = pol_vi.copy()
    plot[tb == 0] = np.nan
    abs_vi = np.abs(plot[np.isfinite(plot)])
    if abs_vi.size == 0:
        return 1.0
    vmax = float(np.percentile(abs_vi, percentile))
    return vmax if np.isfinite(vmax) and vmax > 0 else 1.0
