"""Utilities for ray-tracing and emission maps."""

import numpy as np


def patch_nan_emission_map(emission: np.ndarray, inplace: bool = False) -> np.ndarray:
    """Fill NaN pixels using the nearest non-NaN pixel in each of the four directions.

    For each NaN pixel, finds:
    - pix_left: nearest non-NaN to the left (same row, smaller column)
    - pix_right: nearest non-NaN to the right (same row, larger column)
    - pix_up: nearest non-NaN above (same column, larger row)
    - pix_down: nearest non-NaN below (same column, smaller row)

    Then sets: pix_val_new = (pix_left + pix_right + pix_up + pix_down) / 4.
    If a direction has no non-NaN pixel (e.g. at edges), that direction is omitted
    and the average is over the remaining valid neighbors only.

    Parameters
    ----------
    emission : np.ndarray
        Emission map, 2D (ny, nx) or 3D (ny, nx, nf). NaNs are patched per 2D slice.
    inplace : bool, optional
        If True, modify the array in place. Default False.

    Returns
    -------
    np.ndarray
        Patched emission map (same shape as input).
    """
    out = emission if inplace else np.array(emission, copy=True, dtype=np.float64)
    if out.ndim == 2:
        _patch_nan_2d(out)
        return out
    if out.ndim == 3:
        for k in range(out.shape[2]):
            _patch_nan_2d(out[:, :, k])
        return out
    raise ValueError("emission must be 2D or 3D")


def _patch_nan_2d(a: np.ndarray, max_passes: int = 10) -> None:
    """Patch NaN in 2D array in place. a is (n_row, n_col); row 0 = bottom, row increases = up."""
    ny, nx = a.shape
    for _ in range(max_passes):
        nan_mask = ~np.isfinite(a)
        if not np.any(nan_mask):
            return
        rows, cols = np.where(nan_mask)
        fixed = 0
        for i, j in zip(rows, cols):
            neighbors = []
            # left: same row, smaller col
            for jj in range(j - 1, -1, -1):
                if np.isfinite(a[i, jj]):
                    neighbors.append(a[i, jj])
                    break
            # right: same row, larger col
            for jj in range(j + 1, nx):
                if np.isfinite(a[i, jj]):
                    neighbors.append(a[i, jj])
                    break
            # down: same col, smaller row
            for ii in range(i - 1, -1, -1):
                if np.isfinite(a[ii, j]):
                    neighbors.append(a[ii, j])
                    break
            # up: same col, larger row
            for ii in range(i + 1, ny):
                if np.isfinite(a[ii, j]):
                    neighbors.append(a[ii, j])
                    break
            if neighbors:
                a[i, j] = np.mean(neighbors)
                fixed += 1
        if fixed == 0:
            break
