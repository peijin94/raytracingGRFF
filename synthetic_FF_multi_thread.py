#!/usr/bin/env python
"""
Multi-threaded synthetic free-free emission using GRFF GET_MW_SLICE.

Processes N_pix*N_pix pixels in chunks of N_thread tasks; each chunk is run
in parallel via GET_MW_SLICE (OpenMP inside the library).
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ctypes
from ctypes import c_int, c_void_p
import os

# Constants
R_sun = 6.957e10  # cm
c = 2.998e10  # speed of light, cm/s
kb = 1.38065e-16  # Boltzmann constant, erg/K
sfu2cgs = 1e-19  # SFU to CGS conversion

# GRFF sizes (from MWtransfer.h)
InSize = 15
OutSize = 7
RpSize = 3

GRFF_LIB = '/fast/peijinz/modelSun/GRFF/binaries/GRFF_DEM_Transfer.so'
GRFF_LIB = '/fast/peijinz/modelSun/GRFF/source/GRFF_DEM_Transfer.so'
N_THREAD = 16  # chunk size: pixels per GET_MW_SLICE call


def _load_get_mw_slice(libname):
    """Load GET_MW_SLICE(int argc, void **argv). argc=7, argv = [Lparms_M, Rparms_M, Parms_M, T, DEM_M, DDM_M, RL_M]."""
    lib = ctypes.CDLL(libname)
    func = lib.GET_MW_SLICE
    func.argtypes = [c_int, ctypes.POINTER(c_void_p)]
    func.restype = c_int
    return func


def SyntheticFF(fname_input, freq0, Nfreq, freq_log_step, fname_output, n_thread=N_THREAD):
    """
    Multi-threaded synthetic free-free brightness temperature and V/I using GET_MW_SLICE.

    Parameters
    ----------
    fname_input : str
        Path to LOS npz file (from resampling_MAS.py).
    freq0 : float
        Start frequency in Hz.
    Nfreq : int
        Number of frequencies.
    freq_log_step : float
        log10 step between frequencies.
    fname_output : str
        Base path for output files (no extension).
    n_thread : int
        Number of pixels per chunk (default 16). Each chunk runs in parallel inside GRFF.

    Returns
    -------
    dict
        emission_cube, emission_polVI_cube, frequencies_Hz, x_coords, y_coords.
    """
    print(f"Loading GRFF library from: {GRFF_LIB}")
    GET_MW_SLICE = _load_get_mw_slice(GRFF_LIB)

    data = np.load(fname_input)
    Ne_LOS = data['Ne_LOS']
    Te_LOS = data['Te_LOS']
    B_LOS = data['B_LOS']
    ds_LOS = data['ds_LOS']
    x_coords = data['x_coords']
    y_coords = data['y_coords']

    N_pix = Ne_LOS.shape[0]
    N_z = Ne_LOS.shape[2]
    Nf = Nfreq
    frequencies_Hz = freq0 * (10.0 ** (freq_log_step * np.arange(Nf)))

    pixel_size_Rsun = (x_coords[1] - x_coords[0]) / (R_sun * 1e-2)
    pixel_size_cm = pixel_size_Rsun * R_sun
    area_cm2 = pixel_size_cm * pixel_size_cm

    emission_cube = np.zeros((N_pix, N_pix, Nf), dtype='double')
    emission_polVI_cube = np.zeros((N_pix, N_pix, Nf), dtype='double')

    total_pixels = N_pix * N_pix
    pixel_indices = np.arange(total_pixels)
    n_chunks = (total_pixels + n_thread - 1) // n_thread

    print(f"Image size: {N_pix}x{N_pix}, LOS points: {N_z}, Nf={Nf}")
    print(f"Chunk size: {n_thread} pixels, {n_chunks} chunks")

    distance_cm = 1.49599e13

    for ch in range(n_chunks):
        start = ch * n_thread
        end = min(start + n_thread, total_pixels)
        N_chunk = end - start
        if (ch + 1) % 50 == 0 or ch == 0:
            print(f"Chunk {ch+1}/{n_chunks} ({N_chunk} pixels)...")

        # Lparms_M: [Npix, Nz, Nf, NT, ...]; inner GET_MW uses Lparms_M+1 so gets (Nz, Nf, NT, ...)
        Lparms_M = np.zeros(6, dtype='int32')
        Lparms_M[0] = N_chunk
        Lparms_M[1] = N_z
        Lparms_M[2] = Nf
        Lparms_M[3] = 0   # NT (no DEM)

        # Rparms_M: (3 * N_chunk), same [area, freq0, log_step] for each pixel
        Rparms_M = np.zeros((RpSize * N_chunk,), dtype='double', order='F')
        for p in range(N_chunk):
            Rparms_M[p * RpSize + 0] = area_cm2
            Rparms_M[p * RpSize + 1] = freq0
            Rparms_M[p * RpSize + 2] = freq_log_step

        # Parms_M: (15, Nz, N_chunk) in F-order -> flat 15*Nz*N_chunk
        Parms_M = np.zeros((InSize, N_z, N_chunk), dtype='double', order='F')
        for p in range(N_chunk):
            idx = pixel_indices[start + p]
            i, j = idx // N_pix, idx % N_pix
            ne_los = Ne_LOS[i, j, :].copy()
            te_los = Te_LOS[i, j, :].copy()
            b_los = B_LOS[i, j, :].copy()
            ds_los = ds_LOS[i, j, :].copy()
            valid = ~(np.isnan(ne_los) | np.isnan(te_los) | np.isnan(b_los))
            for k in range(N_z):
                if valid[k]:
                    Parms_M[0, k, p] = ds_los[k]
                    Parms_M[1, k, p] = te_los[k]
                    Parms_M[2, k, p] = ne_los[k]
                    Parms_M[3, k, p] = b_los[k]
                else:
                    Parms_M[0, k, p] = 0.0
                    Parms_M[1, k, p] = 1e4
                    Parms_M[2, k, p] = 1e-8
                    Parms_M[3, k, p] = 0.0
                Parms_M[4, k, p] = 90.0
                Parms_M[5, k, p] = 0.0
                Parms_M[6, k, p] = 1 + 4
                Parms_M[7, k, p] = 30
                Parms_M[8, k, p] = Parms_M[9, k, p] = Parms_M[10, k, p] = 0.0
                Parms_M[11, k, p] = Parms_M[12, k, p] = Parms_M[13, k, p] = Parms_M[14, k, p] = 0

        # RL_M: (7, Nf, N_chunk) F-order
        RL_M = np.zeros((OutSize, Nf, N_chunk), dtype='double', order='F')

        argv = (c_void_p * 7)()
        argv[0] = c_void_p(Lparms_M.ctypes.data)
        argv[1] = c_void_p(Rparms_M.ctypes.data)
        argv[2] = c_void_p(Parms_M.ctypes.data)
        argv[3] = c_void_p(0)  # T_arr not used
        argv[4] = c_void_p(0)  # DEM_arr_M not used
        argv[5] = c_void_p(0)  # DDM_arr_M not used
        argv[6] = c_void_p(RL_M.ctypes.data)

        res = GET_MW_SLICE(7, argv)
        if res != 0:
            print(f"Warning: GET_MW_SLICE returned {res} for chunk {ch}")

        for p in range(N_chunk):
            idx = pixel_indices[start + p]
            i, j = idx // N_pix, idx % N_pix
            for ifreq in range(Nf):
                intensity = RL_M[5, ifreq, p] + RL_M[6, ifreq, p]
                nu_GHz = RL_M[0, ifreq, p]
                nu_Hz = frequencies_Hz[ifreq] if nu_GHz <= 0 else nu_GHz * 1e9
                conversion_factor = (sfu2cgs * c * c / (2.0 * kb * nu_Hz * nu_Hz) / area_cm2) * (distance_cm * distance_cm)
                emission_cube[i, j, ifreq] = intensity * conversion_factor
                denom = RL_M[5, ifreq, p] + RL_M[6, ifreq, p]
                emission_polVI_cube[i, j, ifreq] = (RL_M[5, ifreq, p] - RL_M[6, ifreq, p]) / denom if denom != 0 else 0.0

    print("\nBrightness temperature calculation complete!")
    frequency_first = frequencies_Hz[0]
    emission_map_first = emission_cube[:, :, 0]
    emission_polVI_map_first = emission_polVI_cube[:, :, 0]
    center_size = 16
    center_start = N_pix // 2 - center_size // 2
    center_end = N_pix // 2 + center_size // 2
    center_region = emission_map_first[center_start:center_end, center_start:center_end]
    valid_center = center_region[center_region > 0]
    avg_center_str = f"{np.mean(valid_center):.2e}" if len(valid_center) > 0 else "N/A"
    if len(valid_center) > 0:
        print(f"\nAverage brightness temperature (center {center_size}x{center_size}, first freq): {np.mean(valid_center):.2e} K")

    print("\nSaving brightness temperature cube...")
    np.savez_compressed(fname_output + '.npz',
                        emission_cube=emission_cube, emission_polVI_cube=emission_polVI_cube,
                        frequencies_Hz=frequencies_Hz, x_coords=x_coords, y_coords=y_coords)
    print(f"Brightness temperature cube saved to {fname_output}.npz (shape {N_pix} x {N_pix} x {Nf})")

    print("\nPlotting brightness temperature map (first frequency)...")
    fig, ax = plt.subplots(figsize=(6, 4.8))
    x_range = [x_coords[0] / (R_sun * 1e-2), x_coords[-1] / (R_sun * 1e-2)]
    y_range = [y_coords[0] / (R_sun * 1e-2), y_coords[-1] / (R_sun * 1e-2)]
    emission_plot = emission_map_first.copy()
    emission_plot[emission_plot == 0] = np.nan
    im = ax.imshow(emission_plot, origin='lower', extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                   aspect='equal', cmap='hot', interpolation='bilinear')
    ax.set_xlabel('x (R_sun)')
    ax.set_ylabel('y (R_sun)')
    ax.set_title(f'synthetic $T_b$ map at {frequency_first/1e9:.3f} GHz (multi-thread)')
    plt.colorbar(im, ax=ax, label='T_b (K)')
    ax.text(0.97, 0.97, f'Center $T_b$: {avg_center_str}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(fname_output + '.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Emission map saved to {fname_output}.png")

    fig_tb_vi, (ax_tb, ax_vi) = plt.subplots(1, 2, figsize=(12, 4.2))
    im_tb = ax_tb.imshow(emission_plot, origin='lower', extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                         aspect='equal', cmap='hot', interpolation='bilinear')
    ax_tb.set_xlabel('x (R_sun)')
    ax_tb.set_ylabel('y (R_sun)')
    ax_tb.set_title(f'$T_b$ at {frequency_first/1e9:.3f} GHz')
    plt.colorbar(im_tb, ax=ax_tb, label='T_b (K)')
    pol_vi_plot = emission_polVI_map_first.copy()
    pol_vi_plot[emission_map_first == 0] = np.nan
    vmax_vi = np.nanmax(np.abs(pol_vi_plot))
    if np.isnan(vmax_vi) or vmax_vi == 0:
        vmax_vi = 1.0
    im_vi = ax_vi.imshow(pol_vi_plot, origin='lower', extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                         aspect='equal', cmap='RdBu_r', interpolation='bilinear', vmin=-vmax_vi, vmax=vmax_vi)
    ax_vi.set_xlabel('x (R_sun)')
    ax_vi.set_ylabel('y (R_sun)')
    ax_vi.set_title(f'V/I at {frequency_first/1e9:.3f} GHz')
    plt.colorbar(im_vi, ax=ax_vi, label='V/I')
    plt.tight_layout()
    plt.savefig(fname_output + '_Tb_VI.png', dpi=150, bbox_inches='tight')
    plt.close(fig_tb_vi)
    print(f"T_b and V/I side-by-side plot saved to {fname_output}_Tb_VI.png")

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    im2 = ax2.imshow(emission_plot, origin='lower', extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                     aspect='equal', cmap='hot', interpolation='bilinear',
                     norm=mcolors.LogNorm(vmin=np.nanmin(emission_plot[emission_plot > 0]),
                                          vmax=np.nanmax(emission_plot)))
    ax2.set_xlabel('x (R_sun)')
    ax2.set_ylabel('y (R_sun)')
    ax2.set_title(f'synthetic $T_b$ map at {frequency_first/1e9:.3f} GHz (Log Scale)')
    plt.colorbar(im2, ax=ax2, label='T_b (K)')
    ax2.text(0.97, 0.97, f'Center $T_b$: {avg_center_str}', transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(fname_output + '_log.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Log-scale brightness temperature map saved to {fname_output}_log.png")
    print("\nSynthetic brightness temperature calculation complete!")

    return {
        'emission_cube': emission_cube,
        'emission_polVI_cube': emission_polVI_cube,
        'frequencies_Hz': frequencies_Hz,
        'x_coords': x_coords,
        'y_coords': y_coords,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-threaded synthetic free-free emission via GRFF GET_MW_SLICE.')
    parser.add_argument('--input', '-i', type=str, default='LOS_data.npz',
                        help='Path to LOS npz file (default: LOS_data.npz)')
    parser.add_argument('--output', '-o', type=str, default='emission_map',
                        help='Base path for output files (default: emission_map)')
    parser.add_argument('--freq0', '-f', type=float, default=450e6,
                        help='Start frequency in Hz (default: 450e6)')
    parser.add_argument('--Nfreq', '-n', type=int, default=4,
                        help='Number of frequencies (default: 4)')
    parser.add_argument('--freq-log-step', '-s', type=float, default=0.1,
                        help='log10 step between frequencies (default: 0.1)')
    parser.add_argument('--n-thread', '-t', type=int, default=N_THREAD,
                        help=f'Pixels per chunk / thread count for GET_MW_SLICE (default: {N_THREAD})')
    args = parser.parse_args()
    SyntheticFF(args.input, args.freq0, args.Nfreq, args.freq_log_step, args.output, args.n_thread)
