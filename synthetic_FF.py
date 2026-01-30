#!/usr/bin/env python
"""
Synthetic free-free emission calculation using GRFF.

Calls GRFF_DEM_Transfer.so along each LOS using the resampled MAS data.
Supports multi-frequency: outputs a cube (N_pix, N_pix, N_freq).
Plots use the first frequency only. Uses T_e, N_e, B, and ds from the LOS sampling.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ctypes
from numpy.ctypeslib import ndpointer
import os

# Import GRFF initialization function
# Assuming GRFFcodes.py is in the same directory or in path
try:
    from GRFFcodes import initGET_MW
except ImportError:
    # If GRFFcodes is not available, define it here
    def initGET_MW(libname):
        _intp = ndpointer(dtype=ctypes.c_int32, flags='F')
        _doublep = ndpointer(dtype=ctypes.c_double, flags='F')
        
        libc_mw = ctypes.CDLL(libname)
        mwfunc = libc_mw.PyGET_MW
        mwfunc.argtypes = [_intp, _doublep, _doublep, _doublep, _doublep, _doublep, _doublep]
        mwfunc.restype = ctypes.c_int
        return mwfunc

# Constants
R_sun = 6.957e10  # cm
c = 2.998e10  # speed of light, cm/s
kb = 1.38065e-16  # Boltzmann constant, erg/K
sfu2cgs = 1e-19  # SFU to CGS conversion

# GRFF library path
libname = './GRFF/binaries/GRFF_DEM_Transfer.so'
if not os.path.exists(libname):
    # Try alternative paths
    alt_paths = [
        './GRFF_DEM_Transfer.so',
        '../GRFF/binaries/GRFF_DEM_Transfer.so',
        '/opt/devel/peijin/solarml/GRFF_DEM_Transfer.so'
    ]
    for alt_path in alt_paths:
        if os.path.exists(alt_path):
            libname = alt_path
            break
    else:
        raise FileNotFoundError(f"GRFF library not found. Tried: {libname} and alternatives")

print(f"Loading GRFF library from: {libname}")
GET_MW = initGET_MW(libname)

# Load LOS data
data = np.load('LOS_data.npz')
Ne_LOS = data['Ne_LOS']  # cm^-3
Te_LOS = data['Te_LOS']  # K
B_LOS = data['B_LOS']    # G
ds_LOS = data['ds_LOS']  # cm
x_coords = data['x_coords']
y_coords = data['y_coords']
z_coords = data['z_coords']

N_pix = Ne_LOS.shape[0]  # Should be 512
N_z = Ne_LOS.shape[2]    # Number of points along LOS

print(f"Image size: {N_pix}x{N_pix}")
print(f"LOS points: {N_z}")

# Frequency parameters: multi-frequency
# Start frequency (Hz), log10 step, number of frequencies
freq_start_Hz = 450e6   # 150 MHz
freq_log_step = 0.1     # log10 step between frequencies
Nf = 4                   # number of frequencies
frequencies_Hz = freq_start_Hz * (10.0 ** (freq_log_step * np.arange(Nf)))  # (Nf,) in Hz

# Set up GRFF parameters
Lparms = np.zeros(5, dtype='int32')
Lparms[0] = N_z  # Number of nodes along LOS
Lparms[1] = Nf   # Number of frequencies

Rparms = np.zeros(3, dtype='double')
# Calculate pixel area in cm^2
pixel_size_Rsun = (x_coords[1] - x_coords[0]) / (R_sun * 1e-2)
pixel_size_cm = pixel_size_Rsun * R_sun
Rparms[0] = pixel_size_cm * pixel_size_cm
Rparms[1] = freq_start_Hz   # Starting frequency, Hz
Rparms[2] = freq_log_step   # Logarithmic step

# Initialize output cube: (N_pix, N_pix, Nf)
emission_cube = np.zeros((N_pix, N_pix, Nf), dtype='double')
emission_polVI_cube = np.zeros((N_pix, N_pix, Nf), dtype='double')

print(f"Frequencies: {frequencies_Hz/1e6} MHz (Nf={Nf})")

# Process each pixel
for i in range(N_pix):
    if (i + 1) % 50 == 0:
        print(f"Processing row {i+1}/{N_pix}...")
    
    for j in range(N_pix):
        # Extract LOS data for this pixel and reverse along LOS
        ne_los = Ne_LOS[i, j, :].copy()  # Reverse array index
        te_los = Te_LOS[i, j, :].copy()  # Reverse array index
        b_los = B_LOS[i, j, :].copy()    # Reverse array index
        ds_los = ds_LOS[i, j, :].copy()  # Reverse array index
        
        # Remove NaN values (points outside model domain)
        valid_mask = ~(np.isnan(ne_los) | np.isnan(te_los) | np.isnan(b_los))
        
        if not np.any(valid_mask):
            emission_cube[i, j, :] = 0.0
            continue

        ne_valid = ne_los[valid_mask]
        te_valid = te_los[valid_mask]
        b_valid = b_los[valid_mask]
        ds_valid = ds_los[valid_mask]
        N_valid = len(ne_valid)

        if N_valid == 0:
            emission_cube[i, j, :] = 0.0
            continue
        
        # Set up parameter array for GRFF
        # Parms shape: (15, N_z)
        Parms = np.zeros((15, N_valid), dtype='double', order='F')
        
        for k in range(N_valid):
            Parms[0, k] = ds_valid[k]  # Voxel size along LOS, cm
            Parms[1, k] = te_valid[k]   # Plasma temperature, K
            Parms[2, k] = ne_valid[k]    # Electron density, cm^-3
            Parms[3, k] = b_valid[k]     # Magnetic field, G
            Parms[4, k] = 90.0           # Viewing angle (90 deg = perpendicular to LOS)
            Parms[5, k] = 0.0            # Azimuthal angle, degrees
            Parms[6, k] = 1 + 4          # Emission mechanism flag (free-free + gyroresonance)
            Parms[7, k] = 30             # Maximum harmonic number
            Parms[8, k] = 0.0            # Proton concentration (not used)
            Parms[9, k] = 0.0            # Neutral hydrogen concentration
            Parms[10, k] = 0.0           # Neutral helium concentration
            Parms[11, k] = 0             # Local DEM on/off (off)
            Parms[12, k] = 0             # Local DDM on/off (off)
            Parms[13, k] = 0             # Element abundance code (coronal)
            Parms[14, k] = 0             # Reserved
        
        # Update Lparms for this LOS (number of valid points)
        Lparms_local = Lparms.copy()
        Lparms_local[0] = N_valid
        
        # Dummy arrays for DEM/DDM (not used)
        dummy_T = np.array(0, dtype='double')
        dummy_DEM = np.array(0, dtype='double')
        dummy_DDM = np.array(0, dtype='double')
        
        # Output array: RL shape (7, Nf)
        RL = np.zeros((7, Nf), dtype='double', order='F')
        
        # Call GRFF (returns all Nf frequencies in one call)
        try:
            res = GET_MW(Lparms_local, Rparms, Parms, dummy_T, dummy_DEM, dummy_DDM, RL)

            if res != 0:
                emission_cube[i, j, :] = 0.0
                continue

            # RL[0, :] = frequencies in GHz (GRFF convention), RL[5, :] = I_L, RL[6, :] = I_R
            # Tb = I * sfu2cgs * c^2 / (2*kb*nu^2) * (distance^2/area); nu must be in Hz
            distance_cm = 1.49599e13
            for ifreq in range(Nf):
                intensity = RL[5, ifreq] + RL[6, ifreq]
                circularpol_VI= (RL[5, ifreq] - RL[6, ifreq]) / (RL[5, ifreq] + RL[6, ifreq])

                # GRFF returns frequency in GHz; convert to Hz for Rayleigh-Jeans
                nu_GHz = RL[0, ifreq]
                if nu_GHz <= 0:
                    nu_Hz = frequencies_Hz[ifreq]
                else:
                    nu_Hz = nu_GHz * 1e9  # GHz -> Hz
                conversion_factor = (sfu2cgs * c * c / (2.0 * kb * nu_Hz * nu_Hz) / Rparms[0]) * (distance_cm * distance_cm)
                T_b = intensity * conversion_factor
                emission_cube[i, j, ifreq] = T_b
                emission_polVI_cube[i, j, ifreq] = circularpol_VI

        except Exception as e:
            print(f"Error processing pixel ({i}, {j}): {e}")
            emission_cube[i, j, :] = 0.0
            emission_polVI_cube[i, j, :] = 0.0

print("\nBrightness temperature calculation complete!")

# First frequency for plotting and center stats
frequency_first = frequencies_Hz[0]
emission_map_first = emission_cube[:, :, 0]
emission_polVI_map_first = emission_polVI_cube[:, :, 0]



# Average value of center 16x16 pixels (first frequency)
center_size = 16
center_start = N_pix // 2 - center_size // 2
center_end = N_pix // 2 + center_size // 2
center_region = emission_map_first[center_start:center_end, center_start:center_end]
valid_center = center_region[center_region > 0]
if len(valid_center) > 0:
    avg_center = np.mean(valid_center)
    print(f"\nAverage brightness temperature (center {center_size}x{center_size}, first freq): {avg_center:.2e} K")
    avg_center_str = f"{avg_center:.2e}"
else:
    avg_center_str = "N/A"

# Save multi-frequency cube
print("\nSaving brightness temperature cube...")
np.savez_compressed('emission_map.npz',
                    emission_cube=emission_cube,
                    emission_polVI_cube=emission_polVI_cube,
                    frequencies_Hz=frequencies_Hz,
                    x_coords=x_coords,
                    y_coords=y_coords)
print("Brightness temperature cube saved to emission_map.npz (shape {} x {} x {})".format(N_pix, N_pix, Nf))

# Plot first frequency only
print("\nPlotting brightness temperature map (first frequency)...")
fig, ax = plt.subplots(figsize=(6, 4.8))

x_range = [x_coords[0] / (R_sun * 1e-2), x_coords[-1] / (R_sun * 1e-2)]
y_range = [y_coords[0] / (R_sun * 1e-2), y_coords[-1] / (R_sun * 1e-2)]

emission_plot = emission_map_first.copy()
emission_plot[emission_plot == 0] = np.nan

im = ax.imshow(emission_plot, origin='lower',
               extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
               aspect='equal', cmap='hot', interpolation='bilinear')
ax.set_xlabel('x (R_sun)')
ax.set_ylabel('y (R_sun)')
ax.set_title(f'synthetic $T_b$ map at {frequency_first/1e9:.3f} GHz')
cbar = plt.colorbar(im, ax=ax, label='T_b (K)')

# Add text annotation with center T_b value
ax.text(0.97, 0.97, f'Center $T_b$: {avg_center_str}', 
        transform=ax.transAxes, 
        fontsize=12, 
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('emission_map.png', dpi=150, bbox_inches='tight')
print("Emission map saved to emission_map.png")

# Side-by-side: T_b and V/I (first frequency)
fig_tb_vi, (ax_tb, ax_vi) = plt.subplots(1, 2, figsize=(12, 4.2))

im_tb = ax_tb.imshow(emission_plot, origin='lower',
                     extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                     aspect='equal', cmap='hot', interpolation='bilinear')
ax_tb.set_xlabel('x (R_sun)')
ax_tb.set_ylabel('y (R_sun)')
ax_tb.set_title(f'$T_b$ at {frequency_first/1e9:.3f} GHz')
plt.colorbar(im_tb, ax=ax_tb, label='T_b (K)')

# V/I: mask invalid (zero total intensity)
pol_vi_plot = emission_polVI_map_first.copy()
pol_vi_plot[emission_map_first == 0] = np.nan
vmax_vi = np.nanmax(np.abs(pol_vi_plot))
if np.isnan(vmax_vi) or vmax_vi == 0:
    vmax_vi = 1.0
im_vi = ax_vi.imshow(pol_vi_plot, origin='lower',
                     extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                     aspect='equal', cmap='RdBu_r', interpolation='bilinear',
                     vmin=-vmax_vi, vmax=vmax_vi)
ax_vi.set_xlabel('x (R_sun)')
ax_vi.set_ylabel('y (R_sun)')
ax_vi.set_title(f'V/I at {frequency_first/1e9:.3f} GHz')
cbar_vi = plt.colorbar(im_vi, ax=ax_vi, label='V/I')

plt.tight_layout()
plt.savefig('emission_map_Tb_VI.png', dpi=150, bbox_inches='tight')
plt.close(fig_tb_vi)
print("T_b and V/I side-by-side plot saved to emission_map_Tb_VI.png")

# Also create a log-scale version
fig2, ax2 = plt.subplots(figsize=(10, 10))
im2 = ax2.imshow(emission_plot, origin='lower',
                extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                aspect='equal', cmap='hot', interpolation='bilinear',
                norm=mcolors.LogNorm(vmin=np.nanmin(emission_plot[emission_plot > 0]),
                                    vmax=np.nanmax(emission_plot)))
ax2.set_xlabel('x (R_sun)')
ax2.set_ylabel('y (R_sun)')
ax2.set_title(f'synthetic $T_b$ map at {frequency_first/1e9:.3f} GHz (Log Scale)')
cbar2 = plt.colorbar(im2, ax=ax2, label='T_b (K)')

# Add text annotation with center T_b value
ax2.text(0.97, 0.97, f'Center $T_b$: {avg_center_str}', 
         transform=ax2.transAxes, 
         fontsize=12, 
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('emission_map_log.png', dpi=150, bbox_inches='tight')
print("Log-scale brightness temperature map saved to emission_map_log.png")

print("\nSynthetic brightness temperature calculation complete!")
