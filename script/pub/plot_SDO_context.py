#!/usr/bin/env python
"""Plot SDO context: AIA 304, AIA 171, and HMI magnetogram at a given time.

Uses hvpy to fetch JPEG2000 images from Helioviewer, decodes them, and
produces a 3-panel figure with labels (a), (b), (c).

Example:
    python plot_SDO_context.py --datetime "2025-06-08T20:00:00" -o SDO_context.png
"""

import argparse
import io
import tempfile
from pathlib import Path

import numpy as np

try:
    import hvpy
    from hvpy.datasource import DataSource
except ImportError as e:
    raise ImportError("hvpy is required. Install with: pip install hvpy") from e

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def _jp2_bytes_to_array(jp2_bytes: bytes) -> np.ndarray:
    """Decode JPEG2000 bytes to a numpy array (row, col) for imshow."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(jp2_bytes))
        return np.array(img)
    except Exception:
        pass
    try:
        import glymur
        with tempfile.NamedTemporaryFile(suffix=".jp2", delete=False) as f:
            f.write(jp2_bytes)
            path = f.name
        try:
            j2k = glymur.Jp2k(path)
            arr = j2k[:]
            return np.asarray(arr)
        finally:
            Path(path).unlink(missing_ok=True)
    except Exception as e:
        raise RuntimeError(
            "Could not decode JPEG2000. Install Pillow with openjpeg or glymur: "
            "pip install glymur"
        ) from e


def _parse_datetime(s: str):
    """Parse ISO-like datetime string to datetime (timezone-naive)."""
    from datetime import datetime
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


def fetch_and_decode(dt, source_id: int) -> np.ndarray:
    """Fetch JP2 image for given datetime and source ID; return numpy array."""
    jp2 = hvpy.getJP2Image(dt, source_id)
    if jp2 is None or (isinstance(jp2, (bytes, bytearray)) and len(jp2) == 0):
        raise RuntimeError(f"No image returned for source {source_id} at {dt}")
    return _jp2_bytes_to_array(bytes(jp2))


def main():
    parser = argparse.ArgumentParser(
        description="Plot SDO AIA 304, AIA 171, and HMI magnetogram at a given time (hvpy)."
    )
    parser.add_argument(
        "--datetime",
        "-d",
        default="2025-06-08T20:00:00",
        help='Observation time, e.g. "2025-06-08T20:00:00"',
    )
    parser.add_argument(
        "-o", "--out",
        default="SDO_context.pdf",
        help="Output figure path",
    )
    args = parser.parse_args()

    dt = _parse_datetime(args.datetime)

    # Fetch AIA 304, AIA 171, HMI magnetogram (not continuum)
    aia304 = fetch_and_decode(dt, DataSource.AIA_304.value)
    aia171 = fetch_and_decode(dt, DataSource.AIA_171.value)
    hmi = fetch_and_decode(dt, DataSource.HMI_MAG.value)

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))

    axes[0].imshow(aia304, origin="upper", cmap="gray")
    axes[0].set_title("AIA 304")
    axes[0].axis("off")
    axes[0].text(0.02, 0.98, "(a)", transform=axes[0].transAxes, va="top", ha="left", fontsize=12, fontweight="bold", color="white")

    axes[1].imshow(aia171, origin="upper", cmap="gray")
    axes[1].set_title("AIA 171")
    axes[1].axis("off")
    axes[1].text(0.02, 0.98, "(b)", transform=axes[1].transAxes, va="top", ha="left", fontsize=12, fontweight="bold", color="white")

    # HMI magnetogram: diverging colormap, symmetric about zero (approximate ±1500 G)
    v = np.nanpercentile(hmi, [1, 99])
    print(v)
    vmax = max(abs(v[0]), abs(v[1]), 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    axes[2].imshow(hmi, origin="upper", cmap="RdBu_r")
    axes[2].set_title("HMI magnetogram")
    axes[2].axis("off")
    axes[2].text(0.02, 0.98, "(c)", transform=axes[2].transAxes, va="top", ha="left", fontsize=12, fontweight="bold", color="white")

    fig.suptitle(f"SDO context — {dt.isoformat(timespec='minutes')}", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
