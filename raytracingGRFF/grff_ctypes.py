"""ctypes loader for GRFF ``PyGET_MW`` (``GRFF_DEM_Transfer.so``)."""

from __future__ import annotations

import ctypes
from pathlib import Path

from numpy.ctypeslib import ndpointer

try:
    from GRFFcodes import initGET_MW as _init_from_grffcodes
except ImportError:
    _init_from_grffcodes = None


def initGET_MW(libname: str | Path):
    """Return GRFF ``PyGET_MW`` function bound to ``libname``."""
    if _init_from_grffcodes is not None:
        return _init_from_grffcodes(str(libname))

    _intp = ndpointer(dtype=ctypes.c_int32, flags="F")
    _doublep = ndpointer(dtype=ctypes.c_double, flags="F")
    libc_mw = ctypes.CDLL(str(libname))
    mwfunc = libc_mw.PyGET_MW
    mwfunc.argtypes = [_intp, _doublep, _doublep, _doublep, _doublep, _doublep, _doublep]
    mwfunc.restype = ctypes.c_int
    return mwfunc


def default_grff_lib_path(repo_root: Path | None = None) -> Path:
    """``GRFF/binaries/GRFF_DEM_Transfer.so`` relative to the dev tree (parent of GRFFradioSun)."""
    root = repo_root or Path(__file__).resolve().parents[1]
    return root.parent / "GRFF" / "binaries" / "GRFF_DEM_Transfer.so"
