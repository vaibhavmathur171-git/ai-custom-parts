"""
Export wrappers.

Thin layer around build123d's exporters so the rest of the app doesn't
have to know about file paths or precision flags.
"""

from __future__ import annotations

from pathlib import Path

from build123d import Part, PrecisionMode, Unit, export_step as _build123d_export_step, export_stl as _build123d_export_stl


def export_stl(part: Part, path: str | Path) -> Path:
    """Write a binary STL. STL is mesh — for FDM/SLA printers."""
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    _build123d_export_stl(part, str(path))
    return path


def export_step(part: Part, path: str | Path) -> Path:
    """Write a STEP (AP214) file. STEP is parametric B-Rep — for CAD,
    CNC, injection molding, or any toolchain past 'just print it'."""
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    # Average precision is fine for our part scale; high precision blows up
    # file sizes for no perceptible quality benefit on our 100mm-class parts.
    _build123d_export_step(part, str(path), unit=Unit.MM, precision_mode=PrecisionMode.AVERAGE)
    return path
