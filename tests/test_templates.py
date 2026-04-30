"""
Smoke test: every registered template builds a valid STL at default params.

This is the line of defense that catches a template regression before it
shows up as a broken viewer in the configurator. It runs each template's
geometry function with its default parameter instance, exports to STL, and
verifies the file is a non-trivial binary STL with a finite, non-degenerate
bounding box.
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import pytest
from build123d import export_stl

from templates.registry import TemplateSpec, list_templates


# Minimum STL size that still represents a real solid. Binary STL has an
# 80-byte header + 4-byte triangle count + 50 bytes per triangle. A solid
# with even a few hundred triangles is well above 1 KB.
MIN_STL_BYTES = 1024


def _binary_stl_triangle_count(path: Path) -> int:
    """Read the triangle count from a binary STL header (UINT32 at offset 80)."""
    with path.open("rb") as f:
        f.seek(80)
        (count,) = struct.unpack("<I", f.read(4))
    return count


@pytest.mark.parametrize("spec", list_templates(), ids=lambda s: s.name)
def test_template_default_params_produce_valid_stl(spec: TemplateSpec, tmp_path: Path):
    """Each template must produce a non-trivial STL at its default parameters."""
    # 1. Defaults must pass the template's own validation.
    errors = spec.default_params.validate()
    assert errors == [], (
        f"{spec.name}: default params failed validation: {errors}"
    )

    # 2. Geometry function returns a Part.
    part = spec.make_fn(spec.default_params)
    assert part is not None, f"{spec.name}: make_fn returned None"

    # 3. Bounding box is finite and non-degenerate (every axis has extent).
    bbox = part.bounding_box()
    for axis in ("X", "Y", "Z"):
        size = getattr(bbox.size, axis)
        assert size > 0.1, (
            f"{spec.name}: bounding box {axis} extent is {size}, "
            f"expected > 0.1mm"
        )

    # 4. STL export writes a real file with a sane triangle count.
    stl_path = tmp_path / f"{spec.name}.stl"
    export_stl(part, str(stl_path))
    assert stl_path.exists(), f"{spec.name}: STL was not written"
    assert stl_path.stat().st_size >= MIN_STL_BYTES, (
        f"{spec.name}: STL is suspiciously small "
        f"({stl_path.stat().st_size} bytes)"
    )

    tri_count = _binary_stl_triangle_count(stl_path)
    assert tri_count > 10, (
        f"{spec.name}: STL has {tri_count} triangles; expected > 10"
    )


def test_registry_has_three_templates():
    """Sanity-check the build sequence: Step 2B ships exactly three templates."""
    names = [s.name for s in list_templates()]
    assert set(names) == {"bottle_holder", "hook", "bracket"}, (
        f"Unexpected template set: {names}"
    )
