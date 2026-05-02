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


# ---------------------------------------------------------------------------
# Bottle holder clamp_opening_angle_deg modifier (Step 5C)
# ---------------------------------------------------------------------------


def test_bottle_holder_default_is_closed_clamp():
    """clamp_opening_angle_deg defaults to 0 (closed ring) — backwards
    compatibility with Steps 1-5B."""
    from templates.bottle_holder import BottleHolderParams
    assert BottleHolderParams().clamp_opening_angle_deg == 0.0


def test_bottle_holder_validates_clamp_opening_bounds():
    """[0, 180] inclusive; anything outside must surface a validation error."""
    from templates.bottle_holder import BottleHolderParams
    assert BottleHolderParams(clamp_opening_angle_deg=0).validate() == []
    assert BottleHolderParams(clamp_opening_angle_deg=180).validate() == []
    assert BottleHolderParams(clamp_opening_angle_deg=120).validate() == []
    bad_low = BottleHolderParams(clamp_opening_angle_deg=-1).validate()
    bad_high = BottleHolderParams(clamp_opening_angle_deg=181).validate()
    assert any("clamp_opening" in e for e in bad_low), bad_low
    assert any("clamp_opening" in e for e in bad_high), bad_high


@pytest.mark.parametrize("angle", [90.0, 120.0, 180.0])
def test_bottle_holder_open_clamp_produces_smaller_volume(angle, tmp_path):
    """Opening the clamp must remove material (volume strictly less than
    closed) and the cut must be on the +X side opposite the standoff arm
    (so the bbox max-X shrinks while min-X is unchanged)."""
    from templates.bottle_holder import BottleHolderParams, make_bottle_holder

    closed = make_bottle_holder(BottleHolderParams())
    open_part = make_bottle_holder(BottleHolderParams(clamp_opening_angle_deg=angle))

    assert open_part.volume < closed.volume, (
        f"opening={angle}°: volume {open_part.volume} should be < closed {closed.volume}"
    )

    # Cut is on the +X side opposite the arm, so bbox max.X shrinks while
    # min.X (the cup's far -X edge) stays put.
    closed_bb = closed.bounding_box()
    open_bb = open_part.bounding_box()
    assert open_bb.max.X < closed_bb.max.X - 0.01, (
        f"opening={angle}°: max.X should shrink from {closed_bb.max.X} to less"
    )
    assert abs(open_bb.min.X - closed_bb.min.X) < 0.01, (
        f"opening={angle}°: min.X (cup side) should be unchanged"
    )

    # Larger angle = more material removed.
    if angle > 90:
        smaller = make_bottle_holder(BottleHolderParams(clamp_opening_angle_deg=90))
        assert open_part.volume < smaller.volume, (
            f"opening={angle}° must remove more than 90°"
        )


def test_bottle_holder_open_clamp_passes_manufacturing_checks():
    """A typical snap-on bottle holder (FDM PLA, 120° open) must produce a
    sensible run_checks output with no FAIL — the opening doesn't introduce
    geometry that the existing checks should reject."""
    from manufacturing.checks import run_checks
    from manufacturing.context import Method, ProductionContext
    from templates.bottle_holder import BottleHolderParams, make_bottle_holder

    params = BottleHolderParams(clamp_opening_angle_deg=120.0)
    part = make_bottle_holder(params)
    ctx = ProductionContext(method=Method.FDM, material="PLA", nozzle_dia=0.4)
    results = run_checks("bottle_holder", params, part, ctx)
    statuses = [r.status for r in results]
    assert "fail" not in statuses, (
        f"snap-on bottle holder must not FAIL any check; got {[(r.name, r.status) for r in results]}"
    )
