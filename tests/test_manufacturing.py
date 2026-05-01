"""
Unit tests for the manufacturing layer: checks, context resolution, and
STEP export round-trip. These are deterministic and don't hit the API.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from build123d import Box, import_step

from manufacturing.checks import (
    CheckResult,
    check_overhangs,
    drainage_check,
    hole_edge_distance,
    min_wall_thickness,
    overall_status,
    run_checks,
    wall_thickness_from_value,
)
from manufacturing.context import (
    Method,
    ProductionContext,
    looks_unsure,
    resolve_printer,
    thresholds_for,
)
from manufacturing.export import export_step, export_stl
from templates.bottle_holder import BottleHolderParams, make_bottle_holder
from templates.bracket import BracketParams, make_bracket
from templates.hook import HookParams, make_hook


# ---------------------------------------------------------------------------
# Wall thickness — both face-pair sampling and parametric value paths
# ---------------------------------------------------------------------------


def test_min_wall_thickness_box_pass():
    """A 10×10×5 box has a 5mm minimum face separation; 1mm threshold passes."""
    b = Box(10, 10, 5)
    r = min_wall_thickness(b, 1.0)
    assert r.status == "pass", r.message
    assert "5.00" in r.message


def test_min_wall_thickness_box_fail():
    """Same box, 6mm threshold should hard-fail."""
    b = Box(10, 10, 5)
    r = min_wall_thickness(b, 6.0)
    assert r.status == "fail", r.message
    assert r.suggestion is not None


def test_wall_thickness_from_value_fdm_tiers():
    t = thresholds_for(ProductionContext(method=Method.FDM))
    assert wall_thickness_from_value(0.5, t).status == "fail"
    assert wall_thickness_from_value(1.0, t).status == "warn"
    assert wall_thickness_from_value(1.5, t).status == "pass"


def test_wall_thickness_from_value_cnc_metal():
    t = thresholds_for(ProductionContext(method=Method.CNC_METAL))
    assert wall_thickness_from_value(0.7, t).status == "fail"
    assert wall_thickness_from_value(0.9, t).status == "warn"
    assert wall_thickness_from_value(1.5, t).status == "pass"


def test_wall_thickness_injection_excessive_warns():
    t = thresholds_for(ProductionContext(method=Method.INJECTION_MOLDING))
    r = wall_thickness_from_value(5.0, t)
    assert r.status == "warn"
    assert "uniformity" in r.name.lower() or "exceeds" in r.message.lower()


# ---------------------------------------------------------------------------
# Overhangs
# ---------------------------------------------------------------------------


def test_check_overhangs_box_clean():
    """An axis-aligned box has no overhangs by our heuristic."""
    b = Box(20, 20, 20)
    r = check_overhangs(b, max_angle_deg=45)
    assert r.status == "pass", r.message


def test_check_overhangs_invalid_angle():
    b = Box(10, 10, 10)
    r = check_overhangs(b, max_angle_deg=120)
    assert r.status == "warn"


# ---------------------------------------------------------------------------
# Hole edge distance
# ---------------------------------------------------------------------------


def test_hole_edge_distance_pass():
    r = hole_edge_distance([(15, 15)], (40, 40), hole_dia=4.5, min_factor=1.5)
    assert r.status == "pass"


def test_hole_edge_distance_fail():
    r = hole_edge_distance([(2, 2)], (40, 40), hole_dia=4.5, min_factor=1.5)
    assert r.status == "fail"
    assert "edge" in r.message.lower()


def test_hole_edge_distance_empty_locations():
    r = hole_edge_distance([], (40, 40), hole_dia=4.5, min_factor=1.5)
    assert r.status == "pass"


# ---------------------------------------------------------------------------
# Drainage check
# ---------------------------------------------------------------------------


def test_drainage_check_passes_on_bottle_holder():
    bh = make_bottle_holder(BottleHolderParams())
    r = drainage_check(bh, expected_hole_dia=8.0)
    assert r.status == "pass", r.message


def test_drainage_check_fails_on_solid_box():
    """A plain box has no drainage hole; the check should fail."""
    b = Box(20, 20, 20)
    r = drainage_check(b, expected_hole_dia=8.0)
    assert r.status == "fail"
    assert r.suggestion is not None


# ---------------------------------------------------------------------------
# Orchestrator: methods differentiate as expected
# ---------------------------------------------------------------------------


def test_run_checks_bottle_holder_fdm_includes_drainage_and_overhangs():
    bh_params = BottleHolderParams()
    bh = make_bottle_holder(bh_params)
    ctx = ProductionContext(method=Method.FDM, material="PLA")
    results = run_checks("bottle_holder", bh_params, bh, ctx)
    names = {r.name for r in results}
    assert "Minimum wall thickness" in names
    assert any("Drainage" in n for n in names)
    assert any("Overhang" in n for n in names)


def test_run_checks_bottle_holder_cnc_metal_skips_overhangs():
    bh_params = BottleHolderParams()
    bh = make_bottle_holder(bh_params)
    ctx = ProductionContext(method=Method.CNC_METAL, material="aluminum")
    results = run_checks("bottle_holder", bh_params, bh, ctx)
    names = {r.name for r in results}
    assert not any("Overhang" in n for n in names), (
        f"CNC metal should not run overhang check; got {names}"
    )
    assert any("corner" in n.lower() for n in names), (
        f"CNC metal should add internal corner check; got {names}"
    )


def test_run_checks_bracket_cnc_metal_flags_hole_edges():
    """Default bracket (60×30 plates, edge-margin 6.75mm) violates CNC metal's
    2.5× hole_edge_factor (needs 11.25mm)."""
    p = BracketParams()
    part = make_bracket(p)
    ctx = ProductionContext(method=Method.CNC_METAL)
    results = run_checks("bracket", p, part, ctx)
    hole_results = [r for r in results if "Hole edge" in r.name]
    assert hole_results, "Expected hole_edge results for bracket"
    assert any(r.status == "fail" for r in hole_results), (
        "CNC metal hole-edge factor 2.5× should fail on default bracket"
    )


def test_run_checks_bracket_fdm_passes_hole_edges():
    """Same bracket, FDM 1.5× factor (6.75mm) passes."""
    p = BracketParams()
    part = make_bracket(p)
    ctx = ProductionContext(method=Method.FDM)
    results = run_checks("bracket", p, part, ctx)
    hole_results = [r for r in results if "Hole edge" in r.name]
    assert all(r.status == "pass" for r in hole_results), (
        f"FDM should pass; got {[(r.name, r.status) for r in hole_results]}"
    )


def test_overall_status_picks_worst():
    pass_r = CheckResult(name="a", status="pass", message="")
    warn_r = CheckResult(name="b", status="warn", message="")
    fail_r = CheckResult(name="c", status="fail", message="")
    assert overall_status([pass_r, pass_r]) == "pass"
    assert overall_status([pass_r, warn_r]) == "warn"
    assert overall_status([warn_r, fail_r]) == "fail"


# ---------------------------------------------------------------------------
# STEP export round-trip
# ---------------------------------------------------------------------------


def test_step_export_roundtrip_bottle_holder(tmp_path: Path):
    bh = make_bottle_holder(BottleHolderParams())
    step_path = export_step(bh, tmp_path / "bh.step")
    assert step_path.exists()
    assert step_path.stat().st_size > 1024
    reimported = import_step(str(step_path))
    bb = reimported.bounding_box()
    # Bottle holder default is 144 × 71 × 100 mm; allow a few mm tolerance
    # for B-Rep precision rounding.
    assert 140 < bb.size.X < 150
    assert 65 < bb.size.Y < 75
    assert 95 < bb.size.Z < 105


def test_stl_and_step_both_emitted(tmp_path: Path):
    h = make_hook(HookParams())
    stl = export_stl(h, tmp_path / "h.stl")
    step = export_step(h, tmp_path / "h.step")
    assert stl.stat().st_size > 1024
    assert step.stat().st_size > 1024


# ---------------------------------------------------------------------------
# ProductionContext NL parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected_canonical,expected_method",
    [
        ("I have a Bambu A1", "Bambu A1", Method.FDM),
        ("Elegoo Centauri Carbon, PLA, 0.4mm nozzle", "Elegoo Centauri Carbon", Method.FDM),
        ("centauri carbon", "Elegoo Centauri Carbon", Method.FDM),
        ("Prusa MK4 at home", "Prusa MK4", Method.FDM),
        ("Anycubic Photon Mono", "Anycubic Photon", Method.SLA),
        ("Voron 2.4 with PETG", "Voron 2.4", Method.FDM),
        ("Formlabs Form 3", "Formlabs Form", Method.SLA),
    ],
)
def test_resolve_printer_recognizes_known_models(text, expected_canonical, expected_method):
    info = resolve_printer(text)
    assert info is not None, f"expected to resolve '{text}'"
    assert info.canonical_name == expected_canonical
    assert info.method == expected_method


def test_resolve_printer_returns_none_for_unknown():
    assert resolve_printer("xyz some random thing") is None
    assert resolve_printer("") is None


@pytest.mark.parametrize(
    "phrase",
    ["not sure", "no idea", "just print it normally", "I dunno", "default"],
)
def test_looks_unsure_recognizes_common_phrases(phrase):
    assert looks_unsure(phrase) is True


@pytest.mark.parametrize(
    "phrase",
    ["I have a Bambu A1", "Xometry CNC in aluminum", "shapeways"],
)
def test_looks_unsure_does_not_overmatch(phrase):
    assert looks_unsure(phrase) is False


# ---------------------------------------------------------------------------
# Problem-face fingerprinting + annotation thumbnails
# ---------------------------------------------------------------------------


def test_overhang_check_populates_problem_faces():
    """A bottle holder with default params has bottom-facing overhangs on the
    standoff arm; the warn result should carry those Face objects."""
    part = make_bottle_holder(BottleHolderParams())
    r = check_overhangs(part, max_angle_deg=45.0)
    assert r.status == "warn"
    assert r.problem_faces is not None and len(r.problem_faces) >= 1


def test_min_wall_thickness_fail_populates_worst_pair():
    """When the min-wall sampler fires a fail, it returns the worst pair."""
    b = Box(10, 10, 5)
    r = min_wall_thickness(b, 6.0)
    assert r.status == "fail"
    assert r.problem_faces is not None and len(r.problem_faces) == 2


def test_min_wall_thickness_pass_leaves_problem_faces_none():
    """Pass results don't carry problem_faces — keeps the UI text-only."""
    b = Box(10, 10, 5)
    r = min_wall_thickness(b, 1.0)
    assert r.status == "pass"
    assert r.problem_faces is None


def test_overhangs_pass_leaves_problem_faces_none():
    """A simple box with no problematic overhangs returns None for problem_faces."""
    b = Box(10, 10, 10)
    r = check_overhangs(b, max_angle_deg=45.0)
    assert r.status == "pass"
    assert r.problem_faces is None


def test_drainage_check_never_carries_problem_faces():
    """Drainage check is presence/diameter only — no face highlighting."""
    part = make_bottle_holder(BottleHolderParams())
    r = drainage_check(part, expected_hole_dia=8.0)
    assert r.problem_faces is None


def test_hole_edge_distance_never_carries_problem_faces():
    """Hole-edge check works in 2D plate-local coords; no Face to attach."""
    bad = hole_edge_distance([(2.0, 2.0)], (40.0, 40.0), hole_dia=4.5, min_factor=1.5)
    assert bad.status == "fail"
    assert bad.problem_faces is None
    good = hole_edge_distance([(20.0, 20.0)], (40.0, 40.0), hole_dia=4.5, min_factor=1.5)
    assert good.status == "pass"
    assert good.problem_faces is None


def test_render_problem_thumbnail_produces_png(tmp_path):
    from manufacturing.visualize import render_problem_thumbnail

    part = make_bottle_holder(BottleHolderParams())
    overhangs = check_overhangs(part, max_angle_deg=45.0)
    out = render_problem_thumbnail(part, overhangs.problem_faces, tmp_path / "t.png")
    assert out is not None
    assert out.exists()
    assert out.stat().st_size > 1000  # actual rendered PNG, not an empty file
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_problem_thumbnail_empty_faces_returns_none(tmp_path):
    from manufacturing.visualize import render_problem_thumbnail

    part = make_bottle_holder(BottleHolderParams())
    assert render_problem_thumbnail(part, [], tmp_path / "x.png") is None
    assert render_problem_thumbnail(part, None, tmp_path / "y.png") is None


def test_annotate_check_thumbnails_attaches_paths_and_caches(tmp_path):
    """Two passes over the same checks: first writes the PNG, second is a
    cache hit (reuses existing path, no rewrite)."""
    from manufacturing.visualize import annotate_check_thumbnails

    part = make_bottle_holder(BottleHolderParams())
    stl_path = export_stl(part, tmp_path / "bh.stl")
    ctx = ProductionContext(method=Method.FDM, material="PLA", nozzle_dia=0.4)
    checks = run_checks("bottle_holder", BottleHolderParams(), part, ctx)

    annotate_check_thumbnails(checks, part, stl_path, tmp_path / "thumbs")
    overhang = next(r for r in checks if r.name == "Overhangs")
    assert overhang.thumbnail_path is not None
    p1 = Path(overhang.thumbnail_path)
    assert p1.exists()
    mtime_before = p1.stat().st_mtime_ns

    # Second pass should be a cache hit — same path, same mtime.
    checks2 = run_checks("bottle_holder", BottleHolderParams(), part, ctx)
    annotate_check_thumbnails(checks2, part, stl_path, tmp_path / "thumbs")
    overhang2 = next(r for r in checks2 if r.name == "Overhangs")
    assert overhang2.thumbnail_path == str(p1)
    assert p1.stat().st_mtime_ns == mtime_before


def test_annotate_check_thumbnails_skips_pass_results(tmp_path):
    """Pass results don't get thumbnails even if some renderer is wired in."""
    from manufacturing.visualize import annotate_check_thumbnails

    part = make_bottle_holder(BottleHolderParams())
    stl_path = export_stl(part, tmp_path / "bh.stl")
    ctx = ProductionContext(method=Method.FDM, material="PLA", nozzle_dia=0.4)
    checks = run_checks("bottle_holder", BottleHolderParams(), part, ctx)
    annotate_check_thumbnails(checks, part, stl_path, tmp_path / "thumbs")
    for r in checks:
        if r.status == "pass":
            assert r.thumbnail_path is None
