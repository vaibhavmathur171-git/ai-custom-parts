"""
Manufacturing checks.

Each check is a small, focused function that returns a `CheckResult`. The
orchestrator (`run_checks`) picks the set of checks appropriate for the
template + production context and assembles the results into the list the
UI renders as the Manufacturing Checklist.

Design principles:
- Each check is independently testable against synthetic geometry.
- A check never raises on bad geometry — it returns a CheckResult with
  status='warn' explaining what it could not measure.
- The orchestrator uses the parametric values from the template's params
  dataclass when they're more reliable than face-sampling. Sampling
  routines exist for spec compliance and for parts whose parameters don't
  expose the relevant dimension directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from build123d import GeomType, Part, Vector

from manufacturing.context import (
    Method,
    ProductionContext,
    Thresholds,
    thresholds_for,
)


CheckStatus = str  # "pass" | "warn" | "fail"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    suggestion: str | None = None
    # build123d Face objects identifying the offending geometry. Populated
    # only by checks that can pinpoint specific features (overhangs,
    # measured wall pairs). Parametric/derived checks (the value-based
    # wall check, hole-edge clearance, drainage presence) leave it None
    # — the UI then renders only the text message for those.
    problem_faces: list | None = None
    # Path to a rendered annotation thumbnail. Populated by the agent /
    # orchestrator AFTER the check itself returns; keeps the check
    # functions free of disk I/O and matplotlib dependencies.
    thumbnail_path: str | None = None


def _classify(status_a: CheckStatus, status_b: CheckStatus) -> CheckStatus:
    """Roll up two statuses to the worst of the two ('fail' beats 'warn' beats 'pass')."""
    rank = {"pass": 0, "warn": 1, "fail": 2}
    return status_a if rank[status_a] >= rank[status_b] else status_b


def overall_status(results: Iterable[CheckResult]) -> CheckStatus:
    out: CheckStatus = "pass"
    for r in results:
        out = _classify(out, r.status)
    return out


# ---------------------------------------------------------------------------
# 1. Minimum wall thickness
# ---------------------------------------------------------------------------


def _sample_min_wall_thickness(part: Part) -> tuple[float | None, list]:
    """Pair every planar face with every other planar face whose normal is
    roughly antiparallel; the minimum perpendicular separation across pairs
    is taken as a proxy for the part's minimum wall thickness.

    Returns (measured_min_mm or None, [face_a, face_b]) for the pair that
    produced the minimum, so the caller can highlight them on a thumbnail.
    """
    planar = [f for f in part.faces() if f.geom_type == GeomType.PLANE]
    if len(planar) < 2:
        return None, []

    centers = [f.center() for f in planar]
    normals = [f.normal_at(c) for f, c in zip(planar, centers)]

    measured = math.inf
    worst_pair: list = []
    for i in range(len(planar)):
        n_i = normals[i]
        c_i = centers[i]
        for j in range(i + 1, len(planar)):
            n_j = normals[j]
            if n_i.dot(n_j) > -0.95:
                continue
            d = abs((centers[j] - c_i).dot(n_i))
            if d > 1e-3 and d < measured:
                measured = d
                worst_pair = [planar[i], planar[j]]

    return (None if measured == math.inf else measured), worst_pair


def min_wall_thickness(part: Part, threshold: float) -> CheckResult:
    """Sample the part for its minimum face-pair separation and compare to threshold.

    `threshold` is the hard-fail minimum wall thickness in millimeters.
    """
    measured, worst_pair = _sample_min_wall_thickness(part)
    if measured is None:
        return CheckResult(
            name="Minimum wall thickness",
            status="warn",
            message="Could not measure wall thickness from face geometry (no opposite planar pair).",
            suggestion="Verify the part's wall is at least the recommended minimum manually.",
        )
    if measured + 1e-6 < threshold:
        return CheckResult(
            name="Minimum wall thickness",
            status="fail",
            message=f"Sampled wall thickness {measured:.2f}mm is below the {threshold:.2f}mm minimum.",
            suggestion=f"Increase wall thickness to at least {threshold:.2f}mm.",
            problem_faces=worst_pair or None,
        )
    return CheckResult(
        name="Minimum wall thickness",
        status="pass",
        message=f"Sampled wall thickness {measured:.2f}mm meets the {threshold:.2f}mm minimum.",
    )


def wall_thickness_from_value(measured_mm: float, thresholds: Thresholds) -> CheckResult:
    """Two-tier wall thickness check (fail / warn / pass) based on a known value.

    Used by the orchestrator when the template exposes its wall thickness
    parametrically — more accurate than face sampling on cylindrical walls.
    """
    label = thresholds.method_label
    if measured_mm + 1e-6 < thresholds.wall_fail_below:
        return CheckResult(
            name="Minimum wall thickness",
            status="fail",
            message=(
                f"Wall {measured_mm:.2f}mm is below the {thresholds.wall_fail_below:.2f}mm "
                f"hard minimum for {label}."
            ),
            suggestion=(
                f"Increase to at least {thresholds.default_wall_recommendation:.1f}mm."
            ),
        )
    if measured_mm + 1e-6 < thresholds.wall_warn_below:
        return CheckResult(
            name="Minimum wall thickness",
            status="warn",
            message=(
                f"Wall {measured_mm:.2f}mm is thin for {label} "
                f"(recommended ≥ {thresholds.wall_warn_below:.1f}mm)."
            ),
            suggestion=(
                f"Consider {thresholds.default_wall_recommendation:.1f}mm for safety margin."
            ),
        )
    if (
        thresholds.wall_excessive_above is not None
        and measured_mm > thresholds.wall_excessive_above + 1e-6
    ):
        return CheckResult(
            name="Wall thickness uniformity",
            status="warn",
            message=(
                f"Wall {measured_mm:.2f}mm exceeds the {thresholds.wall_excessive_above:.1f}mm "
                f"upper guideline for {label} (uneven cooling, sink marks)."
            ),
            suggestion=(
                f"Consider coring or thinning to ~{thresholds.default_wall_recommendation:.1f}mm."
            ),
        )
    return CheckResult(
        name="Minimum wall thickness",
        status="pass",
        message=f"Wall {measured_mm:.2f}mm is OK for {label}.",
    )


# ---------------------------------------------------------------------------
# 2. Overhang check
# ---------------------------------------------------------------------------


def check_overhangs(part: Part, max_angle_deg: float = 45.0) -> CheckResult:
    """FDM overhang heuristic.

    A face is considered a problematic overhang if its outward normal points
    sufficiently downward — specifically, if the angle between the normal
    and -Z is less than `max_angle_deg` (i.e., the face's surface is closer
    to horizontal than the slope a slicer can self-support).
    """
    if max_angle_deg <= 0 or max_angle_deg >= 90:
        return CheckResult(
            name="Overhangs",
            status="warn",
            message=f"Invalid max_angle_deg={max_angle_deg}; skipping check.",
        )

    cos_threshold = math.cos(math.radians(max_angle_deg))
    bbox = part.bounding_box()
    bad_faces: list = []
    bad_area = 0.0
    for f in part.faces():
        try:
            n = f.normal_at(f.center())
        except Exception:
            continue
        if n.Z >= -cos_threshold:
            continue  # surface too vertical to be a problem overhang
        # Build-plate-touching faces aren't overhangs.
        if abs(f.center().Z - bbox.min.Z) < 1.0:
            continue
        if f.area < 1.0:
            continue
        bad_faces.append(f)
        bad_area += f.area

    if not bad_faces:
        return CheckResult(
            name="Overhangs",
            status="pass",
            message=f"No problematic overhangs (>{max_angle_deg:.0f}° from vertical).",
        )
    return CheckResult(
        name="Overhangs",
        status="warn",
        message=(
            f"{len(bad_faces)} face(s) overhang past the {max_angle_deg:.0f}° self-support limit "
            f"({bad_area:.0f} mm² total). Supports may be required for FDM."
        ),
        suggestion=(
            "Reorient the part on the build plate, or enable support material in your slicer."
        ),
        problem_faces=bad_faces,
    )


# ---------------------------------------------------------------------------
# 3. Hole edge distance
# ---------------------------------------------------------------------------


def hole_edge_distance(
    hole_locations: Sequence[tuple[float, float]],
    plate_size: tuple[float, float],
    hole_dia: float,
    min_factor: float = 1.5,
) -> CheckResult:
    """Check that every hole center is at least `min_factor * hole_dia` from
    the nearest plate edge.

    `hole_locations` are 2D centers in plate-local coordinates with the
    plate spanning (0, 0) → (plate_w, plate_h). This avoids depending on
    where the part sits in 3D space.
    """
    plate_w, plate_h = plate_size
    margin = min_factor * hole_dia
    bad: list[tuple[tuple[float, float], float]] = []
    for hx, hy in hole_locations:
        dist = min(hx, plate_w - hx, hy, plate_h - hy)
        if dist + 1e-6 < margin:
            bad.append(((hx, hy), dist))

    if not hole_locations:
        return CheckResult(
            name="Hole edge clearance",
            status="pass",
            message="No mounting holes to check.",
        )
    if not bad:
        return CheckResult(
            name="Hole edge clearance",
            status="pass",
            message=(
                f"All {len(hole_locations)} hole(s) are ≥ {margin:.2f}mm "
                f"({min_factor:.1f}× hole_dia) from the nearest edge."
            ),
        )

    worst = min(d for _, d in bad)
    return CheckResult(
        name="Hole edge clearance",
        status="fail",
        message=(
            f"{len(bad)} of {len(hole_locations)} hole(s) sit too close to a plate edge "
            f"(worst: {worst:.2f}mm, need ≥ {margin:.2f}mm)."
        ),
        suggestion=(
            f"Either move the holes inward or increase the plate dimensions so each "
            f"hole has at least {min_factor:.1f}× hole_dia clearance."
        ),
    )


# ---------------------------------------------------------------------------
# 4. Drainage check (bottle holder only)
# ---------------------------------------------------------------------------


def drainage_check(part: Part, expected_hole_dia: float, tolerance: float = 1.0) -> CheckResult:
    """Verify the bottle holder has a circular hole through its lowest face
    of approximately the expected diameter.

    Identifies the lowest planar face, looks at its inner edges (boundary
    loops other than the outer one), and accepts the check if any of those
    inner loops corresponds to a circle of roughly `expected_hole_dia`.
    """
    bbox = part.bounding_box()
    z_min = bbox.min.Z

    # Find planar faces near the bottom (the cup floor).
    floor_candidates = [
        f for f in part.faces()
        if f.geom_type == GeomType.PLANE
        and abs(f.center().Z - z_min) < 0.5
    ]
    if not floor_candidates:
        return CheckResult(
            name="Drainage hole",
            status="warn",
            message="Could not identify a floor face to check for drainage.",
        )

    # Pick the largest of the floor candidates (the actual cup floor).
    floor = max(floor_candidates, key=lambda f: f.area)

    # The drainage hole shows up as a CIRCLE-typed inner edge on the floor.
    inner_circle_radii: list[float] = []
    for edge in floor.edges():
        if edge.geom_type != GeomType.CIRCLE:
            continue
        r = edge.radius
        # The outer perimeter of the floor is also a circle. We're looking
        # for an inner one — i.e., one whose radius is significantly smaller
        # than the floor's bounding-box-derived radius.
        floor_bbox = floor.bounding_box()
        outer_r = max(floor_bbox.size.X, floor_bbox.size.Y) / 2.0
        if r < outer_r - 0.5:
            inner_circle_radii.append(r)

    if not inner_circle_radii:
        return CheckResult(
            name="Drainage hole",
            status="fail",
            message="No drainage hole found in the cup floor — water won't drain.",
            suggestion="Add a drainage hole through the floor of the cup.",
        )

    expected_r = expected_hole_dia / 2.0
    closest = min(inner_circle_radii, key=lambda r: abs(r - expected_r))
    diff = abs(closest - expected_r)
    measured_d = closest * 2
    if diff > tolerance:
        return CheckResult(
            name="Drainage hole",
            status="warn",
            message=(
                f"Drainage hole present but {measured_d:.1f}mm doesn't match expected "
                f"{expected_hole_dia:.1f}mm."
            ),
        )
    return CheckResult(
        name="Drainage hole",
        status="pass",
        message=f"Drainage hole present in the cup floor ({measured_d:.1f}mm).",
    )


# ---------------------------------------------------------------------------
# Internal corner radius (CNC) and draft (injection molding) — coarse
# parametric stand-ins. The geometric versions of these checks need a slicer
# / mold-flow analysis; for the MVP we surface them as method-aware notes.
# ---------------------------------------------------------------------------


def cnc_internal_corner_check(min_radius: float, declared_radius: float | None) -> CheckResult:
    """Surface a method-level note. We don't currently measure inside corners
    on the geometry; the templates in this build use sharp internal corners
    (gusset, bracket inside angle) that would need a tool path radius."""
    if declared_radius is None:
        return CheckResult(
            name="Internal corner radius",
            status="warn",
            message=(
                f"Internal corners are sharp — CNC needs at least {min_radius:.1f}mm "
                f"radius (limited by tool diameter)."
            ),
            suggestion="Add a fillet to internal corners, or accept a slightly larger radius from the shop.",
        )
    if declared_radius + 1e-6 < min_radius:
        return CheckResult(
            name="Internal corner radius",
            status="fail",
            message=f"Internal radius {declared_radius:.2f}mm < {min_radius:.1f}mm minimum.",
            suggestion=f"Increase corner fillets to at least {min_radius:.1f}mm.",
        )
    return CheckResult(
        name="Internal corner radius",
        status="pass",
        message=f"Internal radius {declared_radius:.2f}mm meets {min_radius:.1f}mm minimum.",
    )


def injection_draft_check(declared_draft_deg: float | None) -> CheckResult:
    if declared_draft_deg is None:
        return CheckResult(
            name="Draft angle",
            status="warn",
            message=(
                "No draft angle on vertical faces — parts won't release cleanly from the mold."
            ),
            suggestion="Add 1°–3° draft to all vertical walls.",
        )
    if declared_draft_deg + 1e-6 < 1.0:
        return CheckResult(
            name="Draft angle",
            status="fail",
            message=f"Draft {declared_draft_deg:.1f}° is below the 1° minimum.",
            suggestion="Increase draft to 1°–3°.",
        )
    return CheckResult(
        name="Draft angle",
        status="pass",
        message=f"Draft {declared_draft_deg:.1f}° is within the 1°–3° range.",
    )


# ---------------------------------------------------------------------------
# Orchestrator: per-template check runner
# ---------------------------------------------------------------------------


def _bracket_hole_locations(params) -> tuple[list[tuple[float, float]], list[tuple[float, float]], tuple[float, float], tuple[float, float]]:
    """Reproduce the bracket's hole layout in plate-local 2D coordinates.

    Mirrors templates/bracket.py's `_hole_positions` so the check sees the
    same hole centers the geometry uses. Returns (holes_a_2d, holes_b_2d,
    plate_a_size, plate_b_size).
    """
    from templates.bracket import _hole_positions  # local import to avoid cycles

    edge_margin = max(params.hole_dia * 1.5, params.thickness * 1.5)
    a_xs = _hole_positions(params.plate_a_length, params.holes_a, edge_margin)
    b_zs = _hole_positions(params.plate_b_length, params.holes_b, edge_margin)
    # Plate A: y is centered on 0; in 2D coords we recenter to (0, plate_a_width).
    holes_a_2d = [(x, params.plate_a_width / 2.0) for x in a_xs]
    holes_b_2d = [(params.plate_b_width / 2.0, z) for z in b_zs]
    # Plate A 2D: x_local = x_global, y_local = y_global + plate_a_width/2.
    # So plate A spans (0, plate_a_length) × (0, plate_a_width).
    # Plate B 2D: x_local = y_global + plate_b_width/2, y_local = z_global.
    # So plate B spans (0, plate_b_width) × (0, plate_b_length).
    return (
        holes_a_2d,
        holes_b_2d,
        (params.plate_a_length, params.plate_a_width),
        (params.plate_b_width, params.plate_b_length),
    )


def _hook_flat_hole_locations(params) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    """Reproduce the flat-mount hook's screw-hole layout."""
    inset = params.mount_dim / 4.0
    locs_centered = [
        (inset, inset),
        (-inset, inset),
        (-inset, -inset),
        (inset, -inset),
    ]
    # Recenter to (0,0) → (mount_dim, mount_dim)
    half = params.mount_dim / 2.0
    locs_2d = [(half + x, half + y) for x, y in locs_centered]
    return locs_2d, (params.mount_dim, params.mount_dim)


def run_checks(template_name: str, params, part: Part, context: ProductionContext) -> list[CheckResult]:
    """Pick and run the checks that apply to (template, production method).

    Templates declare which check kinds they care about via META's
    `applicable_checks` list; the orchestrator combines that with the
    method's thresholds (e.g. SLA skips the overhang check entirely).
    """
    from templates.registry import get_template  # local to avoid cycle at import

    spec = get_template(template_name)
    applicable: set[str] = set(spec.applicable_checks)
    thresholds = thresholds_for(context)
    results: list[CheckResult] = []

    # --- Wall thickness ---
    if "wall_thickness" in applicable and spec.wall_param:
        wall_value = getattr(params, spec.wall_param)
        results.append(wall_thickness_from_value(wall_value, thresholds))

    # --- Overhang check: template wants it AND method cares ---
    if "overhangs" in applicable and thresholds.overhang_max_deg is not None:
        results.append(check_overhangs(part, thresholds.overhang_max_deg))

    # --- Hole edge clearance: per-template ---
    if "hole_edge" in applicable:
        if template_name == "bracket":
            holes_a, holes_b, plate_a_size, plate_b_size = _bracket_hole_locations(params)
            if holes_a:
                r = hole_edge_distance(holes_a, plate_a_size, params.hole_dia, thresholds.hole_edge_factor)
                r.name = "Hole edge clearance — plate A"
                results.append(r)
            if holes_b:
                r = hole_edge_distance(holes_b, plate_b_size, params.hole_dia, thresholds.hole_edge_factor)
                r.name = "Hole edge clearance — plate B"
                results.append(r)
        elif template_name == "hook" and params.mount_type == "flat":
            locs, plate_size = _hook_flat_hole_locations(params)
            results.append(hole_edge_distance(locs, plate_size, params.screw_dia, thresholds.hole_edge_factor))

    # --- Drainage (bottle holder only) ---
    if "drainage" in applicable and template_name == "bottle_holder":
        results.append(drainage_check(part, params.drain_dia))

    # --- Method-level checks regardless of template ---
    if thresholds.min_internal_corner_radius is not None:
        results.append(cnc_internal_corner_check(thresholds.min_internal_corner_radius, None))
    if thresholds.draft_required:
        results.append(injection_draft_check(None))

    return results
