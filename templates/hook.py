"""
J-hook template.

Two mount variants:
- "flat": square plate with four corner screw holes, mounts to a flat surface
- "bar":  C-clamp around a horizontal bar (same idea as the bottle holder clamp)

Geometry, after the mount, is a horizontal arm extending along +X and a
J-curve at its end that wraps more than 180 degrees so things hung on it
don't slip off. The arm + curve are produced by sweeping a rectangular
cross-section along a centerline path so the bar has consistent thickness
through the bend.

Coordinate convention:
    +X away from the wall/bar (the direction the hook reaches).
    +Z up. The J curls in the +Z direction.
    Y is depth (left/right when the user faces the hook).
"""

from dataclasses import dataclass
from pathlib import Path
import math

from build123d import (
    Axis,
    BuildLine,
    BuildPart,
    BuildSketch,
    Circle,
    JernArc,
    Line,
    Location,
    Locations,
    Mode,
    Part,
    Plane,
    Rectangle,
    add,
    export_stl,
    extrude,
    sweep,
)


@dataclass
class HookParams:
    """Parameters for the J-hook template."""

    mount_type: str = "flat"      # "flat" plate-with-screws, or "bar" clamp
    mount_dim: float = 30.0       # flat: plate side length; bar: bar diameter
    arm_length: float = 50.0      # how far the hook reaches from the mount
    hook_radius: float = 12.0     # inside radius of the J-curve
    opening: float = 18.0         # chord across the J-curve mouth
    wall_t: float = 3.0           # bar/plate thickness throughout
    screw_dia: float = 4.5        # only used for flat mount

    def validate(self) -> list[str]:
        errors = []

        if self.mount_type not in ("flat", "bar"):
            errors.append(f"mount_type must be 'flat' or 'bar', got '{self.mount_type}'")

        if self.wall_t < 1.0:
            errors.append(f"wall_t ({self.wall_t}mm) must be >= 1.0mm for printability")

        if self.arm_length < 10.0:
            errors.append(f"arm_length ({self.arm_length}mm) must be >= 10mm")

        if self.hook_radius < 4.0:
            errors.append(f"hook_radius ({self.hook_radius}mm) must be >= 4mm")

        if self.opening <= 0:
            errors.append(f"opening ({self.opening}mm) must be > 0")

        # The opening is the chord across the curve's open mouth. For the J to
        # actually curl back past 180 degrees, the chord must be smaller than
        # the curve's diameter.
        if self.opening >= 2 * self.hook_radius:
            errors.append(
                f"opening ({self.opening}mm) must be < 2*hook_radius "
                f"({2 * self.hook_radius}mm) so the J curls back on itself"
            )

        if self.mount_type == "bar":
            if self.mount_dim < 15.0 or self.mount_dim > 60.0:
                errors.append(
                    f"mount_dim (bar diameter, {self.mount_dim}mm) "
                    f"must be between 15mm and 60mm"
                )
        else:
            if self.mount_dim < 15.0:
                errors.append(
                    f"mount_dim (plate side, {self.mount_dim}mm) must be >= 15mm"
                )
            if self.screw_dia < 1.0:
                errors.append(f"screw_dia ({self.screw_dia}mm) must be >= 1mm")
            if self.screw_dia > self.mount_dim / 3:
                errors.append(
                    f"screw_dia ({self.screw_dia}mm) too large for plate "
                    f"of side {self.mount_dim}mm"
                )

        return errors


def _make_flat_mount(params: HookParams) -> Part:
    """Square plate, lying in the YZ plane, with four corner screw holes.

    The +X face of the plate sits at x=0 so the arm joins it cleanly.
    """
    inset = params.mount_dim / 4

    with BuildPart() as plate:
        with BuildSketch(Plane.YZ):
            Rectangle(params.mount_dim, params.mount_dim)
            with Locations(
                (inset, inset),
                (-inset, inset),
                (-inset, -inset),
                (inset, -inset),
            ):
                Circle(params.screw_dia / 2, mode=Mode.SUBTRACT)
        # Plane.YZ extrudes along +X; shift back so +X face lands at x=0.
        extrude(amount=params.wall_t)

    return plate.part.moved(Location((-params.wall_t, 0, 0)))


def _make_bar_clamp(params: HookParams) -> tuple[Part, float]:
    """C-ring clamp; bar axis along Y, slot opens on -X (away from the arm).

    Returns (clamp_part, mount_face_x) where mount_face_x is the +X face of
    the clamp body (where the arm joins).
    """
    bar_dia = params.mount_dim
    clamp_od = bar_dia + 2 * params.wall_t
    clamp_height = max(20.0, bar_dia * 0.8)
    slot_width = max(2.0, params.wall_t)

    with BuildPart() as clamp:
        with BuildSketch(Plane.XZ):
            Circle(clamp_od / 2)
            Circle(bar_dia / 2, mode=Mode.SUBTRACT)
            # Slot on the -X side, away from where the arm emerges.
            with Locations((-clamp_od / 2, 0)):
                Rectangle(clamp_od, slot_width, mode=Mode.SUBTRACT)
        extrude(amount=clamp_height / 2, both=True)

    return clamp.part, clamp_od / 2


def make_hook(params: HookParams) -> Part:
    """Generate a J-hook solid from parameters.

    Raises:
        ValueError: if parameters fail validation
    """
    errors = params.validate()
    if errors:
        raise ValueError(
            "Parameter validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # Sweep arc geometry: chord = opening, want sweep_deg > 180° so the J
    # actually curls back on itself.
    half_angle = math.asin(params.opening / (2 * params.hook_radius))
    sweep_deg = 360.0 - 2.0 * math.degrees(half_angle)

    # Cross-section of the bent bar: arm_depth along Y, bar_thickness along Z.
    bar_thickness = params.wall_t
    arm_depth = max(params.wall_t * 4.0, 12.0)

    overlap = 1.0  # so the union with the mount is clean

    if params.mount_type == "flat":
        mount = _make_flat_mount(params)
        mount_face_x = 0.0
    else:
        mount, mount_face_x = _make_bar_clamp(params)

    arm_start_x = mount_face_x - overlap
    arm_end_x = mount_face_x + params.arm_length

    # Centerline path: straight along +X for the arm, then JernArc curling
    # up (+Z) and back. Tangent (+1, 0) with positive radius puts the arc
    # center directly above the tip at (arm_end_x, hook_radius), which is
    # what we want for the J shape.
    with BuildPart() as hook_body:
        with BuildLine(Plane.XZ) as path:
            Line((arm_start_x, 0), (arm_end_x, 0))
            JernArc(
                start=(arm_end_x, 0),
                tangent=(1, 0),
                radius=params.hook_radius,
                arc_size=sweep_deg,
            )
        # Section perpendicular to the path's start tangent (+X), centered
        # at (arm_start_x, 0, 0) so the swept body's centerline matches.
        with BuildSketch(Plane.YZ.offset(arm_start_x)):
            Rectangle(arm_depth, bar_thickness)
        sweep(path=path.line)

    with BuildPart() as final:
        add(mount)
        add(hook_body.part)

    return final.part


def main():
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    params = HookParams()
    print("Generating hook with parameters:")
    print(f"  mount_type:  {params.mount_type}")
    print(f"  mount_dim:   {params.mount_dim} mm")
    print(f"  arm_length:  {params.arm_length} mm")
    print(f"  hook_radius: {params.hook_radius} mm")
    print(f"  opening:     {params.opening} mm")
    print(f"  wall_t:      {params.wall_t} mm")
    print(f"  screw_dia:   {params.screw_dia} mm")

    part = make_hook(params)

    bbox = part.bounding_box()
    print("\nBounding box (mm):")
    print(f"  X: {bbox.min.X:7.2f} .. {bbox.max.X:7.2f}  ({bbox.size.X:6.2f})")
    print(f"  Y: {bbox.min.Y:7.2f} .. {bbox.max.Y:7.2f}  ({bbox.size.Y:6.2f})")
    print(f"  Z: {bbox.min.Z:7.2f} .. {bbox.max.Z:7.2f}  ({bbox.size.Z:6.2f})")

    output_path = output_dir / "hook.stl"
    export_stl(part, str(output_path))

    file_size = output_path.stat().st_size
    print("\nSTL generated:")
    print(f"  Path: {output_path}")
    print(f"  Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

    return output_path


META = {
    "name": "hook",
    "description": (
        "J-shaped hook. Mounts to a flat surface (with screw holes) or "
        "around a horizontal bar (with a C-clamp), and holds an object "
        "hung from its curve."
    ),
    "typical_use_cases": [
        "Coat or towel hook on a wall",
        "Hang headphones or a key ring",
        "Dog leash hook by the door",
        "Plant or wind chime hanger",
        "Broom or mop hook",
    ],
}


if __name__ == "__main__":
    main()
