"""
L-bracket template.

Two flat plates meeting at 90 degrees, with optional triangular gusset on
the inside of the angle and configurable arrays of mounting holes on each
plate.

Coordinate convention:
    Plate A lies in the XY plane. Its bottom face is at z=0; its top face
    is at z=thickness. It runs from x=0 (the corner of the L) along +X
    and is centered on y=0.

    Plate B lies in the YZ plane. Its outer face is at x=0 (flush with the
    corner of the L); its inner face is at x=thickness. It rises from z=0
    along +Z and is centered on y=0.

    The two plates share material in the small region x in [0, thickness]
    and z in [0, thickness] — that's the inside of the corner.

    Plate A's holes drill through it along Z. Plate B's holes drill
    through it along X.
"""

from dataclasses import dataclass
from pathlib import Path

from build123d import (
    BuildPart,
    BuildSketch,
    Circle,
    Locations,
    Mode,
    Part,
    Plane,
    Polygon,
    Rectangle,
    add,
    export_stl,
    extrude,
)


@dataclass
class BracketParams:
    """Parameters for the L-bracket template."""

    plate_a_length: float = 60.0   # along +X from the corner
    plate_a_width: float = 30.0    # along Y
    plate_b_length: float = 60.0   # along +Z from the corner
    plate_b_width: float = 30.0    # along Y
    thickness: float = 4.0         # plate thickness, both plates
    holes_a: int = 2               # mounting holes through plate A
    holes_b: int = 2               # mounting holes through plate B
    hole_dia: float = 4.5          # hole diameter, both plates
    gusset: bool = True            # triangular stiffener on the inside corner

    def validate(self) -> list[str]:
        errors = []

        if self.thickness < 1.0:
            errors.append(f"thickness ({self.thickness}mm) must be >= 1.0mm for printability")

        for label, value in (
            ("plate_a_length", self.plate_a_length),
            ("plate_a_width", self.plate_a_width),
            ("plate_b_length", self.plate_b_length),
            ("plate_b_width", self.plate_b_width),
        ):
            if value < 10.0:
                errors.append(f"{label} ({value}mm) must be >= 10mm")

        if self.hole_dia <= 0:
            errors.append(f"hole_dia ({self.hole_dia}mm) must be > 0")

        for label, value in (("holes_a", self.holes_a), ("holes_b", self.holes_b)):
            if value < 0:
                errors.append(f"{label} ({value}) must be >= 0")
            if value > 10:
                errors.append(f"{label} ({value}) too many holes; cap is 10")

        # Hole must fit within the plate width with reasonable margin.
        margin = self.hole_dia * 1.5
        if self.holes_a > 0 and self.plate_a_width < 2 * margin:
            errors.append(
                f"plate_a_width ({self.plate_a_width}mm) too narrow for hole_dia "
                f"({self.hole_dia}mm); needs >= {2 * margin}mm"
            )
        if self.holes_b > 0 and self.plate_b_width < 2 * margin:
            errors.append(
                f"plate_b_width ({self.plate_b_width}mm) too narrow for hole_dia "
                f"({self.hole_dia}mm); needs >= {2 * margin}mm"
            )

        # Need enough length for the holes to fit clear of the corner and
        # clear of each other.
        for label, length, count in (
            ("plate_a_length", self.plate_a_length, self.holes_a),
            ("plate_b_length", self.plate_b_length, self.holes_b),
        ):
            if count > 0:
                edge = max(margin, self.thickness * 1.5)
                usable = length - 2 * edge
                if count > 1 and usable < self.hole_dia * 2 * (count - 1):
                    errors.append(
                        f"{label} ({length}mm) too short to fit {count} holes "
                        f"of dia {self.hole_dia}mm"
                    )
                elif count == 1 and length < 2 * edge:
                    errors.append(
                        f"{label} ({length}mm) too short for a hole with edge "
                        f"margin {edge}mm"
                    )

        return errors


def _hole_positions(length: float, count: int, edge_margin: float) -> list[float]:
    """Evenly-spaced hole centers along a 1D length, inset by edge_margin."""
    if count <= 0:
        return []
    if count == 1:
        return [length / 2]
    step = (length - 2 * edge_margin) / (count - 1)
    return [edge_margin + i * step for i in range(count)]


def make_bracket(params: BracketParams) -> Part:
    """Generate an L-bracket solid from parameters.

    Raises:
        ValueError: if parameters fail validation
    """
    errors = params.validate()
    if errors:
        raise ValueError(
            "Parameter validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    edge_margin = max(params.hole_dia * 1.5, params.thickness * 1.5)
    holes_a_positions = _hole_positions(params.plate_a_length, params.holes_a, edge_margin)
    holes_b_positions = _hole_positions(params.plate_b_length, params.holes_b, edge_margin)

    # Plate A: rectangle in XY, extruded +Z by thickness. Sketched centered
    # on (plate_a_length/2, 0) so it spans x in [0, plate_a_length] and y in
    # [-plate_a_width/2, +plate_a_width/2].
    with BuildPart() as plate_a:
        with BuildSketch(Plane.XY):
            with Locations((params.plate_a_length / 2, 0)):
                Rectangle(params.plate_a_length, params.plate_a_width)
            if holes_a_positions:
                with Locations(*[(x, 0) for x in holes_a_positions]):
                    Circle(params.hole_dia / 2, mode=Mode.SUBTRACT)
        extrude(amount=params.thickness)

    # Plate B: rectangle in YZ, extruded +X by thickness. Sketch in YZ has
    # X-axis = global Y and Y-axis = global Z. Center on (0, plate_b_length/2)
    # so it spans y in [-plate_b_width/2, +plate_b_width/2] and z in
    # [0, plate_b_length].
    with BuildPart() as plate_b:
        with BuildSketch(Plane.YZ):
            with Locations((0, params.plate_b_length / 2)):
                Rectangle(params.plate_b_width, params.plate_b_length)
            if holes_b_positions:
                with Locations(*[(0, z) for z in holes_b_positions]):
                    Circle(params.hole_dia / 2, mode=Mode.SUBTRACT)
        extrude(amount=params.thickness)

    parts = [plate_a.part, plate_b.part]

    if params.gusset:
        # Triangle on the inside of the L, in the XZ plane at y=0, extruded
        # along Y to form a thin gusset. Reach is the smaller of 60% of each
        # plate length, capped so it never overruns either plate.
        reach_a = min(params.plate_a_length, params.plate_b_length) * 0.6
        reach_z = reach_a  # 45° gusset
        gusset_y = max(params.thickness, min(params.plate_a_width, params.plate_b_width) / 4)

        with BuildPart() as gusset:
            with BuildSketch(Plane.XZ):
                # Triangle vertices: inside corner, point along plate A's top
                # surface, point along plate B's inner surface. Slight overlap
                # into both plates so the boolean union is clean.
                Polygon(
                    (params.thickness, params.thickness),
                    (params.thickness + reach_a, params.thickness),
                    (params.thickness, params.thickness + reach_z),
                    align=None,
                )
            extrude(amount=gusset_y / 2, both=True)
        parts.append(gusset.part)

    with BuildPart() as final:
        for p in parts:
            add(p)

    return final.part


def main():
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    params = BracketParams()
    print("Generating L-bracket with parameters:")
    print(f"  plate A:    {params.plate_a_length} x {params.plate_a_width} mm, {params.holes_a} holes")
    print(f"  plate B:    {params.plate_b_length} x {params.plate_b_width} mm, {params.holes_b} holes")
    print(f"  thickness:  {params.thickness} mm")
    print(f"  hole_dia:   {params.hole_dia} mm")
    print(f"  gusset:     {params.gusset}")

    part = make_bracket(params)

    bbox = part.bounding_box()
    print("\nBounding box (mm):")
    print(f"  X: {bbox.min.X:7.2f} .. {bbox.max.X:7.2f}  ({bbox.size.X:6.2f})")
    print(f"  Y: {bbox.min.Y:7.2f} .. {bbox.max.Y:7.2f}  ({bbox.size.Y:6.2f})")
    print(f"  Z: {bbox.min.Z:7.2f} .. {bbox.max.Z:7.2f}  ({bbox.size.Z:6.2f})")

    output_path = output_dir / "bracket.stl"
    export_stl(part, str(output_path))

    file_size = output_path.stat().st_size
    print("\nSTL generated:")
    print(f"  Path: {output_path}")
    print(f"  Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

    return output_path


META = {
    "name": "bracket",
    "description": (
        "Right-angle L-bracket joining two perpendicular surfaces. "
        "Each plate has a configurable array of mounting holes; an "
        "optional triangular gusset stiffens the inside corner."
    ),
    "typical_use_cases": [
        "Shelf bracket on a wall",
        "Monitor or light arm mount",
        "Under-desk cable management bracket",
        "Reinforcing a wood-joinery corner",
        "Mounting a small project enclosure",
    ],
}


if __name__ == "__main__":
    main()
