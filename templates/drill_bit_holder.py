"""
Drill bit holder template for pegboard mounting.

Geometry consists of two regions:
1. Body - rectangular block with a row of cylindrical bit holes on the top face
2. Pegboard mount - two cylindrical pegs on the back face that insert into
   standard pegboard holes (1/4" / ~6.35mm on a 25.4mm grid)

Bit holes are sized on a linear gradient from min_hole_dia to max_hole_dia
so a complete set of bits can be sorted small-to-large along the rack.
All regions are boolean-unioned into a single solid.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from build123d import *


@dataclass
class DrillBitHolderParams:
    """Parameters for the drill bit holder template."""

    n_bits: int = 10                  # Number of bit slots
    min_hole_dia: float = 3.5         # Smallest bit hole diameter in mm (3mm bit + clearance)
    max_hole_dia: float = 13.0        # Largest bit hole diameter in mm (12mm bit + clearance)
    bit_pitch: float = 18.0           # Center-to-center spacing between adjacent bit holes in mm
    body_depth: float = 15.0          # Body dimension along the bit axis (how deep bits sit) in mm
    body_height: float = 25.0         # Body dimension front-to-back in mm
    wall_t: float = 3.0               # Minimum wall thickness between/around holes in mm
    peg_dia: float = 6.0              # Pegboard peg diameter in mm (1/4" pegs ~ 6.35mm)
    peg_spacing: float = 25.0         # Pegboard hole grid spacing (center-to-center) in mm
    n_pegs: int = 2                   # Number of pegs that engage the pegboard
    peg_length: float = 10.0          # How far each peg protrudes from the back face in mm

    def validate(self) -> list[str]:
        """Validate parameters and return list of error messages."""
        errors = []

        # Slot count sanity.
        if self.n_bits < 1:
            errors.append(f"n_bits ({self.n_bits}) must be >= 1")

        # Bit hole diameters.
        if self.min_hole_dia <= 0:
            errors.append(f"min_hole_dia ({self.min_hole_dia}mm) must be > 0")
        if self.max_hole_dia < self.min_hole_dia:
            errors.append(
                f"max_hole_dia ({self.max_hole_dia}mm) must be >= "
                f"min_hole_dia ({self.min_hole_dia}mm)"
            )

        # Printability floor.
        if self.wall_t < 1.0:
            errors.append(f"wall_t ({self.wall_t}mm) must be >= 1.0mm for printability")

        # Pitch must leave at least wall_t of material between adjacent biggest holes.
        # We use max_hole_dia for the worst-case neighbor pair.
        min_pitch = self.max_hole_dia + self.wall_t
        if self.bit_pitch < min_pitch:
            errors.append(
                f"bit_pitch ({self.bit_pitch}mm) must be >= max_hole_dia "
                f"({self.max_hole_dia}mm) + wall_t ({self.wall_t}mm) = {min_pitch}mm"
            )

        # Body must be deep enough to grip the bit and leave a floor under it.
        if self.body_depth < 8.0:
            errors.append(f"body_depth ({self.body_depth}mm) must be >= 8mm for meaningful bit grip")

        # Body must be tall enough (front-to-back) to leave wall around the biggest hole
        # and still have material behind for the pegs.
        min_height = self.max_hole_dia + 2 * self.wall_t
        if self.body_height < min_height:
            errors.append(
                f"body_height ({self.body_height}mm) must be >= max_hole_dia "
                f"+ 2*wall_t = {min_height}mm"
            )

        # Pegboard mount sanity.
        if self.peg_dia <= 0:
            errors.append(f"peg_dia ({self.peg_dia}mm) must be > 0")
        if self.peg_dia > 12.0:
            errors.append(f"peg_dia ({self.peg_dia}mm) is unusually large for a pegboard peg")
        if self.n_pegs < 1:
            errors.append(f"n_pegs ({self.n_pegs}) must be >= 1")
        if self.peg_spacing <= self.peg_dia:
            errors.append(
                f"peg_spacing ({self.peg_spacing}mm) must be > peg_dia ({self.peg_dia}mm)"
            )
        if self.peg_length < 4.0:
            errors.append(f"peg_length ({self.peg_length}mm) must be >= 4mm to engage a pegboard")

        # The pegs must fit within the body's overall length.
        body_length = self.n_bits * self.bit_pitch
        peg_span = (self.n_pegs - 1) * self.peg_spacing
        if peg_span + self.peg_dia + 2 * self.wall_t > body_length:
            errors.append(
                f"Peg span ({peg_span + self.peg_dia:.1f}mm including peg diameter) "
                f"plus 2*wall_t ({2 * self.wall_t}mm) exceeds body length "
                f"({body_length}mm = n_bits * bit_pitch). Reduce n_pegs, peg_spacing, "
                f"or increase n_bits/bit_pitch."
            )

        return errors


def make_drill_bit_holder(params: DrillBitHolderParams) -> Part:
    """
    Generate a pegboard-mounted drill bit holder from parameters.

    Coordinate convention:
        +X is along the length of the rack (left-to-right across bit holes).
        +Y is front-to-back; the back face (y = body_height/2) is the
        pegboard-facing side and is where the pegs protrude.
        +Z is up; bits drop down into holes from the top face (z = body_depth).

    Args:
        params: DrillBitHolderParams instance with geometry parameters

    Returns:
        build123d Part representing the complete drill bit holder

    Raises:
        ValueError: if parameters fail validation
    """
    errors = params.validate()
    if errors:
        raise ValueError("Parameter validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    body_length = params.n_bits * params.bit_pitch

    # Small overlap so boolean union is clean across face-coincident features.
    overlap = 0.5

    # Region 1: Body — solid rectangular block.
    # Base is centered on X and Y at the origin; extruded up along +Z.
    with BuildPart() as body_part:
        with BuildSketch(Plane.XY):
            Rectangle(body_length, params.body_height)
        extrude(amount=params.body_depth)

        # Bit holes: drilled down from the top face (+Z), graduated min -> max
        # left-to-right along +X. Leave a floor of wall_t at the bottom.
        # Hole centers are spaced bit_pitch apart and centered on the body length.
        hole_depth = params.body_depth - params.wall_t
        if params.n_bits == 1:
            diameters = [params.min_hole_dia]
        else:
            step = (params.max_hole_dia - params.min_hole_dia) / (params.n_bits - 1)
            diameters = [params.min_hole_dia + i * step for i in range(params.n_bits)]

        for i, dia in enumerate(diameters):
            x_center = -body_length / 2 + params.bit_pitch * (i + 0.5)
            with BuildSketch(Plane.XY.offset(params.body_depth)):
                with Locations((x_center, 0)):
                    Circle(dia / 2)
            # Subtract downward from the top face.
            extrude(amount=-hole_depth, mode=Mode.SUBTRACT)

    body = body_part.part

    # Region 2: Pegs — cylinders extending out the back face along +Y.
    # Peg axes are horizontal (along Y). Center them vertically on the body
    # at z = body_depth / 2 and lay them out symmetrically along X.
    peg_span = (params.n_pegs - 1) * params.peg_spacing
    peg_z = params.body_depth / 2
    back_y = params.body_height / 2

    with BuildPart() as pegs_part:
        for i in range(params.n_pegs):
            x_center = -peg_span / 2 + i * params.peg_spacing
            # Build a cylinder whose axis is along Y, starting just inside the
            # back face (for clean union) and extending peg_length outward.
            with BuildSketch(Plane.XZ.offset(back_y - overlap)):
                with Locations((x_center, peg_z)):
                    Circle(params.peg_dia / 2)
            extrude(amount=params.peg_length + overlap)

    pegs = pegs_part.part

    # Boolean union body + pegs.
    with BuildPart() as final:
        add(body)
        add(pegs)

    return final.part


def main():
    """Generate a drill bit holder STL with default parameters."""
    # Create output directory
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Generate with defaults
    params = DrillBitHolderParams()
    print(f"Generating drill bit holder with parameters:")
    print(f"  n_bits: {params.n_bits}")
    print(f"  min_hole_dia: {params.min_hole_dia}mm")
    print(f"  max_hole_dia: {params.max_hole_dia}mm")
    print(f"  bit_pitch: {params.bit_pitch}mm")
    print(f"  body_depth: {params.body_depth}mm")
    print(f"  peg_dia: {params.peg_dia}mm")
    print(f"  peg_spacing: {params.peg_spacing}mm")

    part = make_drill_bit_holder(params)

    # Export to STL
    output_path = output_dir / "drill_bit_holder.stl"
    export_stl(part, str(output_path))

    # Report file size
    file_size = output_path.stat().st_size
    print(f"\nSTL generated successfully:")
    print(f"  Path: {output_path}")
    print(f"  Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

    return output_path


if __name__ == "__main__":
    main()


# Registry metadata. Read by templates.registry to wire this template into the app.
META = {
    "name": "drill_bit_holder",
    "description": (
        "Pegboard-mounted rack with a row of graduated holes to hold drill bits. "
        "Bit hole diameters scale linearly from min to max so a complete set "
        "can be sorted small-to-large along the rack."
    ),
    "typical_use_cases": [
        "Drill bit organizer on a workshop pegboard",
        "Router bit holder on a pegboard",
        "Driver bit / hex key rack on a pegboard",
        "Small tool shank organizer above a workbench",
    ],
    # Which manufacturing checks apply to this template.
    "applicable_checks": ["wall_thickness", "overhangs"],
    # Which parameter holds the wall thickness.
    "wall_param": "wall_t",
}
