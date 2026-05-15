"""
Paper towel holder template for under-cabinet mounting.

Geometry consists of three regions:
1. Mounting plate - flat rectangular plate with two screw holes; mounts up
   against the underside of a cabinet.
2. Two side arms - vertical walls hanging down from the plate, one at each
   end, holding the dowel ends.
3. Dowel - horizontal cylindrical rod between the two arms that the paper
   towel roll's cardboard core slides onto.

All regions are boolean-unioned into a single solid that prints as one part.
The dowel is co-axial with the roll; the assembly hangs below the cabinet
with the roll axis horizontal.
"""

from dataclasses import dataclass
from pathlib import Path
from build123d import *


@dataclass
class PaperTowelHolderParams:
    """Parameters for the paper towel holder template."""

    # Roll geometry
    roll_width: float = 280.0          # Length of paper towel roll along its axis (mm)
    roll_clearance: float = 15.0       # Extra gap added to roll_width for easy load/unload
    dowel_dia: float = 36.0            # Dowel OD; sized to slide inside ~38mm card core
    roll_outer_dia: float = 130.0      # Approx OD of full roll, used to size arm drop

    # Mounting plate
    plate_length: float = 320.0        # Plate length along roll axis (mm)
    plate_width: float = 60.0          # Plate width perpendicular to roll axis (mm)
    plate_thickness: float = 5.0       # Plate thickness (mm); also wall around screw holes
    screw_dia: float = 4.5             # Clearance hole diameter for #8 wood screw (mm)
    screw_spacing: float = 220.0       # Center-to-center distance between the two screws (mm)

    # Side arms
    arm_thickness: float = 5.0         # Thickness of each side arm wall (mm)
    arm_width: float = 40.0            # Arm depth (front-to-back, perpendicular to roll axis)
    # arm_drop is the distance from the underside of the plate down to the
    # dowel center. Must clear the full roll OD plus a small margin so the
    # roll spins freely.
    arm_drop: float = 90.0

    # General
    wall_t: float = 2.4                # Reference wall thickness (used by mfg checks)

    def validate(self) -> list[str]:
        """Validate parameters and return list of error messages."""
        errors = []

        # Printability floor (Bambu A1 0.4mm nozzle → ≥ 0.8mm; we use 1.0).
        if self.wall_t < 1.0:
            errors.append(
                f"wall_t ({self.wall_t}mm) must be >= 1.0mm for printability"
            )

        # Plate must be thick enough to hold a screw without splitting.
        if self.plate_thickness < self.screw_dia:
            errors.append(
                f"plate_thickness ({self.plate_thickness}mm) must be "
                f">= screw_dia ({self.screw_dia}mm) for screw anchoring"
            )

        # Plate must span the screw spacing with edge margin >= 1× screw_dia.
        min_plate_length = self.screw_spacing + 2 * self.screw_dia + 4.0
        if self.plate_length < min_plate_length:
            errors.append(
                f"plate_length ({self.plate_length}mm) must be "
                f">= screw_spacing + 2×screw_dia + 4mm = {min_plate_length}mm"
            )

        # Plate must be wider than screw_dia + 2× margin so the screw is not on the edge.
        if self.plate_width < self.screw_dia + 6.0:
            errors.append(
                f"plate_width ({self.plate_width}mm) must be "
                f">= screw_dia + 6mm = {self.screw_dia + 6.0}mm"
            )

        # Arms must sit outside the roll plus clearance.
        min_inner_span = self.roll_width + self.roll_clearance
        if self.plate_length < min_inner_span + 2 * self.arm_thickness:
            errors.append(
                f"plate_length ({self.plate_length}mm) must be "
                f">= roll_width + roll_clearance + 2×arm_thickness "
                f"= {min_inner_span + 2 * self.arm_thickness}mm"
            )

        # Arm drop must clear the roll radius with margin.
        min_arm_drop = self.roll_outer_dia / 2 + self.dowel_dia / 2 + 5.0
        if self.arm_drop < min_arm_drop:
            errors.append(
                f"arm_drop ({self.arm_drop}mm) must be >= roll_outer_dia/2 "
                f"+ dowel_dia/2 + 5mm = {min_arm_drop}mm"
            )

        # Dowel must fit inside a residential paper towel core (~38mm).
        if not (20.0 <= self.dowel_dia <= 45.0):
            errors.append(
                f"dowel_dia ({self.dowel_dia}mm) must be between 20mm and 45mm"
            )

        # Roll width sanity range (residential 240-320mm; bulk goes wider).
        if not (150.0 <= self.roll_width <= 320.0):
            errors.append(
                f"roll_width ({self.roll_width}mm) must be between 150mm and 320mm"
            )

        # Bambu A1 build volume is 256mm. Warn if plate exceeds it.
        if self.plate_length > 256.0:
            errors.append(
                f"plate_length ({self.plate_length}mm) exceeds Bambu A1 "
                f"build volume of 256mm; print diagonally or split the part"
            )

        return errors


def make_paper_towel_holder(params: PaperTowelHolderParams) -> Part:
    """
    Generate a paper towel holder solid from parameters.

    Coordinate convention:
        +Z is up (the cabinet underside is at z=0; the holder hangs at -Z).
        +X is along the roll axis. +Y is the front-to-back depth of the plate.
        The plate sits flush against the cabinet at z=0, extending downward
        into negative Z. Side arms drop from the plate ends; the dowel runs
        between them along X.

    Args:
        params: PaperTowelHolderParams instance with geometry parameters

    Returns:
        build123d Part representing the complete paper towel holder

    Raises:
        ValueError: if parameters fail validation
    """
    errors = params.validate()
    if errors:
        raise ValueError(
            "Parameter validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # Small overlap so boolean union seams cleanly.
    overlap = 0.5

    # Region 1: Mounting plate. Centered at origin in X and Y, top face at z=0,
    # extending down by plate_thickness.
    with BuildPart() as plate_part:
        with BuildSketch(Plane.XY.offset(-params.plate_thickness)):
            Rectangle(params.plate_length, params.plate_width)
        extrude(amount=params.plate_thickness)

        # Two screw clearance holes spaced symmetrically along X, centered in Y.
        with BuildSketch(Plane.XY.offset(-params.plate_thickness)):
            with Locations(
                (-params.screw_spacing / 2, 0),
                (params.screw_spacing / 2, 0),
            ):
                Circle(params.screw_dia / 2)
        extrude(amount=params.plate_thickness, mode=Mode.SUBTRACT)

    plate = plate_part.part

    # Region 2: Two side arms. Each arm is a rectangular block hanging from
    # the underside of the plate at the X extremes. Arm centers in X are at
    # +/- (plate_length/2 - arm_thickness/2) so the outer face of the arm is
    # flush with the plate edge.
    arm_x = params.plate_length / 2 - params.arm_thickness / 2
    arm_top_z = -params.plate_thickness + overlap
    arm_bot_z = -params.plate_thickness - params.arm_drop
    arm_height = arm_top_z - arm_bot_z
    arm_center_z = (arm_top_z + arm_bot_z) / 2

    with BuildPart() as arms_part:
        with BuildSketch(Plane.XZ):
            with Locations((arm_x, arm_center_z), (-arm_x, arm_center_z)):
                Rectangle(params.arm_thickness, arm_height)
        extrude(amount=params.arm_width / 2, both=True)

    arms = arms_part.part

    # Region 3: Dowel. Horizontal cylinder along X, between the inner faces
    # of the two arms. Centered vertically at the bottom of the arm drop.
    dowel_z = -params.plate_thickness - params.arm_drop + params.dowel_dia / 2
    # Length spans from inner face of -X arm to inner face of +X arm, plus
    # overlap on each side so the union is clean.
    inner_span = (params.plate_length - 2 * params.arm_thickness) + 2 * overlap

    with BuildPart() as dowel_part:
        with BuildSketch(Plane.YZ):
            Circle(params.dowel_dia / 2)
        extrude(amount=inner_span / 2, both=True)

    dowel = dowel_part.part.moved(Location((0, 0, dowel_z)))

    # Boolean union all three regions.
    with BuildPart() as final:
        add(plate)
        add(arms)
        add(dowel)

    return final.part


def main():
    """Generate a paper towel holder STL with default parameters."""
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    params = PaperTowelHolderParams()
    print("Generating paper towel holder with parameters:")
    print(f"  roll_width: {params.roll_width}mm")
    print(f"  dowel_dia: {params.dowel_dia}mm")
    print(f"  plate_length: {params.plate_length}mm")
    print(f"  plate_thickness: {params.plate_thickness}mm")
    print(f"  arm_drop: {params.arm_drop}mm")

    part = make_paper_towel_holder(params)

    output_path = output_dir / "paper_towel_holder.stl"
    export_stl(part, str(output_path))

    file_size = output_path.stat().st_size
    print("\nSTL generated successfully:")
    print(f"  Path: {output_path}")
    print(f"  Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

    return output_path


if __name__ == "__main__":
    main()


# Registry metadata. Read by templates.registry to wire this template into the app.
META = {
    "name": "paper_towel_holder",
    "description": (
        "Under-cabinet paper towel holder. A flat plate mounts to the "
        "cabinet underside via two screws; two side arms drop down and "
        "support a horizontal dowel that the paper towel roll spins on."
    ),
    "typical_use_cases": [
        "Under-kitchen-cabinet paper towel dispenser",
        "Under-shelf paper towel holder in a workshop",
        "Garage utility roll holder mounted to a workbench underside",
        "Pantry or laundry room roll holder",
    ],
    # Which manufacturing checks apply to this template.
    "applicable_checks": ["wall_thickness", "overhangs"],
    # Which parameter holds the wall thickness (used by manufacturing/agent layer).
    "wall_param": "wall_t",
}
