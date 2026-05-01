"""
Bottle holder template for clamp-on mounting to roll bars.

Geometry consists of three regions:
1. Cup - hollow cylindrical sleeve with closed bottom and drainage hole
2. Standoff arm - rectangular bridge connecting cup to clamp
3. Clamp - C-shaped ring with tightening slot

All regions are boolean-unioned into a single solid.
"""

from dataclasses import dataclass
from pathlib import Path
from build123d import *


@dataclass
class BottleHolderParams:
    """Parameters for the bottle holder template."""

    bottle_dia: float = 63.0      # Bottle diameter in mm
    cup_id: float = 66.0          # Cup inner diameter (bottle_dia + clearance)
    cup_height: float = 100.0     # Cup height in mm
    wall_t: float = 2.5           # Wall thickness in mm
    drain_dia: float = 8.0        # Drainage hole diameter in mm
    bar_dia: float = 28.0         # Roll bar diameter in mm
    clamp_height: float = 25.0    # Clamp vertical extent in mm
    slot_width: float = 4.0       # Tightening slot width in mm
    standoff: float = 40.0        # Cup-to-clamp horizontal gap in mm
    arm_width: float = 20.0       # Standoff arm cross-section width in mm

    def validate(self) -> list[str]:
        """Validate parameters and return list of error messages."""
        errors = []

        # Required clearance
        if self.cup_id <= self.bottle_dia + 1.0:
            errors.append(
                f"cup_id ({self.cup_id}mm) must be > bottle_dia ({self.bottle_dia}mm) + 1mm clearance"
            )

        # Printability floor
        if self.wall_t < 1.0:
            errors.append(f"wall_t ({self.wall_t}mm) must be >= 1.0mm for printability")

        # Sanity range for bar diameter
        if self.bar_dia < 15.0 or self.bar_dia > 60.0:
            errors.append(f"bar_dia ({self.bar_dia}mm) must be between 15mm and 60mm")

        # Minimum cup height
        if self.cup_height < 40.0:
            errors.append(f"cup_height ({self.cup_height}mm) must be >= 40mm for meaningful grip")

        return errors


def make_bottle_holder(params: BottleHolderParams) -> Part:
    """
    Generate a bottle holder solid from parameters.

    Coordinate convention:
        +Z is up. Cup axis runs along Z, with the open end at +Z and the
        drainage hole at z=0. The clamp axis runs along Y (horizontal),
        perpendicular to the cup. The standoff arm runs along +X from
        the cup wall to the clamp.

    Args:
        params: BottleHolderParams instance with geometry parameters

    Returns:
        build123d Part representing the complete bottle holder

    Raises:
        ValueError: if parameters fail validation
    """
    errors = params.validate()
    if errors:
        raise ValueError("Parameter validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    cup_od = params.cup_id + 2 * params.wall_t
    clamp_od = params.bar_dia + 2 * params.wall_t

    # Layout: cup centered on Z axis at origin; clamp centered along Y axis
    # at (clamp_x, 0, arm_z); arm bridges cup outer wall to clamp outer wall.
    arm_z = params.cup_height / 2
    clamp_x = cup_od / 2 + params.standoff + clamp_od / 2

    # Small overlap so boolean union is clean even with curved interfaces.
    overlap = 1.0

    # Region 1: Cup — hollow cylinder, closed bottom, drainage hole at z=0.
    with BuildPart() as cup_part:
        with BuildSketch(Plane.XY):
            Circle(cup_od / 2)
        extrude(amount=params.cup_height)

        # Hollow the interior, leaving a floor of thickness wall_t.
        with BuildSketch(Plane.XY.offset(params.wall_t)):
            Circle(params.cup_id / 2)
        extrude(amount=params.cup_height - params.wall_t, mode=Mode.SUBTRACT)

        # Drainage hole through the floor.
        with BuildSketch(Plane.XY):
            Circle(params.drain_dia / 2)
        extrude(amount=params.wall_t, mode=Mode.SUBTRACT)

    cup = cup_part.part

    # Region 2: Clamp — C-ring with axis along Y.
    # Sketch the C profile in the XZ plane, then extrude symmetrically along Y.
    # Slot is on the +X face of the clamp (the side facing AWAY from the cup).
    with BuildPart() as clamp_part:
        with BuildSketch(Plane.XZ):
            Circle(clamp_od / 2)
            Circle(params.bar_dia / 2, mode=Mode.SUBTRACT)
            # Vertical slot through the +X wall: width slot_width along Z (vertical),
            # length clamp_od along X (cuts cleanly through the wall).
            with Locations((clamp_od / 2, 0)):
                Rectangle(clamp_od, params.slot_width, mode=Mode.SUBTRACT)
        extrude(amount=params.clamp_height / 2, both=True)

    # Place clamp so its axis is along Y at (clamp_x, *, arm_z).
    clamp = clamp_part.part.moved(Location((clamp_x, 0, arm_z)))

    # Region 3: Standoff arm — vertical rib bridging cup outer wall to clamp.
    # Length along X, width (height) along Z, thickness along Y.
    arm_x_start = cup_od / 2 - overlap
    arm_x_end = clamp_x - clamp_od / 2 + overlap
    arm_length = arm_x_end - arm_x_start
    arm_x_center = (arm_x_start + arm_x_end) / 2

    with BuildPart() as arm_part:
        with BuildSketch(Plane.XZ):
            with Locations((arm_x_center, arm_z)):
                Rectangle(arm_length, params.arm_width)
        extrude(amount=params.wall_t / 2, both=True)

    arm = arm_part.part

    # Boolean union all three regions.
    with BuildPart() as final:
        add(cup)
        add(arm)
        add(clamp)

    return final.part


def main():
    """Generate a bottle holder STL with default parameters."""
    # Create output directory
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Generate with defaults
    params = BottleHolderParams()
    print(f"Generating bottle holder with parameters:")
    print(f"  bottle_dia: {params.bottle_dia}mm")
    print(f"  cup_id: {params.cup_id}mm")
    print(f"  cup_height: {params.cup_height}mm")
    print(f"  bar_dia: {params.bar_dia}mm")
    print(f"  wall_t: {params.wall_t}mm")

    part = make_bottle_holder(params)

    # Export to STL
    output_path = output_dir / "bottle_holder.stl"
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
    "name": "bottle_holder",
    "description": (
        "Cylindrical cup that mounts to a horizontal bar via a C-clamp. "
        "Holds a bottle vertically with a drainage hole in the floor."
    ),
    "typical_use_cases": [
        "Drink holder for a child's ride-on toy",
        "Bottle holder for a treadmill handlebar",
        "Stroller cup holder",
        "Gym equipment bottle accessory",
    ],
    # Which manufacturing checks apply to this template. The orchestrator
    # in manufacturing/checks.py uses this list to filter what runs.
    "applicable_checks": ["wall_thickness", "overhangs", "drainage"],
    # Which parameter holds the wall thickness (used by the agent's pushback
    # logic and by wall_thickness_from_value).
    "wall_param": "wall_t",
}
