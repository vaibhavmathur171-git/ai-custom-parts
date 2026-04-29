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

    Args:
        params: BottleHolderParams instance with geometry parameters

    Returns:
        build123d Part representing the complete bottle holder

    Raises:
        ValueError: if parameters fail validation
    """
    # Validate parameters
    errors = params.validate()
    if errors:
        raise ValueError("Parameter validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    # Calculate derived dimensions
    cup_od = params.cup_id + 2 * params.wall_t
    clamp_od = params.bar_dia + 2 * params.wall_t

    # Region 1: Cup - hollow cylinder with closed bottom and drainage hole
    with BuildPart() as cup_part:
        # Outer cylinder
        with BuildSketch():
            Circle(cup_od / 2)
        extrude(amount=params.cup_height)

        # Hollow out the interior
        with BuildSketch(Plane.XY.offset(params.wall_t)):
            Circle(params.cup_id / 2)
        extrude(amount=params.cup_height - params.wall_t, mode=Mode.SUBTRACT)

        # Add drainage hole through the bottom
        with BuildSketch(Plane.XY):
            Circle(params.drain_dia / 2)
        extrude(amount=params.wall_t, mode=Mode.SUBTRACT)

    cup = cup_part.part

    # Region 2: Standoff arm - rectangular bridge
    # Position: connects at midpoint of cup's outer wall, extends horizontally
    # The arm extends from the edge of the cup to the edge of the clamp
    arm_length = params.standoff + cup_od / 2 + clamp_od / 2
    arm_start_x = cup_od / 2
    arm_center_z = params.cup_height / 2

    with BuildPart() as arm_part:
        with BuildSketch(Plane.XZ.offset(0)):
            with Locations((arm_start_x + params.standoff / 2, arm_center_z)):
                Rectangle(params.standoff, params.arm_width)
        extrude(amount=params.wall_t, both=True)

    arm = arm_part.part

    # Region 3: Clamp - C-shaped ring with slot
    # Center the clamp at the end of the standoff arm
    clamp_center_x = arm_start_x + params.standoff
    clamp_center_z = arm_center_z

    with BuildPart() as clamp_part:
        # Create the clamp at origin first
        with BuildSketch(Plane.XY):
            Circle(clamp_od / 2)
            Circle(params.bar_dia / 2, mode=Mode.SUBTRACT)

            # Cut the slot - positioned opposite to the standoff (at -X direction)
            with Locations((-clamp_od / 2, 0)):
                Rectangle(params.slot_width, clamp_od, mode=Mode.SUBTRACT)

        extrude(amount=params.clamp_height)

    clamp = clamp_part.part

    # Boolean union all three regions with proper positioning
    with BuildPart() as final:
        add(cup)
        add(arm)
        # Position clamp at the end of standoff, centered with arm vertically
        clamp_positioned = clamp.moved(
            Location(
                (clamp_center_x, 0, clamp_center_z - params.clamp_height / 2)
            )
        )
        add(clamp_positioned)

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
    part.export_stl(str(output_path))

    # Report file size
    file_size = output_path.stat().st_size
    print(f"\nSTL generated successfully:")
    print(f"  Path: {output_path}")
    print(f"  Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

    return output_path


if __name__ == "__main__":
    main()
