"""
Streamlit configurator for the parametric CAD template library.

A template selector at the top of the sidebar picks which model is active.
The slider panel, the 3D viewer, the validation errors, and the metadata
panel all dispatch on the selected template. No AI yet — that's Step 3.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import streamlit as st
from build123d import export_stl
from streamlit_stl import stl_from_file

from templates.bottle_holder import BottleHolderParams
from templates.bracket import BracketParams
from templates.hook import HookParams
from templates.registry import get_template, list_templates


OUTPUT_DIR = Path(__file__).parent / "output"

TEMPLATE_LABELS = {
    "bottle_holder": "Bottle holder",
    "hook": "J-hook",
    "bracket": "L-bracket",
}


@st.cache_data(show_spinner=False)
def generate_preview(template_name: str, params_json: str) -> tuple[str, int]:
    """Build the part for the given template + params and write a preview STL.

    Cached on (template_name, params_json) so dragging a slider back to a
    prior value is instant. params_json is a sort-keyed JSON dump of the
    params dataclass so the cache key is stable.
    """
    spec = get_template(template_name)
    params_dict = json.loads(params_json)
    params = spec.params_class(**params_dict)
    part = spec.make_fn(params)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{template_name}_preview.stl"
    export_stl(part, str(out_path))
    return str(out_path), out_path.stat().st_size


def _bottle_holder_sliders(defaults: BottleHolderParams) -> BottleHolderParams:
    st.subheader("Bottle & cup")
    bottle_dia = st.slider("Bottle diameter (mm)", 30.0, 100.0, defaults.bottle_dia, 0.5)
    cup_id = st.slider("Cup inner diameter (mm)", 30.0, 110.0, defaults.cup_id, 0.5)
    cup_height = st.slider("Cup height (mm)", 40.0, 200.0, defaults.cup_height, 1.0)
    drain_dia = st.slider("Drainage hole diameter (mm)", 2.0, 20.0, defaults.drain_dia, 0.5)

    st.subheader("Clamp")
    bar_dia = st.slider("Roll bar diameter (mm)", 15.0, 60.0, defaults.bar_dia, 0.5)
    clamp_height = st.slider("Clamp height (mm)", 10.0, 60.0, defaults.clamp_height, 1.0)
    slot_width = st.slider("Slot width (mm)", 1.0, 10.0, defaults.slot_width, 0.5)

    st.subheader("Arm & wall")
    standoff = st.slider("Standoff (mm)", 10.0, 100.0, defaults.standoff, 1.0)
    arm_width = st.slider("Arm width (mm)", 5.0, 60.0, defaults.arm_width, 1.0)
    wall_t = st.slider("Wall thickness (mm)", 1.0, 5.0, defaults.wall_t, 0.1)

    return BottleHolderParams(
        bottle_dia=bottle_dia,
        cup_id=cup_id,
        cup_height=cup_height,
        wall_t=wall_t,
        drain_dia=drain_dia,
        bar_dia=bar_dia,
        clamp_height=clamp_height,
        slot_width=slot_width,
        standoff=standoff,
        arm_width=arm_width,
    )


def _hook_sliders(defaults: HookParams) -> HookParams:
    st.subheader("Mount")
    mount_type = st.radio(
        "Mount type",
        options=["flat", "bar"],
        index=0 if defaults.mount_type == "flat" else 1,
        format_func=lambda v: "Flat plate (screws)" if v == "flat" else "Bar clamp",
        horizontal=True,
    )
    if mount_type == "flat":
        mount_dim = st.slider("Plate side (mm)", 15.0, 80.0, max(defaults.mount_dim, 15.0), 1.0)
        screw_dia = st.slider("Screw hole diameter (mm)", 2.0, 8.0, defaults.screw_dia, 0.1)
    else:
        mount_dim = st.slider("Bar diameter (mm)", 15.0, 60.0, min(max(defaults.mount_dim, 15.0), 60.0), 0.5)
        screw_dia = defaults.screw_dia  # unused for bar mount

    st.subheader("Arm & curve")
    arm_length = st.slider("Arm length (mm)", 10.0, 150.0, defaults.arm_length, 1.0)
    hook_radius = st.slider("Hook inside radius (mm)", 4.0, 40.0, defaults.hook_radius, 0.5)
    # Opening must remain < 2 * hook_radius for the J to curl back.
    max_opening = max(2 * hook_radius - 0.5, 1.0)
    opening = st.slider(
        "Opening at the J mouth (mm)",
        1.0,
        max_opening,
        min(defaults.opening, max_opening),
        0.5,
    )

    st.subheader("Wall")
    wall_t = st.slider("Wall thickness (mm)", 1.0, 6.0, defaults.wall_t, 0.1)

    return HookParams(
        mount_type=mount_type,
        mount_dim=mount_dim,
        arm_length=arm_length,
        hook_radius=hook_radius,
        opening=opening,
        wall_t=wall_t,
        screw_dia=screw_dia,
    )


def _bracket_sliders(defaults: BracketParams) -> BracketParams:
    st.subheader("Plate A (horizontal)")
    plate_a_length = st.slider("Plate A length (mm)", 15.0, 200.0, defaults.plate_a_length, 1.0)
    plate_a_width = st.slider("Plate A width (mm)", 10.0, 120.0, defaults.plate_a_width, 1.0)
    holes_a = st.slider("Holes through plate A", 0, 6, defaults.holes_a, 1)

    st.subheader("Plate B (vertical)")
    plate_b_length = st.slider("Plate B length (mm)", 15.0, 200.0, defaults.plate_b_length, 1.0)
    plate_b_width = st.slider("Plate B width (mm)", 10.0, 120.0, defaults.plate_b_width, 1.0)
    holes_b = st.slider("Holes through plate B", 0, 6, defaults.holes_b, 1)

    st.subheader("Wall & holes")
    thickness = st.slider("Plate thickness (mm)", 1.0, 10.0, defaults.thickness, 0.5)
    hole_dia = st.slider("Hole diameter (mm)", 2.0, 12.0, defaults.hole_dia, 0.1)
    gusset = st.checkbox("Stiffening gusset", value=defaults.gusset)

    return BracketParams(
        plate_a_length=plate_a_length,
        plate_a_width=plate_a_width,
        plate_b_length=plate_b_length,
        plate_b_width=plate_b_width,
        thickness=thickness,
        holes_a=holes_a,
        holes_b=holes_b,
        hole_dia=hole_dia,
        gusset=gusset,
    )


SLIDER_DISPATCH = {
    "bottle_holder": _bottle_holder_sliders,
    "hook": _hook_sliders,
    "bracket": _bracket_sliders,
}


def _bottle_holder_metrics(params: BottleHolderParams):
    cup_od = params.cup_id + 2 * params.wall_t
    clamp_od = params.bar_dia + 2 * params.wall_t
    st.metric("Cup OD", f"{cup_od:.1f} mm")
    st.metric("Clamp OD", f"{clamp_od:.1f} mm")
    st.metric("Bottle clearance", f"{params.cup_id - params.bottle_dia:.1f} mm")


def _hook_metrics(params: HookParams):
    if params.mount_type == "bar":
        clamp_od = params.mount_dim + 2 * params.wall_t
        st.metric("Clamp OD", f"{clamp_od:.1f} mm")
        reach = clamp_od / 2 + params.arm_length
    else:
        st.metric("Plate", f"{params.mount_dim:.0f} × {params.mount_dim:.0f} mm")
        reach = params.arm_length
    st.metric("Reach from mount", f"{reach:.1f} mm")
    st.metric("Curl height", f"{2 * params.hook_radius:.1f} mm")


def _bracket_metrics(params: BracketParams):
    st.metric("Plate A footprint", f"{params.plate_a_length:.0f} × {params.plate_a_width:.0f} mm")
    st.metric("Plate B footprint", f"{params.plate_b_length:.0f} × {params.plate_b_width:.0f} mm")
    st.metric("Holes", f"{params.holes_a} + {params.holes_b}")


METRIC_DISPATCH = {
    "bottle_holder": _bottle_holder_metrics,
    "hook": _hook_metrics,
    "bracket": _bracket_metrics,
}


st.set_page_config(page_title="Parametric CAD Configurator", layout="wide")
st.title("Parametric CAD Configurator")
st.caption(
    "Pick a template, drag sliders to update the parametric model, and "
    "download an STL ready to print."
)

specs = list_templates()
spec_by_name = {s.name: s for s in specs}

with st.sidebar:
    st.header("Template")
    selected_name = st.radio(
        "Choose a template",
        options=[s.name for s in specs],
        format_func=lambda n: TEMPLATE_LABELS.get(n, n),
        label_visibility="collapsed",
    )
    spec = spec_by_name[selected_name]
    st.caption(spec.description)

    st.header("Parameters")
    params = SLIDER_DISPATCH[selected_name](spec.default_params)

    st.header("View")
    color = st.color_picker("Color", "#3a86ff")
    auto_rotate = st.toggle("Auto-rotate", value=False)

errors = params.validate()
if errors:
    st.error("Parameter validation failed:")
    for e in errors:
        st.write(f"- {e}")
    st.stop()

try:
    params_json = json.dumps(asdict(params), sort_keys=True)
    stl_path, file_size = generate_preview(selected_name, params_json)
except Exception as exc:
    st.error(f"Geometry generation failed: {exc}")
    st.stop()

viewer_col, info_col = st.columns([3, 1])

with viewer_col:
    stl_from_file(
        file_path=stl_path,
        color=color,
        material="material",
        auto_rotate=auto_rotate,
        opacity=1,
        height=600,
        cam_distance=0,
    )

with info_col:
    st.metric("STL size", f"{file_size / 1024:.1f} KB")
    METRIC_DISPATCH[selected_name](params)

    with open(stl_path, "rb") as f:
        st.download_button(
            label="Download STL",
            data=f,
            file_name=f"{selected_name}.stl",
            mime="model/stl",
            use_container_width=True,
        )

    with st.expander("Typical use cases"):
        for use_case in spec.typical_use_cases:
            st.write(f"- {use_case}")

    with st.expander("Parameter dump"):
        st.json(asdict(params))
