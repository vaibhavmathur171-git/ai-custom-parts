"""
Streamlit configurator — chat-first.

Primary UI: a conversation with the intake agent. The agent routes the
user to a template, gathers parameters, and calls generate_part. The 3D
viewer shows whichever part was most recently produced — by the agent OR
by the manual-mode sliders in the sidebar.

Manual mode is a power-user / debug surface and is collapsed by default.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

import streamlit as st
from build123d import export_stl
from streamlit_stl import stl_from_file

from conversation.agent import Agent, GeneratedPart
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

ACCEPTED_IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "gif"]


# ---------------------------------------------------------------------------
# Manual-mode sliders (carryover from Step 2B). These render template-specific
# parameter widgets inside the sidebar expander.
# ---------------------------------------------------------------------------


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
        screw_dia = defaults.screw_dia

    st.subheader("Arm & curve")
    arm_length = st.slider("Arm length (mm)", 10.0, 150.0, defaults.arm_length, 1.0)
    hook_radius = st.slider("Hook inside radius (mm)", 4.0, 40.0, defaults.hook_radius, 0.5)
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


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def _ensure_session_state():
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []
    if "current" not in st.session_state:
        # current = dict(template_name, params_dict, stl_path, file_size, source)
        # source ∈ {"agent", "manual"}; updated on every successful generation.
        st.session_state.current = None
    if "agent_error" not in st.session_state:
        st.session_state.agent_error = None
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = Agent(output_dir=OUTPUT_DIR)
        except Exception as exc:
            st.session_state.agent = None
            st.session_state.agent_error = str(exc)


def _record_generation(template_name: str, params, stl_path: Path, file_size: int, source: str):
    st.session_state.current = {
        "template_name": template_name,
        "params_dict": asdict(params),
        "stl_path": str(stl_path),
        "file_size": file_size,
        "source": source,
        "ts": time.time(),
    }


def _generate_manual(template_name: str, params) -> tuple[Path, int]:
    """Build a part from manual sliders and write it to disk."""
    spec = get_template(template_name)
    part = spec.make_fn(params)
    OUTPUT_DIR.mkdir(exist_ok=True)
    stl_path = OUTPUT_DIR / f"{template_name}_manual.stl"
    export_stl(part, str(stl_path))
    return stl_path, stl_path.stat().st_size


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------


st.set_page_config(page_title="Parametric CAD Configurator", layout="wide")
_ensure_session_state()

st.title("Parametric CAD Configurator")
st.caption(
    "Describe what you need to build. The intake agent will ask a few "
    "questions and produce a printable STL."
)

# ---------------------------------------------------------------------------
# Sidebar: manual mode + view options
# ---------------------------------------------------------------------------

specs = list_templates()
spec_by_name = {s.name: s for s in specs}

with st.sidebar:
    st.header("View")
    color = st.color_picker("Color", "#3a86ff")
    auto_rotate = st.toggle("Auto-rotate", value=False)

    with st.expander("Manual mode (override sliders)", expanded=False):
        st.caption(
            "Drive the geometry directly with sliders. Use this to "
            "fine-tune or test outside the conversation."
        )

        manual_template = st.radio(
            "Template",
            options=[s.name for s in specs],
            format_func=lambda n: TEMPLATE_LABELS.get(n, n),
            key="manual_template",
        )
        manual_spec = spec_by_name[manual_template]
        manual_params = SLIDER_DISPATCH[manual_template](manual_spec.default_params)

        manual_errors = manual_params.validate()
        if manual_errors:
            st.error("Manual params invalid:")
            for e in manual_errors:
                st.write(f"- {e}")

        if st.button("Apply manual params to viewer", disabled=bool(manual_errors)):
            try:
                stl_path, file_size = _generate_manual(manual_template, manual_params)
                _record_generation(manual_template, manual_params, stl_path, file_size, "manual")
                st.success(f"Applied {TEMPLATE_LABELS[manual_template]} from manual sliders.")
            except Exception as exc:
                st.error(f"Manual generation failed: {exc}")

    if st.button("Reset conversation"):
        if st.session_state.agent is not None:
            st.session_state.agent.reset()
        st.session_state.chat_log = []
        st.session_state.current = None
        st.rerun()


# ---------------------------------------------------------------------------
# Main area: chat (left), viewer + info (right)
# ---------------------------------------------------------------------------

if st.session_state.agent_error:
    st.error(
        "Conversation agent unavailable: "
        f"{st.session_state.agent_error}\n\n"
        "Manual mode (sidebar) still works. "
        "Add ANTHROPIC_API_KEY to .env to enable chat."
    )

chat_col, viewer_col = st.columns([2, 1])

with chat_col:
    st.subheader("Conversation")

    if not st.session_state.chat_log:
        with st.chat_message("assistant"):
            st.markdown(
                "Hi — describe the problem you're trying to solve and I'll "
                "find a template and gather what I need to make a part. "
                "You can also attach a photo for context."
            )

    for entry in st.session_state.chat_log:
        with st.chat_message(entry["role"]):
            if entry.get("text"):
                st.markdown(entry["text"])
            for img_bytes in entry.get("images", []):
                st.image(img_bytes, width=240)
            if entry.get("generated"):
                gen = entry["generated"]
                st.success(
                    f"✓ Generated **{TEMPLATE_LABELS.get(gen['template_name'], gen['template_name'])}** "
                    f"({gen['file_size'] / 1024:.1f} KB)"
                )
            if entry.get("tool_failures"):
                with st.expander("Validation feedback (sent back to agent)"):
                    for fail in entry["tool_failures"]:
                        st.code(fail, language="text")

with viewer_col:
    st.subheader("Preview")
    current = st.session_state.current
    if current is None:
        st.info("No part generated yet. Start a conversation or use Manual mode.")
    else:
        try:
            stl_from_file(
                file_path=current["stl_path"],
                color=color,
                material="material",
                auto_rotate=auto_rotate,
                opacity=1,
                height=420,
                cam_distance=0,
            )
        except Exception as exc:
            st.error(f"Viewer error: {exc}")

        st.metric("STL size", f"{current['file_size'] / 1024:.1f} KB")
        st.caption(
            f"Source: **{current['source']}** · Template: "
            f"**{TEMPLATE_LABELS.get(current['template_name'], current['template_name'])}**"
        )

        spec = spec_by_name[current["template_name"]]
        params_obj = spec.params_class(**current["params_dict"])
        METRIC_DISPATCH[current["template_name"]](params_obj)

        with open(current["stl_path"], "rb") as f:
            st.download_button(
                label="Download STL",
                data=f,
                file_name=f"{current['template_name']}.stl",
                mime="model/stl",
                use_container_width=True,
            )

        with st.expander("Parameter dump"):
            st.json(current["params_dict"])


# ---------------------------------------------------------------------------
# Chat input — page-level so it pins to the bottom
# ---------------------------------------------------------------------------


prompt = st.chat_input(
    "Describe the problem (you can attach a photo for context)",
    accept_file=True,
    file_type=ACCEPTED_IMAGE_TYPES,
    disabled=st.session_state.agent is None,
)

if prompt is not None:
    text = (prompt.text or "").strip() if hasattr(prompt, "text") else str(prompt).strip()
    raw_files = list(getattr(prompt, "files", []) or [])

    image_payloads: list[tuple[str, bytes]] = []
    image_bytes_for_display: list[bytes] = []
    for f in raw_files:
        data = f.read()
        media_type = f.type or "image/png"
        image_payloads.append((media_type, data))
        image_bytes_for_display.append(data)

    if not text and not image_payloads:
        st.warning("Empty message — type something or attach an image.")
    elif st.session_state.agent is None:
        st.error("Agent unavailable; can't process chat. Use Manual mode.")
    else:
        st.session_state.chat_log.append(
            {"role": "user", "text": text, "images": image_bytes_for_display}
        )
        try:
            turn = st.session_state.agent.send(text, images=image_payloads)
        except Exception as exc:
            st.session_state.chat_log.append(
                {
                    "role": "assistant",
                    "text": f"Sorry, the agent hit an error: `{exc}`",
                    "images": [],
                }
            )
            st.rerun()
        else:
            assistant_entry: dict = {
                "role": "assistant",
                "text": turn.text,
                "images": [],
            }
            tool_failures = [a["message"] for a in turn.tool_attempts if not a["ok"]]
            if tool_failures:
                assistant_entry["tool_failures"] = tool_failures
            if turn.generated_part is not None:
                gp: GeneratedPart = turn.generated_part
                assistant_entry["generated"] = {
                    "template_name": gp.template_name,
                    "file_size": gp.file_size,
                }
                _record_generation(gp.template_name, gp.params, gp.stl_path, gp.file_size, "agent")
            st.session_state.chat_log.append(assistant_entry)
            st.rerun()
