"""
AI Custom Parts — Streamlit configurator.

Layout: chat panel on the LEFT (~30%), 3D viewport on the RIGHT (~70%).
Light theme, soft panels, persistent XYZ axes overlay in the viewport's
bottom-left corner.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path

import streamlit as st
from streamlit_stl import stl_from_file

from conversation.agent import (
    ALL_MODELS,
    Agent,
    GeneratedPart,
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
)
from conversation.system_prompt import OPENING_GREETING
from manufacturing.checks import CheckResult, overall_status, run_checks
from manufacturing.context import (
    METHOD_LABEL,
    Method,
    ProductionContext,
    thresholds_for,
)
from manufacturing.export import export_step, export_stl
from manufacturing.visualize import annotate_check_thumbnails
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
MM_PER_INCH = 25.4

# Model routing pills + sidebar counter are part of the product, not a debug
# toggle: visible by default because they're concrete evidence the per-turn
# routing is real. Set HIDE_MODEL=1 to suppress them for a cleaner UI.
SHOW_MODEL_INDICATOR = os.environ.get("HIDE_MODEL", "0") != "1"

MODEL_SHORT = {
    MODEL_OPUS: "opus",
    MODEL_SONNET: "sonnet",
    MODEL_HAIKU: "haiku",
}


# ---------------------------------------------------------------------------
# Custom CSS. Sections in order:
#   1. Page chrome (header tightening, page header / tagline typography)
#   2. Viewport frame (border, shadow, padding)
#   3. Persistent XYZ axes overlay
#   4. Empty-state hint
#   5. Chat bubbles + chat header (reset button)
#   6. Model-routing indicator pill
#   7. Manufacturing checks panel (single-line title + status pill + cards)
# ---------------------------------------------------------------------------


_PAGE_CSS = """
<style>
  /* --- 1. Page chrome --------------------------------------------------- */
  .block-container {padding-top: 1.2rem; padding-bottom: 1rem; max-width: 100%;}
  header[data-testid="stHeader"] {height: 0; background: transparent;}

  .page-header {margin-bottom: 8px;}
  .page-header h1 {
    font-size: 30px;
    font-weight: 700;
    color: #111827;
    margin: 0 0 6px 0;
    line-height: 1.15;
  }
  .page-header .tagline {
    color: #6a6a6a;
    font-size: 15px;
    margin: 0 0 2px 0;
  }
  .page-header .sub-tagline {
    color: #9aa0a6;
    font-size: 13px;
    margin: 0 0 14px 0;
  }

  .secondary {color: #6a6a6a;}
  .meta-strip {color: #6a6a6a; font-size: 13px; padding: 2px 0 6px 0;}
  .meta-strip .pill {color: #1a1a1a; font-weight: 600; margin-right: 14px;}
  .meta-strip .sep {color: #d0d0d0; margin: 0 10px;}

  /* --- 2. Viewport frame ------------------------------------------------ */
  /* Streamlit assigns the keyed container the class `st-key-{key}`. The
     contains-selector keeps us robust against minor naming changes. */
  [class*="st-key-viewport-frame"] {
    position: relative;
    border: 1px solid #e5e5e7;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    padding: 16px;
    background: #fafafa;
    min-height: 540px;
  }

  /* --- 3. Persistent XYZ corner triad ----------------------------------- */
  .axes-overlay {
    position: absolute;
    bottom: 12px;
    left: 12px;
    width: 70px;
    height: 70px;
    pointer-events: none;
    z-index: 10;
    opacity: 0.85;
  }

  /* --- 4. Empty-state hint --------------------------------------------- */
  .empty-hint {
    color: #9a9a9a;
    font-style: italic;
    font-size: 14px;
    text-align: center;
    padding: 220px 20px;
    user-select: none;
  }

  /* --- 5. Chat bubbles + header ---------------------------------------- */
  div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    background: #e0ecff;
    border-radius: 10px;
    padding: 8px 12px;
  }
  div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 8px 12px;
  }
  div[data-testid="stChatMessage"] {margin-bottom: 8px;}

  .chat-header {display: flex; align-items: center; padding: 4px 0 8px 0;}
  .chat-header .title {font-weight: 600; color: #1a1a1a;}

  /* --- 6. Model indicator (SHOW_MODEL=1 only) -------------------------- */
  .model-pill {
    display: inline-block;
    margin-top: 4px;
    padding: 1px 8px;
    background: #f3f4f6;
    color: #6a6a6a;
    border-radius: 8px;
    font-size: 11px;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
  }

  /* --- 7. Manufacturing checks panel ----------------------------------- */
  .checks-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 14px 0 10px 0;
  }
  .checks-header .title {
    font-weight: 500;
    color: #1a1a1a;
    font-size: 14px;
  }
  .status-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
    line-height: 1.5;
  }
  .status-pill.pass {background: #dcfce7; color: #166534;}
  .status-pill.warn {background: #fef3c7; color: #92400e;}
  .status-pill.fail {background: #fee2e2; color: #991b1b;}

  /* Card-style check rows. Targets only expanders inside the keyed
     container so other expanders (Manual mode, "What's the difference?")
     keep their default look. */
  [class*="st-key-checks-list"] [data-testid="stExpander"] {
    border: 1px solid #e5e5e7;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    margin-bottom: 6px;
    overflow: hidden;
  }
  [class*="st-key-checks-list"] [data-testid="stExpander"] summary {
    padding: 8px 12px;
    font-size: 13.5px;
  }
</style>
"""


# Persistent corner triad — small static SVG positioned bottom-left of the
# viewport frame. Always visible (empty AND rendered states).
_CORNER_TRIAD_SVG = """
<svg viewBox="-10 -60 80 70" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="cR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#dc2626"/>
    </marker>
    <marker id="cG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#16a34a"/>
    </marker>
    <marker id="cB" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
    </marker>
  </defs>
  <circle cx="0" cy="0" r="2" fill="#404040"/>
  <line x1="0" y1="0" x2="32" y2="0" stroke="#dc2626" stroke-width="2" marker-end="url(#cR)"/>
  <text x="38" y="3" fill="#dc2626" font-family="ui-monospace, monospace" font-size="9" font-weight="600">X</text>
  <line x1="0" y1="0" x2="22" y2="-13" stroke="#16a34a" stroke-width="2" marker-end="url(#cG)"/>
  <text x="26" y="-15" fill="#16a34a" font-family="ui-monospace, monospace" font-size="9" font-weight="600">Y</text>
  <line x1="0" y1="0" x2="0" y2="-32" stroke="#2563eb" stroke-width="2" marker-end="url(#cB)"/>
  <text x="-9" y="-37" fill="#2563eb" font-family="ui-monospace, monospace" font-size="9" font-weight="600">Z</text>
</svg>
"""


# ---------------------------------------------------------------------------
# Manual-mode sliders
# ---------------------------------------------------------------------------


def _bottle_holder_sliders(defaults: BottleHolderParams) -> BottleHolderParams:
    bottle_dia = st.slider("Bottle diameter (mm)", 30.0, 100.0, defaults.bottle_dia, 0.5)
    cup_id = st.slider("Cup inner diameter (mm)", 30.0, 110.0, defaults.cup_id, 0.5)
    cup_height = st.slider("Cup height (mm)", 40.0, 200.0, defaults.cup_height, 1.0)
    drain_dia = st.slider("Drainage hole diameter (mm)", 2.0, 20.0, defaults.drain_dia, 0.5)
    bar_dia = st.slider("Roll bar diameter (mm)", 15.0, 60.0, defaults.bar_dia, 0.5)
    clamp_height = st.slider("Clamp height (mm)", 10.0, 60.0, defaults.clamp_height, 1.0)
    slot_width = st.slider("Slot width (mm)", 1.0, 10.0, defaults.slot_width, 0.5)
    standoff = st.slider("Standoff (mm)", 10.0, 100.0, defaults.standoff, 1.0)
    arm_width = st.slider("Arm width (mm)", 5.0, 60.0, defaults.arm_width, 1.0)
    wall_t = st.slider("Wall thickness (mm)", 1.0, 5.0, defaults.wall_t, 0.1)
    return BottleHolderParams(
        bottle_dia=bottle_dia, cup_id=cup_id, cup_height=cup_height, wall_t=wall_t,
        drain_dia=drain_dia, bar_dia=bar_dia, clamp_height=clamp_height,
        slot_width=slot_width, standoff=standoff, arm_width=arm_width,
    )


def _hook_sliders(defaults: HookParams) -> HookParams:
    mount_type = st.radio(
        "Mount type", options=["flat", "bar"],
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
    arm_length = st.slider("Arm length (mm)", 10.0, 150.0, defaults.arm_length, 1.0)
    hook_radius = st.slider("Hook inside radius (mm)", 4.0, 40.0, defaults.hook_radius, 0.5)
    max_opening = max(2 * hook_radius - 0.5, 1.0)
    opening = st.slider("Opening at the J mouth (mm)", 1.0, max_opening, min(defaults.opening, max_opening), 0.5)
    wall_t = st.slider("Wall thickness (mm)", 1.0, 6.0, defaults.wall_t, 0.1)
    return HookParams(
        mount_type=mount_type, mount_dim=mount_dim, arm_length=arm_length,
        hook_radius=hook_radius, opening=opening, wall_t=wall_t, screw_dia=screw_dia,
    )


def _bracket_sliders(defaults: BracketParams) -> BracketParams:
    plate_a_length = st.slider("Plate A length (mm)", 15.0, 200.0, defaults.plate_a_length, 1.0)
    plate_a_width = st.slider("Plate A width (mm)", 10.0, 120.0, defaults.plate_a_width, 1.0)
    holes_a = st.slider("Holes through plate A", 0, 6, defaults.holes_a, 1)
    plate_b_length = st.slider("Plate B length (mm)", 15.0, 200.0, defaults.plate_b_length, 1.0)
    plate_b_width = st.slider("Plate B width (mm)", 10.0, 120.0, defaults.plate_b_width, 1.0)
    holes_b = st.slider("Holes through plate B", 0, 6, defaults.holes_b, 1)
    thickness = st.slider("Plate thickness (mm)", 1.0, 10.0, defaults.thickness, 0.5)
    hole_dia = st.slider("Hole diameter (mm)", 2.0, 12.0, defaults.hole_dia, 0.1)
    gusset = st.checkbox("Stiffening gusset", value=defaults.gusset)
    return BracketParams(
        plate_a_length=plate_a_length, plate_a_width=plate_a_width,
        plate_b_length=plate_b_length, plate_b_width=plate_b_width,
        thickness=thickness, holes_a=holes_a, holes_b=holes_b,
        hole_dia=hole_dia, gusset=gusset,
    )


SLIDER_DISPATCH = {
    "bottle_holder": _bottle_holder_sliders,
    "hook": _hook_sliders,
    "bracket": _bracket_sliders,
}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _fmt(value_mm: float, unit: str, decimals: int | None = None) -> str:
    """Format a millimeter value in the user's preferred unit. Display only."""
    if unit == "in":
        v = value_mm / MM_PER_INCH
        d = 2 if decimals is None else decimals
        return f"{v:.{d}f}\""
    d = 0 if decimals is None else decimals
    return f"{value_mm:.{d}f}mm"


def _key_dimensions(template_name: str, params, unit: str) -> str:
    if template_name == "bottle_holder":
        return (
            f"cup {_fmt(params.cup_id, unit)} × {_fmt(params.cup_height, unit)} tall · "
            f"clamps {_fmt(params.bar_dia, unit)} bar"
        )
    if template_name == "hook":
        if params.mount_type == "bar":
            return (
                f"clamps {_fmt(params.mount_dim, unit)} bar · "
                f"arm {_fmt(params.arm_length, unit)} · "
                f"curl r{_fmt(params.hook_radius, unit)}"
            )
        return (
            f"flat mount {_fmt(params.mount_dim, unit)} · "
            f"arm {_fmt(params.arm_length, unit)} · "
            f"curl r{_fmt(params.hook_radius, unit)}"
        )
    if template_name == "bracket":
        return (
            f"plate A {_fmt(params.plate_a_length, unit)}×{_fmt(params.plate_a_width, unit)} · "
            f"plate B {_fmt(params.plate_b_length, unit)}×{_fmt(params.plate_b_width, unit)} · "
            f"{_fmt(params.thickness, unit, decimals=2 if unit == 'in' else 1)} thick"
        )
    return ""


# ---------------------------------------------------------------------------
# Session state + generation
# ---------------------------------------------------------------------------


def _ensure_session_state():
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []
    if "current" not in st.session_state:
        st.session_state.current = None
    if "agent_error" not in st.session_state:
        st.session_state.agent_error = None
    if "unit_pref" not in st.session_state:
        st.session_state.unit_pref = "mm"
    if "model_counts" not in st.session_state:
        st.session_state.model_counts = {m: 0 for m in ALL_MODELS}
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = Agent(output_dir=OUTPUT_DIR)
        except Exception as exc:
            st.session_state.agent = None
            st.session_state.agent_error = str(exc)


def _record_generation(
    template_name: str,
    params,
    stl_path: Path,
    step_path: Path,
    stl_size: int,
    step_size: int,
    source: str,
    context: ProductionContext,
    check_results: list[CheckResult],
):
    st.session_state.current = {
        "template_name": template_name,
        "params_dict": asdict(params),
        "stl_path": str(stl_path),
        "step_path": str(step_path),
        "stl_size": stl_size,
        "step_size": step_size,
        "source": source,
        "context": {
            "method": context.method.value,
            "material": context.material,
            "nozzle_dia": context.nozzle_dia,
            "printer_model": context.printer_model,
            "notes": context.notes,
        },
        "checks": [
            {
                "name": r.name,
                "status": r.status,
                "message": r.message,
                "suggestion": r.suggestion,
                "thumbnail_path": r.thumbnail_path,
            }
            for r in check_results
        ],
        "overall": overall_status(check_results),
        "ts": time.time(),
    }


def _generate_manual(template_name: str, params, context: ProductionContext) -> tuple[Path, Path, int, int, list[CheckResult]]:
    spec = get_template(template_name)
    part = spec.make_fn(params)
    OUTPUT_DIR.mkdir(exist_ok=True)
    stl_path = export_stl(part, OUTPUT_DIR / f"{template_name}_manual.stl")
    step_path = export_step(part, OUTPUT_DIR / f"{template_name}_manual.step")
    checks = run_checks(template_name, params, part, context)
    try:
        annotate_check_thumbnails(checks, part, stl_path, OUTPUT_DIR / "thumbnails")
    except Exception:
        pass
    return stl_path, step_path, stl_path.stat().st_size, step_path.stat().st_size, checks


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------


st.set_page_config(page_title="AI Custom Parts", layout="wide")
_ensure_session_state()

st.markdown(_PAGE_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="page-header">
      <h1>AI Custom Parts</h1>
      <div class="tagline">For brackets, holders, hooks, and other custom parts you can't find on Amazon.</div>
      <div class="sub-tagline">Describe what you need. The system asks a few questions and generates a 3D-printable file.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

specs = list_templates()
spec_by_name = {s.name: s for s in specs}


# ---------------------------------------------------------------------------
# Sidebar: View options + Manual mode
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("**View**")
    color = st.color_picker("Color", "#3b82f6", label_visibility="collapsed")
    auto_rotate = st.toggle("Auto-rotate", value=False)

    with st.expander("Manual mode", expanded=False):
        st.caption("Drive geometry directly with sliders.")
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

        if st.button("Apply", disabled=bool(manual_errors), use_container_width=True):
            try:
                ctx = (
                    st.session_state.agent.effective_context()
                    if st.session_state.agent
                    else ProductionContext.safe_default()
                )
                stl_p, step_p, stl_sz, step_sz, checks = _generate_manual(
                    manual_template, manual_params, ctx
                )
                _record_generation(
                    manual_template, manual_params, stl_p, step_p, stl_sz, step_sz,
                    "manual", ctx, checks,
                )
            except Exception as exc:
                st.error(f"Manual generation failed: {exc}")

    if SHOW_MODEL_INDICATOR:
        st.markdown("---")
        st.caption("Model routing telemetry")
        counts = st.session_state.model_counts
        st.write(
            f"opus: {counts.get(MODEL_OPUS, 0)} · "
            f"sonnet: {counts.get(MODEL_SONNET, 0)} · "
            f"haiku: {counts.get(MODEL_HAIKU, 0)}"
        )


# ---------------------------------------------------------------------------
# Main layout: chat (LEFT, ~30%), viewport (RIGHT, ~70%)
# ---------------------------------------------------------------------------

if st.session_state.agent_error:
    st.error(
        f"Conversation agent unavailable: {st.session_state.agent_error}. "
        "Manual mode (sidebar) still works. Add ANTHROPIC_API_KEY to .env to enable chat."
    )

chat_col, viewport_col = st.columns([3, 7], gap="medium")


# --- Chat column (LEFT) ---------------------------------------------------

with chat_col:
    header_cols = st.columns([5, 1], vertical_alignment="center")
    with header_cols[0]:
        st.markdown(
            '<div class="chat-header"><span class="title">Conversation</span></div>',
            unsafe_allow_html=True,
        )
    with header_cols[1]:
        if st.button("↺", help="Reset conversation", key="reset_btn"):
            if st.session_state.agent is not None:
                st.session_state.agent.reset()
            st.session_state.chat_log = []
            st.session_state.current = None
            st.session_state.model_counts = {m: 0 for m in ALL_MODELS}
            st.rerun()

    chat_container = st.container(height=540)

    with chat_container:
        if not st.session_state.chat_log:
            with st.chat_message("assistant"):
                st.markdown(OPENING_GREETING)

        for entry in st.session_state.chat_log:
            with st.chat_message(entry["role"]):
                if entry.get("text"):
                    st.markdown(entry["text"])
                for img_bytes in entry.get("images", []):
                    st.image(img_bytes, width=180)
                if entry.get("generated"):
                    gen = entry["generated"]
                    overall_emoji = {"pass": "✅", "warn": "⚠️", "fail": "❌"}.get(
                        gen.get("overall", "pass"), "✅"
                    )
                    st.success(
                        f"{overall_emoji} **{TEMPLATE_LABELS.get(gen['template_name'], gen['template_name'])}** "
                        f"· {gen['stl_size'] / 1024:.0f} KB STL · {gen['step_size'] / 1024:.0f} KB STEP"
                    )
                if entry.get("context_changed"):
                    cc = entry["context_changed"]
                    bits = [cc.get("method", "")]
                    if cc.get("material"):
                        bits.append(cc["material"])
                    if cc.get("printer_model"):
                        bits.append(cc["printer_model"])
                    st.info("📐 " + ", ".join(b for b in bits if b))
                if entry.get("tool_failures"):
                    with st.expander("Validation feedback (sent back to agent)"):
                        for fail in entry["tool_failures"]:
                            st.code(fail, language="text")
                if SHOW_MODEL_INDICATOR and entry.get("models_used"):
                    short = ", ".join(MODEL_SHORT.get(m, m) for m in entry["models_used"])
                    st.markdown(
                        f'<span class="model-pill">🧠 {short}</span>',
                        unsafe_allow_html=True,
                    )

    prompt = st.chat_input(
        "Describe what you'd like to make (you can attach a photo)",
        accept_file=True,
        file_type=ACCEPTED_IMAGE_TYPES,
        disabled=st.session_state.agent is None,
    )


# --- Viewport column (RIGHT) ---------------------------------------------

with viewport_col:
    current = st.session_state.current

    # Top strip: metadata (left) + unit toggle (right) on the same row.
    top_left, top_right = st.columns([6, 1.2], vertical_alignment="center")
    with top_right:
        unit_pref = st.radio(
            "Units",
            options=["mm", "in"],
            horizontal=True,
            label_visibility="collapsed",
            key="unit_pref",
        )

    if current is None:
        with top_left:
            st.markdown(
                '<div class="meta-strip secondary">No part yet</div>',
                unsafe_allow_html=True,
            )

        # Empty viewport frame: just a centered hint plus the corner triad.
        with st.container(key="viewport-frame"):
            st.html('<div class="empty-hint">← Describe what you\'d like to make</div>')
            st.html(f'<div class="axes-overlay">{_CORNER_TRIAD_SVG}</div>')

    else:
        spec = spec_by_name[current["template_name"]]
        params_obj = spec.params_class(**current["params_dict"])
        ctx_data = current["context"]
        method_enum = Method(ctx_data["method"])
        method_text = METHOD_LABEL[method_enum]
        if ctx_data.get("material"):
            method_text += f" · {ctx_data['material']}"
        if ctx_data.get("printer_model"):
            method_text += f" · {ctx_data['printer_model']}"

        with top_left:
            label = TEMPLATE_LABELS.get(current["template_name"], current["template_name"])
            dims = _key_dimensions(current["template_name"], params_obj, unit_pref)
            st.markdown(
                f"""
                <div class="meta-strip">
                    <span class="pill">{label}</span>
                    <span>{dims}</span>
                    <span class="sep">·</span>
                    <span>STL {current['stl_size'] / 1024:.1f} KB · STEP {current['step_size'] / 1024:.1f} KB</span>
                    <span class="sep">·</span>
                    <span>for {method_text}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Rendered viewport: the streamlit-stl widget plus the corner triad.
        with st.container(key="viewport-frame"):
            try:
                stl_from_file(
                    file_path=current["stl_path"],
                    color=color,
                    material="material",
                    auto_rotate=auto_rotate,
                    opacity=1,
                    height=500,
                    cam_distance=0,
                )
            except Exception as exc:
                st.error(f"Viewer error: {exc}")
            st.html(f'<div class="axes-overlay">{_CORNER_TRIAD_SVG}</div>')

        # Action bar
        action_a, action_b, action_c = st.columns([1, 1, 1])
        with action_a:
            with open(current["stl_path"], "rb") as f:
                st.download_button(
                    label="⬇  Download STL  (3D printing)",
                    data=f,
                    file_name=f"{current['template_name']}.stl",
                    mime="model/stl",
                    use_container_width=True,
                    type="primary",
                )
        with action_b:
            with open(current["step_path"], "rb") as f:
                st.download_button(
                    label="⬇  Download STEP  (CAD/CNC)",
                    data=f,
                    file_name=f"{current['template_name']}.step",
                    mime="application/STEP",
                    use_container_width=True,
                )
        with action_c:
            with st.expander("What's the difference?"):
                st.markdown(
                    "**STL** is a triangle mesh. Use it for FDM or SLA "
                    "3D printing — it's what your slicer expects.\n\n"
                    "**STEP** is a parametric solid (B-Rep). Use it for "
                    "CAD work, CNC machining, injection molding quotes, "
                    "or any toolchain that's not 'just print it'."
                )

        # Manufacturing checks panel — single-line title + status pill.
        overall = current["overall"]
        warn_count = sum(1 for r in current["checks"] if r["status"] == "warn")
        fail_count = sum(1 for r in current["checks"] if r["status"] == "fail")
        if overall == "pass":
            pill_text = "All good"
        elif overall == "fail":
            pill_text = f"{fail_count} issue" + ("s" if fail_count != 1 else "")
        else:  # warn
            pill_text = f"{warn_count} warning" + ("s" if warn_count != 1 else "")

        st.markdown(
            f"""
            <div class="checks-header">
              <span class="title">Manufacturing checks for {METHOD_LABEL[method_enum]}</span>
              <span class="status-pill {overall}">{pill_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if (
            current["source"] == "manual"
            and st.session_state.agent is not None
            and st.session_state.agent.production_context is None
        ):
            st.caption("No production context yet — generic FDM defaults applied.")

        if not current["checks"]:
            st.caption("No checks ran for this combination.")
        else:
            with st.container(key="checks-list"):
                for r in current["checks"]:
                    emoji = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[r["status"]]
                    # Pass = collapsed; warn/fail = expanded.
                    with st.expander(f"{emoji} {r['name']}", expanded=(r["status"] != "pass")):
                        thumb = r.get("thumbnail_path")
                        if thumb and Path(thumb).exists():
                            st.image(
                                thumb,
                                caption="Problem area shown in red",
                                width=300,
                            )
                        st.write(r["message"])
                        if r.get("suggestion"):
                            st.caption(f"Suggestion: {r['suggestion']}")


# ---------------------------------------------------------------------------
# Handle a new chat submission
# ---------------------------------------------------------------------------

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
                {"role": "assistant", "text": f"Sorry, the agent hit an error: `{exc}`", "images": []}
            )
            st.rerun()
        else:
            assistant_entry: dict = {
                "role": "assistant",
                "text": turn.text,
                "images": [],
                "models_used": list(turn.models_used),
            }
            tool_failures = [a["message"] for a in turn.tool_attempts if not a.get("ok")]
            if tool_failures:
                assistant_entry["tool_failures"] = tool_failures
            if turn.context_changed is not None:
                cc = turn.context_changed
                assistant_entry["context_changed"] = {
                    "method": cc.method.value,
                    "material": cc.material,
                    "printer_model": cc.printer_model,
                }
            if turn.generated_part is not None:
                gp: GeneratedPart = turn.generated_part
                assistant_entry["generated"] = {
                    "template_name": gp.template_name,
                    "stl_size": gp.stl_size,
                    "step_size": gp.step_size,
                    "overall": gp.overall_status,
                }
                _record_generation(
                    gp.template_name, gp.params, gp.stl_path, gp.step_path,
                    gp.stl_size, gp.step_size, "agent",
                    gp.production_context, gp.check_results,
                )

            for m in turn.models_used:
                st.session_state.model_counts[m] = st.session_state.model_counts.get(m, 0) + 1

            st.session_state.chat_log.append(assistant_entry)
            st.rerun()
