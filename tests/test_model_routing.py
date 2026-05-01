"""
Unit tests for model routing.

Direct, deterministic tests of select_model and build_routing_state. No API
calls — these run fast and lock in the routing rules.
"""

from __future__ import annotations

import pytest

from conversation.agent import (
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
    Phase,
    RoutingState,
    build_routing_state,
    select_model,
)
from manufacturing.context import Method, ProductionContext


# ---------------------------------------------------------------------------
# select_model: pure rule table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phase,has_image,ambiguous,pushback,expected",
    [
        # Image always wins — Opus regardless of phase.
        ("routing", True, False, False, MODEL_OPUS),
        ("parameter", True, False, False, MODEL_OPUS),
        ("post_generation", True, False, False, MODEL_OPUS),
        # Routing phase → Opus.
        ("routing", False, False, False, MODEL_OPUS),
        # Production context: Opus only when ambiguous; otherwise Sonnet.
        ("production_context", False, True, False, MODEL_OPUS),
        ("production_context", False, False, False, MODEL_SONNET),
        # Pushback always upgrades to Opus.
        ("parameter", False, False, True, MODEL_OPUS),
        ("post_generation", False, False, True, MODEL_OPUS),
        # Clean parameter answer → Haiku.
        ("parameter", False, False, False, MODEL_HAIKU),
        # Post-generation default → Sonnet.
        ("post_generation", False, False, False, MODEL_SONNET),
    ],
)
def test_select_model_rule_matrix(
    phase: Phase, has_image: bool, ambiguous: bool, pushback: bool, expected: str
):
    state = RoutingState(
        phase=phase,
        has_image=has_image,
        ambiguous_context=ambiguous,
        pushback_likely=pushback,
    )
    assert select_model(state) == expected


# ---------------------------------------------------------------------------
# build_routing_state: phase + signal detection from agent state
# ---------------------------------------------------------------------------


def test_routing_state_first_message_is_routing_phase():
    state = build_routing_state(
        messages=[],
        production_context=None,
        user_text="I need a bottle holder",
        has_image=False,
    )
    assert state.phase == "routing"
    assert select_model(state) == MODEL_OPUS


def test_routing_state_first_message_with_image_still_opus():
    """Routing + image both route to Opus, no conflict."""
    state = build_routing_state(
        messages=[],
        production_context=None,
        user_text="like this",
        has_image=True,
    )
    assert state.phase == "routing"
    assert state.has_image is True
    assert select_model(state) == MODEL_OPUS


def _opener_then_assistant() -> list[dict]:
    return [
        {"role": "user", "content": [{"type": "text", "text": "I need a bottle holder"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "asks about printer"}]},
    ]


def test_routing_state_clean_printer_answer_is_default_fallback():
    state = build_routing_state(
        messages=_opener_then_assistant(),
        production_context=None,
        user_text="Elegoo Centauri Carbon with PLA",
        has_image=False,
    )
    assert state.phase == "production_context"
    assert state.ambiguous_context is False
    assert select_model(state) == MODEL_SONNET


def test_routing_state_unsure_answer_is_opus():
    state = build_routing_state(
        messages=_opener_then_assistant(),
        production_context=None,
        user_text="not sure honestly",
        has_image=False,
    )
    assert state.phase == "production_context"
    assert state.ambiguous_context is True
    assert select_model(state) == MODEL_OPUS


def test_routing_state_vague_method_answer_is_opus():
    """Answer doesn't name a printer or any method keyword — ambiguous."""
    state = build_routing_state(
        messages=_opener_then_assistant(),
        production_context=None,
        user_text="just print it normally",
        has_image=False,
    )
    assert state.phase == "production_context"
    assert state.ambiguous_context is True
    assert select_model(state) == MODEL_OPUS


def test_routing_state_method_keyword_answer_is_default_fallback():
    """User mentions FDM by name without naming a printer — still concrete."""
    state = build_routing_state(
        messages=_opener_then_assistant(),
        production_context=None,
        user_text="FDM with PETG please",
        has_image=False,
    )
    assert state.phase == "production_context"
    assert state.ambiguous_context is False
    assert select_model(state) == MODEL_SONNET


def _opener_then_context_set() -> list[dict]:
    return [
        {"role": "user", "content": [{"type": "text", "text": "I need a bottle holder"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "asks about printer"}]},
        {"role": "user", "content": [{"type": "text", "text": "Bambu A1 with PLA"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "great — what diameter is the bottle?"}]},
    ]


def test_routing_state_clean_numeric_param_is_haiku():
    ctx = ProductionContext(method=Method.FDM, material="PLA", nozzle_dia=0.4)
    state = build_routing_state(
        messages=_opener_then_context_set(),
        production_context=ctx,
        user_text="63 mm",
        has_image=False,
    )
    assert state.phase == "parameter"
    assert state.pushback_likely is False
    assert select_model(state) == MODEL_HAIKU


def test_routing_state_subthreshold_wall_param_is_pushback_opus():
    ctx = ProductionContext(method=Method.FDM, material="PLA", nozzle_dia=0.4)
    state = build_routing_state(
        messages=_opener_then_context_set(),
        production_context=ctx,
        user_text="use 0.5mm wall please",
        has_image=False,
    )
    assert state.phase == "parameter"
    assert state.pushback_likely is True
    assert select_model(state) == MODEL_OPUS


def test_routing_state_param_phase_with_image_is_opus():
    """Image attachments always upgrade to Opus, even mid-parameter."""
    ctx = ProductionContext(method=Method.FDM, material="PLA", nozzle_dia=0.4)
    state = build_routing_state(
        messages=_opener_then_context_set(),
        production_context=ctx,
        user_text="like this",
        has_image=True,
    )
    assert state.has_image is True
    assert select_model(state) == MODEL_OPUS


def test_routing_state_pushback_threshold_is_method_aware():
    """Same wall value pushes back on CNC metal but not on FDM."""
    fdm = ProductionContext(method=Method.FDM)
    cnc = ProductionContext(method=Method.CNC_PLASTIC)

    # 1.3mm wall: above FDM warn (1.2), below CNC plastic warn (1.5).
    text = "use 1.3 mm walls"
    msgs = _opener_then_context_set()
    fdm_state = build_routing_state(msgs, fdm, text, False)
    cnc_state = build_routing_state(msgs, cnc, text, False)

    assert fdm_state.pushback_likely is False
    assert cnc_state.pushback_likely is True
    assert select_model(fdm_state) == MODEL_HAIKU
    assert select_model(cnc_state) == MODEL_OPUS
