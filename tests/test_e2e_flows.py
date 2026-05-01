"""
End-to-end self-test for the conversation flows.

These hit the live Anthropic API. Skipped if ANTHROPIC_API_KEY isn't set.
The point is structural: each flow must (a) reach a successful tool call,
(b) produce a part of the expected template type, and (c) for production-
context flows, populate Agent.production_context correctly.

Run with: pytest tests/test_e2e_flows.py -v -s
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from PIL import Image, ImageDraw

from conversation.agent import (
    Agent,
    AgentTurn,
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
)
from manufacturing.context import Method


load_dotenv()
SKIP_REASON = "ANTHROPIC_API_KEY not set"


def _drive(
    agent: Agent,
    opener: str,
    scripted_replies: list[str],
    images=None,
    stop_on_part: bool = True,
) -> AgentTurn:
    """Send the opener and scripted replies. Returns the most recent turn.

    By default stops as soon as a part is generated. Set stop_on_part=False
    to keep going (useful for tests that check pushback before generation)."""
    turn = agent.send(opener, images=images)
    print(f"\n[opener] {opener[:120]}")
    print(f"[assistant] {turn.text[:300]}")
    for reply in scripted_replies:
        if stop_on_part and turn.generated_part is not None:
            return turn
        print(f"\n[user] {reply[:120]}")
        turn = agent.send(reply)
        print(f"[assistant] {turn.text[:300]}")
        if agent.production_context is not None:
            print(f"  [context] {agent.production_context}")
        for fail in turn.tool_attempts:
            if not fail["ok"]:
                print(f"  [tool failure] {fail['message'][:200]}")
    return turn


def _make_test_image() -> bytes:
    img = Image.new("RGB", (320, 320), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 200, 280, 260), fill=(140, 140, 140), outline=(40, 40, 40), width=3)
    draw.rectangle((40, 60, 100, 260), fill=(140, 140, 140), outline=(40, 40, 40), width=3)
    for cx in (70, 230):
        draw.ellipse((cx - 6, 224, cx + 6, 236), fill="white", outline=(40, 40, 40), width=2)
    for cy in (90, 160):
        draw.ellipse((64, cy - 6, 76, cy + 6), fill="white", outline=(40, 40, 40), width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason=SKIP_REASON)
def test_flow_bottle_holder_with_production_context():
    """Bottle holder via text — agent must (a) ask production context,
    (b) record an FDM/PLA/Centauri Carbon context after the user answers,
    (c) generate a valid part with manufacturing checks attached."""
    agent = Agent(output_dir="output")
    turn = _drive(
        agent,
        opener=(
            "I need a bottle holder for my daughter's Power Wheels. The bottle "
            "is 63mm wide and 230mm tall, and I want to clamp it to a 28mm roll bar."
        ),
        scripted_replies=[
            "Elegoo Centauri Carbon with PLA.",
            "120mm tall, 66mm cup ID. Use defaults for everything else. Generate.",
            "Yes, generate it now.",
        ],
    )
    assert agent.production_context is not None, (
        "Agent must call set_production_context after the user names a printer"
    )
    assert agent.production_context.method == Method.FDM
    assert agent.production_context.printer_model == "Elegoo Centauri Carbon"
    assert agent.production_context.material is not None
    assert "PLA" in agent.production_context.material

    assert turn.generated_part is not None, (
        f"Bottle holder did not generate. Last text: {turn.text[:300]}"
    )
    gp = turn.generated_part
    assert gp.template_name == "bottle_holder"
    assert gp.stl_path.exists() and gp.stl_size > 1024
    assert gp.step_path.exists() and gp.step_size > 1024
    # Manufacturing checks must run
    assert gp.check_results, "Expected manufacturing checks to be attached"
    check_names = {r.name for r in gp.check_results}
    assert "Minimum wall thickness" in check_names
    # Drainage check is bottle-holder-specific
    assert any("Drainage" in n for n in check_names)


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason=SKIP_REASON)
def test_flow_thin_wall_pushback():
    """User asks for sub-threshold wall thickness. Agent must push back
    in helpful-expertise tone — concrete recommendation, not bureaucratic.
    The agent should NOT call generate_part with the bad value."""
    agent = Agent(output_dir="output")
    turn = _drive(
        agent,
        opener=(
            "I need a small L-bracket for under-shelf cable management. "
            "I'll print on a Bambu A1 with PLA."
        ),
        scripted_replies=[
            "Plates 60×30mm, with 0.5mm thickness please. Two holes per plate.",
        ],
        stop_on_part=False,
    )
    # Context should be set (the user named a printer + material in the opener).
    assert agent.production_context is not None
    assert agent.production_context.method == Method.FDM

    # The agent should NOT have generated a part with the 0.5mm wall — that's
    # the whole point of pushback. If it did generate, the test fails.
    successful_gens = [
        a for a in turn.tool_attempts
        if a.get("tool") == "generate_part" and a.get("ok")
    ]
    assert not successful_gens, (
        "Agent should have pushed back instead of generating with 0.5mm wall"
    )

    text_lower = turn.text.lower()
    # Helpful-expertise signals: agent recommends a concrete higher value,
    # mentions FDM concern, and offers to use the alternative.
    assert any(
        token in text_lower for token in ("0.8mm", "0.8 mm", "1.2", "2mm", "2.0", "2 mm", "3mm", "3 mm")
    ), f"Expected a concrete alternative wall value in the response: {turn.text}"
    assert any(
        token in text_lower for token in ("thin", "fdm", "fail", "flex", "weak", "print poorly")
    ), f"Expected pushback to mention why 0.5mm is bad: {turn.text}"


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason=SKIP_REASON)
def test_flow_routing_picks_right_model_per_turn():
    """Walk through a bottle holder flow turn-by-turn and confirm the
    routing rules fire at the right times: Opus for routing/pushback,
    Sonnet for clean production-context, Haiku for clean parameter answers."""
    agent = Agent(output_dir="output")

    # Turn 1 — opener (routing phase) → Opus.
    t1 = agent.send(
        "I need a bottle holder for my daughter's Power Wheels. "
        "The bottle is 63mm wide, the roll bar is 28mm."
    )
    assert t1.models_used, "expected at least one model call"
    assert t1.models_used[0] == MODEL_OPUS, (
        f"routing turn must use Opus, got {t1.models_used}"
    )

    # Turn 2 — clean production-context answer → Sonnet (default fallback).
    t2 = agent.send("Elegoo Centauri Carbon with PLA.")
    assert t2.models_used[0] == MODEL_SONNET, (
        f"clean production-context turn must use Sonnet, got {t2.models_used}"
    )
    assert agent.production_context is not None
    assert agent.production_context.method == Method.FDM

    # Turn 3 — sub-threshold wall request (pushback) → Opus.
    t3 = agent.send("Use 0.5mm walls please. 100mm tall, 66mm cup ID.")
    assert t3.models_used[0] == MODEL_OPUS, (
        f"pushback turn must use Opus, got {t3.models_used}"
    )
    # The agent should NOT have generated with the bad value.
    successful = [a for a in t3.tool_attempts if a.get("tool") == "generate_part" and a.get("ok")]
    assert not successful, "agent must push back instead of generating with 0.5mm wall"


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason=SKIP_REASON)
def test_flow_routing_unsure_production_context_uses_opus():
    """Ambiguous production reply ('not sure') must route to Opus."""
    agent = Agent(output_dir="output")
    agent.send("I want a hook to hang my keys by the door.")
    t2 = agent.send("Not sure honestly — whatever's easiest.")
    assert t2.models_used[0] == MODEL_OPUS, (
        f"ambiguous production-context turn must use Opus, got {t2.models_used}"
    )


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason=SKIP_REASON)
def test_flow_routing_image_attachment_uses_opus():
    """Any turn with an image attached must use Opus regardless of phase."""
    img_bytes = _make_test_image()
    agent = Agent(output_dir="output")
    t = agent.send("What template would fit this?", images=[("image/png", img_bytes)])
    assert t.models_used[0] == MODEL_OPUS, (
        f"image-attachment turn must use Opus, got {t.models_used}"
    )
    # Counter on the agent should reflect at least one Opus call.
    assert agent.model_call_counts[MODEL_OPUS] >= 1


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason=SKIP_REASON)
def test_flow_bottle_holder_overhang_warning_produces_thumbnail():
    """A successful bottle holder generation should produce annotated
    thumbnails for any warn/fail check that pinpointed problem geometry.
    Default-ish FDM bottle holder always has an overhang warn — verify the
    PNG made it onto the CheckResult."""
    agent = Agent(output_dir="output")
    turn = _drive(
        agent,
        opener=(
            "I need a bottle holder for my daughter's Power Wheels — bottle is "
            "63mm wide and the roll bar is 28mm."
        ),
        scripted_replies=[
            "Elegoo Centauri Carbon with PLA.",
            "120mm tall, 66mm cup ID. Use defaults for everything else. Generate.",
            "Yes, generate it now.",
        ],
    )
    assert turn.generated_part is not None
    gp = turn.generated_part
    overhangs = [r for r in gp.check_results if "Overhang" in r.name]
    assert overhangs, f"Expected an Overhangs check, got {[r.name for r in gp.check_results]}"
    overhang = overhangs[0]
    assert overhang.status in ("warn", "fail"), (
        f"Default bottle holder should fire an overhang warning; got {overhang.status}"
    )
    assert overhang.problem_faces, "Overhang warn must carry problem_faces"
    assert overhang.thumbnail_path, (
        f"Overhang warn must carry a thumbnail_path: {overhang}"
    )
    p = Path(overhang.thumbnail_path)
    assert p.exists() and p.stat().st_size > 1000
    assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason=SKIP_REASON)
def test_flow_square_bar_triggers_shape_pushback():
    """Bottle holder + square mounting bar → agent must flag the round-clamp
    limitation in its reply, not silently push ahead."""
    agent = Agent(output_dir="output")
    turn = _drive(
        agent,
        opener=(
            "I want a bottle holder for my treadmill — the grip bar I'm "
            "clamping to is square, about 25mm on a side."
        ),
        scripted_replies=[],
        stop_on_part=False,
    )
    text = turn.text.lower()
    # The agent must surface the shape concern: mentions round/circular,
    # OR explains it won't grip / will rock on corners.
    shape_signal = any(
        token in text
        for token in (
            "round", "circular", "rock", "corner", "grip", "won't fit",
            "doesn't fit", "won't grip", "non-round", "not round",
        )
    )
    assert shape_signal, (
        f"Agent should flag the square-bar shape limitation. Got: {turn.text}"
    )
    # And it must not have silently generated a part.
    successful = [
        a for a in turn.tool_attempts
        if a.get("tool") == "generate_part" and a.get("ok")
    ]
    assert not successful, "Agent must not silently generate a round clamp on a square bar"


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason=SKIP_REASON)
def test_flow_unknown_printer_sovol_inferred_as_fdm():
    """Sovol SV06 isn't in the registry. The agent must infer it's FDM
    (not refuse, not stall) and either set production context directly or
    proceed to parameter intake."""
    agent = Agent(output_dir="output")
    _drive(
        agent,
        opener="I want a hook to hang my dog leash by the back door.",
        scripted_replies=[
            "Flat plate that screws to the wall.",
            "I have a Sovol SV06 with PLA.",
        ],
        stop_on_part=False,
    )
    # The agent must either (a) call set_production_context with FDM, or
    # (b) at minimum not refuse / claim it doesn't recognize the printer.
    if agent.production_context is not None:
        assert agent.production_context.method == Method.FDM, (
            f"Sovol SV06 should infer to FDM; got {agent.production_context.method}"
        )
    else:
        # If context isn't set yet, the response must not be a refusal.
        # We check the assistant's most recent text.
        last_user_idx = max(
            (i for i, m in enumerate(agent.messages) if m.get("role") == "user"),
            default=-1,
        )
        recent_assistant_text = " ".join(
            block.get("text", "")
            for m in agent.messages[last_user_idx + 1:]
            if m.get("role") == "assistant"
            for block in (m.get("content") or [])
            if isinstance(block, dict) and block.get("type") == "text"
        ).lower()
        refusal_signals = ["don't recognize", "don't know that", "not in my list", "unable to"]
        assert not any(s in recent_assistant_text for s in refusal_signals), (
            f"Agent should not refuse on unknown printer. Got: {recent_assistant_text}"
        )


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason=SKIP_REASON)
def test_flow_image_bracket_with_cnc_metal():
    """Image + bracket request, but for CNC metal — exercises method
    routing through the production context phase."""
    img_bytes = _make_test_image()
    agent = Agent(output_dir="output")
    turn = _drive(
        agent,
        opener=(
            "I want a bracket for this. I'm going to send it to Xometry "
            "for CNC machining in aluminum."
        ),
        images=[("image/png", img_bytes)],
        scripted_replies=[
            "Plate A 80×30mm, plate B 60×30mm, 4mm thick, two 4.5mm holes per plate, with a gusset.",
            "Yes — although given CNC metal, please bump the plate width to 60mm so the holes have enough edge clearance.",
            "Use 60mm plate width on both, and yes, generate with the gusset.",
        ],
    )
    assert agent.production_context is not None
    assert agent.production_context.method == Method.CNC_METAL
    assert agent.production_context.material is not None
    assert "alum" in agent.production_context.material.lower()

    assert turn.generated_part is not None, (
        f"Bracket did not generate. Last text: {turn.text[:300]}"
    )
    gp = turn.generated_part
    assert gp.template_name == "bracket"
    # CNC metal must NOT include the FDM overhang check.
    check_names = {r.name for r in gp.check_results}
    assert not any("Overhang" in n for n in check_names), (
        f"CNC metal must not run overhang checks; got {check_names}"
    )
