"""
End-to-end self-test for the three flows from CLAUDE.md Step 3.

These hit the live Anthropic API. Skipped if ANTHROPIC_API_KEY isn't set.
The point is structural: each flow must (a) reach a successful tool call
and (b) produce a part of the expected template type without crashing.

Run with: pytest tests/test_e2e_flows.py -v -s
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from PIL import Image, ImageDraw

from conversation.agent import Agent, AgentTurn


load_dotenv()
SKIP_REASON = "ANTHROPIC_API_KEY not set"


def _drive(agent: Agent, scripted_replies: list[str], opener: str, images=None, max_turns: int = 6) -> AgentTurn:
    """Send the opener, then scripted replies until a part is generated or turns run out."""
    turn = agent.send(opener, images=images)
    print(f"\n[opener] {opener[:120]}")
    print(f"[assistant] {turn.text[:300]}")
    for reply in scripted_replies:
        if turn.generated_part is not None:
            return turn
        print(f"\n[user] {reply[:120]}")
        turn = agent.send(reply)
        print(f"[assistant] {turn.text[:300]}")
        for fail in turn.tool_attempts:
            if not fail["ok"]:
                print(f"  [tool failure] {fail['message'][:200]}")
    return turn


def _make_test_image() -> bytes:
    """Synthesize a small PNG that vaguely resembles an L-bracket so the
    agent has plausible visual context. Returns PNG bytes."""
    img = Image.new("RGB", (320, 320), "white")
    draw = ImageDraw.Draw(img)
    # Two perpendicular grey rectangles meeting at a corner.
    draw.rectangle((40, 200, 280, 260), fill=(140, 140, 140), outline=(40, 40, 40), width=3)
    draw.rectangle((40, 60, 100, 260), fill=(140, 140, 140), outline=(40, 40, 40), width=3)
    # A few small circles to suggest mounting holes.
    for cx in (70, 230):
        draw.ellipse((cx - 6, 224, cx + 6, 236), fill="white", outline=(40, 40, 40), width=2)
    for cy in (90, 160):
        draw.ellipse((64, cy - 6, 76, cy + 6), fill="white", outline=(40, 40, 40), width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason=SKIP_REASON)
def test_flow_bottle_holder_text_only():
    """Flow 1: bottle holder via plain text intake."""
    agent = Agent(output_dir="output")
    turn = _drive(
        agent,
        opener=(
            "I need a bottle holder for my daughter's Power Wheels. The bottle is "
            "63mm wide and 230mm tall, and I want to clamp it to a 28mm roll bar."
        ),
        scripted_replies=[
            "Use 120mm cup height and a 66mm cup ID. Defaults are fine for the rest. Generate it.",
            "Yes, generate it now.",
        ],
    )
    assert turn.generated_part is not None, (
        f"Bottle holder flow did not produce a part. Last text: {turn.text[:300]}"
    )
    gp = turn.generated_part
    assert gp.template_name == "bottle_holder", f"Wrong template: {gp.template_name}"
    assert gp.stl_path.exists() and gp.file_size > 1024


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason=SKIP_REASON)
def test_flow_hook_via_routing():
    """Flow 2: vague problem statement, agent must route to hook."""
    agent = Agent(output_dir="output")
    turn = _drive(
        agent,
        opener="I need to hang my broom in the garage.",
        scripted_replies=[
            "I'll mount it to a flat drywall surface with screws. The plate "
            "can be 30mm square with 4.5mm screw holes.",
            "Arm length 50mm, hook radius 12mm, opening 18mm, wall thickness 3mm. Generate it.",
            "Yes, generate it.",
        ],
    )
    assert turn.generated_part is not None, (
        f"Hook flow did not produce a part. Last text: {turn.text[:300]}"
    )
    gp = turn.generated_part
    assert gp.template_name == "hook", f"Wrong template: {gp.template_name}"
    assert gp.stl_path.exists() and gp.file_size > 1024


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason=SKIP_REASON)
def test_flow_image_bracket():
    """Flow 3: image upload + 'bracket for this'. Image is context only."""
    img_bytes = _make_test_image()
    agent = Agent(output_dir="output")
    turn = _drive(
        agent,
        opener=(
            "I want a bracket for this. Plate A should be 80x30mm, plate B "
            "60x30mm, both 4mm thick. Two 4.5mm holes per plate, with a "
            "stiffening gusset."
        ),
        images=[("image/png", img_bytes)],
        scripted_replies=[
            "Yes, those numbers are correct. Generate it.",
            "Generate it.",
        ],
    )
    assert turn.generated_part is not None, (
        f"Bracket flow did not produce a part. Last text: {turn.text[:300]}"
    )
    gp = turn.generated_part
    assert gp.template_name == "bracket", f"Wrong template: {gp.template_name}"
    assert gp.stl_path.exists() and gp.file_size > 1024
