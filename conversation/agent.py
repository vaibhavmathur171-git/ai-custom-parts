"""
Conversation agent: drives the Anthropic API, owns the tool-use loop, and
hands off geometry generation to the registered templates.

The agent's job is intake — routing the user to a template and gathering
parameters. The geometry layer is hand-coded; the agent never produces
CAD code. It calls the single tool `generate_part`, our handler validates
and dispatches, and the resulting Part / STL is returned to the UI.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable

from anthropic import Anthropic
from build123d import Part, export_stl
from dotenv import load_dotenv

from conversation.system_prompt import build_system_prompt
from templates.registry import get_template, list_templates


MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 4096
# Cap the tool-use loop so a misbehaving model can't run away with the
# session. Three to four rounds is usually plenty (call → error → retry → ok).
MAX_TOOL_ROUNDS = 6


@dataclass
class GeneratedPart:
    """Result of a successful generate_part tool call."""

    template_name: str
    params: object
    part: Part
    stl_path: Path
    file_size: int


@dataclass
class AgentTurn:
    """What the agent emitted on a single user message → response cycle.

    `text` is the concatenation of all assistant text blocks in this turn,
    suitable for rendering in the chat. `generated_part` is the most recent
    part successfully produced this turn (None if no successful generation).
    `tool_attempts` records every generate_part attempt, including failures,
    so the UI can surface error states transparently.
    """

    text: str
    generated_part: GeneratedPart | None = None
    tool_attempts: list[dict] = field(default_factory=list)


def _build_tool_schema() -> dict:
    """Single tool: generate_part(template, params)."""
    template_names = [s.name for s in list_templates()]
    return {
        "name": "generate_part",
        "description": (
            "Produce a parametric 3D part using one of the registered "
            "templates. The geometry layer validates parameters; if "
            "validation fails you'll receive an error in the tool result "
            "so you can ask the user a clarifying question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "template": {
                    "type": "string",
                    "enum": template_names,
                    "description": "Which template to use. Must be exactly one of the registered names.",
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Parameter values for the chosen template. Include "
                        "only fields that belong to that template's schema. "
                        "Omitted fields fall back to the template's defaults."
                    ),
                },
            },
            "required": ["template", "params"],
        },
    }


def _generate_part_handler(
    tool_input: dict,
    output_dir: Path,
) -> tuple[bool, str, GeneratedPart | None]:
    """Validate and execute a generate_part tool call.

    Returns (ok, message_for_agent, generated_part). The message_for_agent
    becomes the tool_result content the model sees on the next turn — keep
    it actionable (what failed, what to do about it).
    """
    template_name = tool_input.get("template")
    raw_params = tool_input.get("params", {})

    if not isinstance(template_name, str):
        return False, "Missing or non-string 'template' in tool input.", None
    if not isinstance(raw_params, dict):
        return False, "Missing or non-object 'params' in tool input.", None

    try:
        spec = get_template(template_name)
    except KeyError as e:
        return False, str(e), None

    expected_field_names = {f.name for f in fields(spec.params_class)}
    unknown = sorted(set(raw_params.keys()) - expected_field_names)
    if unknown:
        return (
            False,
            (
                f"These keys do not belong on the {template_name} schema: "
                f"{unknown}. Allowed keys are: {sorted(expected_field_names)}. "
                f"Drop the unknown keys and try again."
            ),
            None,
        )

    try:
        params = spec.params_class(**raw_params)
    except TypeError as e:
        return False, f"Could not construct {spec.params_class.__name__}: {e}", None

    errors = params.validate()
    if errors:
        return (
            False,
            (
                "Parameter validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
                + "\nAdjust the offending values and call generate_part again."
            ),
            None,
        )

    try:
        part = spec.make_fn(params)
    except Exception as e:
        return False, f"Geometry generation raised an exception: {e}", None

    output_dir.mkdir(exist_ok=True, parents=True)
    stl_path = output_dir / f"{template_name}_agent.stl"
    export_stl(part, str(stl_path))
    file_size = stl_path.stat().st_size

    summary = {
        "template": template_name,
        "params": asdict(params),
        "stl_size_bytes": file_size,
    }
    return (
        True,
        (
            "Part generated successfully. Confirm to the user with a brief "
            "natural-language summary; do not paste the JSON. Details: "
            + json.dumps(summary)
        ),
        GeneratedPart(
            template_name=template_name,
            params=params,
            part=part,
            stl_path=stl_path,
            file_size=file_size,
        ),
    )


def _user_content_blocks(text: str, images: Iterable[tuple[str, bytes]] | None) -> list[dict]:
    """Build a multimodal user-message content list.

    `images` is an iterable of (media_type, raw_bytes). Images come first
    so the model has visual context before the text question."""
    blocks: list[dict] = []
    if images:
        for media_type, raw_bytes in images:
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.standard_b64encode(raw_bytes).decode("ascii"),
                    },
                }
            )
    blocks.append({"type": "text", "text": text})
    return blocks


def _serialize_assistant_content(content_blocks) -> list[dict]:
    """Convert an SDK Message's content blocks to plain dicts for replay.

    Anthropic accepts either SDK objects or dicts on the next call; we use
    dicts so session_state stays JSON-friendly across Streamlit reruns.
    """
    out: list[dict] = []
    for block in content_blocks:
        if hasattr(block, "model_dump"):
            out.append(block.model_dump(exclude_none=True, by_alias=False))
        else:
            out.append(dict(block))
    return out


class Agent:
    """One agent per session. Holds the full conversation history.

    Streamlit re-runs the script on every interaction; persist the Agent in
    session_state so the history survives across reruns.
    """

    def __init__(self, output_dir: Path | str = "output", api_key: str | None = None):
        # Critical: load .env BEFORE constructing the Anthropic client so the
        # ANTHROPIC_API_KEY env var is set when Anthropic() reads it.
        load_dotenv()
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to .env (see .env.example)."
            )
        self.client = Anthropic(api_key=resolved_key)
        self.system_prompt = build_system_prompt()
        self.tool_schema = _build_tool_schema()
        self.output_dir = Path(output_dir)
        self.messages: list[dict] = []

    def reset(self) -> None:
        self.messages = []

    def send(
        self,
        text: str,
        images: Iterable[tuple[str, bytes]] | None = None,
    ) -> AgentTurn:
        """Send one user message and run the tool-use loop until the model ends its turn."""
        self.messages.append(
            {"role": "user", "content": _user_content_blocks(text, images)}
        )
        return self._run_tool_loop()

    def _run_tool_loop(self) -> AgentTurn:
        emitted_text: list[str] = []
        generated_part: GeneratedPart | None = None
        tool_attempts: list[dict] = []

        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt,
                tools=[self.tool_schema],
                messages=self.messages,
            )

            # Persist the assistant turn before processing tool calls so a
            # mid-loop crash leaves the conversation in a recoverable state.
            self.messages.append(
                {"role": "assistant", "content": _serialize_assistant_content(response.content)}
            )

            tool_uses = []
            for block in response.content:
                if block.type == "text":
                    emitted_text.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            if response.stop_reason != "tool_use":
                break

            tool_results: list[dict] = []
            for tu in tool_uses:
                if tu.name != "generate_part":
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": f"Unknown tool: {tu.name}",
                            "is_error": True,
                        }
                    )
                    tool_attempts.append({"ok": False, "tool": tu.name, "error": "unknown tool"})
                    continue

                ok, message, gp = _generate_part_handler(tu.input, self.output_dir)
                tool_attempts.append(
                    {
                        "ok": ok,
                        "tool": "generate_part",
                        "input": tu.input,
                        "message": message,
                    }
                )
                if ok and gp is not None:
                    generated_part = gp
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": message,
                        "is_error": not ok,
                    }
                )

            self.messages.append({"role": "user", "content": tool_results})

        return AgentTurn(
            text="\n\n".join(t for t in emitted_text if t.strip()),
            generated_part=generated_part,
            tool_attempts=tool_attempts,
        )
