"""
Conversation agent: drives the Anthropic API, owns the tool-use loop, and
hands off geometry generation to the registered templates.

Two tools:
- `set_production_context` — records how the user plans to make the part.
  Updates `Agent.production_context`. Recognized printer model names are
  resolved against the registry to fill in method + nozzle_dia automatically.
- `generate_part` — validates parameters, runs the geometry function,
  exports STL + STEP, and runs the manufacturing checks for the active
  production context. The check results are surfaced back to the model in
  the tool_result so the agent can comment on warnings or failures.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterable, Literal

from anthropic import Anthropic
from build123d import Part
from dotenv import load_dotenv

from conversation.system_prompt import build_system_prompt
from manufacturing.checks import CheckResult, overall_status, run_checks
from manufacturing.context import (
    Method,
    ProductionContext,
    looks_unsure,
    resolve_printer,
    thresholds_for,
)
from manufacturing.export import export_step, export_stl
from manufacturing.visualize import annotate_check_thumbnails
from templates.registry import get_template, list_templates


MODEL_OPUS = "claude-opus-4-7"
MODEL_SONNET = "claude-sonnet-4-5-20250929"
MODEL_HAIKU = "claude-haiku-4-5-20251001"
ALL_MODELS = (MODEL_OPUS, MODEL_SONNET, MODEL_HAIKU)
DEFAULT_MODEL = MODEL_SONNET  # the safety-net fallback in select_model

MAX_TOKENS = 4096
MAX_TOOL_ROUNDS = 6


Phase = Literal["routing", "production_context", "parameter", "post_generation"]


@dataclass
class GeneratedPart:
    template_name: str
    params: object
    part: Part
    stl_path: Path
    step_path: Path
    stl_size: int
    step_size: int
    production_context: ProductionContext
    check_results: list[CheckResult]
    overall_status: str


@dataclass
class AgentTurn:
    text: str
    generated_part: GeneratedPart | None = None
    context_changed: ProductionContext | None = None
    tool_attempts: list[dict] = field(default_factory=list)
    # Models used during this turn — one entry per API call (the tool-use
    # loop can fire multiple). The first entry is the model selected at the
    # start of the turn; subsequent entries are the same unless we ever
    # decide to re-route mid-loop (we currently do not).
    models_used: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------


@dataclass
class RoutingState:
    """Inputs to select_model. Pure data; testable without an API key."""

    phase: Phase
    has_image: bool
    ambiguous_context: bool
    pushback_likely: bool


def select_model(state: RoutingState) -> str:
    """Per-turn model routing. Rules in priority order — first match wins.

    1. Image attached → Opus (vision benefits from the strongest model).
    2. Routing phase → Opus (engineering judgment matters most here).
    3. Ambiguous production-context answer → Opus.
    4. Manufacturing pushback likely → Opus (needs concrete reasoning).
    5. Parameter phase, clean numeric answer → Haiku (transcription is cheap).
    6. Otherwise → Sonnet (safety-net default).
    """
    if state.has_image:
        return MODEL_OPUS
    if state.phase == "routing":
        return MODEL_OPUS
    if state.phase == "production_context" and state.ambiguous_context:
        return MODEL_OPUS
    if state.pushback_likely:
        return MODEL_OPUS
    if state.phase == "parameter":
        return MODEL_HAIKU
    return DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Phase + signal detectors
# ---------------------------------------------------------------------------


def _user_message_count(messages: list[dict]) -> int:
    """Count human-text user messages, ignoring tool_result-only messages."""
    n = 0
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, list):
            # Tool-result-only messages don't count as a "user turn".
            if all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                continue
        n += 1
    return n


def _has_successful_generation(messages: list[dict]) -> bool:
    """True if any prior tool_result indicated a generate_part success."""
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_result" and not b.get("is_error", False):
                # Successful tool_result; could be either tool. We want
                # generate_part specifically.
                text = b.get("content") or ""
                if isinstance(text, list):
                    text = " ".join(t.get("text", "") if isinstance(t, dict) else str(t) for t in text)
                if "Part generated" in text:
                    return True
    return False


def _detect_phase(messages: list[dict], production_context: ProductionContext | None) -> Phase:
    """Approximate phase from the conversation history.

    Called BEFORE the new user message is appended, so:
    - 0 prior user messages → this is the opener (routing).
    - >= 1 prior user message + no context yet → production_context.
    - >= 1 prior user message + context set + no generation → parameter.
    - any successful generation in the past → post_generation.
    """
    if _has_successful_generation(messages):
        return "post_generation"
    user_count = _user_message_count(messages)
    if user_count == 0:
        return "routing"
    if production_context is None:
        return "production_context"
    return "parameter"


_METHOD_KEYWORDS = (
    "fdm", "sla", "cnc", "machin", "inject", "shapeway", "xometry",
    "filament", "resin", "aluminum", "aluminium", "steel", "stainless",
    "petg", "pla", "abs", "tpu", "nylon", "brass", "copper",
    "online", "service",
)


def _looks_ambiguous_context_answer(text: str) -> bool:
    """True if the user's production-context reply doesn't unambiguously
    name a known printer or method keyword."""
    if not text:
        return True
    if looks_unsure(text):
        return True
    if resolve_printer(text) is not None:
        return False
    needle = text.lower()
    return not any(k in needle for k in _METHOD_KEYWORDS)


_NUM_MM = re.compile(r"(\d+(?:\.\d+)?)\s*(?:mm|millimet[er]+s?)\b", re.IGNORECASE)


def _pushback_likely(text: str, context: ProductionContext | None) -> bool:
    """True if the user's message names a sub-threshold dimension that
    will trigger our manufacturability rules and force the agent into
    pushback territory."""
    if not text or context is None:
        return False
    t = thresholds_for(context)
    cutoff = t.wall_warn_below
    for match in _NUM_MM.findall(text):
        try:
            v = float(match)
        except ValueError:
            continue
        if 0 < v < cutoff:
            return True
    return False


def build_routing_state(
    messages: list[dict],
    production_context: ProductionContext | None,
    user_text: str,
    has_image: bool,
) -> RoutingState:
    """Assemble the RoutingState for select_model from the agent's state."""
    phase = _detect_phase(messages, production_context)
    return RoutingState(
        phase=phase,
        has_image=has_image,
        ambiguous_context=(
            phase == "production_context" and _looks_ambiguous_context_answer(user_text)
        ),
        pushback_likely=_pushback_likely(user_text, production_context),
    )


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------


def _set_production_context_tool() -> dict:
    return {
        "name": "set_production_context",
        "description": (
            "Record how the user plans to manufacture the part. Call this "
            "after the user answers the production-method question, before "
            "asking template parameters. You can call it again later if the "
            "user changes their mind. Recognized printer model names "
            "automatically resolve to method + nozzle when omitted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": [m.value for m in Method],
                    "description": (
                        "Manufacturing method. Use 'fdm' for filament 3D "
                        "printing at home, 'sla' for resin, "
                        "'online_service' for Shapeways/JLCPCB-style "
                        "services where method is unknown, 'cnc_plastic' "
                        "or 'cnc_metal' for machining, "
                        "'injection_molding' for tooling-based production."
                    ),
                },
                "material": {
                    "type": "string",
                    "description": "Free text — 'PLA', 'PETG', 'aluminum', 'stainless steel', etc.",
                },
                "nozzle_dia": {
                    "type": "number",
                    "description": "Nozzle diameter in mm for FDM. Default 0.4 if unknown.",
                },
                "printer_model": {
                    "type": "string",
                    "description": "Canonical printer name if the user named a model.",
                },
                "notes": {
                    "type": "string",
                    "description": "Anything else worth remembering about the production setup.",
                },
            },
            "required": ["method"],
        },
    }


def _generate_part_tool() -> dict:
    template_names = [s.name for s in list_templates()]
    return {
        "name": "generate_part",
        "description": (
            "Produce a parametric 3D part using one of the registered "
            "templates. The geometry layer validates parameters and runs "
            "the manufacturing checks for the current production context. "
            "Validation failures and check warnings come back in the tool "
            "result — read them and respond accordingly."
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
                        "Omitted fields fall back to template defaults."
                    ),
                },
            },
            "required": ["template", "params"],
        },
    }


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def _coerce_method(value) -> Method | None:
    if isinstance(value, Method):
        return value
    if not isinstance(value, str):
        return None
    try:
        return Method(value.lower().strip())
    except ValueError:
        return None


def _set_context_handler(tool_input: dict) -> tuple[bool, str, ProductionContext | None]:
    raw_method = tool_input.get("method")
    method = _coerce_method(raw_method)

    raw_printer = tool_input.get("printer_model")
    printer_info = resolve_printer(raw_printer) if raw_printer else None
    if printer_info is not None:
        # Printer match overrides method/nozzle if the agent guessed them differently.
        if method is None:
            method = printer_info.method
        nozzle = tool_input.get("nozzle_dia") or printer_info.nozzle_dia
        printer_canonical = printer_info.canonical_name
    else:
        nozzle = tool_input.get("nozzle_dia")
        printer_canonical = raw_printer

    if method is None:
        return False, (
            f"Could not interpret method='{raw_method}'. "
            f"Use one of: {', '.join(m.value for m in Method)}."
        ), None

    context = ProductionContext(
        method=method,
        material=(tool_input.get("material") or None),
        nozzle_dia=nozzle if nozzle is not None else None,
        printer_model=printer_canonical,
        notes=tool_input.get("notes", "") or "",
    )

    summary_bits = [f"method={context.method.value}"]
    if context.material:
        summary_bits.append(f"material={context.material}")
    if context.nozzle_dia is not None:
        summary_bits.append(f"nozzle={context.nozzle_dia}mm")
    if context.printer_model:
        summary_bits.append(f"printer={context.printer_model}")
    return True, (
        "Production context recorded: "
        + ", ".join(summary_bits)
        + ". Now ask the template-specific measurement questions."
    ), context


def _check_results_to_text(results: list[CheckResult]) -> str:
    if not results:
        return "(no checks ran)"
    lines = []
    for r in results:
        line = f"- [{r.status.upper()}] {r.name}: {r.message}"
        if r.suggestion:
            line += f" (suggestion: {r.suggestion})"
        lines.append(line)
    return "\n".join(lines)


def _generate_part_handler(
    tool_input: dict,
    output_dir: Path,
    context: ProductionContext,
) -> tuple[bool, str, GeneratedPart | None]:
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

    expected = {f.name for f in fields(spec.params_class)}
    unknown = sorted(set(raw_params.keys()) - expected)
    if unknown:
        return (
            False,
            (
                f"These keys do not belong on the {template_name} schema: "
                f"{unknown}. Allowed keys: {sorted(expected)}. "
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
    stl_path = export_stl(part, output_dir / f"{template_name}_agent.stl")
    step_path = export_step(part, output_dir / f"{template_name}_agent.step")

    # Run manufacturing checks for the active production context.
    check_results = run_checks(template_name, params, part, context)
    overall = overall_status(check_results)

    # Render annotation thumbnails for any warn/fail check that pinpointed
    # problem faces. Cached under output_dir/thumbnails/, keyed on the STL
    # bytes + check name + face fingerprints — same part + same warning
    # reuses the prior PNG.
    try:
        annotate_check_thumbnails(
            check_results, part, stl_path, output_dir / "thumbnails"
        )
    except Exception:
        pass

    summary = {
        "template": template_name,
        "params": asdict(params),
        "stl_size_bytes": stl_path.stat().st_size,
        "step_size_bytes": step_path.stat().st_size,
        "production_method": context.method.value,
        "manufacturing_status": overall,
    }

    msg = (
        "Part generated. Confirm to the user with a brief natural-language "
        "summary (no JSON). If any check is WARN or FAIL, surface that "
        "concisely with the suggested fix. Details:\n"
        + json.dumps(summary)
        + "\n\nManufacturing checks:\n"
        + _check_results_to_text(check_results)
    )
    return (
        True,
        msg,
        GeneratedPart(
            template_name=template_name,
            params=params,
            part=part,
            stl_path=stl_path,
            step_path=step_path,
            stl_size=stl_path.stat().st_size,
            step_size=step_path.stat().st_size,
            production_context=context,
            check_results=check_results,
            overall_status=overall,
        ),
    )


# ---------------------------------------------------------------------------
# Multi-modal user content
# ---------------------------------------------------------------------------


def _user_content_blocks(text: str, images: Iterable[tuple[str, bytes]] | None) -> list[dict]:
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
    out: list[dict] = []
    for block in content_blocks:
        if hasattr(block, "model_dump"):
            out.append(block.model_dump(exclude_none=True, by_alias=False))
        else:
            out.append(dict(block))
    return out


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    def __init__(self, output_dir: Path | str = "output", api_key: str | None = None):
        load_dotenv()
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to .env (see .env.example)."
            )
        self.client = Anthropic(api_key=resolved_key)
        self.system_prompt = build_system_prompt()
        self.tools = [_set_production_context_tool(), _generate_part_tool()]
        self.output_dir = Path(output_dir)
        self.messages: list[dict] = []
        # Production context starts unset. The agent should always set it
        # before generation; if the user signals "not sure", the agent
        # records the safe default.
        self.production_context: ProductionContext | None = None
        # Per-turn model routing telemetry.
        self.last_model: str | None = None
        self.last_routing_state: RoutingState | None = None
        self.model_call_counts: dict[str, int] = {m: 0 for m in ALL_MODELS}

    def reset(self) -> None:
        self.messages = []
        self.production_context = None
        self.last_model = None
        self.last_routing_state = None
        self.model_call_counts = {m: 0 for m in ALL_MODELS}

    def effective_context(self) -> ProductionContext:
        """Context to use for generation. Falls back to safe defaults if unset."""
        return self.production_context or ProductionContext.safe_default()

    def send(
        self,
        text: str,
        images: Iterable[tuple[str, bytes]] | None = None,
    ) -> AgentTurn:
        # Materialize the images iterable up front — `Iterable` may be a
        # generator and we need to inspect it for routing AND pass it to
        # the user-content builder.
        images_list = list(images) if images is not None else None

        # Route BEFORE appending the user message so phase detection looks
        # at the conversation as it was when the user spoke.
        routing_state = build_routing_state(
            self.messages,
            self.production_context,
            user_text=text,
            has_image=bool(images_list),
        )
        model = select_model(routing_state)
        self.last_routing_state = routing_state

        self.messages.append({"role": "user", "content": _user_content_blocks(text, images_list)})
        return self._run_tool_loop(model)

    def _run_tool_loop(self, model: str) -> AgentTurn:
        emitted_text: list[str] = []
        generated_part: GeneratedPart | None = None
        context_changed: ProductionContext | None = None
        tool_attempts: list[dict] = []
        models_used: list[str] = []

        for _ in range(MAX_TOOL_ROUNDS):
            self.last_model = model
            self.model_call_counts[model] = self.model_call_counts.get(model, 0) + 1
            models_used.append(model)
            response = self.client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt,
                tools=self.tools,
                messages=self.messages,
            )

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
                if tu.name == "set_production_context":
                    ok, msg, ctx = _set_context_handler(tu.input)
                    tool_attempts.append(
                        {"ok": ok, "tool": tu.name, "input": tu.input, "message": msg}
                    )
                    if ok and ctx is not None:
                        self.production_context = ctx
                        context_changed = ctx
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": msg,
                            "is_error": not ok,
                        }
                    )
                elif tu.name == "generate_part":
                    ok, msg, gp = _generate_part_handler(
                        tu.input, self.output_dir, self.effective_context()
                    )
                    tool_attempts.append(
                        {"ok": ok, "tool": tu.name, "input": tu.input, "message": msg[:300]}
                    )
                    if ok and gp is not None:
                        generated_part = gp
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": msg,
                            "is_error": not ok,
                        }
                    )
                else:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": f"Unknown tool: {tu.name}",
                            "is_error": True,
                        }
                    )
                    tool_attempts.append({"ok": False, "tool": tu.name, "error": "unknown tool"})

            self.messages.append({"role": "user", "content": tool_results})

        return AgentTurn(
            text="\n\n".join(t for t in emitted_text if t.strip()),
            generated_part=generated_part,
            context_changed=context_changed,
            tool_attempts=tool_attempts,
            models_used=models_used,
        )
