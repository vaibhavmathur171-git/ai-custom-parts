# Project: Conversational Parametric CAD for Everyday Problems

## Mission

A web app where someone describes a real-world problem in plain English, has a short conversation with an AI that asks engineer-grade follow-up questions, and gets back a 3D-printable file (STL) and a CAD-grade file (STEP) for a parametric solid that actually solves the problem. Manufacturing constraints are checked deterministically before the file is offered for download.

Hero demo: a custom bottle holder for a child's Power Wheels-class ride-on car.

## Target user

Someone who has a specific physical problem in their home and would never install Fusion or learn CAD. They know what they need but cannot translate that into a manufacturable file. Today they search Amazon, settle for a generic product, or give up. The product is for them, not for engineers.

## Hard architectural rule

**The LLM does not generate geometry code.** The LLM's job is to behave as a smart product engineer doing structured intake — asking the right questions, extracting structured parameters from the user's responses, and calling deterministic, hand-tuned geometry functions.

Geometry functions are hand-coded by humans, version-controlled, and tested. The LLM selects which template to use and what parameter values to pass. It does not write `build123d` code at runtime.

This is non-negotiable. It is the architectural choice that makes the system reliable, explains-to-user, and defensible against foundation models that can already generate raw CAD code. The TDD will explicitly defend this choice.

## Tech stack

- Python 3.11+
- `build123d` for parametric solid geometry (B-Rep, OpenCascade backend)
- `streamlit` for the web UI
- `anthropic` Python SDK for the LLM intake layer
- `streamlit-stl` or embedded `<model-viewer>` for browser rendering
- Use `claude-opus-4-7` model string (Claude Opus 4.7) for the conversation agent

Avoid heavy dependencies. No Docker, no databases, no cloud services. Everything runs locally. Streamlit Cloud is the deploy target.

## File structure

```
cad-design-ai/
├── CLAUDE.md                    # this file
├── README.md                    # human-facing project overview
├── requirements.txt
├── .env.example                 # ANTHROPIC_API_KEY placeholder
├── .gitignore
├── app.py                       # Streamlit entry point
├── conversation/
│   ├── __init__.py
│   ├── agent.py                 # Claude API wrapper, conversation state
│   └── system_prompt.py         # the engineer-personality prompt
├── templates/
│   ├── __init__.py
│   ├── registry.py              # template name → function mapping
│   └── bottle_holder.py         # the hero template
├── manufacturing/
│   ├── __init__.py
│   ├── checks.py                # wall thickness, overhang, drain
│   └── export.py                # STL + STEP export wrappers
├── viewer/
│   └── render.py                # solid → GLB or STL for browser
└── tests/
    └── test_bottle_holder.py    # geometry sanity tests
```

## Build sequence

Work through these in order. Do not jump ahead. Each step ends with a runnable artifact.

### Step 1 — Hand-coded bottle holder (no AI yet)
Implement `templates/bottle_holder.py` with a single function `make_bottle_holder(params: BottleHolderParams) -> Part` returning a `build123d` solid. Parameters are dataclass-defined.

Acceptance: running `python -m templates.bottle_holder` writes `output/bottle_holder.stl` with sensible defaults, viewable in any STL viewer.

### Step 2 — Streamlit shell with sliders (no AI yet)
Build `app.py` with the parameter sliders directly bound to the bottle holder function. Render the resulting STL inline.

Acceptance: `streamlit run app.py` shows a working parametric configurator. Dragging sliders updates the 3D model in under 2 seconds.

### Step 3 — Conversation agent
Implement `conversation/agent.py`. Claude is prompted with the engineer-personality system prompt. It asks questions, builds up a parameter dict, and emits a structured JSON object when ready to generate. The Streamlit chat UI replaces the sliders.

Acceptance: a user can type "I need a bottle holder for my daughter's Power Wheels" and the agent walks them through measurement questions and produces a generated solid.

### Step 4 — Manufacturing checks + STEP export
Implement `manufacturing/checks.py` with:
- `min_wall_thickness(part, threshold=1.0)` — flags faces too thin to print
- `has_drain(part)` — confirms drainage feature exists for a cup
- `check_overhangs(part, max_angle=45)` — for FDM printability

Implement `manufacturing/export.py` with STL and STEP export. Surface check results in the Streamlit sidebar. Add download buttons for both formats.

Acceptance: after generation, the user sees pass/warn/fail indicators and can download both file types.

### Step 5 — Template registry + iteration loop
Refactor `templates/registry.py` so templates are registered by name with metadata (description, parameter schema, default values). The agent selects from the registry. After initial generation, the user can iterate: "make the cup deeper" or "design one for a 38mm bar instead" — the same parameters update, no regeneration from scratch.

Acceptance: at least one round-trip iteration where the user modifies dimensions through chat, the model updates, parameters persist.

### Step 6 — Polish, demo video, docs
PRD (1-2 pages), TDD (1-2 pages), 90-second screen recording, README. Deploy to Streamlit Cloud.

## Bottle holder template specification

This is the hero template. Implement it precisely.

### Geometry

A clamp-on bottle holder with three regions:

1. **Cup**: hollow cylindrical sleeve. Inner diameter `cup_id`, outer diameter `cup_id + 2 * wall_t`, height `cup_height`. Closed bottom with a drainage hole of diameter `drain_dia` centered on the floor.

2. **Standoff arm**: a horizontal rectangular bridge connecting the cup wall to the clamp. Length `standoff`, width `arm_width` (default 20mm), thickness `wall_t`. Connects at the midpoint of the cup's outer wall, perpendicular to the cup axis.

3. **Clamp**: a C-shaped ring with inner diameter `bar_dia`, outer diameter `bar_dia + 2 * wall_t`, height `clamp_height` (default 25mm). The C has a vertical slot of width `slot_width` (default 4mm) on the side opposite the standoff. A small lateral channel through both ends of the C accepts a zip tie or M4 bolt for tightening.

Boolean union the three regions into a single solid.

### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `bottle_dia` | float | 63.0 | Bottle diameter in mm (the 2.5" Thermos) |
| `cup_id` | float | 66.0 | Cup inner diameter (bottle_dia + 3mm clearance) |
| `cup_height` | float | 100.0 | Cup height — grips bottom 40% of a 230mm bottle |
| `wall_t` | float | 2.5 | Wall thickness, FDM-friendly |
| `drain_dia` | float | 8.0 | Drainage hole in cup floor |
| `bar_dia` | float | 28.0 | Roll bar diameter (Bronco Raptor toy default) |
| `clamp_height` | float | 25.0 | Vertical extent of the clamp |
| `slot_width` | float | 4.0 | Tightening slot width |
| `standoff` | float | 40.0 | Cup-to-clamp horizontal gap |
| `arm_width` | float | 20.0 | Standoff arm cross-section width |

### Validation rules

- `cup_id > bottle_dia + 1.0` (mm clearance, refuse otherwise)
- `wall_t >= 1.0` (printability floor; warn if below 1.5)
- `bar_dia >= 15.0 and bar_dia <= 60.0` (sanity range)
- `cup_height >= 40.0` (otherwise no meaningful grip)

## Conversation design

The agent's persona: a senior product engineer at a hardware company doing intake on a custom request. Curious, focused, asks one clear question at a time, never lectures, never asks for information already given.

The system prompt should:
- Define the engineer persona
- Describe the available templates from the registry (start with bottle_holder only)
- Specify the structured parameter schema the agent must produce when ready
- Instruct the agent to ask for measurements explicitly and never invent dimensions
- Allow the agent to suggest typical values ("most ride-on toys have 25-30mm roll bars — does that sound right for yours?") to reduce user friction
- Tell the agent to summarize the parameter set before generating ("Here's what I'm building: cup ID 66mm, cup height 100mm, clamping to a 28mm bar. Generate?")
- After generation, accept iterative edits and update the parameter dict in place — never restart the conversation

The agent communicates with the geometry layer via a structured JSON tool call. Define the tool schema in `conversation/agent.py`.

## Engineering principles for the agent

When working in this repository:

- **Small commits.** Each meaningful change in its own commit with a clear message.
- **Test geometry before claiming done.** Every template change must produce a valid STL that opens in a viewer. Don't trust that build123d succeeded — verify the file exists and has nonzero size.
- **Never let LLM output reach the geometry layer unchecked.** The conversation agent emits structured parameters; those parameters are validated against the template's schema before being passed to the geometry function. Validation failures should be surfaced back to the user as clarifying questions, not as crashes.
- **Prefer clarity over cleverness.** A geometry function with explicit named variables and comments is better than a compact one-liner. Future humans (and this Vaibhav) will read this code.
- **Surface failure modes.** If wall thickness is borderline, say so visibly. Do not hide warnings.
- **No silent fallbacks.** If the template registry is asked for a template it doesn't have, raise. Do not "best-effort" generate something else.

## Known constraints and non-goals

- **Not a general-purpose CAD tool.** The system handles only registered templates. Adding a new template requires hand-coding it. This is a feature, not a bug.
- **Photos are context, not measurement source.** Users can paste photos into the chat for the agent to see; the agent uses them to ask better questions, not to extract dimensions automatically.
- **Single-user, single-session.** No accounts, no persistence, no collaboration. Out of scope.
- **English only.** Out of scope for this build.

## First task

When invoked, your first action is Step 1. Specifically:

1. Create the directory structure above (empty files where appropriate, populated `.gitignore` and `requirements.txt`).
2. Implement `templates/bottle_holder.py` end-to-end with the spec above.
3. Add a `__main__` block that, when the module is run directly, generates an STL file at `output/bottle_holder.stl` using the default parameter values.
4. Run it. Confirm the file exists and is nonzero. Report the file size.
5. Stop and report. Do not proceed to Step 2 until the human reviews the STL.

Do not guess at build123d API. If you are uncertain about a function signature, check the build123d documentation (https://build123d.readthedocs.io/) before writing code.
