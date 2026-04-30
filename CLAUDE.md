# Project: Conversational Parametric CAD for Everyday Problems

## Mission

A web app where someone describes a real-world problem in plain English (optionally with a photo or sketch), has a short conversation with an AI that asks engineer-grade follow-up questions, and gets back a 3D-printable file (STL) and a CAD-grade file (STEP) for a parametric solid that actually solves the problem. Manufacturing constraints are checked deterministically before the file is offered for download.

The wedge is **mounting and holding solutions for everyday objects** — clamps, brackets, hooks, holders. The hero demo is a custom bottle holder for a child's Power Wheels-class ride-on car.

## Target user

Someone who has a specific physical problem in their home and would never install Fusion or learn CAD. They know what they need but cannot translate that into a manufacturable file. Today they search Amazon, settle for a generic product, or give up. The product is for them, not for engineers.

## Hard architectural rule

**The LLM does not generate geometry code.** The LLM's job is to behave as a smart product engineer doing structured intake — listening to the user's problem, identifying which template fits, asking the right parameter questions, extracting structured parameters from the user's responses, and calling deterministic, hand-tuned geometry functions.

Geometry functions are hand-coded by humans, version-controlled, and tested. The LLM selects which template to use and what parameter values to pass. It does not write `build123d` code at runtime.

This is non-negotiable. It is the architectural choice that makes the system reliable, explains-to-user, and defensible against foundation models that can already generate raw CAD code. The TDD will explicitly defend this choice.

## Tech stack

- Python 3.11+
- `build123d` for parametric solid geometry (B-Rep, OpenCascade backend)
- `streamlit` for the web UI
- `streamlit-stl` for inline 3D rendering in the browser
- `anthropic` Python SDK for the LLM intake layer (Sonnet 4.5/4.6 for chat with vision; Opus is too slow and expensive for conversational use)
- `python-dotenv` for loading API keys from .env

Avoid heavy dependencies. No Docker, no databases, no cloud services. Everything runs locally. Streamlit Cloud is the deploy target.

## File structure

```
cad-design-ai/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── app.py
├── conversation/
│   ├── __init__.py
│   ├── agent.py
│   └── system_prompt.py
├── templates/
│   ├── __init__.py
│   ├── registry.py
│   ├── bottle_holder.py
│   ├── hook.py
│   └── bracket.py
├── manufacturing/
│   ├── __init__.py
│   ├── checks.py
│   └── export.py
├── viewer/
│   └── render.py
└── tests/
    └── test_templates.py
```

## Build sequence

### Step 1 — Hand-coded bottle holder ✅ DONE
`templates/bottle_holder.py` with `make_bottle_holder(params)`. Validates parameters, exports STL.

### Step 2A — Streamlit shell with sliders for bottle holder ✅ DONE
`app.py` with parameter sliders. Cup vertical, clamp horizontal. Validation surfaces as errors. STL renders inline.

### Step 2B — Template registry and library expansion (CURRENT)
Refactor `templates/` into a proper registry. Add two new templates: `hook.py` and `bracket.py`. Each template defines:
- A dataclass for its parameters (e.g., `HookParams`, `BracketParams`)
- A geometry function (e.g., `make_hook(params)`, `make_bracket(params)`)
- A `validate()` method on the params class
- A `meta` dict with `name`, `description`, `typical_use_cases`, `default_params`

Update `templates/registry.py` to register all three templates with metadata. Provide a function `get_template(name)` returning the geometry function and parameter class.

Update `app.py`:
- Add a template selector at the top of the sidebar (radio or dropdown).
- The parameter slider panel updates dynamically based on the selected template.
- The 3D viewer renders whichever template is currently active.
- Validation, STL download, and metadata panel all work for any selected template.

Acceptance: a user can pick any of the three templates from the dropdown, drag sliders specific to that template, and see a valid 3D model render. All three templates produce valid STL files at default parameters.

### Step 3 — Conversation agent with template routing
Implement `conversation/agent.py`. The agent's job is to route the user to the correct template and then collect parameters for it.

Two-phase conversation:
1. **Routing phase**: The agent identifies which template fits the user's stated problem. If it can't tell from the initial message, it asks one clarifying question about what they're trying to mount/hold/hang.
2. **Parameter phase**: Once a template is selected, the agent asks template-specific measurement questions one at a time, never skipping ahead.

Input supports text and images. Images are context only (Mode A) — the agent looks at uploaded photos or sketches to understand intent and ask better questions, but never extracts dimensions visually. Mode B (dimension extraction from images) is explicitly out of scope.

The agent emits a structured tool call when ready: `{"template": "bottle_holder", "params": {...}}`. The Streamlit app routes this to the geometry function and renders the result.

The chat interface replaces the slider panel as the primary UI. Sliders remain accessible behind a "show parameters" toggle for power users and debugging.

### Step 4 — Manufacturing checks + STEP export
Each template can specify which checks apply. Common checks:
- `min_wall_thickness(part, threshold)` — flags faces too thin to print
- `check_overhangs(part, max_angle=45)` — for FDM printability
- Template-specific checks declared in template metadata

STL and STEP export both available as download buttons after generation.

### Step 5 — Roadmap UI + polish
Add a "Coming next" section to the sidebar listing future templates as greyed-out cards with brief descriptions. Examples: drawer organizer, knob, plant pot insert, cable clip, phone mount, license plate. Clicking a future template shows a "this template is in development" message.

This is *not* functional code — it's narrative UI that demonstrates the platform vision.

Final polish: README, deploy to Streamlit Cloud.

### Step 6 — Documentation
PRD (1-2 pages), TDD (1-2 pages), 90-second demo video.

## Template specifications

### Template 1: Bottle Holder ✅ Implemented

Mounts a cylindrical bottle to a horizontal bar via a clamp. See `templates/bottle_holder.py`.

Parameters: bottle_dia, cup_id, cup_height, wall_t, drain_dia, bar_dia, clamp_height, slot_width, standoff, arm_width.

Typical use cases: drink holder for ride-on toys, treadmill bottle holder, stroller cup holder, gym equipment accessory.

### Template 2: Hook (TO IMPLEMENT)

A J-shaped hook that mounts to either a flat surface (with screw holes) or to a bar (with a clamp like the bottle holder's). Holds an object that hangs from it.

**Geometry:**
- A mounting plate or clamp (user picks at parameter time)
- An arm extending outward (length: `arm_length`)
- A J-curve at the end (radius `hook_radius`, opening width `opening`) holding the hung object

**Parameters:**
| Name | Type | Default | Description |
|---|---|---|---|
| `mount_type` | str | "flat" | Either "flat" (screw holes) or "bar" (clamp) |
| `mount_dim` | float | 30.0 | If flat: plate width/height; if bar: bar diameter |
| `arm_length` | float | 50.0 | How far the hook sticks out from mount |
| `hook_radius` | float | 12.0 | Inside radius of the J-curve |
| `opening` | float | 18.0 | Gap at the J-curve opening |
| `wall_t` | float | 3.0 | Stock thickness throughout |
| `screw_dia` | float | 4.5 | If flat mount, screw hole diameter (for #8 screws) |

**Typical use cases:** hang a broom, headphones holder, dog leash hook, key hook, plant hanger, towel hook.

### Template 3: L-Bracket (TO IMPLEMENT)

A right-angle bracket that joins two perpendicular surfaces. Used for shelf supports, corner reinforcements, mounting electronics.

**Geometry:**
- Two flat plates meeting at 90 degrees
- Each plate has a configurable number of mounting holes
- Optional triangular gusset between the plates for stiffness

**Parameters:**
| Name | Type | Default | Description |
|---|---|---|---|
| `plate_a_length` | float | 60.0 | Length of first plate |
| `plate_a_width` | float | 30.0 | Width of first plate |
| `plate_b_length` | float | 60.0 | Length of second plate |
| `plate_b_width` | float | 30.0 | Width of second plate |
| `thickness` | float | 4.0 | Plate thickness |
| `holes_a` | int | 2 | Number of mounting holes in plate A |
| `holes_b` | int | 2 | Number of mounting holes in plate B |
| `hole_dia` | float | 4.5 | Mounting hole diameter |
| `gusset` | bool | True | Whether to add a stiffening gusset |

**Typical use cases:** shelf bracket, monitor mount, under-desk cable bracket, joining wood pieces, reinforcing a corner.

## Conversation design

The agent's persona: a senior product engineer at a hardware company doing intake on a custom request. Curious, focused, asks one clear question at a time, never lectures, never asks for information already given.

The system prompt covers:
- Engineer persona and conversation style.
- Brief description of each available template and when each fits.
- Routing logic for handling ambiguity ("sounds like either a hook or a bracket — what's the orientation of the surface you're mounting to?").
- Parameter intake rules per template (only ask for parameters that exist for the chosen template).
- Image handling: acknowledge what's visible, never invent dimensions, always ask measurements explicitly.
- Confirmation pattern: summarize the parameter set before generating.
- Iteration handling: after generation, accept changes ("make it deeper") and update the parameter dict in place.

The agent communicates with the geometry layer via a structured tool call:
```json
{
  "template": "bottle_holder | hook | bracket",
  "params": { ... template-specific parameter values ... }
}
```

The app validates the tool call before execution: template must exist, params must match the template's schema, validation must pass.

## Engineering principles for the agent

- **Small commits.** Each meaningful change in its own commit.
- **Test geometry before claiming done.** Every template change must produce a valid STL that opens in a viewer.
- **Never let LLM output reach the geometry layer unchecked.**
- **Each template is self-contained.** A new template doesn't require modifying existing ones.
- **Prefer clarity over cleverness.**
- **Surface failure modes.**
- **No silent fallbacks.** If a template doesn't exist, raise; don't silently substitute.

## Known constraints and non-goals

- **Three templates in this build.** More templates are roadmap, not deliverable.
- **Photos are context, not measurement source.** Mode A only.
- **Single-user, single-session.** No accounts, no persistence.
- **English only.**
- **The library scales by adding templates, each hand-coded.** This is intentional design — every template encodes design-for-manufacture expertise that black-box generation can't match.
