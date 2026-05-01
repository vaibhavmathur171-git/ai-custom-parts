"""
Template registry.

Each template module exposes:
- a parameter dataclass with a `validate()` method
- a geometry function `make_<thing>(params) -> Part`
- a `META` dict with `name`, `description`, `typical_use_cases`

The registry binds these together so the configurator UI and (later) the
conversation agent can enumerate templates, look up a template by name, and
obtain a default parameter instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type

from build123d import Part

from templates import bottle_holder, bracket, hook


@dataclass(frozen=True)
class TemplateSpec:
    """Everything the app needs to drive a template end-to-end."""

    name: str
    description: str
    typical_use_cases: list[str]
    params_class: Type
    make_fn: Callable[[object], Part]
    default_params: object
    # Manufacturing metadata.
    applicable_checks: tuple[str, ...] = ()
    wall_param: str | None = None


def _spec(module, params_class, make_fn) -> TemplateSpec:
    meta = module.META
    return TemplateSpec(
        name=meta["name"],
        description=meta["description"],
        typical_use_cases=list(meta["typical_use_cases"]),
        params_class=params_class,
        make_fn=make_fn,
        default_params=params_class(),
        applicable_checks=tuple(meta.get("applicable_checks", ())),
        wall_param=meta.get("wall_param"),
    )


# Order here is the order templates appear in the UI selector.
_REGISTRY: dict[str, TemplateSpec] = {}
for _spec_obj in (
    _spec(bottle_holder, bottle_holder.BottleHolderParams, bottle_holder.make_bottle_holder),
    _spec(hook, hook.HookParams, hook.make_hook),
    _spec(bracket, bracket.BracketParams, bracket.make_bracket),
):
    _REGISTRY[_spec_obj.name] = _spec_obj


def list_templates() -> list[TemplateSpec]:
    """All registered templates in registration order."""
    return list(_REGISTRY.values())


def get_template(name: str) -> TemplateSpec:
    """Look up a template by name. Raises KeyError if missing — no silent fallback."""
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys()) or "(none)"
        raise KeyError(f"Template '{name}' not registered. Available: {available}")
    return _REGISTRY[name]


def get_default_params(name: str):
    """Default parameter instance for a template."""
    return get_template(name).default_params
