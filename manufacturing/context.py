"""
Production context: how the user plans to actually make the part.

The `ProductionContext` rides alongside the conversation so the geometry
layer can pick the right manufacturing rules, the agent can push back on
parameters that would fail in production, and the UI can show the user
which checks apply.

This module owns:
- the `Method` enum (FDM, SLA, CNC, ...)
- `ProductionContext` (method + material + nozzle + free-text notes)
- a small registry of well-known printers so the agent can recognize
  "Bambu A1" or "Elegoo Centauri Carbon" without re-asking
- per-method `Thresholds` used by manufacturing/checks.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Method(str, Enum):
    FDM = "fdm"
    SLA = "sla"
    ONLINE_SERVICE = "online_service"
    CNC_PLASTIC = "cnc_plastic"
    CNC_METAL = "cnc_metal"
    INJECTION_MOLDING = "injection_molding"


METHOD_LABEL = {
    Method.FDM: "FDM 3D printing",
    Method.SLA: "SLA / resin printing",
    Method.ONLINE_SERVICE: "online printing service",
    Method.CNC_PLASTIC: "CNC machining (plastic)",
    Method.CNC_METAL: "CNC machining (metal)",
    Method.INJECTION_MOLDING: "injection molding",
}


@dataclass
class ProductionContext:
    """How the user plans to produce the part. All fields except `method`
    are optional — the agent fills in what it knows."""

    method: Method = Method.FDM
    material: str | None = None
    nozzle_dia: float | None = None
    printer_model: str | None = None
    notes: str = ""

    @classmethod
    def safe_default(cls) -> "ProductionContext":
        """Used when the user says 'not sure' or hasn't been asked yet."""
        return cls(
            method=Method.FDM,
            material="PLA",
            nozzle_dia=0.4,
            notes="default — user did not specify, FDM/PLA assumed",
        )

    def label(self) -> str:
        """Short human label for the UI, e.g. 'FDM 3D printing — PLA, Bambu A1'."""
        bits: list[str] = [METHOD_LABEL[self.method]]
        if self.material:
            bits.append(self.material)
        if self.printer_model:
            bits.append(self.printer_model)
        return " — ".join([bits[0]] + ([", ".join(bits[1:])] if len(bits) > 1 else []))


@dataclass(frozen=True)
class Thresholds:
    """Per-method numeric rules used by the checks layer."""

    method: Method
    method_label: str
    wall_fail_below: float        # < this is a hard fail
    wall_warn_below: float        # < this is a warning (>= wall_fail_below)
    wall_excessive_above: float | None  # injection molding: too thick is also a warn
    overhang_max_deg: float | None      # None = method doesn't care about overhangs
    hole_edge_factor: float        # multiple of hole_dia required from edge
    min_internal_corner_radius: float | None  # CNC tool radius minimum
    draft_required: bool           # injection molding
    default_wall_recommendation: float  # what the agent should suggest as a starting wall


THRESHOLDS: dict[Method, Thresholds] = {
    Method.FDM: Thresholds(
        method=Method.FDM,
        method_label="FDM 3D printing",
        wall_fail_below=0.8,
        wall_warn_below=1.2,
        wall_excessive_above=None,
        overhang_max_deg=45.0,
        hole_edge_factor=1.5,
        min_internal_corner_radius=None,
        draft_required=False,
        default_wall_recommendation=2.0,
    ),
    Method.SLA: Thresholds(
        method=Method.SLA,
        method_label="SLA / resin printing",
        wall_fail_below=0.6,
        wall_warn_below=0.8,
        wall_excessive_above=None,
        overhang_max_deg=None,
        hole_edge_factor=1.2,
        min_internal_corner_radius=None,
        draft_required=False,
        default_wall_recommendation=1.5,
    ),
    Method.ONLINE_SERVICE: Thresholds(
        method=Method.ONLINE_SERVICE,
        method_label="online printing service",
        wall_fail_below=0.8,
        wall_warn_below=1.2,
        wall_excessive_above=None,
        overhang_max_deg=45.0,
        hole_edge_factor=1.5,
        min_internal_corner_radius=None,
        draft_required=False,
        default_wall_recommendation=2.0,
    ),
    Method.CNC_PLASTIC: Thresholds(
        method=Method.CNC_PLASTIC,
        method_label="CNC machining (plastic)",
        wall_fail_below=1.0,
        wall_warn_below=1.5,
        wall_excessive_above=None,
        overhang_max_deg=None,
        hole_edge_factor=2.0,
        min_internal_corner_radius=1.0,
        draft_required=False,
        default_wall_recommendation=1.5,
    ),
    Method.CNC_METAL: Thresholds(
        method=Method.CNC_METAL,
        method_label="CNC machining (metal)",
        wall_fail_below=0.8,
        wall_warn_below=1.0,
        wall_excessive_above=None,
        overhang_max_deg=None,
        hole_edge_factor=2.5,
        min_internal_corner_radius=1.0,
        draft_required=False,
        default_wall_recommendation=1.5,
    ),
    Method.INJECTION_MOLDING: Thresholds(
        method=Method.INJECTION_MOLDING,
        method_label="injection molding",
        wall_fail_below=1.0,
        wall_warn_below=1.5,
        wall_excessive_above=3.0,
        overhang_max_deg=None,
        hole_edge_factor=2.5,
        min_internal_corner_radius=None,
        draft_required=True,
        default_wall_recommendation=2.5,
    ),
}


def thresholds_for(context: ProductionContext) -> Thresholds:
    return THRESHOLDS[context.method]


# ---------------------------------------------------------------------------
# Printer registry: the agent uses this to recognize a printer name from the
# user and skip redundant questions. Keys are lower-case substrings; the first
# entry that matches the user's text wins.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrinterInfo:
    canonical_name: str
    method: Method
    nozzle_dia: float | None = None  # None for resin printers


PRINTER_REGISTRY: dict[str, PrinterInfo] = {
    "bambu a1": PrinterInfo("Bambu A1", Method.FDM, 0.4),
    "bambu p1s": PrinterInfo("Bambu P1S", Method.FDM, 0.4),
    "bambu p1p": PrinterInfo("Bambu P1P", Method.FDM, 0.4),
    "bambu x1c": PrinterInfo("Bambu X1C", Method.FDM, 0.4),
    "bambu x1": PrinterInfo("Bambu X1", Method.FDM, 0.4),
    "prusa mk4": PrinterInfo("Prusa MK4", Method.FDM, 0.4),
    "prusa mk3": PrinterInfo("Prusa MK3", Method.FDM, 0.4),
    "prusa core one": PrinterInfo("Prusa Core One", Method.FDM, 0.4),
    "prusa xl": PrinterInfo("Prusa XL", Method.FDM, 0.4),
    "elegoo centauri carbon": PrinterInfo("Elegoo Centauri Carbon", Method.FDM, 0.4),
    "centauri carbon": PrinterInfo("Elegoo Centauri Carbon", Method.FDM, 0.4),
    "elegoo neptune": PrinterInfo("Elegoo Neptune", Method.FDM, 0.4),
    "creality ender 3": PrinterInfo("Creality Ender 3", Method.FDM, 0.4),
    "creality k1": PrinterInfo("Creality K1", Method.FDM, 0.4),
    "creality k2": PrinterInfo("Creality K2", Method.FDM, 0.4),
    "voron 2.4": PrinterInfo("Voron 2.4", Method.FDM, 0.4),
    "voron trident": PrinterInfo("Voron Trident", Method.FDM, 0.4),
    "voron 0": PrinterInfo("Voron 0", Method.FDM, 0.4),
    "anycubic photon": PrinterInfo("Anycubic Photon", Method.SLA, None),
    "elegoo mars": PrinterInfo("Elegoo Mars", Method.SLA, None),
    "elegoo saturn": PrinterInfo("Elegoo Saturn", Method.SLA, None),
    "formlabs form": PrinterInfo("Formlabs Form", Method.SLA, None),
}


def resolve_printer(text: str) -> PrinterInfo | None:
    """Look up a printer model from a substring of user-provided text.

    Case-insensitive. The first registry key that appears as a substring wins.
    Longer keys are checked first so 'centauri carbon' wins over partial
    matches against 'elegoo'.
    """
    if not text:
        return None
    needle = text.lower()
    # Sort by descending length so the most specific match wins.
    for key in sorted(PRINTER_REGISTRY.keys(), key=len, reverse=True):
        if key in needle:
            return PRINTER_REGISTRY[key]
    return None


# ---------------------------------------------------------------------------
# Convenience NL-parsing helpers (used mainly by the agent's tool handler
# and by tests). The agent itself does most of the parsing, but these give
# us a deterministic fallback for common phrases.
# ---------------------------------------------------------------------------


_UNSURE_PHRASES = (
    "not sure",
    "no idea",
    "don't know",
    "dont know",
    "no clue",
    "whatever",
    "just print",
    "default",
    "i dunno",
)


def looks_unsure(text: str) -> bool:
    """True if the user's answer signals they don't know — agent should use safe defaults."""
    if not text:
        return False
    needle = text.lower()
    return any(p in needle for p in _UNSURE_PHRASES)
