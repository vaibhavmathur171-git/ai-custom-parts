"""
Annotated thumbnails for manufacturing warnings.

When a check fingerprints specific problem geometry (overhanging faces, the
worst face-pair on a thin wall), this module turns that into a small PNG
showing the part with the problem regions highlighted in semi-transparent
red. The PNG is what the user sees inside the warning expander.

We use matplotlib's mplot3d because it's already a transitive dependency
(via build123d's stack) and it's deterministic across machines. The
renderer never raises into the caller — failures return None and the UI
gracefully falls back to text-only.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

# Lazy: matplotlib is heavy and we want imports of this module to stay
# cheap. We import inside the renderer.

# Colors. Normal faces are translucent so the red problem faces remain
# visible even when they sit on the underside of the part — for overhang
# warnings on a bottle holder, the offenders are bottom-facing and would
# otherwise hide behind the cup body.
_NORMAL_RGBA = (0.231, 0.510, 0.965, 0.45)   # #3b82f6, translucent blue
_PROBLEM_RGBA = (0.863, 0.149, 0.149, 0.92)  # #dc2626, near-opaque red
_EDGE_RGBA = (0.30, 0.30, 0.30, 0.25)
_BG = "#fafafa"


def _face_fingerprint(face) -> tuple:
    """A light identity for a Face we can compare across iterations of
    `part.faces()` (which may return fresh wrapper objects each call).

    We fingerprint by (rounded center, rounded normal, rounded area). This
    is good enough for our parts — no two distinct faces share all three.
    """
    c = face.center()
    n = face.normal_at(c)
    return (
        round(c.X, 3), round(c.Y, 3), round(c.Z, 3),
        round(n.X, 3), round(n.Y, 3), round(n.Z, 3),
        round(face.area, 3),
    )


def _stable_hash(*parts: bytes | str) -> str:
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, str):
            p = p.encode("utf-8")
        h.update(p)
        h.update(b"|")
    return h.hexdigest()[:16]


def thumbnail_cache_key(stl_bytes: bytes, check_name: str, problem_faces: Iterable) -> str:
    """Cache key incorporating part geometry, check identity, and problem set."""
    fps = "|".join(repr(_face_fingerprint(f)) for f in problem_faces)
    return _stable_hash(stl_bytes, check_name, fps)


def render_problem_thumbnail(
    part,
    problem_faces: Iterable,
    output_path: str | Path,
    *,
    size_px: int = 400,
    tolerance: float = 0.5,
) -> Path | None:
    """Render `part` with `problem_faces` highlighted in red.

    Returns the output path on success, or None on any failure (matplotlib
    error, no problem faces, empty geometry). Callers must treat `None` as
    "render fell back, just show text".
    """
    problem_list = list(problem_faces) if problem_faces is not None else []
    if not problem_list:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np
    except Exception:
        return None

    try:
        problem_fps = {_face_fingerprint(f) for f in problem_list}
    except Exception:
        return None

    normal_polys: list = []
    problem_polys: list = []

    try:
        for face in part.faces():
            try:
                verts, tris = face.tessellate(tolerance, 5.0)
            except Exception:
                continue
            if not verts or not tris:
                continue
            try:
                fp = _face_fingerprint(face)
            except Exception:
                fp = None
            target = problem_polys if fp in problem_fps else normal_polys
            pts = np.array([(v.X, v.Y, v.Z) for v in verts])
            for tri in tris:
                target.append(pts[list(tri)])
    except Exception:
        return None

    if not normal_polys and not problem_polys:
        return None

    try:
        bbox = part.bounding_box()
        size_in = size_px / 100.0  # matplotlib uses inches × dpi
        fig = plt.figure(figsize=(size_in, size_in), facecolor=_BG)
        ax = fig.add_subplot(111, projection="3d", facecolor=_BG)
        ax.set_axis_off()

        # Two collections so the red problem faces draw on top of the
        # translucent body rather than getting z-sorted underneath it.
        if normal_polys:
            ax.add_collection3d(Poly3DCollection(
                normal_polys,
                facecolors=[_NORMAL_RGBA] * len(normal_polys),
                edgecolors=_EDGE_RGBA,
                linewidths=0.25,
            ))
        if problem_polys:
            problem_coll = Poly3DCollection(
                problem_polys,
                facecolors=[_PROBLEM_RGBA] * len(problem_polys),
                edgecolors=(0.55, 0.10, 0.10, 0.85),
                linewidths=0.5,
            )
            # Force the problem faces to render last so they sit visually on top.
            problem_coll.set_zorder(10)
            ax.add_collection3d(problem_coll)

        ax.set_xlim(bbox.min.X, bbox.max.X)
        ax.set_ylim(bbox.min.Y, bbox.max.Y)
        ax.set_zlim(bbox.min.Z, bbox.max.Z)
        # Equal-aspect 3D so cylinders don't squash.
        ax.set_box_aspect([
            max(bbox.size.X, 1e-3),
            max(bbox.size.Y, 1e-3),
            max(bbox.size.Z, 1e-3),
        ])
        # Slightly low isometric (elev=-12, azim=-55) so bottom-facing
        # surfaces — the most common kind of overhang — peek into view
        # without flipping the part upside down.
        ax.view_init(elev=-12, azim=-55)

        out = Path(output_path)
        out.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(out, dpi=100, bbox_inches="tight", pad_inches=0.05, facecolor=_BG)
        plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None

    if not out.exists() or out.stat().st_size == 0:
        return None
    return out


def annotate_check_thumbnails(
    results,
    part,
    stl_path: str | Path,
    cache_dir: str | Path,
) -> None:
    """Render annotated PNGs for any warn/fail check with problem_faces and
    write the resulting path back onto `result.thumbnail_path`.

    Cached by hash of (STL bytes + check name + face fingerprints) under
    `cache_dir/thumb_{key}.png`, so the same part + same warning reuses the
    prior PNG across turns. All errors are swallowed — a missing thumbnail
    just means the UI shows text only.
    """
    cache = Path(cache_dir)
    try:
        cache.mkdir(exist_ok=True, parents=True)
        stl_bytes = Path(stl_path).read_bytes()
    except Exception:
        return

    for r in results:
        if getattr(r, "status", "pass") == "pass":
            continue
        problem = getattr(r, "problem_faces", None)
        if not problem:
            continue
        try:
            key = thumbnail_cache_key(stl_bytes, r.name, problem)
        except Exception:
            continue
        out_path = cache / f"thumb_{key}.png"
        if out_path.exists() and out_path.stat().st_size > 0:
            r.thumbnail_path = str(out_path)
            continue
        rendered = render_problem_thumbnail(part, problem, out_path)
        if rendered is not None:
            r.thumbnail_path = str(rendered)
