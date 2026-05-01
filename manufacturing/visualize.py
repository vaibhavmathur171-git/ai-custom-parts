"""
Annotated thumbnails for manufacturing warnings.

When a check fingerprints specific problem geometry (overhanging faces, the
worst face-pair on a thin wall), this module renders a small PNG showing
the part with the problem regions highlighted in vivid opaque red against
a soft translucent blue body. The PNG is what the user sees inside the
warning expander.

Design priorities, in order:
1. Red regions are the visual focal point — opaque, saturated, and rendered
   on top of the rest of the part.
2. Camera direction is derived from the problem-face normals, so the view
   automatically frames whatever the problem actually is — overhang
   undersides come into view from below, side-facing thin walls come into
   view from the side, etc.
3. The non-problem geometry fades back: lighter blue, ~70% alpha, no
   wireframe, soft Lambert shading so the part still reads as a 3D object.

We use matplotlib's mplot3d because it's pinned in requirements.txt and
deterministic across machines. The renderer never raises into the caller —
failures return None and the UI gracefully falls back to text-only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

# Lazy: matplotlib is heavy and we want imports of this module to stay
# cheap. We import inside the renderer.

# --- Colors --------------------------------------------------------------

# Soft, slightly desaturated blue for the non-problem body. Quite
# translucent — matplotlib uses painter's-algorithm depth sorting, so
# stacked translucent polys composite multiplicatively and a body alpha
# at the spec'd 0.6–0.8 ends up fully opaque after 4–5 layers. ~0.32 keeps
# the body silhouette visible while letting red problem faces inside the
# part read through cleanly.
_BODY_RGB = (0.576, 0.773, 0.992)     # #93c5fd
_BODY_ALPHA = 0.32

# Vivid, fully opaque red for problem faces — focal point of the image.
_PROBLEM_RGB = (0.863, 0.149, 0.149)  # #dc2626
_PROBLEM_ALPHA = 1.0

# Off-white background to match the app's viewport panel.
_BG = "#fafafa"

# Soft directional light from upper-front-left. Used for per-triangle
# Lambert shading so the part reads as a 3D object instead of flat color.
_LIGHT_DIR = (-0.45, -0.55, 0.72)

# Map the human-facing check name to a short label rendered in the corner
# of the thumbnail. Optional — empty string means no label is drawn.
_CHECK_TYPE_LABELS: dict[str, str] = {
    "Overhangs": "overhang",
    "Minimum wall thickness": "thin wall",
    "Wall thickness uniformity": "thick wall",
}


def _label_for_check(check_name: str) -> str:
    return _CHECK_TYPE_LABELS.get(check_name, "")


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


def _camera_from_problem(problem_faces, part_center, np_module) -> tuple[float, float]:
    """Pick (elev, azim) so the camera looks at the problem region with the
    least body occlusion.

    Combines two signals:
    - Average problem normal → preferred view direction (we want to look
      back along it so the problem faces present maximum projected area).
    - Problem-face centroid relative to the part center → picks the azimuth
      when normals are nearly pole-aligned (overhang on a downward face has
      avg normal (0,0,-1), which carries no XY information; we instead
      orbit the camera to the side of the part where the problem is).
    """
    np = np_module
    try:
        normal_acc = np.zeros(3, dtype=float)
        centroid_acc = np.zeros(3, dtype=float)
        n_faces = 0
        for f in problem_faces:
            try:
                n = f.normal_at(f.center())
                c = f.center()
            except Exception:
                continue
            normal_acc += np.array([n.X, n.Y, n.Z], dtype=float)
            centroid_acc += np.array([c.X, c.Y, c.Z], dtype=float)
            n_faces += 1
        if n_faces == 0:
            return 22.0, -55.0
        centroid = centroid_acc / n_faces
        offset = centroid - np.array(part_center, dtype=float)
        offset_xy = float(np.hypot(offset[0], offset[1]))
        normal_mag = float(np.linalg.norm(normal_acc))

        # Pick a base "look-at" azimuth from whichever signal is strongest:
        # the offset to the problem region (best when the problem is on
        # one side of the part), then average normal XY, then a default.
        if offset_xy > 1e-3:
            base_azim = float(np.degrees(np.arctan2(offset[1], offset[0])))
        elif normal_mag > 0.05:
            avg_n = normal_acc / normal_mag
            if float(np.hypot(avg_n[0], avg_n[1])) > 1e-3:
                base_azim = float(np.degrees(np.arctan2(avg_n[1], avg_n[0])))
            else:
                base_azim = 0.0
        else:
            base_azim = 0.0
        # Rotate 55° off the look-at direction so the camera ends up at a
        # 3/4 view of the problem region — head-on flattens the geometry
        # and obscures depth.
        azim = base_azim - 55.0

        # Elevation — driven by the average normal's vertical component.
        # Strong downward normal (overhang) → camera below part, looking
        # up. Strong upward normal → camera above. Sideways normal →
        # gentle tilt.
        if normal_mag > 0.05:
            avg_n = normal_acc / normal_mag
            z = float(avg_n[2])
            horiz = float(np.hypot(avg_n[0], avg_n[1]))
            if horiz < 0.25:
                # Pole-aligned: tilt 40° off-pole — enough to clearly see
                # the underside and the side silhouette together.
                elev = -40.0 if z < 0 else 40.0
            else:
                elev = float(np.degrees(np.arcsin(max(-1.0, min(1.0, z)))))
                elev = float(max(-40.0, min(40.0, elev * 0.55)))
        else:
            elev = 22.0
        return elev, azim
    except Exception:
        return 22.0, -55.0


def _shade(rgb: tuple[float, float, float], brightness: float) -> tuple[float, float, float]:
    """Lambert-style shade: scale the RGB by `brightness` and clamp."""
    return (
        max(0.0, min(1.0, rgb[0] * brightness)),
        max(0.0, min(1.0, rgb[1] * brightness)),
        max(0.0, min(1.0, rgb[2] * brightness)),
    )


def render_problem_thumbnail(
    part,
    problem_faces: Iterable,
    output_path: str | Path,
    *,
    size_px: int = 600,
    tolerance: float = 0.2,
    check_type: str = "",
) -> Path | None:
    """Render `part` with `problem_faces` highlighted in opaque red.

    Returns the output path on success, or None on any failure (matplotlib
    error, no problem faces, empty geometry). Callers must treat `None` as
    "render fell back, just show text".

    Parameters
    ----------
    size_px
        Square output edge in pixels. Default 600 — Streamlit displays the
        result at 300px width, so 600 gives a crisp 2× look.
    tolerance
        build123d Face.tessellate linear tolerance in mm. Lower = smoother
        cylinders, more triangles. 0.3 is a good balance for our parts.
    check_type
        Short label drawn in the upper-left corner — e.g. "overhang",
        "thin wall". Empty string suppresses the label.
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

    light = np.array(_LIGHT_DIR, dtype=float)
    light = light / max(float(np.linalg.norm(light)), 1e-9)

    body_polys: list = []
    body_colors: list = []
    problem_polys: list = []
    problem_colors: list = []

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
            is_problem = fp in problem_fps
            pts = np.array([(v.X, v.Y, v.Z) for v in verts], dtype=float)
            for tri in tris:
                tri_pts = pts[list(tri)]
                # Right-handed face normal of this triangle.
                edge1 = tri_pts[1] - tri_pts[0]
                edge2 = tri_pts[2] - tri_pts[0]
                n = np.cross(edge1, edge2)
                n_len = float(np.linalg.norm(n))
                if n_len < 1e-12:
                    continue
                n = n / n_len
                # Lambert + ambient. Ambient term keeps shadowed faces from
                # going black; diffuse term adds form.
                lambert = max(0.0, float(np.dot(n, light)))
                brightness = 0.62 + 0.38 * lambert
                if is_problem:
                    rgb = _shade(_PROBLEM_RGB, brightness)
                    problem_polys.append(tri_pts)
                    problem_colors.append(rgb + (_PROBLEM_ALPHA,))
                else:
                    rgb = _shade(_BODY_RGB, brightness)
                    body_polys.append(tri_pts)
                    body_colors.append(rgb + (_BODY_ALPHA,))
    except Exception:
        return None

    if not body_polys and not problem_polys:
        return None

    bbox = part.bounding_box()
    part_center = (
        (bbox.min.X + bbox.max.X) / 2.0,
        (bbox.min.Y + bbox.max.Y) / 2.0,
        (bbox.min.Z + bbox.max.Z) / 2.0,
    )
    elev, azim = _camera_from_problem(problem_list, part_center, np)

    try:
        # matplotlib uses inches × dpi for pixel output. dpi=150 with 4-inch
        # figure gives 600 px square.
        dpi = 150
        size_in = size_px / float(dpi)
        fig = plt.figure(figsize=(size_in, size_in), facecolor=_BG, dpi=dpi)
        ax = fig.add_subplot(111, projection="3d", facecolor=_BG)
        ax.set_axis_off()

        # Hide the panes and gridlines that mpl 3D draws by default. These
        # are the boxy walls behind the part — pure noise here.
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.pane.set_visible(False)
                axis.pane.set_edgecolor((1, 1, 1, 0))
                axis.pane.set_facecolor((1, 1, 1, 0))
                axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
            except Exception:
                pass

        if body_polys:
            body_coll = Poly3DCollection(
                body_polys,
                facecolors=body_colors,
                edgecolors="none",
            )
            body_coll.set_zorder(1)
            ax.add_collection3d(body_coll)
        if problem_polys:
            problem_coll = Poly3DCollection(
                problem_polys,
                facecolors=problem_colors,
                edgecolors="none",
            )
            # Force the problem faces to draw last so they sit visually on
            # top of the translucent body.
            problem_coll.set_zorder(20)
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
        ax.view_init(elev=elev, azim=azim)

        # Optional small corner label naming the problem type. Falls back
        # silently if the figure is too small to fit it.
        label = check_type or _label_for_check("")  # caller may pass it directly
        if label:
            try:
                fig.text(
                    0.04, 0.95, label,
                    fontsize=11, fontweight="bold",
                    color="#b91c1c",
                    family="DejaVu Sans Mono",
                    ha="left", va="top",
                )
            except Exception:
                pass

        out = Path(output_path)
        out.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.05, facecolor=_BG)
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
        rendered = render_problem_thumbnail(
            part, problem, out_path,
            check_type=_label_for_check(r.name),
        )
        if rendered is not None:
            r.thumbnail_path = str(rendered)
