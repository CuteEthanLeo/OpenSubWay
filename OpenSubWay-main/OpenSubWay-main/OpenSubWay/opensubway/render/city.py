"""Procedural daytime city: buildings with glass windows beside the track.

A grid of city blocks fills the ground around the line; blocks that fall on the
rail corridor are left open so the railway runs through the city. Windows are
dark glass quads (they pick up specular/fresnel sky sheen from the shader); a
few are lit from inside. Half the window slots are omitted to keep the
triangle count down — it reads as facade variation.
"""

from __future__ import annotations

import math
import random

from .. import config
from . import mesh
from .mesh import Mesh

_BASE_COLORS = [
    (0.62, 0.58, 0.52),   # sandstone
    (0.68, 0.64, 0.58),   # cream render
    (0.55, 0.42, 0.36),   # brick
    (0.58, 0.60, 0.63),   # concrete
    (0.52, 0.55, 0.58),   # blue-grey panels
]
_GLASS = (0.25, 0.33, 0.45)     # dark daytime glass (specular does the rest)
_LIT = (0.95, 0.85, 0.60)       # the odd interior light


def _dist_to_track(px, pz, pts):
    best = 1e18
    n = len(pts)
    for i in range(n - 1):
        a = pts[i]
        b = pts[i + 1]
        dx, dz = b.x - a.x, b.z - a.z
        l2 = dx * dx + dz * dz
        if l2 <= 1e-9:
            t = 0.0
        else:
            t = ((px - a.x) * dx + (pz - a.z) * dz) / l2
            t = max(0.0, min(1.0, t))
        cx, cz = a.x + t * dx, a.z + t * dz
        best = min(best, math.hypot(px - cx, pz - cz))
    return best


def _window(rng):
    """Return (color, emissive) — mostly dark glass, occasionally lit inside."""
    if rng.random() < 0.08:
        return _LIT, 0.45
    return _GLASS, 0.0


def _windows_along_z(parts, xface, nx, cx, cz, w, d, h, rng):
    """Windows on an X-facing wall (varies along Z, up Y)."""
    cols = min(5, max(1, int((d - 1.5) / 2.6)))
    rows = min(16, max(1, int((h - 2.0) / 3.0)))
    x = xface + 0.06 * nx
    for r in range(rows):
        yc = 1.6 + r * 3.0
        if yc + 0.8 > h:
            break
        for c in range(cols):
            if rng.random() < 0.5:
                continue
            zc = cz - d / 2 + (c + 0.5) * (d / cols)
            z0, z1 = zc - 0.55, zc + 0.55
            y0, y1 = yc - 0.7, yc + 0.7
            corners = [(x, y0, z0), (x, y0, z1), (x, y1, z1), (x, y1, z0)]
            col, emi = _window(rng)
            parts.append(mesh.make_quad(corners, (nx, 0.0, 0.0), col, emi, gloss=0.95))


def _windows_along_x(parts, zface, nz, cx, cz, w, d, h, rng):
    """Windows on a Z-facing wall (varies along X, up Y)."""
    cols = min(5, max(1, int((w - 1.5) / 2.6)))
    rows = min(16, max(1, int((h - 2.0) / 3.0)))
    z = zface + 0.06 * nz
    for r in range(rows):
        yc = 1.6 + r * 3.0
        if yc + 0.8 > h:
            break
        for c in range(cols):
            if rng.random() < 0.5:
                continue
            xc = cx - w / 2 + (c + 0.5) * (w / cols)
            x0, x1 = xc - 0.55, xc + 0.55
            y0, y1 = yc - 0.7, yc + 0.7
            corners = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
            col, emi = _window(rng)
            parts.append(mesh.make_quad(corners, (0.0, 0.0, nz), col, emi, gloss=0.95))


def _building(parts, cx, cz, w, d, h, rng):
    base = rng.choice(_BASE_COLORS)
    radius = min(w, d) * rng.uniform(0.08, 0.18)
    # Rounded-plan facade and a wider podium eliminate the old shoebox skyline.
    parts.append(mesh.make_rounded_panel(
        w, d, h, base, center=(cx, h / 2, cz), normal_axis="y",
        corner_radius=radius, corner_segments=5, gloss=0.15,
    ))
    podium_h = min(5.2, max(2.8, h * 0.15))
    parts.append(mesh.make_rounded_panel(
        w * 1.10, d * 1.10, podium_h,
        tuple(min(1.0, c * 0.88) for c in base),
        center=(cx, podium_h / 2, cz), normal_axis="y",
        corner_radius=radius * 1.08, corner_segments=5, gloss=0.20,
    ))
    # Dark parapet, mechanical penthouse and occasional illuminated crown.
    roof_color = (base[0] * 0.48, base[1] * 0.50, base[2] * 0.52)
    parts.append(mesh.make_rounded_panel(
        w * 0.96, d * 0.96, 0.55, roof_color,
        center=(cx, h + 0.275, cz), normal_axis="y",
        corner_radius=radius * 0.75, corner_segments=4, gloss=0.26,
    ))
    if h > 34.0:
        crown_h = rng.uniform(2.2, 5.8)
        parts.append(mesh.make_rounded_panel(
            w * 0.58, d * 0.58, crown_h,
            (0.22, 0.29, 0.34),
            center=(cx, h + crown_h / 2 + 0.5, cz), normal_axis="y",
            corner_radius=radius * 0.45, corner_segments=4,
            gloss=0.82, emissive=0.06,
        ))
    else:
        for dx in (-w * 0.18, w * 0.18):
            parts.append(mesh.make_rounded_panel(
                min(2.4, w * 0.24), min(2.0, d * 0.24), 0.75,
                (0.30, 0.32, 0.33), center=(cx + dx, h + 0.72, cz),
                normal_axis="y", corner_radius=0.22, corner_segments=4,
                gloss=0.28,
            ))
    # Slender facade fins provide vertical scale and catch highlights at speed.
    for xoff in (-w * 0.43, w * 0.43):
        for zoff in (-d * 0.43, d * 0.43):
            parts.append(mesh.make_cylinder(
                radius=0.075, length=max(1.0, h - 1.2), axis="y", segments=8,
                color=tuple(max(0.05, c * 0.60) for c in base),
                center=(cx + xoff, h * 0.5 + 0.5, cz + zoff), gloss=0.40,
            ))
    _windows_along_z(parts, cx + w / 2, 1.0, cx, cz, w, d, h, rng)
    _windows_along_z(parts, cx - w / 2, -1.0, cx, cz, w, d, h, rng)
    _windows_along_x(parts, cz + d / 2, 1.0, cx, cz, w, d, h, rng)
    _windows_along_x(parts, cz - d / 2, -1.0, cx, cz, w, d, h, rng)


def build_city_mesh(line) -> Mesh:
    rng = random.Random(config.CITY_SEED)
    parts: list[Mesh] = []

    # Stream scenery along the full 59 km corridor rather than filling a huge
    # square grid. This keeps both airport approaches populated without tens of
    # thousands of off-route buildings.
    distance = 0.0
    while distance < line.total:
        pos, fwd = line.position_at(distance)
        # Keep both airport approach/runway zones open.  Dense generic towers
        # here would intersect the authored Hongqiao and Pudong landmarks.
        if distance < 1450.0 or distance > line.total - 1450.0:
            distance += 110.0
            continue
        perp_x, perp_z = -fwd.z, fwd.x
        for side in (-1.0, 1.0):
            rows = 2 if rng.random() < 0.42 else 1
            for row in range(rows):
                offset = 42.0 + row * 42.0 + rng.uniform(0.0, 18.0)
                along = rng.uniform(-45.0, 45.0)
                bx = pos.x + perp_x * side * offset + fwd.x * along
                bz = pos.z + perp_z * side * offset + fwd.z * along
                step = config.CITY_BLOCK
                w = rng.uniform(step * 0.58, step * 1.15)
                depth = rng.uniform(step * 0.58, step * 1.15)
                roll = rng.random()
                if roll < 0.48:
                    h = rng.uniform(10.0, 20.0)
                elif roll < 0.86:
                    h = rng.uniform(22.0, 44.0)
                else:
                    h = rng.uniform(48.0, 86.0)
                _building(parts, bx, bz, w, depth, h, rng)
        distance += (rng.uniform(72.0, 112.0) if distance < 7200.0
                     else rng.uniform(145.0, 220.0))

    if not parts:  # safety (shouldn't happen)
        parts.append(mesh.make_box(size=(1, 1, 1), color=(0.1, 0.1, 0.1)))
    return mesh.combine(parts)
