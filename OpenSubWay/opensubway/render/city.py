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
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
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
    parts.append(mesh.make_box(size=(w, h, d), color=base, center=(cx, h / 2, cz),
                               gloss=0.12))
    # A darker flat-roof cap (tar/gravel look under daylight).
    parts.append(
        mesh.make_box(size=(w * 0.96, 0.6, d * 0.96),
                      color=(base[0] * 0.55, base[1] * 0.55, base[2] * 0.55),
                      center=(cx, h + 0.3, cz))
    )
    _windows_along_z(parts, cx + w / 2, 1.0, cx, cz, w, d, h, rng)
    _windows_along_z(parts, cx - w / 2, -1.0, cx, cz, w, d, h, rng)
    _windows_along_x(parts, cz + d / 2, 1.0, cx, cz, w, d, h, rng)
    _windows_along_x(parts, cz - d / 2, -1.0, cx, cz, w, d, h, rng)


def build_city_mesh(line) -> Mesh:
    rng = random.Random(config.CITY_SEED)
    c = line.centroid()
    half = config.CITY_RADIUS
    step = config.CITY_BLOCK
    parts: list[Mesh] = []

    n = int((2 * half) / step) + 1
    for ix in range(n):
        x = c.x - half + ix * step
        for iz in range(n):
            z = c.z - half + iz * step
            bx = x + rng.uniform(-2.0, 2.0)
            bz = z + rng.uniform(-2.0, 2.0)
            if _dist_to_track(bx, bz, line.points) <= config.CITY_TRACK_CLEARANCE:
                continue
            if rng.random() > 0.82:
                continue
            w = rng.uniform(step * 0.5, step * 0.82)
            d = rng.uniform(step * 0.5, step * 0.82)
            roll = rng.random()
            if roll < 0.5:
                h = rng.uniform(9.0, 18.0)
            elif roll < 0.85:
                h = rng.uniform(20.0, 38.0)
            else:
                h = rng.uniform(42.0, 70.0)
            _building(parts, bx, bz, w, d, h, rng)

    if not parts:  # safety (shouldn't happen)
        parts.append(mesh.make_box(size=(1, 1, 1), color=(0.1, 0.1, 0.1)))
    return mesh.combine(parts)
