"""Procedural geometry.

Vertices are interleaved: position(3) + normal(3) + color(3) + emissive(1)
+ material(2). ``emissive`` is 0 for normally-lit surfaces and up to 1 for
self-illuminated ones (e.g. destination signs, headlights, the odd lit
window). ``material`` packs (gloss, translucency): gloss 0 = rough matte
(soil, grass), 1 = polished (train paint, glass, rail steel); translucency
lets sunlight bleed through thin volumes (tree canopies).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLOATS_PER_VERTEX = 12


@dataclass
class Mesh:
    vertices: np.ndarray  # shape (N, 12) float32
    indices: np.ndarray   # shape (M,) uint32

    @property
    def index_count(self) -> int:
        return int(self.indices.shape[0])

    def vertex_bytes(self) -> bytes:
        return np.ascontiguousarray(self.vertices, dtype=np.float32).tobytes()

    def index_bytes(self) -> bytes:
        return np.ascontiguousarray(self.indices, dtype=np.uint32).tobytes()


# Six faces: (normal, four corner offsets in CCW order seen from outside)
_FACES = [
    ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),      # +Z
    ((0, 0, -1), [(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)]),  # -Z
    ((1, 0, 0), [(1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1)]),       # +X
    ((-1, 0, 0), [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)]),  # -X
    ((0, 1, 0), [(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1)]),       # +Y
    ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),  # -Y
]


def make_box(size=(1.0, 1.0, 1.0), color=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0),
             emissive=0.0, gloss=0.0, translucency=0.0) -> Mesh:
    """Axis-aligned box. ``size`` is the full extent along (x, y, z)."""
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    cx, cy, cz = center
    verts = []
    idx = []
    for normal, corners in _FACES:
        base = len(verts)
        for ox, oy, oz in corners:
            verts.append(
                [cx + ox * hx, cy + oy * hy, cz + oz * hz, *normal, *color,
                 emissive, gloss, translucency]
            )
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))


def make_quad(corners, normal, color, emissive=0.0, gloss=0.0, translucency=0.0) -> Mesh:
    """A single quad from four CCW corner points (glm.vec3-like)."""
    verts = [
        [c[0], c[1], c[2], *normal, *color, emissive, gloss, translucency]
        for c in corners
    ]
    idx = [0, 1, 2, 0, 2, 3]
    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))


def make_ground(size, center=(0.0, 0.0), color=(0.3, 0.4, 0.3), y=0.0, emissive=0.0,
                gloss=0.0) -> Mesh:
    """A flat ground quad on the XZ plane, normal +Y."""
    cx, cz = center
    h = size / 2
    n = (0.0, 1.0, 0.0)
    verts = [
        [cx - h, y, cz - h, *n, *color, emissive, gloss, 0.0],
        [cx + h, y, cz - h, *n, *color, emissive, gloss, 0.0],
        [cx + h, y, cz + h, *n, *color, emissive, gloss, 0.0],
        [cx - h, y, cz + h, *n, *color, emissive, gloss, 0.0],
    ]
    idx = [0, 1, 2, 0, 2, 3]
    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))


def _loop_perps(points):
    """Unit XZ-perpendicular at each vertex of a closed polyline (list of vec3-likes)."""
    import glm

    n = len(points)
    perps = []
    for i in range(n):
        prev = points[(i - 1) % n]
        cur = points[i]
        nxt = points[(i + 1) % n]
        din = glm.normalize(cur - prev) if glm.length(cur - prev) > 0 else glm.vec3(1, 0, 0)
        dout = glm.normalize(nxt - cur) if glm.length(nxt - cur) > 0 else din
        tangent = din + dout
        if glm.length(tangent) < 1e-6:
            tangent = dout
        tangent = glm.normalize(tangent)
        perp = glm.normalize(glm.vec3(-tangent.z, 0.0, tangent.x))
        perps.append(perp)
    return perps


def make_loop_ribbon(points, half_width, y, color, gloss=0.0) -> Mesh:
    """A flat ribbon (normal +Y) of the given half-width following a closed loop.

    ``points`` is a list of glm.vec3 (ground positions). Used for the track bed
    and the two rails.
    """
    perps = _loop_perps(points)
    n = len(points)
    up = (0.0, 1.0, 0.0)
    verts = []
    for i in range(n):
        p = points[i]
        perp = perps[i]
        left = p + perp * half_width
        right = p - perp * half_width
        verts.append([left.x, y, left.z, *up, *color, 0.0, gloss, 0.0])
        verts.append([right.x, y, right.z, *up, *color, 0.0, gloss, 0.0])
    idx = []
    for i in range(n):
        a = 2 * i          # left_i
        b = 2 * i + 1      # right_i
        c = 2 * ((i + 1) % n)      # left_{i+1}
        d = 2 * ((i + 1) % n) + 1  # right_{i+1}
        idx += [a, b, d, a, d, c]
    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))


def offset_loop(points, distance):
    """Return points shifted along their per-vertex perpendicular by ``distance``."""
    perps = _loop_perps(points)
    return [points[i] + perps[i] * distance for i in range(len(points))]


def combine(meshes) -> Mesh:
    """Merge several meshes into one, offsetting indices."""
    all_verts = []
    all_idx = []
    offset = 0
    for m in meshes:
        all_verts.append(m.vertices)
        all_idx.append(m.indices + offset)
        offset += m.vertices.shape[0]
    return Mesh(
        np.concatenate(all_verts, axis=0).astype(np.float32),
        np.concatenate(all_idx, axis=0).astype(np.uint32),
    )
